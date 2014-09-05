#include "BGSSubsenseM.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
/*
*
* Intrinsic parameters for our method are defined here; tuning these for better
* performance should not be required in most cases -- although improvements in
* very specific scenarios are always possible.
*
* Note that the current configuration was used to obtain the results presented
* in our paper, in conjunction with the 2014 CVPRW on Change Detection.
*
*/
//! defines the threshold value(s) used to detect long-term ghosting and trigger the fast edge-based absorption heuristic
#define GHOSTDET_D_MAX (0.010f) // defines 'negligible' change here
#define GHOSTDET_S_MIN (0.995f) // defines the required minimum local foreground saturation value
//! parameter used to scale dynamic distance threshold adjustments ('R(x)')
#define FEEDBACK_R_VAR (0.01f)
//! parameters used to adjust the variation step size of 'v(x)'
#define FEEDBACK_V_INCR  (1.000f)
#define FEEDBACK_V_DECR  (0.100f)
//! parameters used to scale dynamic learning rate adjustments  ('T(x)')
#define FEEDBACK_T_DECR  (0.2500f)
#define FEEDBACK_T_INCR  (0.5000f)
#define FEEDBACK_T_LOWER (2.0000f)
#define FEEDBACK_T_UPPER (256.00f)
//! parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local dist threshold values
#define UNSTABLE_REG_RATIO_MIN 0.100f
#define UNSTABLE_REG_RDIST_MIN 3.000f
//! parameters used to scale the relative LBSP intensity threshold used for internal comparisons
#define LBSPDESC_NONZERO_RATIO_MIN 0.100f
#define LBSPDESC_NONZERO_RATIO_MAX 0.500f
//! parameters used to define model reset/learning rate boosts in our frame-level component
#define FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD 15
#define FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO 8

// local define used for debug purposes only
#define DISPLAY_SUBSENSE_DEBUG_INFO 0
// local define used to specify the default internal frame size
#define DEFAULT_FRAME_SIZE cv::Size(320,240)
// local define used to specify the color dist threshold offset used for unstable regions
#define STAB_COLOR_DIST_OFFSET m_nMinColorDistThreshold/5
// local define used to specify the desc dist threshold offset used for unstable regions
#define UNSTAB_DESC_DIST_OFFSET m_nDescDistThreshold
// local define used to determine the median blur kernel size
#define DEFAULT_MEDIAN_BLUR_KERNEL_SIZE (9)

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;


void BGSSubsenseM::cloneModels()
{
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	w_oUpdateRateFrame = m_oUpdateRateFrame.clone();
	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	w_oDistThresholdFrame = m_oDistThresholdFrame.clone();
	//! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
	w_oVariationModulatorFrame = m_oVariationModulatorFrame.clone();
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	w_oMeanLastDistFrame = m_oMeanLastDistFrame.clone();
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	w_oMeanMinDistFrame_LT = m_oMeanMinDistFrame_LT.clone();
	w_oMeanMinDistFrame_ST = m_oMeanMinDistFrame_ST.clone();
	//! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
	w_oMeanDownSampledLastDistFrame_LT = m_oMeanDownSampledLastDistFrame_LT.clone();
	w_oMeanDownSampledLastDistFrame_ST = m_oMeanDownSampledLastDistFrame_ST.clone();
	//! per-pixel mean raw segmentation results
	w_oMeanRawSegmResFrame_LT = m_oMeanRawSegmResFrame_LT.clone();
	w_oMeanRawSegmResFrame_ST = m_oMeanRawSegmResFrame_ST.clone();
	//! per-pixel mean final segmentation results
	w_oMeanFinalSegmResFrame_LT = m_oMeanFinalSegmResFrame_LT.clone();
	w_oMeanFinalSegmResFrame_ST = m_oMeanFinalSegmResFrame_ST.clone();
	//! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	w_oUnstableRegionMask = m_oUnstableRegionMask.clone();
	//! per-pixel blink detection results ('Z(x)')
	w_oBlinksFrame = m_oBlinksFrame.clone();

	for( int i=0; i<m_nBGSamples; i++)
	{
		w_voBGColorSamples[i] = m_voBGColorSamples[i].clone();
		w_voBGDescSamples[i] = m_voBGDescSamples[i].clone();
	}
}

//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
void BGSSubsenseM::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints)
{
	//init points for tracking
	const unsigned blockX = 24;
	const unsigned blockY = 32;
	unsigned rSize = oInitImg.rows/blockX;
	unsigned cSize = oInitImg.cols/blockY;
	for(int i=0; i<=oInitImg.rows/blockX; i++)
	{
		for(int j=0; j<=oInitImg.cols/blockY; j++)
			m_points[0].push_back(cv::Point2f(j*blockY,i*blockX));
	}
	m_warpMask.create(oInitImg.size(),CV_8U);
	m_warpMask = cv::Scalar(0.0f);
	BackgroundSubtractorSuBSENSE::initialize(oInitImg,voKeyPoints);
	m_voTKeyPoints.resize(m_voKeyPoints.size());
	/*	for(int i=0; i<m_nBGSamples; i++)
	{
	m_modelsPtr.push_back(&m_voBGColorSamples[i]);
	m_modelsPtr.push_back(&m_voBGDescSamples[i]);
	}

	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	m_modelsPtr.push_back(&m_oUpdateRateFrame);
	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	m_modelsPtr.push_back(&m_oDistThresholdFrame);
	//! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
	m_modelsPtr.push_back(&m_oVariationModulatorFrame);
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	m_modelsPtr.push_back(&m_oMeanLastDistFrame);
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	m_modelsPtr.push_back(&m_oMeanMinDistFrame_LT);
	m_modelsPtr.push_back(&m_oMeanMinDistFrame_ST);
	//! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
	m_modelsPtr.push_back(&m_oMeanDownSampledLastDistFrame_LT);
	m_modelsPtr.push_back(&m_oMeanDownSampledLastDistFrame_ST);
	//! per-pixel mean raw segmentation results
	m_modelsPtr.push_back(&m_oMeanRawSegmResFrame_LT);
	m_modelsPtr.push_back(&m_oMeanRawSegmResFrame_ST);
	//! per-pixel mean final segmentation results
	m_modelsPtr.push_back(&m_oMeanFinalSegmResFrame_LT);
	m_modelsPtr.push_back(&m_oMeanFinalSegmResFrame_ST);
	//! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	m_modelsPtr.push_back(&m_oUnstableRegionMask);
	//! per-pixel blink detection results ('Z(x)')
	m_modelsPtr.push_back(&m_oBlinksFrame);
	//! pre-allocated matrix used to downsample (1/8) the input frame when needed
	m_modelsPtr.push_back(&m_oDownSampledColorFrame);
	//! copy of previously used pixel intensities used to calculate 'D_last(x)'
	m_modelsPtr.push_back(&m_oLastColorFrame);
	//! copy of previously used descriptors used to calculate 'D_last(x)'
	m_modelsPtr.push_back(&m_oLastDescFrame);
	//! the foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	m_modelsPtr.push_back(&m_oRawFGMask_last);
	//! the foreground mask generated by the method at [t-1] (with post-proc)
	m_modelsPtr.push_back(&m_oFGMask_last);
	*/

	//warped models

	w_voBGColorSamples.resize(m_nBGSamples);
	w_voBGDescSamples.resize(m_nBGSamples);
	cloneModels();
}

//! refreshes all samples based on the last analyzed frame
void BGSSubsenseM::refreshModel(float fSamplesRefreshFrac)
{
	BackgroundSubtractorSuBSENSE::refreshModel(fSamplesRefreshFrac);
}

void BGSSubsenseM::getHomography(const cv::Mat& image, cv::Mat&  homography)
{
	// convert to gray-level image
	if (image.channels() ==3)
	{
		cv::cvtColor(image, m_gray, CV_BGR2GRAY); 
	}
	else
		m_gray = image;
	if (m_preGray.empty())
	{
		m_gray.copyTo(m_preGray);
	}

	// 2. track features
	cv::calcOpticalFlowPyrLK(m_gray,m_preGray,  // 2 consecutive images
		m_points[0], // input point position in first image
		m_points[1], // output point postion in the second image
		m_status,    // tracking success
		m_err);      // tracking error

	// 2. loop over the tracked points to reject the undesirables
	int k=0;
	std::vector<cv::Point2f> prev;
	prev.reserve(m_points[1].size());
	for( int i= 0; i < m_points[1].size(); i++ ) {

		// do we keep this point?
		if (m_status[i] == 1) {

			// keep this point in vector
			prev.push_back(m_points[0][i]);
			m_points[1][k++] = m_points[1][i];
		}
	}

	// eliminate unsuccesful points
	m_points[1].resize(k);
	std::vector<uchar> inliers(k,0);
	homography= cv::findHomography(
		cv::Mat(prev), // corresponding
		cv::Mat(m_points[1]), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		1.);

	cv::swap(m_preGray, m_gray);
}

void BGSSubsenseM::motionCompensate()
{
	//cv::Mat warpGray;
	//cv::imshow("before warp",m_preGray);
	//cv::warpPerspective(m_preGray, // input image
	//		warpGray,			// output image
	//		m_homography,		// homography
	//		m_preGray.size()); // size of output image
	//cv::imshow("after warp",warpGray);
	//cv::imshow("current",m_gray);
	for(int i=0; i< m_modelsPtr.size(); i++)
	{
		//cv::imshow("before warp",*m_modelsPtr[i]);
		cv::warpPerspective(*m_modelsPtr[i], // input image
			*m_modelsPtr[i],			// output image
			m_homography,		// homography
			m_modelsPtr[i]->size()); // size of output image

		//cv::imshow("after warp",*m_modelsPtr[i]);
	}
}
//! primary model update function; the learning param is used to override the internal learning thresholds (ignored when <= 0)
void BGSSubsenseM::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride)
{
	getHomography(_image.getMat(),m_homography);
	cv::Mat wImage(_image.getMat().size(),_image.getMat().type());
	//std::cout<<m_homography;
	cloneModels();
	//std::ofstream file("homo.txt");
	for(int i=0; i<m_voKeyPoints.size(); i++)
	{
		double* ptr = (double*)m_homography.data;
		float x,y,w;
		x = m_voKeyPoints[i].pt.x*ptr[0] + m_voKeyPoints[i].pt.y*ptr[1] + ptr[2];
		y = m_voKeyPoints[i].pt.x*ptr[3] + m_voKeyPoints[i].pt.y*ptr[4] + ptr[5];
		w = m_voKeyPoints[i].pt.x*ptr[6] + m_voKeyPoints[i].pt.y*ptr[7] + ptr[8];
		x /=w;
		y/=w;
		//std::cout<<x<<","<<y<<std::endl;
		if (x<2 || x>= m_oImgSize.width-2 || y<2 || y>=m_oImgSize.height-2)
			m_warpMask.at<uchar>((int)m_voKeyPoints[i].pt.y,(int)m_voKeyPoints[i].pt.x) = 0;
		else
			m_warpMask.at<uchar>((int)m_voKeyPoints[i].pt.y,(int)m_voKeyPoints[i].pt.x) = 255;
		m_voTKeyPoints[i] = cv::KeyPoint(x,y,1.f);
		//wImage.at<cv::Vec3b>((int)m_voKeyPoints[i].pt.y,(int)m_voKeyPoints[i].pt.x) = _image.getMat().at<cv::Vec3b>((int)y,(int)x);
		//file<<m_voKeyPoints[i].pt.x<<","<<m_voKeyPoints[i].pt.y<<"--->"<<m_voTKeyPoints[i].pt.x<<","<<m_voTKeyPoints[i].pt.y<<std::endl;
	}
	//file.close();
	/*char iname[20];
	sprintf(iname,"warped%d.jpg",m_nFrameIndex);
	imwrite(iname,wImage);*/
	//imshow("warp mask",m_warpMask);
	// == process
	CV_DbgAssert(m_bInitialized);
	cv::Mat oInputImg = _image.getMat();
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	size_t nNonZeroDescCount = 0;
	const float fRollAvgFactor_LT = 1.0f/std::min(++m_nFrameIndex,m_nSamplesForMovingAvgs*4);
	const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIndex,m_nSamplesForMovingAvgs);
	if(m_nImgChannels==1) {
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;			

			//变换后在前一帧图像中的位置
			const int wx = (int)(m_voTKeyPoints[k].pt.x+0.5);
			const int wy = (int)(m_voTKeyPoints[k].pt.y+0.5);
			const size_t idx_uchar = m_oImgSize.width*wy + wx;
			const size_t idx_ushrt = idx_uchar*2;
			const size_t idx_flt32 = idx_uchar*4;

			const size_t oidx_uchar = m_oImgSize.width*y + x;
			const uchar nCurrColor = oInputImg.data[oidx_uchar];
			uchar warpMask = m_warpMask.data[oidx_uchar];
			if (warpMask == 0)
			{
				//相机移动后在原模型中不存在的部分，不处理
				continue;
			}
			size_t nMinDescDist = s_nDescMaxDataRange_1ch;
			size_t nMinSumDist = s_nColorMaxDataRange_1ch;
			float* pfCurrDistThresholdFactor = (float*)(w_oDistThresholdFrame.data+idx_flt32);
			float* pfCurrVariationFactor = (float*)(w_oVariationModulatorFrame.data+idx_flt32);
			float* pfCurrLearningRate = ((float*)(w_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanLastDist = ((float*)(w_oMeanLastDistFrame.data+idx_flt32));
			float* pfCurrMeanMinDist_LT = ((float*)(w_oMeanMinDistFrame_LT.data+idx_flt32));
			float* pfCurrMeanMinDist_ST = ((float*)(w_oMeanMinDistFrame_ST.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_LT = ((float*)(w_oMeanRawSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_ST = ((float*)(w_oMeanRawSegmResFrame_ST.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_LT = ((float*)(w_oMeanFinalSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_ST = ((float*)(w_oMeanFinalSegmResFrame_ST.data+idx_flt32));
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_ushrt));
			uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!w_oUnstableRegionMask.data[idx_uchar])*STAB_COLOR_DIST_OFFSET))/2;
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(w_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			w_oUnstableRegionMask.data[idx_uchar] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar& nBGColor = w_voBGColorSamples[nSampleIdx].data[idx_uchar];
				{
					const size_t nColorDist = absdiff_uchar(nCurrColor,nBGColor);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					const ushort& nBGIntraDesc = *((ushort*)(w_voBGDescSamples[nSampleIdx].data+idx_ushrt));
					const size_t nIntraDescDist = hdist_ushort_8bitLUT(nCurrIntraDesc,nBGIntraDesc);
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,x,y,m_anLBSPThreshold_8bitLUT[nBGColor],nCurrInterDesc);
					const size_t nInterDescDist = hdist_ushort_8bitLUT(nCurrInterDesc,nBGIntraDesc);
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					if(nDescDist>nCurrDescDistThreshold)
						goto failedcheck1ch;
					const size_t nSumDist = std::min((nDescDist/4)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					if(nMinDescDist>nDescDist)
						nMinDescDist = nDescDist;
					if(nMinSumDist>nSumDist)
						nMinSumDist = nSumDist;
					nGoodSamplesCount++;
				}
failedcheck1ch:
				nSampleIdx++;
			}
			const float fNormalizedLastDist = ((float)absdiff_uchar(nLastColor,nCurrColor)/s_nColorMaxDataRange_1ch+(float)hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc)/s_nDescMaxDataRange_1ch)/2;
			*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				// == foreground
				const float fNormalizedMinDist = std::min(1.0f,((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
				oCurrFGMask.data[oidx_uchar] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(w_voBGDescSamples[s_rand].data+idx_ushrt)) = nCurrIntraDesc;
					w_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
			}
			else {
				// == background
				const float fNormalizedMinDist = ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2;
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(w_voBGDescSamples[s_rand].data+idx_ushrt)) = nCurrIntraDesc;
					w_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
				int x_rand,y_rand;
				const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar];
				if(bCurrUsing3x3Spread)
					getRandNeighborPosition_3x3(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				else
					getRandNeighborPosition_5x5(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*y_rand + x_rand;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist = *((float*)(w_oMeanLastDistFrame.data+idx_rand_flt32));
				const float fRandMeanRawSegmRes = *((float*)(w_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
						const size_t idx_rand_ushrt = idx_rand_uchar*2;
						const size_t s_rand = rand()%m_nBGSamples;
						*((ushort*)(w_voBGDescSamples[s_rand].data+idx_rand_ushrt)) = nCurrIntraDesc;
						w_voBGColorSamples[s_rand].data[idx_rand_uchar] = nCurrColor;
				}
			}
			if(m_oFGMask_last.data[idx_uchar] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[oidx_uchar])) {
				if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
					*pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
			}
			else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
				*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
			if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
				*pfCurrLearningRate = m_fCurrLearningRateLowerCap;
			else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
				*pfCurrLearningRate = m_fCurrLearningRateUpperCap;
			if(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && w_oBlinksFrame.data[idx_uchar])
				(*pfCurrVariationFactor) += FEEDBACK_V_INCR;
			else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
				(*pfCurrVariationFactor) -= m_oFGMask_last.data[idx_uchar]?FEEDBACK_V_DECR/4:w_oUnstableRegionMask.data[idx_uchar]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
				if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
					(*pfCurrVariationFactor) = FEEDBACK_V_DECR;
			}
			if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
				(*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
			else {
				(*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
				if((*pfCurrDistThresholdFactor)<1.0f)
					(*pfCurrDistThresholdFactor) = 1.0f;
			}
			if(popcount_ushort_8bitsLUT(nCurrIntraDesc)>=2)
				++nNonZeroDescCount;
			nLastIntraDesc = nCurrIntraDesc;
			nLastColor = nCurrColor;
		}
	}
	else { //m_nImgChannels==3
		
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t oidx_uchar = m_oImgSize.width*y + x;	
			const size_t oidx_flt32 = oidx_uchar*4;
			const size_t oidx_uchar_rgb = oidx_uchar*3;
			const size_t oidx_ushrt_rgb = oidx_uchar_rgb*2;
			const uchar* const anCurrColor = oInputImg.data+oidx_uchar_rgb;
			
			const int wx = (int)(m_voTKeyPoints[k].pt.x+0.5);
			const int wy = (int)(m_voTKeyPoints[k].pt.y+0.5);
			const size_t idx_uchar = m_oImgSize.width*wy +wx;
			const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			uchar warpMask = m_warpMask.data[oidx_uchar];
			if (warpMask == 0)
			{
				//相机移动后在原模型中不存在的部分，不处理
				continue;
			}
			size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
			size_t nMinTotSumDist=s_nColorMaxDataRange_3ch;
			float* pfCurrDistThresholdFactor = (float*)(w_oDistThresholdFrame.data+idx_flt32);
			float* pfCurrVariationFactor = (float*)(w_oVariationModulatorFrame.data+idx_flt32);
			float* pfCurrLearningRate = ((float*)(w_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanLastDist = ((float*)(w_oMeanLastDistFrame.data+idx_flt32));
			float* pfCurrMeanMinDist_LT = ((float*)(w_oMeanMinDistFrame_LT.data+idx_flt32));
			float* pfCurrMeanMinDist_ST = ((float*)(w_oMeanMinDistFrame_ST.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_LT = ((float*)(w_oMeanRawSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_ST = ((float*)(w_oMeanRawSegmResFrame_ST.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_LT = ((float*)(w_oMeanFinalSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_ST = ((float*)(w_oMeanFinalSegmResFrame_ST.data+idx_flt32));
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;


			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!w_oUnstableRegionMask.data[idx_uchar])*STAB_COLOR_DIST_OFFSET));
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(w_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
			const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
			const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			w_oUnstableRegionMask.data[idx_uchar] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGIntraDesc = (ushort*)(w_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar* const anBGColor = w_voBGColorSamples[nSampleIdx].data+idx_uchar_rgb;
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anBGColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
					size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					const size_t nSumDist = std::min((nDescDist/2)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					nTotDescDist += nDescDist;
					nTotSumDist += nSumDist;
				}
				if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
					goto failedcheck3ch;
				if(nMinTotDescDist>nTotDescDist)
					nMinTotDescDist = nTotDescDist;
				if(nMinTotSumDist>nTotSumDist)
					nMinTotSumDist = nTotSumDist;
				nGoodSamplesCount++;
failedcheck3ch:
				nSampleIdx++;
			}
			const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
			*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				// == foreground
				const float fNormalizedMinDist = std::min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
				oCurrFGMask.data[oidx_uchar] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+oidx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+oidx_uchar_rgb+c) = anCurrColor[c];
					}
				}
			}
			else {
				// == background
				const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2;
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+oidx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+oidx_uchar_rgb+c) = anCurrColor[c];
					}
				}
				int x_rand,y_rand;
				const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar];
				if(bCurrUsing3x3Spread)
					getRandNeighborPosition_3x3(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				else
					getRandNeighborPosition_5x5(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*y_rand + x_rand;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist = *((float*)(w_oMeanLastDistFrame.data+idx_rand_flt32));
				const float fRandMeanRawSegmRes = *((float*)(w_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
						const size_t idx_rand_uchar_rgb = idx_rand_uchar*3;
						const size_t idx_rand_ushrt_rgb = idx_rand_uchar_rgb*2;
						const size_t s_rand = rand()%m_nBGSamples;
						for(size_t c=0; c<3; ++c) {
							*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
							*(m_voBGColorSamples[s_rand].data+idx_rand_uchar_rgb+c) = anCurrColor[c];
						}
				}
			}
			if(m_oFGMask_last.data[idx_uchar] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[oidx_uchar])) {
				if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
					*pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
			}
			else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
				*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
			if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
				*pfCurrLearningRate = m_fCurrLearningRateLowerCap;
			else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
				*pfCurrLearningRate = m_fCurrLearningRateUpperCap;
			if(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[idx_uchar])
				(*pfCurrVariationFactor) += FEEDBACK_V_INCR;
			else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
				(*pfCurrVariationFactor) -= m_oFGMask_last.data[idx_uchar]?FEEDBACK_V_DECR/4:w_oUnstableRegionMask.data[idx_uchar]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
				if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
					(*pfCurrVariationFactor) = FEEDBACK_V_DECR;
			}
			if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
				(*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
			else {
				(*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
				if((*pfCurrDistThresholdFactor)<1.0f)
					(*pfCurrDistThresholdFactor) = 1.0f;
			}
			if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>=4)
				++nNonZeroDescCount;
			for(size_t c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColor[c] = anCurrColor[c];
			}
		}
		
		/*cv::imwrite(wname,w_voBGColorSamples[1]);*/
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			uchar warpMask = m_warpMask.data[idx_uchar];
			
			const size_t idx_ushrt = idx_uchar*2;
			const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_ushrt*3;
			//变换后在前一帧图像中的位置
			const int wx = (int)m_voTKeyPoints[k].pt.x;
			const int wy = (int)m_voTKeyPoints[k].pt.y;
			const size_t widx_uchar = m_oImgSize.width*wy + wx;
			const size_t widx_ushrt = widx_uchar*2;
			const size_t widx_flt32 = widx_uchar*4;
			const size_t widx_uchar_rgb = widx_uchar*3;
			const size_t widx_ushrt_rgb = widx_ushrt*3;

			
			//warped models
			float* wpfCurrDistThresholdFactor = (float*)(w_oDistThresholdFrame.data+widx_flt32);
			float* wpfCurrVariationFactor = (float*)(w_oVariationModulatorFrame.data+widx_flt32);
			float* wpfCurrLearningRate = ((float*)(w_oUpdateRateFrame.data+widx_flt32));
			float* wpfCurrMeanLastDist = ((float*)(w_oMeanLastDistFrame.data+widx_flt32));
			float* wpfCurrMeanMinDist_LT = ((float*)(w_oMeanMinDistFrame_LT.data+widx_flt32));
			float* wpfCurrMeanMinDist_ST = ((float*)(w_oMeanMinDistFrame_ST.data+widx_flt32));
			float* wpfCurrMeanRawSegmRes_LT = ((float*)(w_oMeanRawSegmResFrame_LT.data+widx_flt32));
			float* wpfCurrMeanRawSegmRes_ST = ((float*)(w_oMeanRawSegmResFrame_ST.data+widx_flt32));
			float* wpfCurrMeanFinalSegmRes_LT = ((float*)(w_oMeanFinalSegmResFrame_LT.data+widx_flt32));
			float* wpfCurrMeanFinalSegmRes_ST = ((float*)(w_oMeanFinalSegmResFrame_ST.data+widx_flt32));

			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			float* pfCurrVariationFactor = (float*)(m_oVariationModulatorFrame.data+idx_flt32);
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+idx_flt32));
			float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+idx_flt32));
			if (warpMask == 0)
			{
				const uchar* const anCurrColor = oInputImg.data+idx_uchar*3;
				//相机移动后在原模型中不存在的部分，不处理
			/*	const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);*/
				/*if((rand()%nLearningRate)==0) */
				{
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						//*((ushort*)(w_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(w_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
					}
				}
				continue;
			}
			*pfCurrDistThresholdFactor = *wpfCurrDistThresholdFactor;
			*pfCurrVariationFactor = *wpfCurrVariationFactor;
			*pfCurrLearningRate = *wpfCurrLearningRate;
			*pfCurrMeanLastDist = *wpfCurrMeanLastDist;
			*pfCurrMeanMinDist_LT = *wpfCurrMeanMinDist_LT;
			*pfCurrMeanMinDist_ST = *wpfCurrMeanMinDist_ST;
			*pfCurrMeanRawSegmRes_LT = *wpfCurrMeanRawSegmRes_LT;
			*pfCurrMeanRawSegmRes_ST = *wpfCurrMeanRawSegmRes_ST;
			*pfCurrMeanFinalSegmRes_LT = *wpfCurrMeanFinalSegmRes_LT;
			*pfCurrMeanFinalSegmRes_ST = *wpfCurrMeanFinalSegmRes_ST;
				
		}		
	}
	char name[50];
	char wname[50];
	sprintf(name,"sample%d_frame%d.jpg",1,m_nFrameIndex);
	/*sprintf(wname,"wsample%d_frame%d.jpg",1,m_nFrameIndex);*/
	cv::Mat avgBGColor(m_oImgSize,CV_8UC3);
	avgBGColor =  cv::Scalar_<uchar>::all(0);
	for(int i=0; i<m_nBGSamples; i++)
		cv::addWeighted(m_voBGColorSamples[i],1.0/m_nBGSamples,avgBGColor,1.0,0.0,avgBGColor);
	cv::imwrite(name,avgBGColor);
#if DISPLAY_SUBSENSE_DEBUG_INFO
	std::cout << std::endl;
	cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
	cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame_ST.copyTo(oMeanMinDistFrameNormalized);
	cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
	cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanRawSegmResFrameNormalized; m_oMeanRawSegmResFrame_ST.copyTo(oMeanRawSegmResFrameNormalized);
	cv::circle(oMeanRawSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanRawSegmResFrameNormalized,oMeanRawSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("s_avg(x)",oMeanRawSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  s_avg(" << dbgpt << ") = " << m_oMeanRawSegmResFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanFinalSegmResFrameNormalized; m_oMeanFinalSegmResFrame_ST.copyTo(oMeanFinalSegmResFrameNormalized);
	cv::circle(oMeanFinalSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanFinalSegmResFrameNormalized,oMeanFinalSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("z_avg(x)",oMeanFinalSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  z_avg(" << dbgpt << ") = " << m_oMeanFinalSegmResFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,0.25f,-0.25f);
	cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "      r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oVariationModulatorFrameNormalized; cv::normalize(m_oVariationModulatorFrame,oVariationModulatorFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::circle(oVariationModulatorFrameNormalized,dbgpt,5,cv::Scalar(255));
	cv::resize(oVariationModulatorFrameNormalized,oVariationModulatorFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("v(x)",oVariationModulatorFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "      v(" << dbgpt << ") = " << m_oVariationModulatorFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/FEEDBACK_T_UPPER,-FEEDBACK_T_LOWER/FEEDBACK_T_UPPER);
	cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,DEFAULT_FRAME_SIZE);
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "      t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
#endif //DISPLAY_SUBSENSE_DEBUG_INFO
	cv::Mat wRawFGMask_last;
	cv::Mat wRawFGBlinkMask_last;
	//warp last fgmask to current 
	cv::Mat invHomo = m_homography.inv();
	cv::warpPerspective(m_oRawFGMask_last,wRawFGMask_last,invHomo,m_oRawFGMask_last.size());
	cv::warpPerspective(m_oRawFGBlinkMask_last,wRawFGBlinkMask_last,invHomo,wRawFGBlinkMask_last.size());
	cv::bitwise_xor(oCurrFGMask,wRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,wRawFGBlinkMask_last,m_oBlinksFrame);
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oRawFGMask_last);
	//cv::morphologyEx(oCurrFGMask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	//m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	//cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	//cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	//cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	//cv::bitwise_or(oCurrFGMask,m_oFGMask_FloodedHoles,oCurrFGMask);
	//cv::bitwise_or(oCurrFGMask,m_oFGMask_PreFlood,oCurrFGMask);
	//cv::medianBlur(oCurrFGMask,m_oFGMask_last,m_nMedianBlurKernelSize);
	//cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat(),cv::Point(-1,-1),3);
	//cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	//cv::bitwise_not(m_oFGMask_last_dilated,m_oFGMask_last_dilated_inverted);
	//cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	//m_oFGMask_last.copyTo(oCurrFGMask);
	cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
	cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);
	const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/m_nKeyPoints;
	if(fCurrNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN) {
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			if(m_anLBSPThreshold_8bitLUT[t]>cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4)))
				--m_anLBSPThreshold_8bitLUT[t];
	}
	else if(fCurrNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX) {
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			if(m_anLBSPThreshold_8bitLUT[t]<cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+UCHAR_MAX*m_fRelLBSPThreshold))
				++m_anLBSPThreshold_8bitLUT[t];
	}
	m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
	if(m_bLearningRateScalingEnabled) {
		cv::resize(oInputImg,m_oDownSampledColorFrame,m_oDownSampledFrameSize,0,0,cv::INTER_AREA);
		cv::accumulateWeighted(m_oDownSampledColorFrame,m_oMeanDownSampledLastDistFrame_LT,fRollAvgFactor_LT);
		cv::accumulateWeighted(m_oDownSampledColorFrame,m_oMeanDownSampledLastDistFrame_ST,fRollAvgFactor_ST);
		size_t nTotColorDiff = 0;
		for(int i=0; i<m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
			const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0]*i;
			for(int j=0; j<m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
				const size_t idx2 = idx1+m_oMeanDownSampledLastDistFrame_ST.step.p[1]*j;
				nTotColorDiff += (m_nImgChannels==1)?
					(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2)))/2
					:  //(m_nImgChannels==3)
				std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2))),
					std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+4))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+4))),
					(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+8))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+8)))));
			}
		}
		const float fCurrColorDiffRatio = (float)nTotColorDiff/(m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
		if(m_bAutoModelResetEnabled) {
			if(m_nFramesSinceLastReset>1000)
				m_bAutoModelResetEnabled = false;
			else if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD && m_nModelResetCooldown==0) {
				m_nFramesSinceLastReset = 0;
				refreshModel(0.1f); // reset 10% of the bg model
				m_nModelResetCooldown = m_nSamplesForMovingAvgs;
				m_oUpdateRateFrame = cv::Scalar(1.0f);
			}
			else
				++m_nFramesSinceLastReset;
		}
		else if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD*2) {
			m_nFramesSinceLastReset = 0;
			m_bAutoModelResetEnabled = true;
		}
		if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD/2) {
			m_fCurrLearningRateLowerCap = (float)std::max((int)FEEDBACK_T_LOWER>>(int)(fCurrColorDiffRatio/2),1);
			m_fCurrLearningRateUpperCap = (float)std::max((int)FEEDBACK_T_UPPER>>(int)(fCurrColorDiffRatio/2),1);
		}
		else {
			m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
			m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
		}
		if(m_nModelResetCooldown>0)
			--m_nModelResetCooldown;
	}
}


