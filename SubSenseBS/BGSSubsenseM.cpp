#include "BGSSubsenseM.h"
#include <iostream>
#include <fstream>
#include "LSBSP.h"
#include "LBP.h"
#include "STAT.h"
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
	// == init
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
	if(oInitImg.type()==CV_8UC3) {
		std::vector<cv::Mat> voInitImgChannels;
		cv::split(oInitImg,voInitImgChannels);
		bool eq = std::equal(voInitImgChannels[0].begin<uchar>(), voInitImgChannels[0].end<uchar>(), voInitImgChannels[1].begin<uchar>())
				&& std::equal(voInitImgChannels[1].begin<uchar>(), voInitImgChannels[1].end<uchar>(), voInitImgChannels[2].begin<uchar>());
		if(eq)
			std::cout << std::endl << "\tBackgroundSubtractorSuBSENSE : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance." << std::endl;
	}
	std::vector<cv::KeyPoint> voNewKeyPoints;
	if(voKeyPoints.empty()) {
		cv::DenseFeatureDetector oKPDDetector(1.f, 1, 1.f, 1, 0, true, false);
		voNewKeyPoints.reserve(oInitImg.rows*oInitImg.cols);
		oKPDDetector.detect(cv::Mat(oInitImg.size(),oInitImg.type()),voNewKeyPoints);
	}
	else
		voNewKeyPoints = voKeyPoints;
	const size_t nOrigKeyPointsCount = voNewKeyPoints.size();
	CV_Assert(nOrigKeyPointsCount>0);
	LBSP::validateKeyPoints(voNewKeyPoints,oInitImg.size());
	//MBLBP::validateKeyPoints(voNewKeyPoints,oInitImg.size());

	CV_Assert(!voNewKeyPoints.empty());
	m_voKeyPoints = voNewKeyPoints;
	m_nKeyPoints = m_voKeyPoints.size();
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nFrameIndex = 0;
	m_nFramesSinceLastReset = 0;
	m_nModelResetCooldown = 0;
	m_fLastNonZeroDescRatio = 0.0f;
	const int nTotImgPixels = m_oImgSize.height*m_oImgSize.width;
	if((int)nOrigKeyPointsCount>=nTotImgPixels/2 && nTotImgPixels>=DEFAULT_FRAME_SIZE.area()) {
		m_bLearningRateScalingEnabled = true;
		m_bAutoModelResetEnabled = true;
		m_bUse3x3Spread = !(nTotImgPixels>DEFAULT_FRAME_SIZE.area()*2);
		const int nRawMedianBlurKernelSize = std::min((int)floor((float)nTotImgPixels/DEFAULT_FRAME_SIZE.area()+0.5f)+DEFAULT_MEDIAN_BLUR_KERNEL_SIZE,14);
		m_nMedianBlurKernelSize = (nRawMedianBlurKernelSize%2)?nRawMedianBlurKernelSize:nRawMedianBlurKernelSize-1;
		m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
		m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
	}
	else {
		m_bLearningRateScalingEnabled = false;
		m_bAutoModelResetEnabled = false;
		m_bUse3x3Spread = true;
		m_nMedianBlurKernelSize = DEFAULT_MEDIAN_BLUR_KERNEL_SIZE;
		m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER*2;
		m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER*2;
	}
	//std::cout << m_oImgSize << " => m_nMedianBlurKernelSize=" << m_nMedianBlurKernelSize << ", with 3x3Spread=" << m_bUse3x3Spread << ", with Tscaling=" << m_bLearningRateScalingEnabled << std::endl;
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(m_fCurrLearningRateLowerCap);
	m_oDistThresholdFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdFrame = cv::Scalar(1.0f);
	m_oVariationModulatorFrame.create(m_oImgSize,CV_32FC1);
	m_oVariationModulatorFrame = cv::Scalar(10.0f); // should always be >= FEEDBACK_V_DECR
	m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame = cv::Scalar(0.0f);
	m_oMeanMinDistFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanMinDistFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame_ST = cv::Scalar(0.0f);
	m_oDownSampledFrameSize = cv::Size(m_oImgSize.width/FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO,m_oImgSize.height/FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO);
	m_oMeanDownSampledLastDistFrame_LT.create(m_oDownSampledFrameSize,CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanDownSampledLastDistFrame_ST.create(m_oDownSampledFrameSize,CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_ST = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_LT = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_ST = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame_LT = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame_ST = cv::Scalar(0.0f);
	m_oUnstableRegionMask.create(m_oImgSize,CV_8UC1);
	m_oUnstableRegionMask = cv::Scalar_<uchar>(0);
	m_oBlinksFrame.create(m_oImgSize,CV_8UC1);
	m_oBlinksFrame = cv::Scalar_<uchar>(0);
	m_oDownSampledColorFrame.create(m_oDownSampledFrameSize,CV_8UC((int)m_nImgChannels));
	m_oDownSampledColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	m_oRawFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oRawFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated_inverted.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated_inverted = cv::Scalar_<uchar>(0);
	m_oFGMask_FloodedHoles.create(m_oImgSize,CV_8UC1);
	m_oFGMask_FloodedHoles = cv::Scalar_<uchar>(0);
	m_oFGMask_PreFlood.create(m_oImgSize,CV_8UC1);
	m_oFGMask_PreFlood = cv::Scalar_<uchar>(0);
	m_oRawFGBlinkMask_curr.create(m_oImgSize,CV_8UC1);
	m_oRawFGBlinkMask_curr = cv::Scalar_<uchar>(0);
	m_oRawFGBlinkMask_last.create(m_oImgSize,CV_8UC1);
	m_oRawFGBlinkMask_last = cv::Scalar_<uchar>(0);
	m_voBGColorSamples.resize(m_nBGSamples);
	m_voBGDescSamples.resize(m_nBGSamples);
	for(size_t s=0; s<m_nBGSamples; ++s) {
		m_voBGColorSamples[s].create(m_oImgSize,CV_8UC((int)m_nImgChannels));
		m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);
		m_voBGDescSamples[s].create(m_oImgSize,CV_16UC((int)m_nImgChannels));
		m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);
	}
	if(m_nImgChannels==1) {
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((m_nLBSPThresholdOffset+t*m_fRelLBSPThreshold)/3);
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_color = m_oLastColorFrame.cols*y_orig + x_orig;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			m_oLastColorFrame.data[idx_color] = oInitImg.data[idx_color];
			//LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[idx_color],x_orig,y_orig,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
			LBP::computeGrayscaleDescriptor(oInitImg,x_orig,y_orig,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
		}
	}
	else { //m_nImgChannels==3
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+t*m_fRelLBSPThreshold);
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_color = 3*(m_oLastColorFrame.cols*y_orig + x_orig);
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			
			size_t anCurrIntraLBSPThresholds[3]; 
			for(size_t c=0; c<3; ++c) {
				const uchar nCurrBGInitColor = oInitImg.data[idx_color+c];
				m_oLastColorFrame.data[idx_color+c] = nCurrBGInitColor;
				anCurrIntraLBSPThresholds[c] = m_anLBSPThreshold_8bitLUT[nCurrBGInitColor];
				//LBSP::computeSingleRGBDescriptor(oInitImg,nCurrBGInitColor,x_orig,y_orig,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);
			}
			//LBP::computeRGBDescriptor(oInitImg,x_orig,y_orig,anCurrIntraLBSPThresholds,((ushort*)(m_oLastDescFrame.data+idx_desc)));
			LBP::computeRGBDescriptor(oInitImg,x_orig,y_orig,anCurrIntraLBSPThresholds,((ushort*)(m_oLastDescFrame.data+idx_desc)));
		}
	}
	m_voTKeyPoints.resize(m_voKeyPoints.size());
	m_bInitializedInternalStructs = true;
	refreshModel(1.0f);


	w_voBGColorSamples.resize(m_nBGSamples);
	w_voBGDescSamples.resize(m_nBGSamples);
	m_nOutPixels = 0;
	cloneModels();
	m_features = cv::Mat(m_oImgSize,CV_8U);
	m_features = cv::Scalar(0);
	m_bInitialized = true;
	
}

//! refreshes all samples based on the last analyzed frame
void BGSSubsenseM::refreshModel(float fSamplesRefreshFrac)
{
	std::cout<<m_nFrameIndex<<": refresh model"<<std::endl;
	BackgroundSubtractorSuBSENSE::refreshModel(fSamplesRefreshFrac);
}
void BGSSubsenseM::resetPara()
{

	m_oUpdateRateFrame = cv::Scalar(m_fCurrLearningRateLowerCap);

	m_oDistThresholdFrame = cv::Scalar(1.0f);

	m_oVariationModulatorFrame = cv::Scalar(10.0f); // should always be >= FEEDBACK_V_DECR

	m_oMeanLastDistFrame = cv::Scalar(0.0f);

	m_oMeanMinDistFrame_LT = cv::Scalar(0.0f);

	m_oMeanMinDistFrame_ST = cv::Scalar(0.0f);

	m_oMeanDownSampledLastDistFrame_LT = cv::Scalar(0.0f);

	m_oMeanDownSampledLastDistFrame_ST = cv::Scalar(0.0f);

	m_oMeanRawSegmResFrame_LT = cv::Scalar(0.0f);

	m_oMeanRawSegmResFrame_ST = cv::Scalar(0.0f);

	m_oMeanFinalSegmResFrame_LT = cv::Scalar(0.0f);

	m_oMeanFinalSegmResFrame_ST = cv::Scalar(0.0f);

	m_oUnstableRegionMask = cv::Scalar_<uchar>(0);

	m_oBlinksFrame = cv::Scalar_<uchar>(0);

	m_oDownSampledColorFrame = cv::Scalar_<uchar>::all(0);

	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);

	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);

	m_oRawFGMask_last = cv::Scalar_<uchar>(0);



	m_oRawFGBlinkMask_curr = cv::Scalar_<uchar>(0);

	m_oRawFGBlinkMask_last = cv::Scalar_<uchar>(0);


}
void BGSSubsenseM::getHomography(const cv::Mat& image, cv::Mat&  homography)
{
	m_features = cv::Scalar(0);
	// convert to gray-level image
	if (image.channels() ==3)
	{
		cv::cvtColor(image, m_gray, CV_BGR2GRAY); 
	}
	else
		m_gray = image;


	cv::Mat edges,edges1;
	C
	//cv::dilate(m_edges,m_edges,cv::Mat(),cv::Point(-1,-1));
	if (m_preEdges.empty())
	{
		m_edges.copyTo(m_preEdges);
	}
	cv::bitwise_or(m_edges,m_preEdges,m_mixEdges);
	if (m_preGray.empty())
	{
		m_gray.copyTo(m_preGray);
	}
	std::vector<cv::Point2f> initial;   // initial position of tracked points
	std::vector<cv::Point2f> features;  // detected features
	int max_count(5000);	  // maximum number of features to detect
	double qlevel(0.05);    // quality level for feature detection,越大质量越高
	double minDist(2.);   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking

	// detect the features
	cv::goodFeaturesToTrack(m_gray, // the image 
		m_points[0],   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features
	/*cv::Mat tmp;
	m_gray.copyTo(tmp);*/

	//for(int i=0; i<m_points[0].size(); i++)
	//{
	//	cv::circle(tmp,m_points[0][i],3,cv::Scalar(255));
	//}
	//cv::imwrite("gf.jpg",tmp);

	// 2. track features
	cv::calcOpticalFlowPyrLK(m_gray, m_preGray, // 2 consecutive images
		m_points[0], // input point position in first image
		m_points[1], // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	// 2. loop over the tracked points to reject the undesirables
	int k=0;

	for( int i= 0; i < m_points[1].size(); i++ ) {

		// do we keep this point?
		if (status[i] == 1) {

			m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
			// keep this point in vector
			m_points[0][k] = m_points[0][i];
			m_points[1][k++] = m_points[1][i];
		}
	}
	// eliminate unsuccesful points
	m_points[0].resize(k);
	m_points[1].resize(k);

	//perspective transform
	std::vector<uchar> inliers(m_points[0].size(),0);
	homography= cv::findHomography(
		cv::Mat(m_points[0]), // corresponding
		cv::Mat(m_points[1]), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
	
	for(int i=0; i<m_points[0].size(); i++)
	{
		if (inliers[i] == 1)
			m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] =100;
	}
	char filename[50];
	sprintf(filename,"edge%d.jpg",m_nFrameIndex);
	cv::imwrite(filename,m_mixEdges);
	cv::swap(m_preGray, m_gray);	
	cv::swap(m_preEdges,m_edges);
}
void BGSSubsenseM::UpdateBackground(float* pfCurrLearningRate, int x, int y, size_t idx_ushrt, size_t idx_uchar, const ushort* anCurrIntraDesc, const uchar* anCurrColor)
{
	const size_t nLearningRate = (size_t)ceil(*pfCurrLearningRate);
	if((rand()%nLearningRate)==0) {
		const size_t s_rand = rand()%m_nBGSamples;
		for(size_t c=0; c<m_nImgChannels; ++c) {
			*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt+2*c)) = anCurrIntraDesc[c];
			*(m_voBGColorSamples[s_rand].data+idx_uchar+c) = anCurrColor[c];
		}

	}
	int x_rand,y_rand;
	const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar/3];
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
			for(size_t c=0; c<m_nImgChannels; ++c) {
				*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt+2*c)) = anCurrIntraDesc[c];
				*(m_voBGColorSamples[s_rand].data+idx_rand_uchar+c) = anCurrColor[c];
			}				
	}
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

template <typename T>
void LinearInterData(int width, int height, T* data, float x, float y,T* out, int step = 3)
{
	
		if ( x >=0 && x <width && y>=0&& y< height)
		{
			int sx = (int)x;
			int sy = (int)y;
			int bx = sx +1;
			int by = sy +1;
			float tx = x - sx;
			float ty = y - sy;
			size_t idx_rgb_lu = (sx+sy*width)*step;
			size_t idx_rgb_ru = (bx+sy*width)*step;
			size_t idx_rgb_ld =(sx+by*width)*step;
			size_t idx_rgb_rd = (bx+by*width)*step;
			for(int c=0; c<3; c++)
			{
				out[c] =(1- ty)*((1-tx)*data[idx_rgb_lu+c]+tx*data[idx_rgb_ru+c]) + ty*((1-tx)*data[idx_rgb_ld+c] + tx*data[idx_rgb_rd+c]);
			}

		}
	
	
}
//! primary model update function; the learning param is used to override the internal learning thresholds (ignored when <= 0)
void BGSSubsenseM::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride)
{
	
	getHomography(_image.getMat(),m_homography);
	cv::Mat invHomo = m_homography.inv();
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
		{
			m_warpMask.at<uchar>((int)m_voKeyPoints[i].pt.y,(int)m_voKeyPoints[i].pt.x) = 0;
			m_nOutPixels ++;
		}
		else
			m_warpMask.at<uchar>((int)m_voKeyPoints[i].pt.y,(int)m_voKeyPoints[i].pt.x) = 255;
		//在s*s的区域内搜索与原图像最接近的点
		/*int s = 1;
		float alpha = 0.7;
		float min = 16384;
		int wwx = x;
		int wwy = y;
		uchar grad = m_preGrad.data[(int)m_voKeyPoints[i].pt.y*m_oImgSize.width+(int)m_voKeyPoints[i].pt.x];
		for(int m=-s; m<=s; m++)
		{
		for(int n=-s; n<=s; n++)
		{
		int mx = m+x;
		int ny = n+y;
		if (mx >=0 && mx<m_oImgSize.width && ny>=0 && ny<m_oImgSize.height)
		{
		int idx = mx+ny*m_oImgSize.width;
		float diff = std::abs(m_grad.data[idx] - grad);
		if (diff<min)
		{
		min = diff;
		wwx = mx;
		wwy = ny;
		}
		}
		}
		}*/
		m_voTKeyPoints[i] = cv::KeyPoint(x,y,1.f);
		ptr = (double*)invHomo.data;
		x = m_voKeyPoints[i].pt.x*ptr[0] + m_voKeyPoints[i].pt.y*ptr[1] + ptr[2];
		y = m_voKeyPoints[i].pt.x*ptr[3] + m_voKeyPoints[i].pt.y*ptr[4] + ptr[5];
		w = m_voKeyPoints[i].pt.x*ptr[6] + m_voKeyPoints[i].pt.y*ptr[7] + ptr[8];
		x /=w;
		y/=w;
		//std::cout<<x<<","<<y<<std::endl;
		if (x<2 || x>= m_oImgSize.width-2 || y<2 || y>=m_oImgSize.height-2)
		{
			m_nOutPixels ++;
		}
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
	//cv::GaussianBlur(oInputImg,oInputImg,cv::Size(3,3),0.1);
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
			const size_t oidx_ushrt = oidx_uchar * 2;
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
				UpdateBackground(pfCurrLearningRate,x,y,oidx_ushrt,oidx_uchar,(const ushort*)(&nCurrIntraDesc),&nCurrColor);

				/*const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
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
				}*/
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
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			uchar warpMask = m_warpMask.data[idx_uchar];

			const size_t idx_ushrt = idx_uchar*2;
			const size_t idx_flt32 = idx_uchar*4;

			//变换后在前一帧图像中的位置
			const int wx = (int)(m_voTKeyPoints[k].pt.x+0.5);
			const int wy = (int)(m_voTKeyPoints[k].pt.y+0.5);
			const size_t widx_uchar = m_oImgSize.width*wy + wx;
			const size_t widx_ushrt = widx_uchar*2;
			const size_t widx_flt32 = widx_uchar*4;

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
				const uchar* const anCurrColor = oInputImg.data+idx_uchar;
				ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt));
				//相机移动后在原模型中不存在的部分，不处理
				/*	const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);*/
				/*if((rand()%nLearningRate)==0) */
				{
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt)) = anLastIntraDesc[0];
					*(m_voBGColorSamples[s_rand].data+idx_uchar) = anCurrColor[0];

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
	else { //m_nImgChannels==3
		/*cv::Mat smask(m_oImgSize,CV_8U);
		smask = cv::Scalar(0);*/
		std::cout<<m_nFrameIndex<<std::endl;
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
			/*if (x>2 && x< m_oImgSize.width-3 && y>2 && y<m_oImgSize.height-3
				&& wx>2 && wx< m_oImgSize.width-3 && wy>2 && wy<m_oImgSize.height-3
				&& m_mixEdges.data[x+y*m_oImgSize.width] == 0xff)
			{


			}*/
			const size_t idx_uchar = m_oImgSize.width*wy +wx;
			const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			uchar warpMask = m_warpMask.data[oidx_uchar];
			//m_oUnstableRegionMask.data[oidx_uchar] = m_mixEdges.data[oidx_uchar] >20 ? 1 :0;
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

			
			size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!w_oUnstableRegionMask.data[idx_uchar])*STAB_COLOR_DIST_OFFSET));
			size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(w_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
			const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
			const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			//LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			LBP::computeRGBDescriptor(oInputImg,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			w_oUnstableRegionMask.data[idx_uchar] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGIntraDesc = (ushort*)(w_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar* const anBGColor = w_voBGColorSamples[nSampleIdx].data+idx_uchar_rgb;
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				bool pass = true;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anBGColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
					{
						pass = false;
						break;
					}
					size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
					/*LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
					size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;*/
					const size_t nDescDist = nIntraDescDist/2;
					const size_t nSumDist = std::min((nDescDist/2)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrSCColorDistThreshold)
					{
						pass = false; 
						break;
					}
					nTotDescDist += nDescDist;
					nTotSumDist += nSumDist;
					//nTotSumDist += nColorDist;
				}

				if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
					pass =false;
				if (pass)
				{
					if(nMinTotDescDist>nTotDescDist)
						nMinTotDescDist = nTotDescDist;
					if(nMinTotSumDist>nTotSumDist)
						nMinTotSumDist = nTotSumDist;
					nGoodSamplesCount++;
				}
				//if (nGoodSamplesCount < m_nRequiredBGSamples && (m_preEdges.data[oidx_uchar] > 20))
				//{
				//	int s = 1;
				//	for(int m=-s; m<=s; m++)
				//	{
				//		for(int n=-s; n<=s; n++)
				//		{
				//			int rx = m+wx;
				//			int ry = n+wy;
				//			size_t ridx = rx + ry*m_oImgSize.width;
				//			size_t ridx_rgb = ridx * 3;
				//			uchar* bgColor = 	w_voBGColorSamples[nSampleIdx].data+ridx_rgb;
				//			bool pass = true;
				//			for(size_t c=0;c<3; ++c) {
				//				const size_t nColorDist = absdiff_uchar(anCurrColor[c],bgColor[c]);
				//				if(nColorDist>nCurrSCColorDistThreshold)
				//				{
				//					pass = false;
				//					break;
				//				}
				//			}
				//			if (pass)
				//			{
				//				nGoodSamplesCount++;
				//				//std::cout<<"neighbour background "<<": "<<x<<" , "<<y<<std::endl;
				//			}
				//		}
				//	}
				//	nGoodSamplesCount /=2;
				//}


				nSampleIdx++;
			}
			if (x==8 && y==18 && m_nFrameIndex == 2)
			{

				std::cout<<m_nFrameIndex<<":"<<nGoodSamplesCount<<std::endl;
				int s = 2;
			
				for(int c=0; c<3; c++)
				{	
					std::cout<<"current"<<std::endl;
					for(int n=-s; n<=s; n++)
					{
						for(int m=-s; m<=s; m++)
						{
							int rx = m+x;
							int ry = n+y;
							size_t ridx = rx + ry*m_oImgSize.width;
							std::cout<<(int)oInputImg.data[ridx*3+c]<<" ";
						}
						std::cout<<std::endl;
					}
					std::cout<<"pre"<<std::endl;
					for(int n=-s; n<=s; n++)
					{
						for(int m=-s; m<=s; m++)
						{
							int rx = m+wx;
							int ry = n+wy;
							size_t ridx = rx + ry*m_oImgSize.width;
							std::cout<<(int)m_oLastColorFrame.data[ridx*3+c]<<" ";
						}
						std::cout<<std::endl;
					}
					std::cout<<"desc = "<<anCurrIntraDesc[c]<<std::endl;
				}

			}
			//const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
			const float fNormalizedLastDist = (float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch;
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
				UpdateBackground(pfCurrLearningRate,x,y,oidx_ushrt_rgb,oidx_uchar_rgb,(const ushort*)(anCurrIntraDesc),anCurrColor);
				/*	const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
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
				}*/
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
			const int wx = (int)(m_voTKeyPoints[k].pt.x+0.5);
			const int wy = (int)(m_voTKeyPoints[k].pt.y+0.5);
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
				ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
				//相机移动后在原模型中不存在的部分，不处理
				/*	const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);*/
				/*if((rand()%nLearningRate)==0) */
				{
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anLastIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
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
	/*	char wname[50];
		sprintf(wname,"smask%d.jpg",m_nFrameIndex);
		cv::imwrite(wname,smask);*/
	}
	//char name[50];
	//char wname[50];
	//sprintf(name,"sample%d_frame%d.jpg",1,m_nFrameIndex);
	///*sprintf(wname,"wsample%d_frame%d.jpg",1,m_nFrameIndex);*/
	//cv::Mat avgBGColor(m_oImgSize,CV_8UC3);
	//avgBGColor =  cv::Scalar_<uchar>::all(0);
	//for(int i=0; i<m_nBGSamples; i++)
	//	cv::addWeighted(m_voBGColorSamples[i],1.0/m_nBGSamples,avgBGColor,1.0,0.0,avgBGColor);
	//cv::imwrite(name,avgBGColor);
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
	cv::warpPerspective(m_oRawFGMask_last,wRawFGMask_last,invHomo,m_oRawFGMask_last.size());
	cv::warpPerspective(m_oRawFGBlinkMask_last,wRawFGBlinkMask_last,invHomo,wRawFGBlinkMask_last.size());
	cv::bitwise_xor(oCurrFGMask,wRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,wRawFGBlinkMask_last,m_oBlinksFrame);	
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);	
	//BlockMaskHomographyTest(oCurrFGMask,m_preGray,m_gray,m_homography);
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
	//MaskHomographyTest(oCurrFGMask,m_preGray,m_gray,m_homography);
	cv::dilate(m_features,m_features,cv::Mat(),cv::Point(-1,-1),3);
	char filename[200];
	sprintf(filename,"..\\result\\subsensem\\ptz\\input3\\features\\features%06d.jpg",m_nFrameIndex);
	cv::imwrite(filename,m_features);
	/*cv::Mat nedge;
	cv::bitwise_not(m_mixEdges,nedge);
	cv::bitwise_and(nedge,oCurrFGMask,oCurrFGMask);*/
	
	/*char filename[150];*/
	//sprintf(filename,"unstable%d.jpg",m_nFrameIndex);
	//cv::Mat us = w_oUnstableRegionMask.clone();
	//for(int i=0; i<us.rows; i++)
	//{
	//	for(int j=0; j<us.cols; j++)
	//	{
	//		if (us.data[j+i*us.cols] == 1)
	//			us.data[j+i*us.cols] = 0xff;
	//	}
	//}
	//cv::imwrite(filename,us);

	
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
				//refreshModel(0.1f); // reset 10% of the bg model
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
	if (m_nOutPixels > 0.4*m_oImgSize.height*m_oImgSize.width)
	{
		refreshModel(0.1);
		resetPara();
		m_nOutPixels = 0;
	}
}


