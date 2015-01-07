#define _USE_MATH_DEFINES
#include "GpuBackgroundSubtractor.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include "CudaBSOperator.h"
#include "LBSP.h"
#include "GpuTimer.h"
#include "MotionEstimate.h"
#define LBSP_PATCH_SIZE 5
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
void postProcessa(const Mat& img, Mat& mask)
{
	cv::Mat m_oFGMask_PreFlood(img.size(),CV_8U);
	cv::Mat m_oFGMask_FloodedHoles(img.size(),CV_8U);
	cv::morphologyEx(mask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_or(mask,m_oFGMask_FloodedHoles,mask);
	cv::bitwise_or(mask,m_oFGMask_PreFlood,mask);
	cv::medianBlur(mask,mask,3);
	
}
void postProcessSegments(const Mat& img, Mat& mask)
{
	int niters = 3;

	vector<vector<Point> > contours,imgContours;
	vector<Vec4i> hierarchy,imgHierarchy;
	
	Mat temp;


	dilate(mask, temp, Mat(), Point(-1,-1), niters);//ÅòÕÍ£¬3*3µÄelement£¬µü´ú´ÎÊýÎªniters
	erode(temp, temp, Mat(), Point(-1,-1), niters*2);//¸¯Ê´
	dilate(temp, temp, Mat(), Point(-1,-1), niters);
	
	findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//ÕÒÂÖÀª
	
	
	if( contours.size() == 0 )
		return;

	cv::Mat cimg(mask.size(),CV_8UC3);
	double minArea = 15*15;
	Scalar color( 255, 255, 255 );
	for( int i = 0; i< contours.size(); i++ )
	{
		const vector<Point>& c = contours[i];
		double area = fabs(contourArea(Mat(c)));
		if( area > minArea )
		{
			drawContours( cimg, contours, i, color, 1, 8, hierarchy, 0, Point() );
			
		}
		
	}
	cv::cvtColor(cimg,mask,CV_BGR2GRAY);
}
float AvgColor(const cv::Mat& img, int row, int col, int size)
{
	cv::Mat gray = img;
	if (img.channels() == 3)
		cv::cvtColor(img,gray,CV_BGR2GRAY);
	int width = img.cols;
	int height = img.rows;
	int step = size/2;
	size_t avgColor = 0;
	int c = 0;
	for(int i= -step; i<=step; i++)
	{
		for(int j=-step; j<=step; j++)
		{
			int x = col+ i;
			int y = row +j;
			if ( x>=0 && x<width && y>=0 && y<height)
			{
				int idx = y*width + x;
				avgColor += gray.data[idx];
				c++;
			}
		}
	}
	avgColor/=c;
	return avgColor;
}
GpuBackgroundSubtractor::GpuBackgroundSubtractor(	 float fRelLBSPThreshold
															,size_t nMinDescDistThreshold
															,size_t nMinColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples
															,size_t nSamplesForMovingAvgs)
	:	 //BackgroundSubtractorLBSP(fRelLBSPThreshold,nMinDescDistThreshold)
		m_bInitializedInternalStructs(false)
		,m_nMinColorDistThreshold(nMinColorDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nSamplesForMovingAvgs(nSamplesForMovingAvgs)
		,m_nFrameIndex(SIZE_MAX)
		,m_nFramesSinceLastReset(0)
		,m_nModelResetCooldown(0)
		,m_fLastNonZeroDescRatio(0.0f)
		,m_bAutoModelResetEnabled(true)
		,m_bLearningRateScalingEnabled(true)
		,m_fCurrLearningRateLowerCap(FEEDBACK_T_LOWER)
		,m_fCurrLearningRateUpperCap(FEEDBACK_T_UPPER)
		,m_nMedianBlurKernelSize(DEFAULT_MEDIAN_BLUR_KERNEL_SIZE)
		,m_bUse3x3Spread(true)
		,m_fRelLBSPThreshold(0.333)
		,m_nDescDistThreshold(3)
		,m_nLBSPThresholdOffset(0),
		LIST_SIZE(20){
	CV_Assert(m_nBGSamples>0 && m_nRequiredBGSamples<=m_nBGSamples);
	CV_Assert(m_nMinColorDistThreshold>=STAB_COLOR_DIST_OFFSET);
	m_trackDist = 1;
}

GpuBackgroundSubtractor::~GpuBackgroundSubtractor() 
{
	m_ofstream.close();
	cudaFree(d_anLBSPThreshold_8bitLUT);
	delete m_gpuDetector;
	cudaFree(d_homoPtr);
	cudaFree(d_outMaskPtr);
	delete[] m_outMaskPtr;
	delete m_optimizer;
	delete m_gs;
	cudaFree(d_randStates);
	delete[] m_distance;
}
void GpuBackgroundSubtractor::WarpInitialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints)
{
	m_ASAP = new ASAPWarping(oInitImg.cols,oInitImg.rows,8,1.0);
	cv::Mat img;
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
	//LSTBP::validateKeyPoints(voNewKeyPoints,oInitImg.size());

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
	w_voBGColorSamples.resize(m_nBGSamples);
	m_voBGDescSamples.resize(m_nBGSamples);
	w_voBGDescSamples.resize(m_nBGSamples);
	
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
			LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[idx_color],x_orig,y_orig,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
			//LBP::computeGrayscaleDescriptor(oInitImg,x_orig,y_orig,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
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
				LBSP::computeSingleRGBDescriptor(oInitImg,nCurrBGInitColor,x_orig,y_orig,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);

			}
			//VLBP::computeVLBPRGBDescriptor(oInitImg,x_orig,y_orig,oInitImg,x_orig,y_orig,anCurrIntraLBSPThresholds,((ushort*)(m_oLastDescFrame.data+idx_desc)));
			//LSTBP::computeRGBDescriptor(m_dx,m_dy,x_orig,y_orig,(ushort*)(m_oLastDescFrame.data+idx_desc));
			//LGBP::computeRGBDescriptor(m_dx,m_dy,x_orig,y_orig,NULL,(ushort*)(m_oLastDescFrame.data+idx_desc));
			//LBP::computeRGBDescriptor(oInitImg,x_orig,y_orig,anCurrIntraLBSPThresholds,((ushort*)(m_oLastDescFrame.data+idx_desc)));
			//LBP::computeRGBDescriptor(oInitImg,x_orig,y_orig,anCurrIntraLBSPThresholds,((ushort*)(m_oLastDescFrame.data+idx_desc)));
		}
	}

	m_bInitializedInternalStructs = true;
	warpRefreshModel(1.0f);

	
	w_voBGColorSamples.resize(m_nBGSamples);
	w_voBGDescSamples.resize(m_nBGSamples);
	m_nOutPixels = 0;
	m_features = cv::Mat(m_oImgSize,CV_8U);
	m_features = cv::Scalar(0);
	m_bInitialized = true;
	

	//warped last color frame and last desc frame
	w_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	w_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	w_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	w_oLastDescFrame = cv::Scalar_<ushort>::all(0);

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
	
	m_gs = new GpuSuperpixel(m_oImgSize.width,m_oImgSize.height,5);
	m_optimizer = new MRFOptimize(m_oImgSize.width,m_oImgSize.height,5);
}
void GpuBackgroundSubtractor::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints) {
	WarpInitialize(oInitImg,voKeyPoints);
	return;
	// == init
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
	m_ofstream = std::ofstream("file.txt");
	if(oInitImg.type()==CV_8UC3) {
		std::vector<cv::Mat> voInitImgChannels;
		cv::split(oInitImg,voInitImgChannels);
		bool eq = std::equal(voInitImgChannels[0].begin<uchar>(), voInitImgChannels[0].end<uchar>(), voInitImgChannels[1].begin<uchar>())
				&& std::equal(voInitImgChannels[1].begin<uchar>(), voInitImgChannels[1].end<uchar>(), voInitImgChannels[2].begin<uchar>());
		if(eq)
			std::cout << std::endl << "\tBackgroundSubtractorSuBSENSE : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance." << std::endl;
	}
	m_nPixels = oInitImg.rows*oInitImg.cols;
	cv::Mat initImg(oInitImg.size(),CV_8UC4);
	cv::cvtColor(oInitImg,initImg,CV_BGR2BGRA);
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
	//LBSP::validateKeyPoints(voNewKeyPoints,oInitImg.size());
	cv::KeyPointsFilter::runByImageBorder(voNewKeyPoints,oInitImg.size(),LBSP_PATCH_SIZE/2);
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
	m_voBGColorSamples.resize(m_nBGSamples);
	
	m_voBGDescSamples.resize(m_nBGSamples);
	
	for(size_t s=0; s<m_nBGSamples; ++s) {
		m_voBGColorSamples[s].create(m_oImgSize,CV_8UC(4));
		m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);		
		
		
		m_voBGDescSamples[s].create(m_oImgSize,CV_16UC(4));
		m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);
		
	}
	//50 samples + last frame in one big 1* n matrix
	d_voBGColorSamples.create(1,(m_nBGSamples+1)*m_oImgSize.width*m_oImgSize.height,CV_8UC4);
	d_wvoBGColorSamples.create(d_voBGColorSamples.size(),d_voBGColorSamples.type());
	d_voBGDescSamples.create(d_voBGColorSamples.size(),CV_16UC4);
	d_wvoBGDescSamples.create(d_voBGDescSamples.size(),d_voBGDescSamples.type());
	d_fModels.create(1,10*m_nPixels,CV_32FC1);
	d_wfModels.create(d_fModels.size(),d_fModels.type());
	d_bModels.create(1,2*m_nPixels,CV_8UC1);
	d_wbModels.create(d_bModels.size(),d_bModels.type());

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
	m_oDownSampledFrameSize = cv::Size(m_oImgSize.width/FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO,m_oImgSize.height/FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO);
	m_oDownSampledColorFrame.create(m_oDownSampledFrameSize,CV_8UC((int)m_nImgChannels));
	m_oDownSampledColorFrame = cv::Scalar_<uchar>::all(0);
	m_oMeanDownSampledLastDistFrame_LT.create(m_oDownSampledFrameSize,CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanDownSampledLastDistFrame_ST.create(m_oDownSampledFrameSize,CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_ST = cv::Scalar(0.0f);
	//d_oDownSampledColorFrame.upload(m_oDownSampledColorFrame );
	//d_ColorModels.push_back(d_oDownSampledColorFrame);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC(4));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC(4));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	m_oRawFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oRawFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated = cv::Scalar_<uchar>(0);
	d_oFGMask_last_dilated.upload(m_oFGMask_last_dilated);
	m_oFGMask_last_dilated_inverted.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated_inverted = cv::Scalar_<uchar>(0);
	d_oFGMask_last_dilated_inverted.upload(m_oFGMask_last_dilated_inverted);
	m_oFGMask_FloodedHoles.create(m_oImgSize,CV_8UC1);
	m_oFGMask_FloodedHoles = cv::Scalar_<uchar>(0);
	d_oFGMask_FloodedHoles.upload(m_oFGMask_FloodedHoles);
	m_oFGMask_PreFlood.create(m_oImgSize,CV_8UC1);
	m_oFGMask_PreFlood = cv::Scalar_<uchar>(0);
	d_oFGMask_PreFlood.upload(m_oFGMask_PreFlood);
	m_oRawFGBlinkMask_curr.create(m_oImgSize,CV_8UC1);
	m_oRawFGBlinkMask_curr = cv::Scalar_<uchar>(0);
	d_oRawFGBlinkMask_curr.upload(m_oRawFGBlinkMask_curr);
	m_oRawFGBlinkMask_last.create(m_oImgSize,CV_8UC1);
	m_oRawFGBlinkMask_last = cv::Scalar_<uchar>(0);
	d_oRawFGBlinkMask_last.upload(m_oRawFGBlinkMask_last);
	
	if(m_nImgChannels==1) {
	/*	for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((m_nLBSPThresholdOffset+t*m_fRelLBSPThreshold)/3);*/
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_color = m_oLastColorFrame.cols*y_orig + x_orig;
			//CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			//const size_t idx_desc = idx_color*2;
			m_oLastColorFrame.data[idx_color] = oInitImg.data[idx_color];
			//LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[idx_color],x_orig,y_orig,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
		}
	}
	else { //m_nImgChannels==3
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+t*m_fRelLBSPThreshold);
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			//CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_color = 4*(m_oLastColorFrame.cols*y_orig + x_orig);
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			for(size_t c=0; c<3; ++c) {
				const uchar nCurrBGInitColor = initImg.data[idx_color+c];
				m_oLastColorFrame.data[idx_color+c] = nCurrBGInitColor;
				LBSP::computeSingleRGBDescriptor(initImg,nCurrBGInitColor,x_orig,y_orig,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);
			}
		}
	}
	
	cudaMalloc(&d_anLBSPThreshold_8bitLUT ,sizeof(size_t)*(UCHAR_MAX+1));
	cudaMemcpy(d_anLBSPThreshold_8bitLUT,m_anLBSPThreshold_8bitLUT,sizeof(size_t)*(UCHAR_MAX+1),cudaMemcpyHostToDevice);

	
	d_CurrentColorFrame = gpu::createContinuous(oInitImg.size(),CV_8UC4);
	//d_CurrentColorFrame.create(oInitImg.size(),CV_8U);
	d_FGMask.create(oInitImg.size(),CV_8U);
	d_FGMask_last.create(oInitImg.size(),CV_8U);
	d_outMask.create(oInitImg.size(),CV_8U);
	d_features.create(oInitImg.size(),CV_8U);
	cv::Mat iniImg,outImg;
	cv::cvtColor(oInitImg,iniImg,CV_BGR2BGRA);
	uchar* dataPtr = d_voBGColorSamples.data + (m_nPixels*4*m_nBGSamples);	
	cudaMemcpy(dataPtr,m_oLastColorFrame.data,m_nPixels*4,cudaMemcpyHostToDevice);
	dataPtr = d_voBGDescSamples.data + (m_nPixels*8*m_nBGSamples);
	cudaMemcpy(dataPtr,m_oLastDescFrame.data,m_nPixels*8,cudaMemcpyHostToDevice);
	
		
		
	//d_DescModels = d_voDESCSamples;
	m_bInitializedInternalStructs = true;
	refreshModel(1.0f);
	cudaMalloc ( &d_randStates, m_oImgSize.width*m_oImgSize.height*sizeof( curandState ) );    
	InitRandState(m_oImgSize.width,m_oImgSize.height,d_randStates);
	//GpuTimer timer;
	//timer.Start();
	CudaRefreshModel(d_randStates,1.f, m_oImgSize.width,m_oImgSize.height, d_voBGColorSamples,d_voBGDescSamples,d_fModels,d_bModels);
	//timer.Stop();
	//std::cout<<"refresh model "<<timer.Elapsed()<<"ms"<<std::endl;
	//for(int i=0; i<m_voBGColorSamples.size(); i++)
	//{
	//	//d_voBGColorSamples[i].upload(m_voBGColorSamples[i]);
	//	d_voBGDescSamples[i].upload(m_voBGDescSamples[i]);
	//}
	/*cv::Mat h_tmp;	

	h_tmp.create(m_oImgSize,CV_8UC4);
	cv::Mat diff;
	char filename[20];
	cudaMemcpy(h_tmp.data, d_voBGColorSamples.data + (m_nPixels*4*50),m_nPixels*4,cudaMemcpyDeviceToHost);
	cv::imwrite("lastFrame.jpg",h_tmp);
	cv::gpu::GpuMat d_tmp,d_ctmp;
	d_tmp.create(m_oImgSize,CV_16UC4);
	
	d_ctmp.create(m_oImgSize,CV_8UC4);*/
	
	//for(int i=0;i <50; i++)
	//{			
	//	DownloadModel(m_oImgSize.width,m_oImgSize.height,d_voBGDescSamples,50,i,d_tmp);
	//	d_tmp.download(h_tmp);
	//	sprintf(filename,"gpu%ddescmodel.jpg",i);		
	//	imwrite(filename, h_tmp);
	//	uchar* dataPtr = d_voBGColorSamples.data + (m_nPixels*4*i);
	//	/*d_voBGDescSamples[i].download(h_tmp);
	//	cv::absdiff(m_voBGDescSamples[i],h_tmp,diff);
	//	sprintf(filename,"%dmodel_diff.jpg",i);
	//	imwrite(filename, diff);
	//	d_voBGColorSamples[i].download(h_tmp);*/
	//	
	//	DownloadColorModel(m_oImgSize.width,m_oImgSize.height,d_voBGColorSamples,50,i,d_ctmp);
	//	d_ctmp.download(h_tmp);
	//	//cudaMemcpy(h_tmp.data,dataPtr,m_nPixels*4,cudaMemcpyDeviceToHost);
	//	sprintf(filename,"gpu%dmodel.jpg",i);
	//	
	//	imwrite(filename, h_tmp);
	//	sprintf(filename,"cpu%dmodel.jpg",i);
	//	imwrite(filename, m_voBGColorSamples[i]);
	//	sprintf(filename,"cpu%descmodel.jpg",i);
	//	imwrite(filename, m_voBGDescSamples[i]);
	//}
	
	m_gpuDetector = new cv::gpu::GoodFeaturesToTrackDetector_GPU(50000,0.05);
	//±£´æhomographyÒÔ¼°invhomography
	cudaMalloc(&d_homoPtr,sizeof(double)*18);
	cudaMalloc(&d_outMaskPtr,m_nPixels);
	 m_outMaskPtr= new uchar[m_nPixels];
	 m_nOutPixels = 0;
	InitConstantMem(m_anLBSPThreshold_8bitLUT);
	m_thetaMat = cv::Mat(m_oImgSize,CV_32FC2);
	m_preThetaMat =cv::Mat(m_oImgSize,CV_32FC2);

	m_gs = new GpuSuperpixel(m_oImgSize.width,m_oImgSize.height,5);
	m_optimizer = new MRFOptimize(m_oImgSize.width,m_oImgSize.height,5);
	m_rawFGMask.create(m_oImgSize,CV_8UC1);
	m_distance = new float[m_oImgSize.width*m_oImgSize.height];
	m_bInitialized = true;
}
void GpuBackgroundSubtractor::MotionEstimate(const cv::Mat& image, cv::Mat& homography)
{
	if (m_features.empty())
		m_features = cv::Mat(image.size(),CV_8UC1);
	m_features = cv::Scalar(0);
	d_CurrentColorFrame.upload(image);
	//¼ÆËã³¬ÏñËØ
	m_optimizer->ComuteSuperpixel(m_gs,d_CurrentColorFrame.ptr<uchar4>());
	int* labels(NULL);
	int nPixels(0);
	SLICClusterCenter* centers = NULL;
	float avgE(0);
	m_optimizer->GetSuperpixelResult(nPixels,labels,centers,avgE);

	cv::gpu::cvtColor(d_CurrentColorFrame,d_gray,CV_BGRA2GRAY);
	(*m_gpuDetector)(d_gray,d_currPts);
	if (d_preGray.empty())
	{
		d_gray.copyTo(d_preGray);
		
	}
	d_pyrLk.sparse(d_gray,d_preGray,d_currPts,d_prevPts,d_status);
	download(d_status,m_status);
	download(d_currPts,m_points[0]);
	download(d_prevPts,m_points[1]);
		// 2. loop over the tracked points to reject the undesirables
	int k=0;

	for( int i= 0; i < m_points[1].size(); i++ ) {

		// do we keep this point?
		if (m_status[i] == 1) {

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
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
	//std::cout<<homography<<std::endl;

	d_gray.download(m_gray);
	if (m_preGray.empty())
		m_gray.copyTo(m_preGray);
	cudaMemcpy(d_homoPtr,(double*)homography.data,sizeof(double)*9,cudaMemcpyHostToDevice);
	cv::Mat invhomography = homography.inv();
	cudaMemcpy(d_homoPtr+9,(double*)invhomography.data,sizeof(double)*9,cudaMemcpyHostToDevice);

	std::vector<cv::Point2f> spCenters,matched;
	std::vector<uchar> status;
	std::vector<float> err;
	for(int i=0; i<nPixels; i++)
	{
		spCenters.push_back(cv::Point2f(centers[i].xy.x,centers[i].xy.y));
	}

	cv::calcOpticalFlowPyrLK(m_gray,m_preGray,spCenters,matched,status,err);
	//upload(spCenters,d_currPts);
	//d_pyrLk.sparse(d_gray,d_preGray,d_currPts,d_prevPts,d_status);
	//download(d_status,m_status);
	//download(d_currPts,m_points[0]);
	//download(d_prevPts,m_points[1]);
	k=0;
	
	for( int i= 0; i < matched.size(); i++ ) {

		// do we keep this point?
		if (status[i] == 1) {

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
			// keep this point in vector
			spCenters[k] = spCenters[i];
			matched[k++] = matched[i];
		}
	}
	// eliminate unsuccesful points
	spCenters.resize(k);
	matched.resize(k);
	
	double* data = (double*)homography.data;
	float threshold = 0.06;
	std::vector<int> resLabels;
	for(int i=0; i<k; i++)
	{
		cv::Point2f pt = spCenters[i];
		float x = data[0]*pt.x + data[1]*pt.y + data[2];
		float y = data[3]*pt.x + data[4]*pt.y + data[5];
		float w = data[6]*pt.x + data[7]*pt.y + data[8];
		x /= w;
		y /= w;
		float d = abs(matched[i].x-x) + abs(matched[i].y - y);
		if (d<threshold)
		{
			resLabels.push_back(labels[(int)pt.x+(int)pt.y*m_oImgSize.width]);
		}
	}
	//std::cout<<"k= "<<k<<" inliers "<<resLabels.size()<<std::endl;

	SuperPixelRegionGrowing(m_oImgSize.width,m_oImgSize.height,5,resLabels,labels,centers,m_features,avgE);
	/*char filename[200];	
	sprintf(filename,"..\\result\\subsensex\\ptz\\input0\\features\\features%06d.jpg",m_nFrameIndex+1);
	cv::imwrite(filename,m_features);*/
	

	cv::gpu::swap(d_gray,d_preGray);
	cv::swap(m_preGray, m_gray);	

	//cv::imshow("pregray",m_preGray);
	//cv::imshow("gray",m_gray);
	//cv::waitKey();
}
void GpuBackgroundSubtractor::getHomography(const cv::Mat& image, cv::Mat&  homography)
{
	if (m_features.empty())
		m_features = cv::Mat(image.size(),CV_8UC1);
	m_features = cv::Scalar(0);
	d_CurrentColorFrame.upload(image);
	//¼ÆËã³¬ÏñËØ
	m_optimizer->ComuteSuperpixel(m_gs,d_CurrentColorFrame.ptr<uchar4>());
	cv::gpu::cvtColor(d_CurrentColorFrame,d_gray,CV_BGRA2GRAY);
	(*m_gpuDetector)(d_gray,d_currPts);
	

	if (d_preGray.empty())
		d_gray.copyTo(d_preGray);
	d_pyrLk.sparse(d_gray,d_preGray,d_currPts,d_prevPts,d_status);
	download(d_status,m_status);
	download(d_currPts,m_points[0]);
	download(d_prevPts,m_points[1]);
		// 2. loop over the tracked points to reject the undesirables
	int k=0;

	for( int i= 0; i < m_points[1].size(); i++ ) {

		// do we keep this point?
		if (m_status[i] == 1) {

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
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
	//std::cout<<homography<<std::endl;
	d_gray.download(m_gray);
	cv::Canny(m_gray,m_edges,100,300);
	ExtractEdgePoint(m_gray,m_edges,m_thetaMat,m_edgePoints);

	//cv::dilate(m_edges,m_edges,cv::Mat(),cv::Point(-1,-1));
	if (m_edgeList.size()< LIST_SIZE)
	{
		m_grayList.push_back(m_gray.clone());
		m_thetaMatList.push_back(m_thetaMat.clone());
		m_edgeList.push_back(m_edges.clone());
		m_edgePointList.push_back(m_edgePoints);
	}
	else if(m_edgeList.size() == LIST_SIZE)
	{
		m_grayList.pop_front();
		m_grayList.push_back(m_gray.clone());
		m_thetaMatList.pop_front();
		m_thetaMatList.push_back(m_thetaMat.clone());
		m_edgeList.pop_front();
		m_edgeList.push_back(m_edges.clone());
		m_edgePointList.pop_front();
		m_edgePointList.push_back(m_edgePoints);

	}
	if(m_edgeList.size() < m_trackDist+1)
	{
		m_edges.copyTo(m_preEdges);
		m_thetaMat.copyTo(m_preThetaMat);
		m_preEdgePoints = m_edgePoints;
		m_gray.copyTo(m_lastGray);
	}
	else
	{
		size_t id = m_edgeList.size()-1-m_trackDist;
		std::list<cv::Mat>::iterator itr = std::next(m_edgeList.begin(), id);			
		m_preEdges = *itr;
		itr = std::next(m_grayList.begin(),id);
		m_lastGray = *itr;
		itr = std::next(m_thetaMatList.begin(),id);
		m_preThetaMat = *(itr);
		std::list<vector<EdgePoint>>::iterator eitr = std::next(m_edgePointList.begin(),id);
		m_preEdgePoints = *(eitr);
	}
	cv::Mat affineM,homoM;
	GetTransformMatrix(m_gray,m_lastGray,homoM,affineM);
	double theta(0);
	if (!affineM.empty())
		theta = atan(affineM.at<double>(1,0)/(affineM.at<double>(1,1)+1e-6))/M_PI*180;

	MapEdgePoint(m_edgePoints,m_preEdges,m_preThetaMat,homoM,theta, m_features);	
	cudaMemcpy(d_homoPtr,(double*)homography.data,sizeof(double)*9,cudaMemcpyHostToDevice);
	homography = homography.inv();
	cudaMemcpy(d_homoPtr+9,(double*)homography.data,sizeof(double)*9,cudaMemcpyHostToDevice);
	for(int i=0; i<inliers.size(); i++)
	{
		if (inliers[i] == 1)
			m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] =0xff;
	}
	cv::dilate(m_features,m_features,cv::Mat(),cv::Point(-1,-1),2);
	//cv::bitwise_or(m_features,m_preFeatures,m_mixFeatures);
	char filename[200];	
	sprintf(filename,".\\features\\input0\\features%06d.jpg",m_nFrameIndex+1);
	//cv::imwrite(filename,m_features);
	m_features = cv::imread(filename);
	if (m_features.channels() == 3)
		cv::cvtColor(m_features,m_features,CV_BGR2GRAY);
	/*cv::swap(m_preEdges,m_edges);
	cv::swap(m_preThetaMat,m_thetaMat);
	cv::swap(m_preFeatures,m_features);
	std::swap(m_preEdgePoints,m_edgePoints);*/
	cv::gpu::swap(d_gray,d_preGray);
	cv::swap(m_preGray, m_gray);	
}
void GpuBackgroundSubtractor::WarpModels()
{
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	
	cv::swap(m_oUpdateRateFrame,w_oUpdateRateFrame);
	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	
	cv::swap(m_oDistThresholdFrame,w_oDistThresholdFrame);
	//! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
	
	cv::swap(m_oVariationModulatorFrame,w_oVariationModulatorFrame);
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	
	cv::swap(m_oMeanLastDistFrame,w_oMeanLastDistFrame);
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::swap(m_oMeanMinDistFrame_LT,w_oMeanMinDistFrame_LT);
	cv::swap(m_oMeanMinDistFrame_ST,w_oMeanMinDistFrame_ST);
	
	//! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
	cv::swap(w_oMeanDownSampledLastDistFrame_LT, m_oMeanDownSampledLastDistFrame_LT);
	cv::swap(w_oMeanDownSampledLastDistFrame_ST,m_oMeanDownSampledLastDistFrame_ST);
	//! per-pixel mean raw segmentation results
	cv::swap(w_oMeanRawSegmResFrame_LT, m_oMeanRawSegmResFrame_LT);
	cv::swap(w_oMeanRawSegmResFrame_ST, m_oMeanRawSegmResFrame_ST);
	
	//! per-pixel mean final segmentation results
	cv::swap(w_oMeanFinalSegmResFrame_LT,m_oMeanFinalSegmResFrame_LT);
	cv::swap(w_oMeanFinalSegmResFrame_ST,m_oMeanFinalSegmResFrame_ST);

	//! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	cv::swap(w_oUnstableRegionMask,m_oUnstableRegionMask);
	//! per-pixel blink detection results ('Z(x)')
	cv::swap(w_oBlinksFrame,m_oBlinksFrame);
	//cv::swap(m_oLastColorFrame,w_oLastColorFrame);
	cv::swap(m_oLastDescFrame,w_oLastDescFrame);
	std::swap(w_voBGColorSamples,m_voBGColorSamples);
	std::swap(w_voBGDescSamples,m_voBGDescSamples);
	
	
}
void GpuBackgroundSubtractor::WarpImage(const cv::Mat image, cv::Mat& warpedImg)
{
	/*cv::warpPerspective(img,warpedImg,m_homography,m_oImgSize);*/
	
	if (image.channels() ==3)
	{
		cv::cvtColor(image, m_gray, CV_BGR2GRAY); 
	}
	else
		m_gray = image;
	if (m_preGray.empty())
		m_gray.copyTo(m_preGray);
	
	KLTFeaturesMatching(m_gray,m_preGray,m_points[0],m_points[1]);
	//FeaturePointsRefineHistogram(m_gray.cols,m_gray.rows,m_points[0],m_points[1]);
	FeaturePointsRefineRANSAC(m_points[0],m_points[1],m_homography);
	m_ASAP->SetControlPts(m_points[0],m_points[1]);
	m_ASAP->Solve();
	m_ASAP->Warp(image,warpedImg);
	m_ASAP->Reset();
	//char fileName[50];
	//sprintf(fileName,"in%06d_warped.jpg",m_nFrameIndex+1);
	//
	//cv::imwrite(fileName,warpedImg);
	//sprintf(fileName,"in%06d_gray.jpg",m_nFrameIndex+1);
	//cv::imwrite(fileName,m_gray);	
	//sprintf(fileName,"in%06d_preGray.jpg",m_nFrameIndex+1);
	//cv::imwrite(fileName,m_preGray);

	//std::vector<uchar> inliers(m_points[0].size());
	//m_homography = cv::findHomography(
	//	m_points[0], // corresponding
	//	m_points[1], // points
	//	inliers, // outputted inliers matches
	//	CV_RANSAC, // RANSAC method
	//	0.1); // max distance to reprojection point
	cv::swap(m_gray,m_preGray);
}
void GpuBackgroundSubtractor::WarpBSOperator(cv::InputArray _image, cv::OutputArray _fgmask)
{
	std::cout<<m_nFrameIndex<<std::endl;
	cv::Mat outMask(m_oImgSize,CV_8U);
	outMask = cv::Scalar(0);
	w_oLastColorFrame = cv::Scalar(0);
	/*char filename[30];
	sprintf(filename,"lastColor%d.jpg",m_nFrameIndex);
	cv::imwrite(filename,m_oLastColorFrame);*/
	
	cv::Mat oInputImg;
	cv::Mat img = _image.getMat();
	//getHomography(_image.getMat(),m_homography);
	/*std::cout<<"homo \n";
	std::cout<<m_homography<<std::endl;*/
	WarpImage(img,oInputImg);
	m_invHomography = m_homography.inv();
	
	
	/*sprintf(filename,"curColor%d.jpg",m_nFrameIndex);
	cv::imwrite(filename,oInputImg);*/
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	size_t nNonZeroDescCount = 0;
	const float fRollAvgFactor_LT = 1.0f/std::min(++m_nFrameIndex,m_nSamplesForMovingAvgs*4);
	const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIndex,m_nSamplesForMovingAvgs);

	if(m_nImgChannels==1) {
		cout<<"currently not supported!\n";		
	}
	else { //m_nImgChannels==3
		float* mapXPtr = (float*)m_ASAP->getMapX().data;
		float* mapYPtr = (float*)m_ASAP->getMapY().data;
		float* invMapXPtr = (float*)m_ASAP->getInvMapX().data;
		float* invMapYPtr = (float*)m_ASAP->getInvMapY().data;
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			float mapX = *(mapXPtr + idx_uchar);
			float mapY = *(mapYPtr + idx_uchar);
			float invMapX = *(invMapXPtr + idx_uchar);
			float invMapY = *(invMapYPtr + idx_uchar);
			float fx = invMapX;
			float fy = invMapY;
			int wx = (int)(mapX+0.5);
			int wy = (int)(mapY+0.5);
		/*	double* ptr = (double*)m_invHomography.data;
			fx = x*ptr[0] + y*ptr[1] + ptr[2];
			fy = x*ptr[3] + y*ptr[4] + ptr[5];
			fw = x*ptr[6] + y*ptr[7] + ptr[8];
			fx /=fw;
			fy/=fw;
			wx = (int)(fx+0.5);
			wy = (int)(fy+0.5);*/
			
			if (wx<2 || wx>= m_oImgSize.width-2 || wy<2 || wy>=m_oImgSize.height-2)
			{					
				//m_features.data[oidx_uchar] = 0xff;
				//outMask.data[x+y*m_oImgSize.width] = 0xff;
				m_nOutPixels ++;
				continue;
			}
			else
			{
				//·´±ä»»
			/*	double* ptr = (double*)m_homography.data;
				fx = x*ptr[0] + y*ptr[1] + ptr[2];
				fy = x*ptr[3] + y*ptr[4] + ptr[5];
				fw = x*ptr[6] + y*ptr[7] + ptr[8];
				fx /=fw;
				fy/=fw;*/
				//std::cout<<x<<","<<y<<std::endl;
				if (fx<2 || fx>= m_oImgSize.width-2 || fy<2 || fy>=m_oImgSize.height-2)
				{
					m_nOutPixels ++;
					outMask.data[x+y*m_oImgSize.width] = 0xff;
					size_t anCurrIntraLBSPThresholds[3]; 
					for(size_t c=0; c<3; ++c) {
						const uchar nCurrBGInitColor = img.data[idx_uchar_rgb+c];
						m_oLastColorFrame.data[idx_uchar_rgb+c] = nCurrBGInitColor;
						anCurrIntraLBSPThresholds[c] = m_anLBSPThreshold_8bitLUT[nCurrBGInitColor];
						LBSP::computeSingleRGBDescriptor(img,nCurrBGInitColor,x,y,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(w_oLastDescFrame.data+idx_ushrt_rgb))[c]);

					}
				}
			}
			

			const size_t widx_uchar = m_oImgSize.width*wy +wx;
			const size_t widx_flt32 = widx_uchar*4;
			const size_t widx_uchar_rgb = widx_uchar*3;
			const size_t widx_ushrt_rgb = widx_uchar_rgb*2;

			const uchar* const anCurrColor = oInputImg.data+idx_uchar_rgb;
			size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
			size_t nMinTotSumDist=s_nColorMaxDataRange_3ch;
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
			uchar* pbUnstableRegion = (uchar*)(m_oUnstableRegionMask.data+idx_uchar);

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
			uchar* wpbUnstableRegion = (uchar*)(w_oUnstableRegionMask.data +widx_uchar);

			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;
			uchar* rColor = img.data + idx_uchar_rgb;
			ushort* wanLastIntraDesc = ((ushort*)(w_oLastDescFrame.data+widx_ushrt_rgb));
			uchar* wanLastColor = w_oLastColorFrame.data+widx_uchar_rgb;
			
			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_uchar])*STAB_COLOR_DIST_OFFSET));
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
			const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
			const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			m_oUnstableRegionMask.data[idx_uchar] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+idx_uchar_rgb;
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
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
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
				
			}
			if(m_oFGMask_last.data[idx_uchar] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[idx_uchar])) {
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
				(*pfCurrVariationFactor) -= m_oFGMask_last.data[idx_uchar]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[idx_uchar]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
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
				/*anLastIntraDesc[c] = anCurrIntraDesc[c];*/
				anLastColor[c] = rColor[c];
				//wanLastColor[c] = rColor[c];
				wanLastIntraDesc[c] = anCurrIntraDesc[c];
			}
			*wpfCurrDistThresholdFactor =  *pfCurrDistThresholdFactor;
			*wpfCurrVariationFactor = *pfCurrVariationFactor;
			*wpfCurrLearningRate = *pfCurrLearningRate;
			*wpfCurrMeanLastDist = *pfCurrMeanLastDist;
			*wpfCurrMeanMinDist_LT = *pfCurrMeanMinDist_LT;
			*wpfCurrMeanMinDist_ST = *pfCurrMeanMinDist_ST;
			*wpfCurrMeanRawSegmRes_LT = *pfCurrMeanRawSegmRes_LT; 
			*wpfCurrMeanRawSegmRes_ST = *pfCurrMeanRawSegmRes_ST;
			*wpfCurrMeanFinalSegmRes_LT = *pfCurrMeanFinalSegmRes_LT;
			*wpfCurrMeanFinalSegmRes_ST = *pfCurrMeanFinalSegmRes_ST;
			*wpbUnstableRegion = *pbUnstableRegion;

			for(int i=0; i< m_nBGSamples; i++)
			{
				uchar* wbgColor = w_voBGColorSamples[i].data +widx_uchar_rgb;
				uchar* bgColor = m_voBGColorSamples[i].data + idx_uchar_rgb;
				ushort* wbgDesc = (ushort*)(w_voBGDescSamples[i].data + widx_ushrt_rgb);
				ushort* bgDesc = (ushort*)(m_voBGDescSamples[i].data+idx_ushrt_rgb);
				for(int c=0; c<3; c++)
				{
					wbgColor[c] = bgColor[c];
					wbgDesc[c] = bgDesc[c];
				}
			}

		}
	}
	cv::bitwise_xor(oCurrFGMask,m_oRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,m_oRawFGBlinkMask_last,m_oBlinksFrame);
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oRawFGMask_last);
	/*cv::morphologyEx(oCurrFGMask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_or(oCurrFGMask,m_oFGMask_FloodedHoles,oCurrFGMask);
	cv::bitwise_or(oCurrFGMask,m_oFGMask_PreFlood,oCurrFGMask);
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,m_nMedianBlurKernelSize);
	cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	cv::bitwise_not(m_oFGMask_last_dilated,m_oFGMask_last_dilated_inverted);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	m_oRawFGMask_last.copyTo(oCurrFGMask);*/
	//warp mask to curr Frame
	//cv::warpPerspective(m_oFGMask_last,m_oFGMask_last,m_invHomography,m_oImgSize);
	//cv::warpPerspective(m_oRawFGMask_last,m_oRawFGMask_last,m_invHomography,m_oImgSize);
	cv::remap(m_oFGMask_last,m_oFGMask_last,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	cv::remap(oCurrFGMask,oCurrFGMask,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	
	//MaskHomographyTest(oCurrFGMask,m_preGray,m_gray,m_homography);
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
			//refreshEdgeModel(0.1);
		
	}
	/*sprintf(filename,"outmask%d.jpg",m_nFrameIndex-1);
	cv::imwrite(filename,outMask);*/
	
	m_optimizer->Optimize(m_gs,img,m_oRawFGMask_last,m_features,oCurrFGMask);
	postProcessa(img,oCurrFGMask);
	WarpModels();
	UpdateModel(img,oCurrFGMask);
	//if (m_nOutPixels > 0.4*m_oImgSize.height*m_oImgSize.width)
	//{
	//	refreshModel(0.1);
	//	//resetPara();
	//	m_nOutPixels = 0;
	//}
	//refreshModel(outMask,0.1);
}
void GpuBackgroundSubtractor::cloneModels()
{
	
	cv::gpu::swap(d_voBGColorSamples,d_wvoBGColorSamples);
	cv::gpu::swap(d_voBGDescSamples,d_wvoBGDescSamples);
	cv::gpu::swap(d_fModels,d_wfModels);
	cv::gpu::swap(d_bModels,d_wbModels);
	/*d_voBGColorSamples.copyTo(d_wvoBGColorSamples);
	d_voBGDescSamples.copyTo(d_wvoBGDescSamples);
	d_fModels.copyTo(d_wfModels);
	d_bModels.copyTo(d_wbModels);*/

}

void GpuBackgroundSubtractor::GpuBSOperator(cv::InputArray _image, cv::OutputArray _fgmask)
{
	cv::Mat oInputImg = _image.getMat();
	
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	cv::Mat img;
	cv::cvtColor(oInputImg,img,CV_BGR2BGRA);
	
	//getHomography(img,m_homography);
	MotionEstimate(img,m_homography);
	
	cloneModels();
	d_features.upload(m_features);
	/*GpuTimer gtimer;
	gtimer.Start();*/
	//d_CurrentColorFrame.upload(img);
	//CudaBSOperator(d_CurrentColorFrame, d_randStates,d_homoPtr,++m_nFrameIndex,d_voBGColorSamples, d_wvoBGColorSamples,
	//	d_voBGDescSamples,d_wvoBGDescSamples,d_bModels,d_wbModels,d_fModels,d_wfModels,d_FGMask, d_FGMask_last,d_outMaskPtr,m_fCurrLearningRateLowerCap,m_fCurrLearningRateUpperCap, d_anLBSPThreshold_8bitLUT);
	CudaBSOperator(d_CurrentColorFrame, d_features,d_randStates,d_homoPtr,++m_nFrameIndex,d_voBGColorSamples, d_wvoBGColorSamples,
		d_voBGDescSamples,d_wvoBGDescSamples,d_bModels,d_wbModels,d_fModels,d_wfModels,d_FGMask, d_FGMask_last,d_outMaskPtr,m_fCurrLearningRateLowerCap,m_fCurrLearningRateUpperCap);
	
	
	cudaMemcpy(m_outMaskPtr,d_outMaskPtr,m_nPixels,cudaMemcpyDeviceToHost);
	//d_outMask.download(htmp);
	m_nOutPixels += std::count(m_outMaskPtr,m_outMaskPtr+m_nPixels,0xff);
	std::cout<<m_nFrameIndex<<std::endl;
	if ( m_nOutPixels > m_nPixels*0.4)
	{
		std::cout<<"refresh model"<<std::endl;
		CudaRefreshModel(d_randStates,0.1f, m_oImgSize.width,m_oImgSize.height, d_voBGColorSamples,d_voBGDescSamples,d_fModels,d_bModels);
		m_nOutPixels = 0;
	}

	/*char filename[20];
	sprintf(filename,"%d_outmask.jpg",m_nFrameIndex);
	cv::imwrite(filename,htmp);*/
	/*gtimer.Stop();
	std::cout<<"CudaBSOperator kernel "<<gtimer.Elapsed()<<"ms"<<std::endl;*/

	
	//gtimer.Start();
	/*cv::gpu::bitwise_xor(d_FGMask,d_oRawFGMask_last,d_oRawFGBlinkMask_curr);
	cv::gpu::bitwise_or(d_oRawFGBlinkMask_curr,d_oRawFGBlinkMask_last,d_oBlinksFrame);
	d_oRawFGBlinkMask_curr.copyTo(d_oRawFGBlinkMask_last);
	d_FGMask.copyTo(d_oRawFGMask_last);
	cv::gpu::morphologyEx(d_FGMask,d_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());	
	d_oFGMask_PreFlood.copyTo(d_oFGMask_FloodedHoles);
	d_oFGMask_FloodedHoles.download(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);	
	d_oFGMask_FloodedHoles.upload(m_oFGMask_FloodedHoles);
	cv::gpu::bitwise_not(d_oFGMask_FloodedHoles,d_oFGMask_FloodedHoles);
	cv::gpu::erode(d_oFGMask_PreFlood,d_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	cv::gpu::bitwise_or(d_FGMask,d_oFGMask_FloodedHoles,d_FGMask);	
	cv::gpu::bitwise_or(d_FGMask,d_oFGMask_PreFlood,d_FGMask);	
	cv::gpu::GaussianBlur(d_FGMask,d_oFGMask_last,cv::Size(3,3),1.0);	
	cv::gpu::dilate(d_oFGMask_last,d_oFGMask_last_dilated,cv::Mat(),cv::Point(-1,-1),3);
	cv::gpu::bitwise_and(d_oBlinksFrame,d_oFGMask_last_dilated_inverted,d_oBlinksFrame);
	cv::gpu::bitwise_not(d_oFGMask_last_dilated,d_oFGMask_last_dilated_inverted);
	cv::gpu::bitwise_and(d_oBlinksFrame,d_oFGMask_last_dilated_inverted,d_oBlinksFrame);
	d_oFGMask_last.copyTo(d_FGMask);
	d_FGMask.download(oCurrFGMask);
	const float fRollAvgFactor_LT = 1.0f/std::min(++m_nFrameIndex,m_nSamplesForMovingAvgs*4);
	const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIndex,m_nSamplesForMovingAvgs);
	d_oMeanFinalSegmResFrame_LT.download(m_oMeanFinalSegmResFrame_LT);
	d_oMeanFinalSegmResFrame_ST.download(m_oMeanFinalSegmResFrame_ST);
	cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
	cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);
	d_oMeanFinalSegmResFrame_LT.upload(m_oMeanFinalSegmResFrame_LT);
	d_oMeanFinalSegmResFrame_ST.upload(m_oMeanFinalSegmResFrame_ST);*/

	d_FGMask.download(m_rawFGMask);
	//memset(m_distance,0,sizeof(float)*m_oImgSize.height*m_oImgSize.width);
	MaskHomographyTest(m_rawFGMask,m_preGray,m_gray,m_homography, m_distance);
	//d_FGMask.copyTo(d_FGMask_last);
	
	//CudaRefreshModel(d_randStates,0.1f,m_oImgSize.width,m_oImgSize.height,d_features, d_voBGColorSamples,d_voBGDescSamples,d_fModels,d_bModels);
	/*uchar4* d_ptr = d_CurrentColorFrame.ptr<uchar4>();
	uchar4* h_ptr = new uchar4[m_oImgSize.width*m_oImgSize.height];
	cudaMemcpy(h_ptr,d_ptr,sizeof(uchar4)*m_oImgSize.width*m_oImgSize.height,cudaMemcpyDeviceToHost);
	cv::Mat himg(m_oImgSize,CV_8UC4,h_ptr);
	cv::imwrite("test.jpg",himg);*/	
	m_optimizer->Optimize(m_rawFGMask,m_features,oCurrFGMask);
	//m_optimizer->Optimize(m_gs,oInputImg,m_rawFGMask,m_preFeatures,oCurrFGMask);
	/*if (m_nFrameIndex == 99)
	{
		cv::imwrite("input.jpg",oInputImg);
		cv::imwrite("features.jpg",m_features);
		cv::imwrite("result.jpg",oCurrFGMask);
	}*/

	/*gtimer.Stop();
	std::cout<<"gpu mat post process  "<<gtimer.Elapsed()<<"ms"<<std::endl;*/
	//const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/m_nKeyPoints;
	//if(fCurrNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN) {
	//    for(size_t t=0; t<=UCHAR_MAX; ++t)
	//        if(m_anLBSPThreshold_8bitLUT[t]>cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4)))
	//            --m_anLBSPThreshold_8bitLUT[t];
	//}
	//else if(fCurrNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX) {
	//    for(size_t t=0; t<=UCHAR_MAX; ++t)
	//        if(m_anLBSPThreshold_8bitLUT[t]<cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+UCHAR_MAX*m_fRelLBSPThreshold))
	//            ++m_anLBSPThreshold_8bitLUT[t];
	//}
	//m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
	const float fRollAvgFactor_LT = 1.0f/std::min(m_nFrameIndex,m_nSamplesForMovingAvgs*4);
	const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIndex,m_nSamplesForMovingAvgs);
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
			//m_ofstream<<m_nFrameIndex<<" "<<fCurrColorDiffRatio<<" updateing m_fCurrLearningRateLowerCap " <<m_fCurrLearningRateLowerCap<<" m_fCurrLearningRateUpperCap "<<m_fCurrLearningRateUpperCap<<std::endl;
		}
		else {
			m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
			m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
			//m_ofstream<<m_nFrameIndex<<" "<<fCurrColorDiffRatio<<" updateing m_fCurrLearningRateLowerCap " <<m_fCurrLearningRateLowerCap<<" m_fCurrLearningRateUpperCap "<<m_fCurrLearningRateUpperCap<<std::endl;
		}
		if(m_nModelResetCooldown>0)
			--m_nModelResetCooldown;
	}
	
	//postProcessa(m_gray,oCurrFGMask);
	postProcessSegments(m_gray,oCurrFGMask);
	//MaskHomographyTest(oCurrFGMask,m_gray,m_preGray,m_homography,NULL);
}
void GpuBackgroundSubtractor::warpRefreshModel(float fSamplesRefreshFrac)
{
	std::cout<<m_nFrameIndex<<": refresh model"<<std::endl;
	// == refresh
	CV_Assert(m_bInitializedInternalStructs);
	CV_Assert(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f);
	const size_t nBGSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
	const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
	if(m_nImgChannels==1) {
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_orig_color = m_oLastColorFrame.cols*y_orig + x_orig;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = m_oLastColorFrame.cols*y_sample + x_sample;
				const size_t idx_sample_desc = idx_sample_color*2;
				const size_t idx_sample = s%m_nBGSamples;
				m_voBGColorSamples[idx_sample].data[idx_orig_color] = m_oLastColorFrame.data[idx_sample_color];
				*((ushort*)(m_voBGDescSamples[idx_sample].data+idx_orig_desc)) = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
			}
		}
	}
	else { //m_nImgChannels==3
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_orig_color = 3*(m_oLastColorFrame.cols*y_orig + x_orig);
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = 3*(m_oLastColorFrame.cols*y_sample + x_sample);
				const size_t idx_sample_desc = idx_sample_color*2;
				const size_t idx_sample = s%m_nBGSamples;
				uchar* bg_color_ptr = m_voBGColorSamples[idx_sample].data+idx_orig_color;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDescSamples[idx_sample].data+idx_orig_desc);
				
				const uchar* const init_color_ptr = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const init_desc_ptr = (ushort*)(m_oLastDescFrame.data+idx_sample_desc);
				
				for(size_t c=0; c<3; ++c) {
					bg_color_ptr[c] = init_color_ptr[c];
					bg_desc_ptr[c] = init_desc_ptr[c];
					
				}
			}
		}
	}
}
void GpuBackgroundSubtractor::refreshModel(float fSamplesRefreshFrac) {
	// == refresh
	CV_Assert(m_bInitializedInternalStructs);
	CV_Assert(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f);
	const size_t nBGSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
	const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
	if(m_nImgChannels==1) {
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_orig_color = m_oLastColorFrame.cols*y_orig + x_orig;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP_PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = m_oLastColorFrame.cols*y_sample + x_sample;
				const size_t idx_sample_desc = idx_sample_color*2;
				const size_t idx_sample = s%m_nBGSamples;
				m_voBGColorSamples[idx_sample].data[idx_orig_color] = m_oLastColorFrame.data[idx_sample_color];
				//*((ushort*)(m_voBGDescSamples[idx_sample].data+idx_orig_desc)) = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
			}
		}
	}
	else { //m_nImgChannels==3
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			//CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_orig_color = 4*(m_oLastColorFrame.cols*y_orig + x_orig);
			//CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP_PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = 4*(m_oLastColorFrame.cols*y_sample + x_sample);
				const size_t idx_sample_desc = idx_sample_color*2;
				const size_t idx_sample = s%m_nBGSamples;
				uchar* bg_color_ptr = m_voBGColorSamples[idx_sample].data+idx_orig_color;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDescSamples[idx_sample].data+idx_orig_desc);
				const uchar* const init_color_ptr = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const init_desc_ptr = (ushort*)(m_oLastDescFrame.data+idx_sample_color);
				for(size_t c=0; c<3; ++c) {
					bg_color_ptr[c] = init_color_ptr[c];
					bg_desc_ptr[c] = init_desc_ptr[c];
				}
			}
		}
	}
}

void GpuBackgroundSubtractor::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {
	//GpuBSOperator(_image,_fgmask);
	WarpBSOperator(_image,_fgmask);
	return;
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
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_ushrt = idx_uchar*2;
			const size_t idx_flt32 = idx_uchar*4;
			const uchar nCurrColor = oInputImg.data[idx_uchar];
			//size_t nMinDescDist = s_nDescMaxDataRange_1ch;
			size_t nMinSumDist = s_nColorMaxDataRange_1ch;
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
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_ushrt));
			uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_uchar])*STAB_COLOR_DIST_OFFSET))/2;
			//const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			//ushort nCurrInterDesc, nCurrIntraDesc;
			//LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			m_oUnstableRegionMask.data[idx_uchar] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar& nBGColor = m_voBGColorSamples[nSampleIdx].data[idx_uchar];
				{
					const size_t nColorDist = absdiff_uchar(nCurrColor,nBGColor);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					//const ushort& nBGIntraDesc = *((ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt));
					//const size_t nIntraDescDist = hdist_ushort_8bitLUT(nCurrIntraDesc,nBGIntraDesc);
					//LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,x,y,m_anLBSPThreshold_8bitLUT[nBGColor],nCurrInterDesc);
					//const size_t nInterDescDist = hdist_ushort_8bitLUT(nCurrInterDesc,nBGIntraDesc);
					//const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					//if(nDescDist>nCurrDescDistThreshold)
					//	goto failedcheck1ch;
					//const size_t nSumDist = std::min((nDescDist/4)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					//if(nSumDist>nCurrColorDistThreshold)
					//	goto failedcheck1ch;
					//if(nMinDescDist>nDescDist)
					//	nMinDescDist = nDescDist;
					//if(nMinSumDist>nSumDist)
					//	nMinSumDist = nSumDist;
					if(nMinSumDist>nColorDist)
						nMinSumDist = nColorDist;
					nGoodSamplesCount++;
				}
				failedcheck1ch:
				nSampleIdx++;
			}
			//const float fNormalizedLastDist = ((float)absdiff_uchar(nLastColor,nCurrColor)/s_nColorMaxDataRange_1ch+(float)hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc)/s_nDescMaxDataRange_1ch)/2;
			const float fNormalizedLastDist = (float)absdiff_uchar(nLastColor,nCurrColor)/s_nColorMaxDataRange_1ch;
			*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				// == foreground
				//const float fNormalizedMinDist = std::min(1.0f,((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				const float fNormalizedMinDist = std::min(1.0f,((float)nMinSumDist/s_nColorMaxDataRange_1ch) + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					//*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
			}
			else {
				// == background
				//const float fNormalizedMinDist = ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2;
				const float fNormalizedMinDist = (float)nMinSumDist/s_nColorMaxDataRange_1ch;
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					//*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
				int x_rand,y_rand;
				const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar];
				if(bCurrUsing3x3Spread)
					getRandNeighborPosition_3x3(x_rand,y_rand,x,y,LBSP_PATCH_SIZE/2,m_oImgSize);
				else
					getRandNeighborPosition_5x5(x_rand,y_rand,x,y,LBSP_PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*y_rand + x_rand;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
				const float fRandMeanRawSegmRes = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
					const size_t idx_rand_ushrt = idx_rand_uchar*2;
					const size_t s_rand = rand()%m_nBGSamples;
					//*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_rand_uchar] = nCurrColor;
				}
			}
			if(m_oFGMask_last.data[idx_uchar] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[idx_uchar])) {
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
				(*pfCurrVariationFactor) -= m_oFGMask_last.data[idx_uchar]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[idx_uchar]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
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
			/*if(popcount_ushort_8bitsLUT(nCurrIntraDesc)>=2)
				++nNonZeroDescCount;
			nLastIntraDesc = nCurrIntraDesc;*/
			nLastColor = nCurrColor;
		}
	}
	else { //m_nImgChannels==3
		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			const uchar* const anCurrColor = oInputImg.data+idx_uchar_rgb;
			//size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
			size_t nMinTotSumDist=s_nColorMaxDataRange_3ch;
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
			//ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;
			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_uchar])*STAB_COLOR_DIST_OFFSET));
			//const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
			//const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
			const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
			//ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			//const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			//LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			m_oUnstableRegionMask.data[idx_uchar] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				//const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+idx_uchar_rgb;
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anBGColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					//size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
					//LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
					//size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
					//const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					//const size_t nSumDist = std::min((nDescDist/2)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					//if(nSumDist>nCurrSCColorDistThreshold)
					//	goto failedcheck3ch;
					//nTotDescDist += nDescDist;
					//nTotSumDist += nSumDist;
					nTotSumDist += nColorDist;
				}
				//if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
				if(nTotSumDist>nCurrTotColorDistThreshold)
					goto failedcheck3ch;
				/*if(nMinTotDescDist>nTotDescDist)
					nMinTotDescDist = nTotDescDist;*/
				if(nMinTotSumDist>nTotSumDist)
					nMinTotSumDist = nTotSumDist;
				nGoodSamplesCount++;
				failedcheck3ch:
				nSampleIdx++;
			}
			//const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
			const float fNormalizedLastDist = (float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch;
			*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				// == foreground
				//const float fNormalizedMinDist = std::min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				const float fNormalizedMinDist = std::min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch) + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						//*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
					}
				}
			}
			else {
				// == background
				//const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2;
				const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						//*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
					}
				}
				int x_rand,y_rand;
				const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar];
				if(bCurrUsing3x3Spread)
					getRandNeighborPosition_3x3(x_rand,y_rand,x,y,LBSP_PATCH_SIZE/2,m_oImgSize);
				else
					getRandNeighborPosition_5x5(x_rand,y_rand,x,y,LBSP_PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*y_rand + x_rand;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
				const float fRandMeanRawSegmRes = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
					const size_t idx_rand_uchar_rgb = idx_rand_uchar*3;
					const size_t idx_rand_ushrt_rgb = idx_rand_uchar_rgb*2;
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						//*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_rand_uchar_rgb+c) = anCurrColor[c];
					}
				}
			}
		
			if(m_oFGMask_last.data[idx_uchar] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[idx_uchar])) {
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
				(*pfCurrVariationFactor) -= m_oFGMask_last.data[idx_uchar]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[idx_uchar]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
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
			///*if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>=4)
			//	++nNonZeroDescCount;*/
			//for(size_t c=0; c<3; ++c) {
			//	//anLastIntraDesc[c] = anCurrIntraDesc[c];
			//	anLastColor[c] = anCurrColor[c];
			//}
		}
	}
	
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
	cv::bitwise_xor(oCurrFGMask,m_oRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,m_oRawFGBlinkMask_last,m_oBlinksFrame);
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oRawFGMask_last);
	cv::morphologyEx(oCurrFGMask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());	
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);	
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_or(oCurrFGMask,m_oFGMask_FloodedHoles,oCurrFGMask);	
	cv::bitwise_or(oCurrFGMask,m_oFGMask_PreFlood,oCurrFGMask);	
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,m_nMedianBlurKernelSize);	
	cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	cv::bitwise_not(m_oFGMask_last_dilated,m_oFGMask_last_dilated_inverted);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	m_oFGMask_last.copyTo(oCurrFGMask);
	cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
	cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);
	//const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/m_nKeyPoints;
	//if(fCurrNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN) {
	//    for(size_t t=0; t<=UCHAR_MAX; ++t)
	//        if(m_anLBSPThreshold_8bitLUT[t]>cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4)))
	//            --m_anLBSPThreshold_8bitLUT[t];
	//}
	//else if(fCurrNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX) {
	//    for(size_t t=0; t<=UCHAR_MAX; ++t)
	//        if(m_anLBSPThreshold_8bitLUT[t]<cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+UCHAR_MAX*m_fRelLBSPThreshold))
	//            ++m_anLBSPThreshold_8bitLUT[t];
	//}
	//m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
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

void GpuBackgroundSubtractor::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	for(size_t s=0; s<m_nBGSamples; ++s) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_nimg = m_voBGColorSamples[s].step.p[0]*y + m_voBGColorSamples[s].step.p[1]*x;
				const size_t idx_flt32 = idx_nimg*4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+idx_flt32);
				const uchar* const oBGImgPtr = m_voBGColorSamples[s].data+idx_nimg;
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nBGSamples;
			}
		}
	}
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}

//ÌáÈ¡±ßÔµµã
void GpuBackgroundSubtractor::ExtractEdgePoint(const cv::Mat& img, const cv::Mat& edge, cv::Mat& edgeThetaMat,std::vector<EdgePoint>& edgePoints)
{
	using namespace cv;
	edgePoints.clear();
	Mat dx,dy;
	edgeThetaMat = cv::Scalar(0);
	cv::Sobel(img,dx,0,1,0);
	cv::Sobel(img,dy,0,0,1);
	for(int i=0; i< img.rows; i++)
	{
		for(int j=0; j<img.cols; j++)
		{
			int idx = i*img.cols + j;
			if (edge.data[idx] == 0xff)
			{				
				float theta = atan(dy.data[idx]*1.0/(dx.data[idx]+1e-6))/M_PI*180;
				/*std::cout<<theta<<std::endl;*/
				float avgColor = AvgColor(img,i,j);
				float* ptr = edgeThetaMat.ptr<float>(i)+2*j;
				*ptr= theta;
				*(ptr+1) = avgColor;
				edgePoints.push_back(EdgePoint(j,i,theta,avgColor));
			}
		}
	}

}

//±ßÔµµãÆ¥Åä
void GpuBackgroundSubtractor::MapEdgePoint(const std::vector<EdgePoint>& ePoints1, const cv::Mat& edge2,const cv::Mat edgeThetamat, const const cv::Mat& transform, float deltaTheta, cv::Mat& matchMask)
{
	using namespace cv;
	double * ptr = (double*)transform.data;
	int r = 0;//ËÑËØ·¶Î§
	int width = edge2.cols;
	int height = edge2.rows;
	//matchMask = Scalar(0);
	float thetaDist = 0.5;
	float colorDist = 10;
	for(int i=0; i<ePoints1.size(); i++)
	{
		int ox = ePoints1[i].x;
		int oy = ePoints1[i].y;
		float theta = ePoints1[i].theta;
		float color = ePoints1[i].color;
		float x,y,w;
		x = ox*ptr[0] + oy*ptr[1] + ptr[2];
		y = ox*ptr[3] + oy*ptr[4] + ptr[5];
		w = ox*ptr[6] + oy*ptr[7] + ptr[8];
		x /=w;
		y/=w;
		int wx = int(x+0.5);
		int wy = int(y+0.5);
		for(int m=-r; m<=r; m++)
		{
			for(int n=-r; n<=r; n++)
			{
				int nx  = wx + m;
				int ny = wy + n;
				if (nx>=0 && nx<width && ny >=0 && ny<height)
				{
					int id = nx + ny*width;
					int tid = ny*edgeThetamat.step.p[0]+nx*edgeThetamat.step.p[1];
					 float* angColorPtr = (float*)(edgeThetamat.data+tid);
					if (edge2.data[id]==255  && 
						abs( angColorPtr[0] - theta-deltaTheta) < thetaDist &&
						 abs(color - angColorPtr[1]) < colorDist)
					{
						//match
						matchMask.data[ox+oy*width] = UCHAR_MAX;
					}
				}
			}
		}
	}
}

void GpuBackgroundSubtractor::setAutomaticModelReset(bool b) {
	m_bAutoModelResetEnabled = b;
}


cv::AlgorithmInfo* GpuBackgroundSubtractor::info() const {
	return nullptr;
}



std::vector<cv::KeyPoint> GpuBackgroundSubtractor::getBGKeyPoints() const {
	return m_voKeyPoints;
}

void GpuBackgroundSubtractor::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	//LBSP::validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voKeyPoints = keypoints;
	m_nKeyPoints = keypoints.size();
}
void GpuBackgroundSubtractor::UpdateBackground(float* pfCurrLearningRate, int x, int y, size_t idx_ushrt, size_t idx_uchar, const ushort* anCurrIntraDesc, const uchar* anCurrColor)
{
	float fSamplesRefreshFrac = 0.1;
	const size_t nBGSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
	const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
	const size_t nLearningRate = 1;
	for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = 3*(m_oLastColorFrame.cols*y_sample + x_sample);
				const size_t idx_sample_desc = idx_sample_color*2;
				const size_t idx_sample = s%m_nBGSamples;
		
		for(size_t c=0; c<m_nImgChannels; ++c) {
			*((ushort*)(m_voBGDescSamples[idx_sample].data+idx_sample_desc+2*c)) = anCurrIntraDesc[c];
			*(m_voBGColorSamples[idx_sample].data+idx_sample_color+c) = anCurrColor[c];
		}

	}
	
}
void GpuBackgroundSubtractor::UpdateModel(const cv::Mat& curImg, const cv::Mat& curMask)
{
	for(size_t k=0; k<m_nKeyPoints; ++k) 
	{
		const int x = (int)m_voKeyPoints[k].pt.x;
		const int y = (int)m_voKeyPoints[k].pt.y;

		int idx_uchar = x+y*m_oImgSize.width;
		int idx_uchar_rgb = idx_uchar*3;
		int idx_ushrt_rgb = idx_uchar_rgb*2;
		int idx_flt32 = idx_uchar*4;
		float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
		uchar* anCurrColor = curImg.data + idx_uchar_rgb;
		ushort anCurrIntraDesc[3];
		const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
		LBSP::computeRGBDescriptor(curImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
		if (curMask.data[idx_uchar] == 0xff)
		{
			//update foreground
		}
		else
		{
			//update background
			UpdateBackground(pfCurrLearningRate,x,y,idx_ushrt_rgb,idx_uchar_rgb,anCurrIntraDesc,anCurrColor);
		}
	}

}