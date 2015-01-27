#include "WarpBackgroundSubtractor.h"
#define _USE_MATH_DEFINES
#include "WarpBackgroundSubtractor.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include "CudaBSOperator.h"
#include "LBSP.h"
#include "timer.h"
#include "GpuTimer.h"
#include "MotionEstimate.h"
#include "FeaturePointRefine.h"
#include "FGMaskPostProcess.h"
#include "Common.h"
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


WarpBackgroundSubtractor::WarpBackgroundSubtractor(	 float fRelLBSPThreshold
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
	,m_nLBSPThresholdOffset(0){
		CV_Assert(m_nBGSamples>0 && m_nRequiredBGSamples<=m_nBGSamples);
		CV_Assert(m_nMinColorDistThreshold>=STAB_COLOR_DIST_OFFSET);

}

WarpBackgroundSubtractor::~WarpBackgroundSubtractor() 
{
	delete m_ASAP;
	//delete m_DOFP;
	delete m_SPComputer;

	delete m_optimizer;
	delete m_gs;


}
void WarpBackgroundSubtractor::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints)
{

	//m_DOFP = new EPPMDenseOptialFlow();
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
	refreshModel(1.0f);
	//loadModels();
	/*cv::imwrite("cmlastdescfram.png",m_oLastDescFrame);
	char filename[20];
	for(int i=0; i<m_voBGDescSamples.size(); i++)
	{	
	sprintf(filename,"cpu%ddescmodel.png",i);		
	cv::imwrite(filename,m_voBGDescSamples[i]);
	}*/

	w_voBGColorSamples.resize(m_nBGSamples);
	w_voBGDescSamples.resize(m_nBGSamples);

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
	m_SPComputer = new SuperpixelComputer(m_oImgSize.width,m_oImgSize.height,5);
	m_optimizer = new MRFOptimize(m_oImgSize.width,m_oImgSize.height,5);

	//gpu
	d_CurrentColorFrame = gpu::createContinuous(oInitImg.size(),CV_8UC4);

	if (m_oImgSize.area() <= 320*240)
		m_rgThreshold = 0.40;
	else
		m_rgThreshold = 0.8;
}

void WarpBackgroundSubtractor::WarpModels()
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
void WarpBackgroundSubtractor::saveModels()
{
	char filename[200];

	for(int i=0;i <1; i++)
	{			

		sprintf(filename,"cpu%ddescmodel_%d.png",i,m_nFrameIndex);		
		imwrite(filename, m_voBGDescSamples[i]);




		sprintf(filename,"cpu%dmodel_%d.png",i,m_nFrameIndex);		
		imwrite(filename, m_voBGDescSamples[i]);

		/*sprintf(filename,"cpu%dmodel.jpg",i);
		imwrite(filename, m_voBGColorSamples[i]);
		sprintf(filename,"cpu%descmodel.jpg",i);
		imwrite(filename, m_voBGDescSamples[i]);*/
	}
}
void WarpBackgroundSubtractor::loadModels()
{
	char filename[200];

	for(int i=0;i <m_nBGSamples; i++)
	{			

		sprintf(filename,"desc_model_%d.png",i);			
		m_voBGDescSamples[i] = cv::imread(filename,-1);




		sprintf(filename,"color_model_%d.png",i);				
		m_voBGColorSamples[i] = cv::imread(filename,-1);
		/*sprintf(filename,"cpu%dmodel.jpg",i);
		imwrite(filename, m_voBGColorSamples[i]);
		sprintf(filename,"cpu%descmodel.jpg",i);
		imwrite(filename, m_voBGDescSamples[i]);*/
	}
}
void WarpBackgroundSubtractor::WarpImage(const cv::Mat image, cv::Mat& warpedImg)
{


	/*cv::warpPerspective(img,warpedImg,m_homography,m_oImgSize);*/
	m_img = image.clone();
	if (m_preImg.empty())
	{
		m_preImg = m_img.clone();

	}
	if (image.channels() ==3)
	{
		cv::cvtColor(image, m_gray, CV_BGR2GRAY); 
	}
	else
		m_gray = image;
	cv::Mat rgbaImg;
	cv::cvtColor(image,rgbaImg,CV_BGR2BGRA);
	d_CurrentColorFrame.upload(rgbaImg);
	cv::gpu::cvtColor(d_CurrentColorFrame,d_gray,CV_BGRA2GRAY);

	if (d_preGray.empty())
	{
		d_gray.copyTo(d_preGray);

	}

	if (m_preGray.empty())
		m_gray.copyTo(m_preGray);

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

	//calculate dense flow 
	/*m_DOFP->DenseOpticalFlow(m_gray,m_preGray,m_flow);*/
	//计算超像素



	int * labels(NULL), *preLabels(NULL);
	SLICClusterCenter* centers(NULL), *preCenters(NULL);
	int num(0);
	float avgE(0);
	int step(5);
	int spHeight = (m_oImgSize.height+step-1)/step;
	int spWidth = (m_oImgSize.width+step-1)/step;
	cv::Mat spFlow;
	m_SPComputer->ComputeSuperpixel(d_CurrentColorFrame.ptr<uchar4>(),num,labels,centers);

#ifndef REPORT
	nih::Timer cpuTimer;
	cpuTimer.start();
#endif
	m_points[0].clear();
	cv::goodFeaturesToTrack(m_gray,m_points[0],100,0.08,10);
	int nf = m_points[0].size();
	for(int i=0; i<num; i++)
	{
		m_points[0].push_back(cv::Point2f(centers[i].xy.x,centers[i].xy.y));
	}
	upload(m_points[0],d_currPts);
#ifndef REPORT
	cpuTimer.stop();
	std::cout<<"	goodFeaturesToTrack  "<<cpuTimer.seconds()*1000<<" ms"<<std::endl;
#endif
#ifndef REPORT
	cpuTimer.start();
#endif
	std::vector<uchar>status;
	std::vector<float> err;
	//cv::calcOpticalFlowPyrLK(m_gray,m_preGray,m_points[0],m_points[1],status,err);
	d_pyrLk.sparse(d_gray,d_preGray,d_currPts,d_prevPts,d_status);
	download(d_status,m_status);
	download(d_currPts,m_points[0]);
	download(d_prevPts,m_points[1]);
#ifndef REPORT
	cpuTimer.stop();
	std::cout<<"	calcOpticalFlowPyrLK  "<<cpuTimer.seconds()*1000<<" ms"<<std::endl;
#endif

#ifndef REPORT
	cpuTimer.start();
#endif

	SuperpixelFlow(spWidth,spHeight,num,centers,nf,m_points[0],m_points[1],m_status,spFlow);

#ifndef REPORT
	cpuTimer.stop();
	std::cout<<"	Superpixel Flow "<<cpuTimer.seconds()*1000<<" ms"<<std::endl;
#endif	

#ifndef REPORT
	cpuTimer.start();
#endif
	m_goodFeatures[0].resize(nf);
	m_goodFeatures[1].resize(nf);
	size_t size = sizeof(cv::Point2f)*nf;
	memcpy(&m_goodFeatures[0][0],&m_points[0][0],size);
	memcpy(&m_goodFeatures[1][0],&m_points[1][0],size);
	FeaturePointsRefineRANSAC(nf,m_goodFeatures[0],m_goodFeatures[1],m_homography);
	//FeaturePointsRefineHistogram(nf,m_oImgSize.width,m_oImgSize.height,m_points[0],m_points[1]);
#ifndef REPORT
	cpuTimer.stop();
	std::cout<<"	FeaturePointsRefineRANSAC "<<cpuTimer.seconds()*1000<<" ms"<<std::endl;

#endif	

#ifndef REPORT
	cpuTimer.start();

#endif		
	m_ASAP->SetControlPts(m_goodFeatures[0],m_goodFeatures[1]);
	m_ASAP->Solve();
	m_ASAP->Warp(image,warpedImg);
	m_ASAP->Reset();
	m_ASAP->getFlow(m_wflow);
#ifndef REPORT
	cpuTimer.stop();
	std::cout<<"	ASAP Warping "<<cpuTimer.seconds()*1000<<std::endl;
#endif


#ifndef REPORT
	cpuTimer.start();

#endif	
	m_SPComputer->GetPreSuperpixelResult(num,preLabels,preCenters);
	int rows = m_oImgSize.height;
	int cols = m_oImgSize.width;
	SuperpixelMatching(labels,centers,m_img,preLabels,preCenters,m_preImg,num,step,cols,rows,spFlow,m_matchedId);

#ifndef REPORT
	cpuTimer.stop();
	std::cout<<"	superpixel matching "<<cpuTimer.seconds()*1000<<std::endl;
#endif

#ifndef REPORT
	cpuTimer.start();
#endif
	avgE = m_SPComputer->ComputAvgColorDistance();
	std::vector<int> resLabels;
	for(int i=nf,j=0; i<m_points[0].size(); i++,j++)
	{

		cv::Point2f pt = m_points[0][i];
		//比较superpixel flow 与 warping的结果，若一致则认为是背景超像素
		int idx = (int)pt.x+(int)(pt.y)*m_oImgSize.width;
		float2* flow = (float2*)(spFlow.data+labels[idx]*8);
		float2* wflow = (float2*)(m_wflow.data+idx*8);
		if (abs(flow->x-wflow->x) + abs(flow->y-wflow->y) < m_rgThreshold)
			resLabels.push_back(labels[idx]);

	}
	int * rgResult(NULL);
	m_SPComputer->RegionGrowingFast(resLabels,2.0*avgE,rgResult);
	/*m_SPComputer->GetRegionGrowingImg(m_features);	
	char filename[200];	
	sprintf(filename,".\\features\\input3\\features%06d.jpg",m_nFrameIndex+1);
	cv::imwrite(filename,m_features);*/
#ifndef REPORT
	cpuTimer.stop();
	std::cout<<"	superpixel Regiongrowing "<<cpuTimer.seconds()*1000<<std::endl;
#endif




	cv::swap(m_gray,m_preGray);
	cv::swap(m_preImg,m_img);
	cv::gpu::swap(d_gray,d_preGray);
}
void WarpBackgroundSubtractor::BSOperator(cv::InputArray _image, cv::OutputArray _fgmask)
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


#ifndef REPORT
	nih::Timer timer;
	timer.start();
#endif
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


			if (wx<2 || wx>= m_oImgSize.width-2 || wy<2 || wy>=m_oImgSize.height-2)
			{					
				//m_features.data[oidx_uchar] = 0xff;
				outMask.data[idx_uchar] = 0xff;

				continue;
			}
			else
			{

				if (fx<2 || fx>= m_oImgSize.width-2 || fy<2 || fy>=m_oImgSize.height-2)
				{

					outMask.data[idx_uchar] = 0xff;
					/*size_t anCurrIntraLBSPThresholds[3]; 
					for(size_t c=0; c<3; ++c) {
						const uchar nCurrBGInitColor = img.data[idx_uchar_rgb+c];
						m_oLastColorFrame.data[idx_uchar_rgb+c] = nCurrBGInitColor;
						anCurrIntraLBSPThresholds[c] = m_anLBSPThreshold_8bitLUT[nCurrBGInitColor];
						LBSP::computeSingleRGBDescriptor(img,nCurrBGInitColor,x,y,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(w_oLastDescFrame.data+idx_ushrt_rgb))[c]);
					}*/
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
			/*if(m_oFGMask_last.data[idx_uchar] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[idx_uchar])) {
			if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
			*pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
			}
			else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
			*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
			if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
			*pfCurrLearningRate = m_fCurrLearningRateLowerCap;
			else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
			*pfCurrLearningRate = m_fCurrLearningRateUpperCap;*/
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
			/*	if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>=4)
			++nNonZeroDescCount;*/
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
#ifndef REPORT
	timer.stop();
	std::cout<<"bs operator "<<timer.seconds()*1000<<std::endl;
#endif
	/*char filename[50];
	sprintf(filename,"bin%06d.jpg",m_nFrameIndex);
	cv::imwrite(filename,oCurrFGMask);*/
	/*cv::imshow("mask",oCurrFGMask);
	cv::waitKey();*/
	/*cv::remap(m_oRawFGMask_last,m_oRawFGMask_last,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	cv::bitwise_xor(oCurrFGMask,m_oRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::remap(m_oRawFGBlinkMask_last,m_oRawFGBlinkMask_last,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,m_oRawFGBlinkMask_last,m_oBlinksFrame);
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oRawFGMask_last);*/
	/*cv::bitwise_xor(oCurrFGMask,m_oRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,m_oRawFGBlinkMask_last,m_oBlinksFrame);
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);*/

#ifndef REPORT
	timer.start();
#endif
	
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
	//cv::remap(m_oFGMask_last,m_oFGMask_last,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	cv::remap(oCurrFGMask,oCurrFGMask,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	oCurrFGMask.copyTo(m_oRawFGMask_last);
	//MaskHomographyTest(oCurrFGMask,m_preGray,m_gray,m_homography);
	/*cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
	cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);*/
	/*const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/m_nKeyPoints;
	if(fCurrNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN) {
	for(size_t t=0; t<=UCHAR_MAX; ++t)
	if(m_anLBSPThreshold_8bitLUT[t]>cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4)))
	--m_anLBSPThreshold_8bitLUT[t];
	}
	else if(fCurrNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX) {
	for(size_t t=0; t<=UCHAR_MAX; ++t)
	if(m_anLBSPThreshold_8bitLUT[t]<cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+UCHAR_MAX*m_fRelLBSPThreshold))
	++m_anLBSPThreshold_8bitLUT[t];
	}*/
	//m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
	//if(m_bLearningRateScalingEnabled) {
	//	cv::resize(oInputImg,m_oDownSampledColorFrame,m_oDownSampledFrameSize,0,0,cv::INTER_AREA);
	//	cv::accumulateWeighted(m_oDownSampledColorFrame,m_oMeanDownSampledLastDistFrame_LT,fRollAvgFactor_LT);
	//	cv::accumulateWeighted(m_oDownSampledColorFrame,m_oMeanDownSampledLastDistFrame_ST,fRollAvgFactor_ST);
	//	size_t nTotColorDiff = 0;
	//	for(int i=0; i<m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
	//		const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0]*i;
	//		for(int j=0; j<m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
	//			const size_t idx2 = idx1+m_oMeanDownSampledLastDistFrame_ST.step.p[1]*j;
	//			nTotColorDiff += (m_nImgChannels==1)?
	//				(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2)))/2
	//				:  //(m_nImgChannels==3)
	//			std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2))),
	//				std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+4))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+4))),
	//				(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+8))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+8)))));
	//		}
	//	}
	//	const float fCurrColorDiffRatio = (float)nTotColorDiff/(m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
	//	if(m_bAutoModelResetEnabled) {
	//		if(m_nFramesSinceLastReset>1000)
	//			m_bAutoModelResetEnabled = false;
	//		else if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD && m_nModelResetCooldown==0) {
	//			m_nFramesSinceLastReset = 0;
	//			//refreshModel(0.1f); // reset 10% of the bg model
	//			m_nModelResetCooldown = m_nSamplesForMovingAvgs;
	//			m_oUpdateRateFrame = cv::Scalar(1.0f);
	//		}
	//		else
	//			++m_nFramesSinceLastReset;
	//	}
	//	else if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD*2) {
	//		m_nFramesSinceLastReset = 0;
	//		m_bAutoModelResetEnabled = true;
	//	}
	//	if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD/2) {
	//		m_fCurrLearningRateLowerCap = (float)std::max((int)FEEDBACK_T_LOWER>>(int)(fCurrColorDiffRatio/2),1);
	//		m_fCurrLearningRateUpperCap = (float)std::max((int)FEEDBACK_T_UPPER>>(int)(fCurrColorDiffRatio/2),1);
	//	}
	//	else {
	//		m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
	//		m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
	//	}
	//	if(m_nModelResetCooldown>0)
	//		--m_nModelResetCooldown;
	//		//refreshEdgeModel(0.1);
	//	
	//}
	/*sprintf(filename,"outmask%d.jpg",m_nFrameIndex-1);
	cv::imwrite(filename,outMask);*/

	//m_optimizer->Optimize(m_gs,img,m_oRawFGMask_last,m_features,oCurrFGMask);
	//m_optimizer->Optimize(m_gs,img,m_oRawFGMask_last,m_flow,m_wflow,oCurrFGMask);
	//m_optimizer->Optimize(m_oRawFGMask_last,m_features,oCurrFGMask);
	//m_optimizer->Optimize(m_SPComputer,m_oRawFGMask_last,m_features,m_matchedId,oCurrFGMask);
	m_optimizer->Optimize(m_SPComputer,m_oRawFGMask_last,m_matchedId,oCurrFGMask);
	postProcessSegments(img,oCurrFGMask);
	WarpModels();
#ifndef REPORT
	timer.stop();
	std::cout<<"optimize "<<timer.seconds()*1000<<" ms\n";
#endif
	//saveModels();
	/*cv::remap(m_fgCounter,m_fgCounter,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	int threshold = 10;
	for(int i=0; i<m_oImgSize.height; i++)
	{
	ushort* cPtr = m_fgCounter.ptr<ushort>(i);
	uchar* mPtr = oCurrFGMask.ptr<uchar>(i);
	for(int j=0; j<m_oImgSize.width; j++)
	{
	if (mPtr[j] == 0xff)
	{	
	cPtr[j]++;
	if (cPtr[j] > threshold)
	{
	mPtr[j] = 100;
	cPtr[j] = 0;
	}
	}
	else
	cPtr[j] = 0;

	}
	}*/
	//postProcessSegments(img,oCurrFGMask);
#ifndef REPORT
	timer.start();
#endif
	cv::Mat refeshMask;	
	cv::bitwise_or(outMask,oCurrFGMask,refeshMask);
	UpdateModel(img,refeshMask);
#ifndef REPORT
	timer.stop();
	std::cout<<"update model "<<timer.seconds()*1000<<" ms\n";
#endif
	//if (m_nOutPixels > 0.4*m_oImgSize.height*m_oImgSize.width)
	//{
	//	std::cout<<"refresh model\n";
	//	cv::Mat empty(oCurrFGMask.size(),oCurrFGMask.type(),cv::Scalar(0));
	//	UpdateModel(img,empty);
	//	//resetPara();
	//	m_nOutPixels = 0;
	//}
	//refreshModel(outMask,0.1);
}



void WarpBackgroundSubtractor::refreshModel(float fSamplesRefreshFrac)
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
			//CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_orig_color = 3*(m_oLastColorFrame.cols*y_orig + x_orig);
			//CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
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


void WarpBackgroundSubtractor::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {

	BSOperator(_image,_fgmask);
	return;

}

void WarpBackgroundSubtractor::getBackgroundImage(cv::OutputArray backgroundImage) const {
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



void WarpBackgroundSubtractor::setAutomaticModelReset(bool b) {
	m_bAutoModelResetEnabled = b;
}


cv::AlgorithmInfo* WarpBackgroundSubtractor::info() const {
	return nullptr;
}



std::vector<cv::KeyPoint> WarpBackgroundSubtractor::getBGKeyPoints() const {
	return m_voKeyPoints;
}

void WarpBackgroundSubtractor::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	//LBSP::validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voKeyPoints = keypoints;
	m_nKeyPoints = keypoints.size();
}
void WarpBackgroundSubtractor::UpdateBackground(float* pfCurrLearningRate, int x, int y, size_t idx_ushrt, size_t idx_uchar, const ushort* anCurrIntraDesc, const uchar* anCurrColor)
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
void WarpBackgroundSubtractor::UpdateModel(const cv::Mat& curImg, const cv::Mat& curMask)
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
			if((rand()%(size_t)FEEDBACK_T_LOWER)==0) {
			const size_t s_rand = rand()%m_nBGSamples;
			for(size_t c=0; c<3; ++c) {
			*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
			*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
			}
			}
		}
		else
		{
			//update background
			UpdateBackground(pfCurrLearningRate,x,y,idx_ushrt_rgb,idx_uchar_rgb,anCurrIntraDesc,anCurrColor);
		}
	}

}

void GpuWarpBackgroundSubtractor::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints)
{

	//m_DOFP = new EPPMDenseOptialFlow();
	m_ASAP = new ASAPWarping(oInitImg.cols,oInitImg.rows,8,1.0);


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
	m_nPixels = oInitImg.rows*oInitImg.cols;
	cv::Mat initImg(oInitImg.size(),CV_8UC4);
	cv::cvtColor(oInitImg,initImg,CV_BGR2BGRA);

	m_oImgSize = oInitImg.size();
	m_nPixels = oInitImg.rows*oInitImg.cols;
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nFrameIndex = 0;
	m_nFramesSinceLastReset = 0;
	m_nModelResetCooldown = 0;
	m_fLastNonZeroDescRatio = 0.0f;
	const int nTotImgPixels = m_oImgSize.height*m_oImgSize.width;
	std::vector<cv::KeyPoint> voNewKeyPoints;
	if(voKeyPoints.empty()) {
		cv::DenseFeatureDetector oKPDDetector(1.f, 1, 1.f, 1, 0, true, false);
		voNewKeyPoints.reserve(oInitImg.rows*oInitImg.cols);
		oKPDDetector.detect(cv::Mat(oInitImg.size(),oInitImg.type()),voNewKeyPoints);
	}
	LBSP::validateKeyPoints(voNewKeyPoints,oInitImg.size());
	m_voKeyPoints = voNewKeyPoints;
	m_nKeyPoints = m_voKeyPoints.size();

	m_bLearningRateScalingEnabled = false;
	m_bAutoModelResetEnabled = false;
	m_bUse3x3Spread = true;
	m_nMedianBlurKernelSize = DEFAULT_MEDIAN_BLUR_KERNEL_SIZE;
	m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER*2;
	m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER*2;


	//50 samples + last frame in one big 1* n matrix
	d_voBGColorSamples.create(1,(m_nBGSamples+1)*m_oImgSize.width*m_oImgSize.height,CV_8UC4);
	d_wvoBGColorSamples.create(d_voBGColorSamples.size(),d_voBGColorSamples.type());
	d_voBGDescSamples.create(d_voBGColorSamples.size(),CV_16UC4);
	d_wvoBGDescSamples.create(d_voBGDescSamples.size(),d_voBGDescSamples.type());
	d_fModels.create(1,10*m_nPixels,CV_32FC1);
	d_wfModels.create(d_fModels.size(),d_fModels.type());
	d_bModels.create(1,2*m_nPixels,CV_8UC1);
	d_wbModels.create(d_bModels.size(),d_bModels.type());
	m_FGMask.create(m_oImgSize,CV_8U);
	m_outMask.create(m_oImgSize,CV_8U);
	//std::cout << m_oImgSize << " => m_nMedianBlurKernelSize=" << m_nMedianBlurKernelSize << ", with 3x3Spread=" << m_bUse3x3Spread << ", with Tscaling=" << m_bLearningRateScalingEnabled << std::endl;

	m_oLastColorFrame.create(m_oImgSize,CV_8UC(4));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC(4));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);



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
			//CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_color = 4*(m_oLastColorFrame.cols*y_orig + x_orig);
			//CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;

			size_t anCurrIntraLBSPThresholds[3]; 
			for(size_t c=0; c<3; ++c) {
				const uchar nCurrBGInitColor = initImg.data[idx_color+c];
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
	//gpu side
	cudaMalloc(&d_anLBSPThreshold_8bitLUT ,sizeof(size_t)*(UCHAR_MAX+1));
	cudaMemcpy(d_anLBSPThreshold_8bitLUT,m_anLBSPThreshold_8bitLUT,sizeof(size_t)*(UCHAR_MAX+1),cudaMemcpyHostToDevice);
	d_CurrentColorFrame = gpu::createContinuous(oInitImg.size(),CV_8UC4);

	d_CurrWarpedColorFrame = gpu::createContinuous(oInitImg.size(),CV_8UC4);
	//d_CurrentColorFrame.create(oInitImg.size(),CV_8U);
	d_FGMask.create(oInitImg.size(),CV_8U);
	d_FGMask_last = cv::gpu::GpuMat(oInitImg.size(),CV_8U,cv::Scalar(0));
	
	d_outMask.create(oInitImg.size(),CV_8U);
	d_Map.create(oInitImg.size(),CV_32FC2);
	d_invMap.create(oInitImg.size(),CV_32FC2);
	cv::Mat iniImg,outImg;
	cv::cvtColor(oInitImg,iniImg,CV_BGR2BGRA);

	uchar* dataPtr = d_voBGColorSamples.data + (m_nPixels*4*m_nBGSamples);	
	cudaMemcpy(dataPtr,m_oLastColorFrame.data,m_nPixels*4,cudaMemcpyHostToDevice);

	dataPtr = d_voBGDescSamples.data + (m_nPixels*8*m_nBGSamples);
	cudaMemcpy(dataPtr,m_oLastDescFrame.data,m_nPixels*8,cudaMemcpyHostToDevice);
	/*cv::Mat tmp;
	cv::cvtColor(m_oLastDescFrame,tmp,CV_BGRA2BGR);
	cv::imwrite("mLastDesc.png",tmp);*/

	cudaMalloc ( &d_randStates, m_oImgSize.width*m_oImgSize.height*sizeof( curandState ) );   
	InitRandState(m_oImgSize.width,m_oImgSize.height,d_randStates);
	InitConstantMem(m_anLBSPThreshold_8bitLUT);


	m_bInitializedInternalStructs = true;
	

	cv::Mat rfmask = cv::Mat::zeros(m_oImgSize,CV_8U);
	cv::gpu::GpuMat drfmask(rfmask);
	d_CurrentColorFrame.upload(initImg);
	CudaBindImgTexture(d_CurrentColorFrame);
	cv::gpu::GpuMat d_refreshMask;
	cv::Mat zeros = cv::Mat::zeros(m_oImgSize,CV_8U);
	d_refreshMask.upload(zeros);
	GpuTimer gtimer;
	gtimer.Start();
	CudaRefreshModel(d_randStates,1.f, m_oImgSize.width,m_oImgSize.height, d_voBGColorSamples,d_voBGDescSamples,d_fModels,d_bModels);
	//CudaUpdateModel(d_randStates,d_CurrentColorFrame,m_oImgSize.width,m_oImgSize.height,d_refreshMask,d_voBGColorSamples,d_voBGDescSamples);
	gtimer.Stop();
	std::cout<<gtimer.Elapsed()<<" ms\n";

	//saveModels();


	m_bInitialized = true;	
	m_gs = new GpuSuperpixel(m_oImgSize.width,m_oImgSize.height,5);
	m_SPComputer = new SuperpixelComputer(m_oImgSize.width,m_oImgSize.height,5);
	m_optimizer = new MRFOptimize(m_oImgSize.width,m_oImgSize.height,5);

	cudaMalloc(&d_outMaskPtr,m_nPixels);

	if (m_oImgSize.area() <= 320*240)
		m_rgThreshold = 0.1;
	else
		m_rgThreshold = 0.8;

}


void GpuWarpBackgroundSubtractor::BSOperator(cv::InputArray _image, cv::OutputArray _fgmask)
{
#ifndef REPORT
	std::cout<<m_nFrameIndex<<"---\n";
#endif
	cv::Mat oInputImg;
	cv::Mat img = _image.getMat();	
#ifndef REPORT
	nih::Timer cpuTimer;
	cpuTimer.start();
#endif
	WarpImage(img,oInputImg);

#ifndef REPORT
	cpuTimer.stop();
	std::cout<<"warp image "<<cpuTimer.seconds()*1000<<" ms\n";
#endif

#ifndef REPORT
	GpuTimer gtimer;
	gtimer.Start();
#endif
	cv::Mat mat2[] = {m_ASAP->getMapX(),m_ASAP->getMapY()};	
	cv::Mat map;
	cv::merge(mat2,2,map);


	cv::Mat invMat2[] = {m_ASAP->getInvMapX(),m_ASAP->getInvMapY()};
	cv::Mat invMap;
	cv::merge(invMat2,2,invMap);
	d_Map.upload(map);
	d_invMap.upload(invMap);
	cv::Mat wimg;
	cv::cvtColor(oInputImg,wimg,CV_BGR2BGRA);
	d_CurrWarpedColorFrame.upload(wimg);
	/*cv::imshow("warped img",oInputImg);
	cv::waitKey();*/

	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	
	WarpCudaBSOperator(d_CurrentColorFrame,d_CurrWarpedColorFrame, d_randStates,d_Map,d_invMap,++m_nFrameIndex,d_voBGColorSamples, d_wvoBGColorSamples,
		d_voBGDescSamples,d_wvoBGDescSamples,d_bModels,d_wbModels,d_fModels,d_wfModels,d_FGMask, d_FGMask_last,d_outMaskPtr,m_fCurrLearningRateLowerCap,m_fCurrLearningRateUpperCap);

	d_FGMask.download(oCurrFGMask);
	cv::remap(oCurrFGMask,oCurrFGMask,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	oCurrFGMask.copyTo(m_oRawFGMask_last);
	//cv::remap(oCurrFGMask,oCurrFGMask,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	cudaMemcpy(m_outMask.data,d_outMaskPtr,m_nPixels,cudaMemcpyDeviceToHost);

#ifndef REPORT
	gtimer.Stop();
	std::cout<<"gpu bs operator "<<gtimer.Elapsed()<<std::endl;
#endif
	/*char filename[50];
	sprintf(filename,"bin%06d.jpg",m_nFrameIndex);
	cv::imwrite(filename,oCurrFGMask);*/
	/*cv::imshow("mask",oCurrFGMask);
	cv::imshow("out mask",m_outMask);
	cv::waitKey();*/
#ifndef REPORT
	cpuTimer.start();
#endif
	m_optimizer->Optimize(m_SPComputer,oCurrFGMask,m_matchedId,oCurrFGMask);
	postProcessSegments(img,oCurrFGMask);
	/*cv::imshow("after optimize ",oCurrFGMask);
	cv::waitKey();*/
	swapModels();
	cv::Mat refeshMask;

	cv::bitwise_or(m_outMask,oCurrFGMask,refeshMask);
	//cv::imshow("refresh mask", refeshMask);
	//cv::waitKey();
#ifndef REPORT
	cpuTimer.stop();
	std::cout<<" optimize "<<cpuTimer.seconds()*1000<<"ms\n";

	gtimer.Start();
	
#endif
	d_FGMask_last = d_FGMask.clone();
	CudaUpdateModel(d_randStates,d_CurrentColorFrame,m_oImgSize.width,m_oImgSize.height,d_FGMask,d_voBGColorSamples,d_voBGDescSamples);
#ifndef REPORT
	gtimer.Stop();
	std::cout<<"gpu update model "<<gtimer.Elapsed()<<std::endl;
#endif	//saveModels();
}
void GpuWarpBackgroundSubtractor::loadModels()
{

}
void GpuWarpBackgroundSubtractor::saveModels()
{
	cv::Mat h_tmp,tmp;	
	h_tmp.create(m_oImgSize,CV_16UC4);
	cv::Mat diff;
	char filename[200];
	/*cudaMemcpy(h_tmp.data, d_voBGDescSamples.data + (m_nPixels*8*50),m_nPixels*2*4,cudaMemcpyDeviceToHost);
	cv::imwrite("lastDescFrame.png",h_tmp);
	cv::cvtColor(h_tmp,h_tmp,CV_BGRA2BGR);
	cv::imwrite("lastDescFrame3.png",h_tmp);*/

	cv::gpu::GpuMat d_tmp,d_ctmp;
	d_tmp.create(m_oImgSize,CV_16UC4);	
	d_ctmp.create(m_oImgSize,CV_8UC4);

	for(int i=0;i <50; i++)
	{			
		DownloadModel(m_oImgSize.width,m_oImgSize.height,d_voBGDescSamples,50,i,d_tmp);
		d_tmp.download(h_tmp);
		cv::cvtColor(h_tmp,tmp,CV_BGRA2BGR);
		sprintf(filename,"desc_model_%d.png",i);		
		imwrite(filename, tmp);



		DownloadColorModel(m_oImgSize.width,m_oImgSize.height,d_voBGColorSamples,50,i,d_ctmp);
		d_ctmp.download(h_tmp);
		cv::cvtColor(h_tmp,tmp,CV_BGRA2BGR);
		sprintf(filename,"color_model_%d.png",i);		
		imwrite(filename, tmp);

		/*sprintf(filename,"cpu%dmodel.jpg",i);
		imwrite(filename, m_voBGColorSamples[i]);
		sprintf(filename,"cpu%descmodel.jpg",i);
		imwrite(filename, m_voBGDescSamples[i]);*/
	}
}
void GpuWarpBackgroundSubtractor::refreshModel(float fSamplesRefreshFrac)
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
			//CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_orig_color = 4*(m_oLastColorFrame.cols*y_orig + x_orig);
			//CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
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
GpuWarpBackgroundSubtractor::~GpuWarpBackgroundSubtractor()
{
	cudaFree(d_anLBSPThreshold_8bitLUT);
	cudaFree(d_outMaskPtr);	
	cudaFree(d_randStates);


}

void WarpSPBackgroundSubtractor::WarpSPImg()
{
	cv::cvtColor(m_spDSImg,m_spDSGray,CV_BGR2GRAY);
	if (m_preSPDSGray.empty())
	{
		m_preSPDSGray = m_spDSGray.clone();
	}
	std::vector<cv::Point2f> f0,f1;
	std::vector<float> err;
	std::vector<uchar> status;
	cv::goodFeaturesToTrack(m_spDSGray,f0,500,0.05,2);
	cv::calcOpticalFlowPyrLK(m_spDSGray,m_preSPDSGray,f0,f1,status,err);
	int k=0;
	for(int i=0; i<f0.size(); i++)
	{
		if (status[i] == 1)
		{
			f0[k] = f0[i];
			f1[k] = f1[i];
			k++;
		}
	}
	f0.resize(k);
	f1.resize(k);
	status.resize(k);
	cv::findHomography(f0,f1,status,CV_RANSAC,0.1);
	k=0;
	for(int i=0; i<f0.size(); i++)
	{
		if (status[i] == 1)
		{
			f0[k] = f0[i];
			f1[k] = f1[i];
			k++;
		}
	}
	f0.resize(k);
	f1.resize(k);
	m_spASAP->SetControlPts(f0,f1);
	m_spASAP->Solve();
	m_spASAP->Warp(m_spDSImg,m_wspDSImg);
	m_spASAP->Reset();
	cv::swap(m_preSPDSGray,m_spDSGray);

}
void WarpSPBackgroundSubtractor::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints)
{
	m_step = 5;
	m_SPComputer = new SuperpixelComputer(oInitImg.cols,oInitImg.rows,m_step);
	m_SPComputer->ComputeSuperpixel(oInitImg);
	m_SPComputer->GetSuperpixelDownSampleImg(m_spDSImg);
	m_spWidth = (oInitImg.cols+m_step-1)/m_step;
	m_spHeight = (oInitImg.rows + m_step -1)/ m_step;
	m_spImgSize = cv::Size(m_spWidth,m_spHeight);
	m_ASAP = new ASAPWarping(oInitImg.cols,oInitImg.rows,8,1.0);
	m_spASAP = new ASAPWarping(m_spWidth,m_spHeight,8,1.0);
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
		voNewKeyPoints.reserve(m_spWidth*m_spHeight);
		oKPDDetector.detect(cv::Mat(m_spImgSize,oInitImg.type()),voNewKeyPoints);
	}
	else
		voNewKeyPoints = voKeyPoints;
	const size_t nOrigKeyPointsCount = voNewKeyPoints.size();
	CV_Assert(nOrigKeyPointsCount>0);
	LBSP::validateKeyPoints(voNewKeyPoints,m_spImgSize);
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
	m_oUpdateRateFrame.create(m_spImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(m_fCurrLearningRateLowerCap);
	m_oDistThresholdFrame.create(m_spImgSize,CV_32FC1);
	m_oDistThresholdFrame = cv::Scalar(1.0f);
	m_oVariationModulatorFrame.create(m_spImgSize,CV_32FC1);
	m_oVariationModulatorFrame = cv::Scalar(10.0f); // should always be >= FEEDBACK_V_DECR
	m_oMeanLastDistFrame.create(m_spImgSize,CV_32FC1);
	m_oMeanLastDistFrame = cv::Scalar(0.0f);
	m_oMeanMinDistFrame_LT.create(m_spImgSize,CV_32FC1);
	m_oMeanMinDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanMinDistFrame_ST.create(m_spImgSize,CV_32FC1);
	m_oMeanMinDistFrame_ST = cv::Scalar(0.0f);

	m_oMeanRawSegmResFrame_LT.create(m_spImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_LT = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame_ST.create(m_spImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_ST = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame_LT.create(m_spImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame_LT = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame_ST.create(m_spImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame_ST = cv::Scalar(0.0f);
	m_oUnstableRegionMask.create(m_spImgSize,CV_8UC1);
	m_oUnstableRegionMask = cv::Scalar_<uchar>(0);
	m_oBlinksFrame.create(m_spImgSize,CV_8UC1);
	m_oBlinksFrame = cv::Scalar_<uchar>(0);

	m_oLastColorFrame.create(m_spImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_spImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	m_oRawFGMask_last.create(m_spImgSize,CV_8UC1);
	m_oRawFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last.create(m_spImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);

	m_oRawFGBlinkMask_curr.create(m_spImgSize,CV_8UC1);
	m_oRawFGBlinkMask_curr = cv::Scalar_<uchar>(0);
	m_oRawFGBlinkMask_last.create(m_spImgSize,CV_8UC1);
	m_oRawFGBlinkMask_last = cv::Scalar_<uchar>(0);
	m_voBGColorSamples.resize(m_nBGSamples);
	w_voBGColorSamples.resize(m_nBGSamples);
	m_voBGDescSamples.resize(m_nBGSamples);
	w_voBGDescSamples.resize(m_nBGSamples);

	for(size_t s=0; s<m_nBGSamples; ++s) {
		m_voBGColorSamples[s].create(m_spImgSize,CV_8UC((int)m_nImgChannels));
		m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);

		m_voBGDescSamples[s].create(m_spImgSize,CV_16UC((int)m_nImgChannels));
		m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);


	}
	if(m_nImgChannels==1) {
		std::cout<<"not supported yet!\n";
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
				const uchar nCurrBGInitColor = m_spDSImg.data[idx_color+c];
				m_oLastColorFrame.data[idx_color+c] = nCurrBGInitColor;
				anCurrIntraLBSPThresholds[c] = m_anLBSPThreshold_8bitLUT[nCurrBGInitColor];
				LBSP::computeSingleRGBDescriptor(m_spDSImg,nCurrBGInitColor,x_orig,y_orig,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);

			}			
		}
	}

	m_bInitializedInternalStructs = true;
	refreshModel(1.0f);
	//loadModels();
	/*cv::imwrite("cmlastdescfram.png",m_oLastDescFrame);
	char filename[20];
	for(int i=0; i<m_voBGDescSamples.size(); i++)
	{	
	sprintf(filename,"cpu%ddescmodel.png",i);		
	cv::imwrite(filename,m_voBGDescSamples[i]);
	}*/

	w_voBGColorSamples.resize(m_nBGSamples);
	w_voBGDescSamples.resize(m_nBGSamples);

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


	m_optimizer = new MRFOptimize(m_oImgSize.width,m_oImgSize.height,m_step);

	//gpu
	d_CurrentColorFrame = gpu::createContinuous(oInitImg.size(),CV_8UC4);
}
void WarpSPBackgroundSubtractor::UpdateModel(const cv::Mat& curImg, const cv::Mat& curMask)
{
	for(size_t k=0; k<m_nKeyPoints; ++k) 
	{
		const int x = (int)m_voKeyPoints[k].pt.x;
		const int y = (int)m_voKeyPoints[k].pt.y;

		int idx_uchar = x+y*m_spWidth;
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
			/*if((rand()%(size_t)FEEDBACK_T_LOWER)==0) {
			const size_t s_rand = rand()%m_nBGSamples;
			for(size_t c=0; c<3; ++c) {
			*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
			*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
			}
			}*/
		}
		else
		{
			//update background
			float fSamplesRefreshFrac = 0.1;
			const size_t nBGSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
			const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x,y,LBSP::PATCH_SIZE/2,m_spImgSize);
				const size_t idx_sample_color = 3*(m_oLastColorFrame.cols*y_sample + x_sample);
				const size_t idx_sample_desc = idx_sample_color*2;
				const size_t idx_sample = s%m_nBGSamples;

				for(size_t c=0; c<m_nImgChannels; ++c) {
					*((ushort*)(m_voBGDescSamples[idx_sample].data+idx_sample_desc+2*c)) = anCurrIntraDesc[c];
					*(m_voBGColorSamples[idx_sample].data+idx_sample_color+c) = anCurrColor[c];
				}

			}
		}
	}
}
void WarpSPBackgroundSubtractor::BSOperator(cv::InputArray _image, cv::OutputArray _fgmask)
{
	std::cout<<m_nFrameIndex<<std::endl;
	cv::Mat spOutMask(m_spImgSize,CV_8U);
	spOutMask = cv::Scalar(0);
	
	/*w_oLastColorFrame = cv::Scalar(0);
	char filename[30];
	sprintf(filename,"lastColor%d.jpg",m_nFrameIndex);
	cv::imwrite(filename,m_oLastColorFrame);*/

	cv::Mat oInputImg;
	cv::Mat img = _image.getMat();
	cv::Mat warpImg;
	WarpImage(img,warpImg);
#ifndef REPORT
	nih::Timer timer;
	timer.start();
#endif
	/*sprintf(filename,"curColor%d.jpg",m_nFrameIndex);
	cv::imwrite(filename,oInputImg);*/
	int * preLabels, * labels;
	int num(0);
	SLICClusterCenter* preCenters, * centers;
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	cv::Mat spFGmask = cv::Mat::zeros(m_spHeight,m_spWidth,CV_8U);
	size_t nNonZeroDescCount = 0;
	const float fRollAvgFactor_LT = 1.0f/std::min(++m_nFrameIndex,m_nSamplesForMovingAvgs*4);
	const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIndex,m_nSamplesForMovingAvgs);

	if(m_nImgChannels==1) {
		cout<<"currently not supported!\n";		
	}
	else { //m_nImgChannels==3

		m_SPComputer->GetPreSuperpixelResult(num,preLabels,preCenters);
		m_SPComputer->GetSuperpixelResult(num,labels,centers);
		m_SPComputer->GetSuperpixelDownSampleImg(preLabels, preCenters,warpImg,m_ASAP->getMapX(),m_ASAP->getMapY(), 
			m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),m_spDSImg,m_spDSMapXImg,m_spDSMapYImg,m_spDSIMapXImg,m_spDSIMapYImg);
		cv::imwrite("dscolor.jpg",m_spDSImg);
		WarpSPImg();
		float* mapXPtr = (float*)m_spDSMapXImg.data;
		float* mapYPtr = (float*)m_spDSMapYImg.data;
		float* invMapXPtr = (float*)m_spDSIMapXImg.data;
		float* invMapYPtr = (float*)m_spDSIMapYImg.data;
		float* spMapXPtr = (float*)m_spASAP->getMapX().data;
		float* spMapYPtr = (float*)m_spASAP->getMapY().data;

		for(size_t k=0; k<m_nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_spWidth*y + x;
			const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			float mapX = *(mapXPtr + idx_uchar);
			float mapY = *(mapYPtr + idx_uchar);
			float spMapX = *(spMapXPtr + idx_uchar);
			float spMapY = *(spMapYPtr + idx_uchar);
			float invMapX = *(invMapXPtr + idx_uchar);
			float invMapY = *(invMapYPtr + idx_uchar);
			float fx = invMapX;
			float fy = invMapY;
			int wx = (int)(mapX+0.5);
			int wy = (int)(mapY+0.5);


			if (wx<0 || wx>= m_oImgSize.width-2 || wy<2 || wy>=m_oImgSize.height-2)
			{					
				//m_features.data[oidx_uchar] = 0xff;
				spOutMask.data[idx_uchar] = 0xff;

				continue;
			}
			else
			{

				if (fx<2 || fx>= m_oImgSize.width-2 || fy<2 || fy>=m_oImgSize.height-2)
				{

					spOutMask.data[idx_uchar] = 0xff;
					size_t anCurrIntraLBSPThresholds[3]; 
					for(size_t c=0; c<3; ++c) {
						const uchar nCurrBGInitColor = m_spDSImg.data[idx_uchar_rgb+c];
						m_oLastColorFrame.data[idx_uchar_rgb+c] = nCurrBGInitColor;
						anCurrIntraLBSPThresholds[c] = m_anLBSPThreshold_8bitLUT[nCurrBGInitColor];
						LBSP::computeSingleRGBDescriptor(img,nCurrBGInitColor,x,y,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(w_oLastDescFrame.data+idx_ushrt_rgb))[c]);
					}
				}
			}


			const size_t widx_uchar = wy*m_spWidth+wx;
			const size_t widx_flt32 = widx_uchar*4;
			const size_t widx_uchar_rgb = widx_uchar*3;
			const size_t widx_ushrt_rgb = widx_uchar_rgb*2;

			const uchar* const anCurrColor = m_spDSImg.data+idx_uchar_rgb;
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
			LBSP::computeRGBDescriptor(m_spDSImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
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
					LBSP::computeSingleRGBDescriptor(m_spDSImg,anBGColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
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
				spFGmask.data[idx_uchar] = UCHAR_MAX;
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
			/*if(m_oFGMask_last.data[idx_uchar] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[idx_uchar])) {
			if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
			*pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
			}
			else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
			*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
			if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
			*pfCurrLearningRate = m_fCurrLearningRateLowerCap;
			else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
			*pfCurrLearningRate = m_fCurrLearningRateUpperCap;*/
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
			/*	if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>=4)
			++nNonZeroDescCount;*/
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
#ifndef REPORT
	timer.stop();
	std::cout<<"bs operator "<<timer.seconds()*1000<<std::endl;
#endif
	/*cv::imshow("mask",spFGmask);*/

	/*cv::remap(m_oRawFGMask_last,m_oRawFGMask_last,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	cv::bitwise_xor(oCurrFGMask,m_oRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::remap(m_oRawFGBlinkMask_last,m_oRawFGBlinkMask_last,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,m_oRawFGBlinkMask_last,m_oBlinksFrame);
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oRawFGMask_last);*/
	/*cv::bitwise_xor(oCurrFGMask,m_oRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,m_oRawFGBlinkMask_last,m_oBlinksFrame);
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);*/

#ifndef REPORT
	timer.start();
#endif

	m_SPComputer->GetSuperpixelUpSampleImg(preLabels,preCenters,spFGmask,m_oRawFGMask_last);
	//m_SPComputer->GetSuperpixelUpSampleImg(preLabels,preCenters,spOutMask,outMask);
	/*cv::imshow("upsample",m_oRawFGMask_last);
	cv::waitKey();*/
	
	//warp mask to curr Frame
	
	cv::remap(m_oRawFGMask_last,oCurrFGMask,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	//cv::remap(oCurrFGMask,oCurrFGMask,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	
	//MaskHomographyTest(oCurrFGMask,m_preGray,m_gray,m_homography);
	/*cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
	cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);*/
	/*const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/m_nKeyPoints;
	if(fCurrNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN) {
	for(size_t t=0; t<=UCHAR_MAX; ++t)
	if(m_anLBSPThreshold_8bitLUT[t]>cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4)))
	--m_anLBSPThreshold_8bitLUT[t];
	}
	else if(fCurrNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX) {
	for(size_t t=0; t<=UCHAR_MAX; ++t)
	if(m_anLBSPThreshold_8bitLUT[t]<cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+UCHAR_MAX*m_fRelLBSPThreshold))
	++m_anLBSPThreshold_8bitLUT[t];
	}*/
	//m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
	//if(m_bLearningRateScalingEnabled) {
	//	cv::resize(oInputImg,m_oDownSampledColorFrame,m_oDownSampledFrameSize,0,0,cv::INTER_AREA);
	//	cv::accumulateWeighted(m_oDownSampledColorFrame,m_oMeanDownSampledLastDistFrame_LT,fRollAvgFactor_LT);
	//	cv::accumulateWeighted(m_oDownSampledColorFrame,m_oMeanDownSampledLastDistFrame_ST,fRollAvgFactor_ST);
	//	size_t nTotColorDiff = 0;
	//	for(int i=0; i<m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
	//		const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0]*i;
	//		for(int j=0; j<m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
	//			const size_t idx2 = idx1+m_oMeanDownSampledLastDistFrame_ST.step.p[1]*j;
	//			nTotColorDiff += (m_nImgChannels==1)?
	//				(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2)))/2
	//				:  //(m_nImgChannels==3)
	//			std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2))),
	//				std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+4))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+4))),
	//				(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+8))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+8)))));
	//		}
	//	}
	//	const float fCurrColorDiffRatio = (float)nTotColorDiff/(m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
	//	if(m_bAutoModelResetEnabled) {
	//		if(m_nFramesSinceLastReset>1000)
	//			m_bAutoModelResetEnabled = false;
	//		else if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD && m_nModelResetCooldown==0) {
	//			m_nFramesSinceLastReset = 0;
	//			//refreshModel(0.1f); // reset 10% of the bg model
	//			m_nModelResetCooldown = m_nSamplesForMovingAvgs;
	//			m_oUpdateRateFrame = cv::Scalar(1.0f);
	//		}
	//		else
	//			++m_nFramesSinceLastReset;
	//	}
	//	else if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD*2) {
	//		m_nFramesSinceLastReset = 0;
	//		m_bAutoModelResetEnabled = true;
	//	}
	//	if(fCurrColorDiffRatio>=FRAMELEVEL_COLOR_DIFF_RESET_THRESHOLD/2) {
	//		m_fCurrLearningRateLowerCap = (float)std::max((int)FEEDBACK_T_LOWER>>(int)(fCurrColorDiffRatio/2),1);
	//		m_fCurrLearningRateUpperCap = (float)std::max((int)FEEDBACK_T_UPPER>>(int)(fCurrColorDiffRatio/2),1);
	//	}
	//	else {
	//		m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
	//		m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
	//	}
	//	if(m_nModelResetCooldown>0)
	//		--m_nModelResetCooldown;
	//		//refreshEdgeModel(0.1);
	//	
	//}
	/*sprintf(filename,"outmask%d.jpg",m_nFrameIndex-1);
	cv::imwrite(filename,outMask);*/

	//m_optimizer->Optimize(m_gs,img,m_oRawFGMask_last,m_features,oCurrFGMask);
	//m_optimizer->Optimize(m_gs,img,m_oRawFGMask_last,m_flow,m_wflow,oCurrFGMask);
	//m_optimizer->Optimize(m_oRawFGMask_last,m_features,oCurrFGMask);
	//m_optimizer->Optimize(m_SPComputer,m_oRawFGMask_last,m_features,m_matchedId,oCurrFGMask);
	m_optimizer->Optimize(m_SPComputer,m_oRawFGMask_last,m_matchedId,oCurrFGMask);
	postProcessSegments(img,oCurrFGMask);
	WarpModels();
#ifndef REPORT
	timer.stop();
	std::cout<<"optimize "<<timer.seconds()*1000<<" ms\n";
#endif
	//saveModels();
	/*cv::remap(m_fgCounter,m_fgCounter,m_ASAP->getInvMapX(),m_ASAP->getInvMapY(),0);
	int threshold = 10;
	for(int i=0; i<m_oImgSize.height; i++)
	{
	ushort* cPtr = m_fgCounter.ptr<ushort>(i);
	uchar* mPtr = oCurrFGMask.ptr<uchar>(i);
	for(int j=0; j<m_oImgSize.width; j++)
	{
	if (mPtr[j] == 0xff)
	{	
	cPtr[j]++;
	if (cPtr[j] > threshold)
	{
	mPtr[j] = 100;
	cPtr[j] = 0;
	}
	}
	else
	cPtr[j] = 0;

	}
	}*/
	//postProcessSegments(img,oCurrFGMask);
#ifndef REPORT
	timer.start();
#endif
	cv::Mat refreshMask;	
	m_SPComputer->GetSuperpixelDownSampleGrayImg(labels,centers,oCurrFGMask,refreshMask);
	//cv::bitwise_or(outMask,oCurrFGMask,refeshMask);
	/*cv::imshow("outMask" ,refreshMask);
	cv::waitKey(0);*/
	UpdateModel(m_spDSImg,refreshMask);
#ifndef REPORT
	timer.stop();
	std::cout<<"update model "<<timer.seconds()*1000<<" ms\n";
#endif
	//if (m_nOutPixels > 0.4*m_oImgSize.height*m_oImgSize.width)
	//{
	//	std::cout<<"refresh model\n";
	//	cv::Mat empty(oCurrFGMask.size(),oCurrFGMask.type(),cv::Scalar(0));
	//	UpdateModel(img,empty);
	//	//resetPara();
	//	m_nOutPixels = 0;
	//}
	//refreshModel(outMask,0.1);

}