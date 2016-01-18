#pragma once
#undef min
#undef max
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <thrust\device_vector.h>
#include "GpuSuperpixel.h"
#include "MRFOptimize.h"
#include "ASAPWarping.h"
#include "FlowComputer.h"
#include "SuperpixelComputer.h"
#include <fstream>
#include <curand_kernel.h>
#include "BlockWarping.h"
//! defines the default value for BackgroundSubtractorLBSP::m_fRelLBSPThreshold
#define BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.333f)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD (3)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nMinColorDistThreshold
#define BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nBGSamples
#define BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES (50)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nRequiredBGSamples
#define BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nSamplesForMovingAvgs
#define BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS (25)

class WarpBackgroundSubtractor : public cv::BackgroundSubtractor {
public:
	//! full constructor
	WarpBackgroundSubtractor(float rggThreshold = 2.0,
									float rggSeedThreshold = 0.8,
									float modelConfidence = 0.8,
									float tcConfidence = 0.25,
									float scConfidence = 0.35,
									float fRelLBSPThreshold=BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
									size_t nMinDescDistThreshold=BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD,
									size_t nMinColorDistThreshold=BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD,
									size_t nBGSamples=BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES,
									size_t nRequiredBGSamples=BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES,
									size_t nSamplesForMovingAvgs=BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS
									);
	//! default destructor
	virtual ~WarpBackgroundSubtractor();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! gpu Background subtraction operator
	virtual void BSOperator(cv::InputArray image, cv::OutputArray fgmask);
	
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=0);
	//! unused, always returns nullptr
	virtual cv::AlgorithmInfo* info() const;
	//! returns a copy of the latest reconstructed background descriptors image
	virtual void getBackgroundImage(cv::OutputArray backgroundDescImage) const;
	//! returns the keypoints list used for descriptor extraction (note: by default, these are generated from the DenseFeatureDetector class, and the border points are removed)
	virtual std::vector<cv::KeyPoint> getBGKeyPoints() const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (note: this function will remove all border keypoints)
	virtual void setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints);
	virtual void refreshModel(float fSamplesRefreshFrac);
	
	//! turns automatic model reset on or off
	void setAutomaticModelReset(bool);
	
	
	void upload(std::vector<cv::Point2f>& vec, cv::gpu::GpuMat& d_mat)
	{
		cv::Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
		d_mat.upload(mat);
	}
	void download(const cv::gpu::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
 	{
 	    vec.resize(d_mat.cols);
 	    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
 	    d_mat.download(mat);
 	}
	void download(const cv::gpu::GpuMat& d_mat, std::vector<uchar>& vec)
 	{
 	    vec.resize(d_mat.cols);
 	    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
 	    d_mat.download(mat);
 	}
	void UpdateBackground(float* pfCurrLearningRate, int x, int y,size_t idx_ushrt, size_t idx_uchar, const ushort* nCurrIntraDesc, const uchar* nCurrColor);
	//更新模型
	void UpdateModel(const cv::Mat& curImg, const cv::Mat& curMask);
	void WarpModels();
	bool WarpImage(const cv::Mat img, cv::Mat& warpedImg);
	
protected:
	virtual void saveModels();
	virtual void loadModels();
	//! background model keypoints used for LBSP descriptor extraction (specific to the input image size)
	std::vector<cv::KeyPoint> m_voKeyPoints;
	//! defines the current number of used keypoints (always tied to m_voKeyPoints)
	size_t m_nKeyPoints;
	//! input image size
	cv::Size m_oImgSize;
	//! input image channel size
	size_t m_nImgChannels;
	//! input image type
	int m_nImgType;
	//! indicates whether internal structures have already been initialized (LBSP lookup tables, samples, etc.)
	bool m_bInitializedInternalStructs;
	//! absolute minimal color distance threshold ('R' or 'radius' in the original ViBe paper, used as the default/initial 'R(x)' value here, paired with BackgroundSubtractorLBSP::m_nDescDistThreshold)
	const size_t m_nMinColorDistThreshold;
	//! number of different samples per pixel/block to be taken from input frames to build the background model (same as 'N' in ViBe/PBAS)
	const size_t m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background' (same as '#_min' in ViBe/PBAS)
	const size_t m_nRequiredBGSamples;
	//! number of samples to use to compute the learning rate of moving averages
	const size_t m_nSamplesForMovingAvgs;
	//! current frame index, frame count since last model reset & model reset cooldown counters
	size_t m_nFrameIndex, m_nFramesSinceLastReset, m_nModelResetCooldown;
	//! last calculated non-zero desc ratio
	float m_fLastNonZeroDescRatio;
	//! specifies whether automatic model reset is enabled or not
	bool m_bAutoModelResetEnabled;
	//! specifies whether Tmin/Tmax scaling is enabled or not
	bool m_bLearningRateScalingEnabled;
	//! current learning rate caps
	float m_fCurrLearningRateLowerCap, m_fCurrLearningRateUpperCap;
	//! current kernel size for median blur post-proc filtering
	int m_nMedianBlurKernelSize;
	//! specifies the px update spread range
	bool m_bUse3x3Spread;
	//! specifies the downsampled frame size (used for cam motion analysis)
	cv::Size m_oDownSampledFrameSize;
	//! absolute descriptor distance threshold
	const size_t m_nDescDistThreshold;
	//! LBSP internal threshold offset value -- used to reduce texture noise in dark regions
	const size_t m_nLBSPThresholdOffset;
	//! LBSP relative internal threshold (kept here since we don't keep an LBSP object)
	const float m_fRelLBSPThreshold;
	//! pre-allocated internal LBSP threshold values for all possible 8-bit intensity values
	size_t m_anLBSPThreshold_8bitLUT[UCHAR_MAX+1];
	size_t* d_anLBSPThreshold_8bitLUT;

	//! background model pixel color intensity samples (equivalent to 'B(x)' in PBAS, but also paired with BackgroundSubtractorLBSP::m_voBGDescSamples to create our complete model)
	std::vector<cv::Mat> m_voBGColorSamples;
	std::vector<cv::Mat> w_voBGColorSamples;
	
	//! background model descriptors samples (tied to m_voKeyPoints but shaped like the input frames)
	std::vector<cv::Mat> m_voBGDescSamples;
	std::vector<cv::Mat> w_voBGDescSamples;
	
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	cv::Mat m_oUpdateRateFrame;
	cv::Mat w_oUpdateRateFrame;
	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	cv::Mat m_oDistThresholdFrame;
	cv::Mat w_oDistThresholdFrame;
	//! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
	cv::Mat m_oVariationModulatorFrame;
	cv::Mat w_oVariationModulatorFrame;
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanLastDistFrame;
	cv::Mat w_oMeanLastDistFrame;
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::Mat m_oMeanMinDistFrame_LT, m_oMeanMinDistFrame_ST;
	cv::Mat w_oMeanMinDistFrame_LT, w_oMeanMinDistFrame_ST;
	//! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
	cv::Mat m_oMeanDownSampledLastDistFrame_LT, m_oMeanDownSampledLastDistFrame_ST;
	cv::Mat w_oMeanDownSampledLastDistFrame_LT, w_oMeanDownSampledLastDistFrame_ST;
	//! per-pixel mean raw segmentation results
	cv::Mat m_oMeanRawSegmResFrame_LT, m_oMeanRawSegmResFrame_ST;
	cv::Mat w_oMeanRawSegmResFrame_LT, w_oMeanRawSegmResFrame_ST;
	//! per-pixel mean final segmentation results
	cv::Mat m_oMeanFinalSegmResFrame_LT, m_oMeanFinalSegmResFrame_ST;
	cv::Mat w_oMeanFinalSegmResFrame_LT, w_oMeanFinalSegmResFrame_ST;
	//! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	cv::Mat m_oUnstableRegionMask;
	cv::Mat w_oUnstableRegionMask;
	//! per-pixel blink detection results ('Z(x)')
	cv::Mat m_oBlinksFrame;
	cv::Mat w_oBlinksFrame;
	//! pre-allocated matrix used to downsample (1/8) the input frame when needed
	cv::Mat m_oDownSampledColorFrame;
	
	//! copy of previously used pixel intensities used to calculate 'D_last(x)'
	cv::Mat m_oLastColorFrame;
	cv::Mat w_oLastColorFrame;
	//! copy of previously used descriptors used to calculate 'D_last(x)'
	cv::Mat m_oLastDescFrame;
	cv::Mat w_oLastDescFrame;
	//! the foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	cv::Mat m_oRawFGMask_last;
	cv::gpu::GpuMat d_oFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc)
	cv::Mat m_oFGMask_last;
	cv::gpu::GpuMat d_FGMask_last;
	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
	size_t m_nPixels;
	//! pre-allocated CV_8UC1 matrices used to speed up morph ops
	cv::Mat m_oFGMask_PreFlood;	
	cv::Mat m_oFGMask_FloodedHoles;
	cv::Mat m_oFGMask_last_dilated;
	cv::Mat m_oFGMask_last_dilated_inverted;	
	cv::Mat m_oRawFGBlinkMask_curr;	
	cv::Mat m_oRawFGBlinkMask_last;
	cv::Mat m_homography,m_invHomography;
	std::vector<uchar> m_status; // status of tracked features
	std::vector<cv::Point2f> m_points[2];	
	std::vector<cv::Point2f> m_goodFeatures[2];
	cv::Mat m_preGray,m_gray;	
	//保存特征点跟踪情况
	cv::Mat m_features,m_preFeatures;	
	GpuSuperpixel* m_gs;
	MRFOptimize* m_optimizer;	
	cv::Mat m_warpedImg;
	ImageWarping* m_imgWarper;
	BlockWarping* m_blkWarping;
	NBlockWarping* m_nblkWarping;
	ASAPWarping* m_ASAP;
	GlobalWarping* m_glbWarping;
	//DenseOpticalFlowProvier* m_DOFP;
	cv::Mat m_flow,m_wflow;
	//计算像素连续被判为前景的次数，若大于某门限可以改判为背景
	cv::Mat m_fgCounter;
	SuperpixelComputer * m_SPComputer;
	//超像素匹配结果
	std::vector<int> m_matchedId;
	cv::Mat m_img,m_preImg;
	//region growing 的 门限
	float m_rgThreshold;
	//! the input color frame
	cv::gpu::GpuMat d_CurrentColorFrame;
	cv::gpu::PyrLKOpticalFlow  d_pyrLk;
	cv::gpu::GpuMat d_gray;
	cv::gpu::GpuMat d_preGray;
	cv::gpu::GpuMat d_prevPts;
	cv::gpu::GpuMat d_currPts;
	cv::gpu::GpuMat d_status;

	//region growing的门限
	float m_rggThreshold;
	//region growing 种子点门限
	float m_rggSeedThreshold;
	//模型可靠度（优化时模型的可靠性）
	float m_modelConfidence;
	//优化时时域连续性
	float m_TCConfidence;
	//空域连续性
	float m_SCConfidence;
};

class WarpSPBackgroundSubtractor : public WarpBackgroundSubtractor
{
public:
	WarpSPBackgroundSubtractor(float fRelLBSPThreshold=BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
									size_t nMinDescDistThreshold=BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD,
									size_t nMinColorDistThreshold=BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD,
									size_t nBGSamples=BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES,
									size_t nRequiredBGSamples=BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES,
									size_t nSamplesForMovingAvgs=BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS):WarpBackgroundSubtractor(fRelLBSPThreshold,nMinDescDistThreshold,nMinColorDistThreshold,nBGSamples,nRequiredBGSamples,nSamplesForMovingAvgs)
	{
		

	}
	//! default destructor
	virtual ~WarpSPBackgroundSubtractor()
	{
		delete m_spASAP;
	}
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! gpu Background subtraction operator
	virtual void BSOperator(cv::InputArray image, cv::OutputArray fgmask);
	virtual void UpdateModel(const cv::Mat& curImg, const cv::Mat& curMask);
	void WarpSPImg();
protected:
	int m_spWidth;
	int m_spHeight;
	int m_spSize;
	int m_step;
	cv::Size m_spImgSize;
	cv::Mat m_spDSImg;
	cv::Mat m_spDSGray,m_preSPDSGray;
	cv::Mat m_spDSMapXImg;
	cv::Mat m_spDSMapYImg;
	cv::Mat m_spDSIMapXImg;
	cv::Mat m_spDSIMapYImg;
	ASAPWarping* m_spASAP;
	cv::Mat m_wspDSImg;
};
class GpuWarpBackgroundSubtractor : public WarpBackgroundSubtractor
{
public:
	GpuWarpBackgroundSubtractor(int warpId = 0,
		                            float rggThreshold = 1.0,
									float rggSeedThreshold = 0.4,
									float modelConfidence = 0.75,
									float tcConfidence = 0.15,
									float scConfidence = 0.35,
									float fRelLBSPThreshold=BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
									size_t nMinDescDistThreshold=BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD,
									size_t nMinColorDistThreshold=BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD,
									size_t nBGSamples=BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES,
									size_t nRequiredBGSamples=BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES,
									size_t nSamplesForMovingAvgs=BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS
									):m_warpId(warpId),WarpBackgroundSubtractor(rggThreshold,rggSeedThreshold,modelConfidence,tcConfidence, scConfidence)
	{
		

	}
	//! default destructor
	virtual ~GpuWarpBackgroundSubtractor();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! gpu Background subtraction operator
	virtual void BSOperator(cv::InputArray image, cv::OutputArray fgmask);
	virtual void refreshModel(float fSamplesRefreshFrac);
	virtual bool WarpImage(const cv::Mat image, cv::Mat& warpedImg);
protected:
	virtual void saveModels();
	virtual void loadModels();
	void swapModels()
	{
		cv::gpu::swap(d_voBGColorSamples,d_wvoBGColorSamples);
		cv::gpu::swap(d_voBGDescSamples,d_wvoBGDescSamples);
		cv::gpu::swap(d_fModels,d_wfModels);
		cv::gpu::swap(d_bModels,d_wbModels);
	}
	int m_warpId;
	cv::Mat m_FGMask,m_outMask;
	size_t* d_anLBSPThreshold_8bitLUT;
	cv::gpu::GpuMat d_voBGColorSamples,d_wvoBGColorSamples;
	cv::gpu::GpuMat d_voBGDescSamples,d_wvoBGDescSamples;
	cv::gpu::GpuMat d_fModels, d_wfModels, d_bModels, d_wbModels;
	//! the input color frame
	//cv::gpu::GpuMat d_CurrentColorFrame;
	cv::gpu::GpuMat d_CurrWarpedColorFrame;
	cv::gpu::GpuMat d_Map,d_invMap;
	//! output foreground mask
	cv::gpu::GpuMat d_FGMask;
	//因相机运动，新出现的像素或者消失的像素。
	cv::gpu::GpuMat d_outMask;
	uchar* d_outMaskPtr;
	curandState* d_randStates;
};

