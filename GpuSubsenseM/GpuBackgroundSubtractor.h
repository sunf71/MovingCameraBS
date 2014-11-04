#pragma once

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <thrust\device_vector.h>
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

class GpuBackgroundSubtractor : public cv::BackgroundSubtractor {
public:
	//! full constructor
	GpuBackgroundSubtractor(float fRelLBSPThreshold=BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
									size_t nMinDescDistThreshold=BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD,
									size_t nMinColorDistThreshold=BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD,
									size_t nBGSamples=BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES,
									size_t nRequiredBGSamples=BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES,
									size_t nSamplesForMovingAvgs=BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS);
	//! default destructor
	virtual ~GpuBackgroundSubtractor();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! gpu Background subtraction operator
	virtual void GpuBSOperator(cv::InputArray image, cv::OutputArray fgmask);
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

protected:
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
	std::vector<cv::gpu::GpuMat> d_voBGColorSamples;
	//! background model descriptors samples (tied to m_voKeyPoints but shaped like the input frames)
	std::vector<cv::Mat> m_voBGDescSamples;
	std::vector<cv::gpu::GpuMat> d_voBGDescSamples;
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	cv::Mat m_oUpdateRateFrame;
	cv::gpu::GpuMat d_oUpdateRateFrame,d_woUpdateRateFrame;
	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	cv::Mat m_oDistThresholdFrame;
	cv::gpu::GpuMat d_oDistThresholdFrame,d_woDistThresholdFrame;
	//! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
	cv::Mat m_oVariationModulatorFrame;
	cv::gpu::GpuMat d_oVariationModulatorFrame,d_woVariationModulatorFrame;
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanLastDistFrame;
	cv::gpu::GpuMat d_oMeanLastDistFrame,d_woMeanLastDistFrame;
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::Mat m_oMeanMinDistFrame_LT, m_oMeanMinDistFrame_ST;
	cv::gpu::GpuMat d_oMeanMinDistFrame_LT, d_woMeanMinDistFrame_LT,d_oMeanMinDistFrame_ST,d_woMeanMinDistFrame_ST;
	//! per-pixel mean raw segmentation results
	cv::Mat m_oMeanRawSegmResFrame_LT, m_oMeanRawSegmResFrame_ST;
	cv::gpu::GpuMat d_oMeanRawSegmResFrame_LT, d_woMeanRawSegmResFrame_LT,d_oMeanRawSegmResFrame_ST,d_woMeanRawSegmResFrame_ST;
	//! per-pixel mean final segmentation results
	cv::Mat m_oMeanFinalSegmResFrame_LT, m_oMeanFinalSegmResFrame_ST;
	cv::gpu::GpuMat d_oMeanFinalSegmResFrame_LT, d_woMeanFinalSegmResFrame_LT,d_oMeanFinalSegmResFrame_ST,d_woMeanFinalSegmResFrame_ST;
	//! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	cv::Mat m_oUnstableRegionMask;
	cv::gpu::GpuMat d_oUnstableRegionMask;
	//! per-pixel blink detection results ('Z(x)')
	cv::Mat m_oBlinksFrame;
	cv::gpu::GpuMat d_oBlinksFrame;
	//! pre-allocated matrix used to downsample (1/8) the input frame when needed
	cv::Mat m_oDownSampledColorFrame;
	cv::gpu::GpuMat d_oDownSampledColorFrame;
	//! copy of previously used pixel intensities used to calculate 'D_last(x)'
	cv::Mat m_oLastColorFrame;
	cv::gpu::GpuMat d_oLastColorFrame;
	//! copy of previously used descriptors used to calculate 'D_last(x)'
	cv::Mat m_oLastDescFrame;
	cv::gpu::GpuMat d_oLastDescFrame;
	//! the foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	cv::Mat m_oRawFGMask_last;
	cv::gpu::GpuMat d_oRawFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc)
	cv::Mat m_oFGMask_last;
	cv::gpu::GpuMat d_oFGMask_last;
	//! the input color frame
	cv::gpu::GpuMat d_CurrentColorFrame;
	//! output foreground mask
	cv::gpu::GpuMat d_FGMask;

	std::vector<cv::gpu::PtrStepf> d_FModels;
	std::vector<cv::gpu::PtrStepf> d_wFModels;
	std::vector<cv::gpu::PtrStepb> d_BModels;
	std::vector<cv::gpu::PtrStep<uchar4>> d_ColorModels;
	std::vector<cv::gpu::PtrStep<ushort4>> d_DescModels;

	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
	//! pre-allocated CV_8UC1 matrices used to speed up morph ops
	cv::Mat m_oFGMask_PreFlood;
	cv::gpu::GpuMat d_oFGMask_PreFlood;
	cv::Mat m_oFGMask_FloodedHoles;
	cv::gpu::GpuMat d_oFGMask_FloodedHoles;
	cv::Mat m_oFGMask_last_dilated;
	cv::gpu::GpuMat d_oFGMask_last_dilated;
	cv::Mat m_oFGMask_last_dilated_inverted;
	cv::gpu::GpuMat d_oFGMask_last_dilated_inverted;
	cv::Mat m_oRawFGBlinkMask_curr;
	cv::gpu::GpuMat d_oRawFGBlinkMask_curr;
	cv::Mat m_oRawFGBlinkMask_last;
	cv::gpu::GpuMat d_oRawFGBlinkMask_last;
};

