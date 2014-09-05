#pragma once

#include "BackgroundSubtractorSuBSENSE.h"
#include "DistanceUtils.h"
#include "RandUtils.h"

class BGSSubsenseM : public BackgroundSubtractorSuBSENSE
{
public:
	BGSSubsenseM():BackgroundSubtractorSuBSENSE(),m_qlevel(0.01),m_minDist(10.){}
	virtual ~BGSSubsenseM()
	{
		
	}
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! refreshes all samples based on the last analyzed frame
	virtual void refreshModel(float fSamplesRefreshFrac);
	//! primary model update function; the learning param is used to override the internal learning thresholds (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0);
	virtual void motionCompensate();
	virtual void getHomography(const cv::Mat& oInitImg, cv::Mat&  homography);
	void cloneModels();
protected:
	//! points used to compute the homography matrix between two continuous frames
	std::vector<cv::Point2f> m_points[2];
	cv::Mat m_gray;			// current gray-level image
	cv::Mat m_preGray;		// previous gray-level image
	double m_qlevel;    // quality level for feature detection
	double m_minDist;   // minimum distance between two feature points
	std::vector<uchar> m_status; // status of tracked features
	std::vector<float> m_err;    // error in tracking
	cv::Mat m_homography; // 

	std::vector<cv::Mat*> m_modelsPtr;
	std::vector<cv::KeyPoint> m_voTKeyPoints;// transformed keypoints
	cv::Mat m_warpMask;


	//warped models
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	cv::Mat w_oUpdateRateFrame;
	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	cv::Mat w_oDistThresholdFrame;
	//! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
	cv::Mat w_oVariationModulatorFrame;
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat w_oMeanLastDistFrame;
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::Mat w_oMeanMinDistFrame_LT;
	cv::Mat w_oMeanMinDistFrame_ST;
	//! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
	cv::Mat w_oMeanDownSampledLastDistFrame_LT;
	cv::Mat w_oMeanDownSampledLastDistFrame_ST;
	//! per-pixel mean raw segmentation results
	cv::Mat w_oMeanRawSegmResFrame_LT;
	cv::Mat w_oMeanRawSegmResFrame_ST;
	//! per-pixel mean final segmentation results
	cv::Mat w_oMeanFinalSegmResFrame_LT;
	cv::Mat w_oMeanFinalSegmResFrame_ST;
	//! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	cv::Mat w_oUnstableRegionMask;
	//! per-pixel blink detection results ('Z(x)')
	cv::Mat w_oBlinksFrame;

	std::vector<cv::Mat> w_voBGColorSamples;
	std::vector<cv::Mat> w_voBGDescSamples;
};