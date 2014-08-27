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
};