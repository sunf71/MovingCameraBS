#pragma once
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "BackgroundSubtractorSuBSENSE.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
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
	//检测所求出前景的运动是否与背景一致，去掉错误前景
	void MaskHomographyTest(cv::Mat& mCurr, cv::Mat& curr, cv::Mat & prev, cv::Mat& homography)
	{
		/*std::cout<<homography;*/
		float threshold = 0.6;
		std::vector<cv::Point2f> currPoints, trackedPoints;
		std::vector<uchar> status; // status of tracked features
		std::vector<float> err;    // error in tracking
		for(int i=0; i<mCurr.cols; i++)
		{
			for(int j=0; j<mCurr.rows; j++)
				if(mCurr.data[i + j*mCurr.cols] == 0xff)
					currPoints.push_back(cv::Point2f(i,j));
		}
		if (currPoints.size() <=0)
			return;
		// 2. track features
		cv::calcOpticalFlowPyrLK(curr, prev, // 2 consecutive images
			currPoints, // input point position in first image
			trackedPoints, // output point postion in the second image
			status,    // tracking success
			err);      // tracking error

		// 2. loop over the tracked points to reject the undesirables
		int k=0;

		for( int i= 0; i < currPoints.size(); i++ ) {

			// do we keep this point?
			if (status[i] == 1) {

				// keep this point in vector
				currPoints[k] = currPoints[i];
				trackedPoints[k++] = trackedPoints[i];
			}
		}
		// eliminate unsuccesful points
		currPoints.resize(k);
		trackedPoints.resize(k);

		float distance = 0;
		for(int i=0; i<k; i++)
		{
			cv::Point2f pt = currPoints[i];
			double* data = (double*)homography.data;
			float x = data[0]*pt.x + data[1]*pt.y + data[2];
			float y = data[3]*pt.x + data[4]*pt.y + data[5];
			float w = data[6]*pt.x + data[7]*pt.y + data[8];
			x /= w;
			y /= w;
			float d = abs(trackedPoints[i].x-x) + abs(trackedPoints[i].y - y);
			distance += d;
			if (d < threshold)
			{
				mCurr.data[(int)currPoints[i].x+(int)currPoints[i].y*mCurr.cols] = 0x0;
				const unsigned char idx_uchar_rgb = (pt.x + pt.y* this->m_oImgSize.width)*3;
				unsigned char* anCurrColor = curr.data + idx_uchar_rgb;
				//update model
				const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						//*((ushort*)(w_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
					}
			}
		}		
	}
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