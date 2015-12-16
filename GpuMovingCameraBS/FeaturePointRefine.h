#pragma once
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\features2d.hpp>
//Get the feature points matching result( draw a line from every src feature point to dst feature point)
void MatchingResult(const cv::Mat& simg, const cv::Mat& timg, const std::vector<cv::Point2f>& features1, const std::vector<cv::Point2f>& features2,cv::Mat& matchingRst);

//Refine( elemate the mismatching feature points and feature points from the foreground) with Ransac
void FeaturePointsRefineRANSAC(std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography,float threshold = 0.1);

void FeaturePointsRefineRANSAC(int& nf, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography,float threshold = 0.1);

//build optical flow histogram based on optical flow orientation and distance
void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16, float thetaMin = 0, float thetaMax = 360);

//build optical flow histogram based on optical flow orientation
void OpticalFlowHistogramO(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int thetaBins);

void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, std::vector<float>& rads, std::vector<float>& thetas,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16, float thetaMin = 0, float thetaMax = 360);

float FeedbackOpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, std::vector<float>& rads, std::vector<float>& thetas,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int& DistSize, int thetaSize = 16, float thetaMin = 0, float thetaMax = 360);

void OpticalFlowHistogram(std::vector<float>& dist, std::vector<float>& theta,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16, float thetaMin = 0, float thetaMax = 360);

void OpticalFlowHistogram(const cv::Mat& flow, std::vector<float>& histogram,
	std::vector<float>&avgDx, std::vector<float>& avgDy,
	std::vector<std::vector<int>>& ids, cv::Mat& flowIdx,int DistSize = 16,int thetaSize = 16);

//Refine with histogram voting method
void FeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2,int distSize = 10, int thetaSize = 90);

//Refine with histogram voting method, only consider the optical orientation
void FeaturePointsRefineHistogramO(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, std::vector<uchar>& inliers, int thetaSize = 5);

void FeaturePointsRefineHistogramO(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int thetaSize = 5);

void FeaturePointsRefineHistogram(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, std::vector<uchar>& inliers, int distSize = 10, int thetaSize = 90);

void FeaturePointsRefineHistogram(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int distSize = 5, int thetaSize = 5);

void C2FFeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int radSize1=16, int thetaSize1=36, int radSize2=4, int thetaSize2=4);

//Refine with histogram voting method(block based coarse to fine)
void BC2FFeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2,  std::vector<float>& blkWeight,int quadWidth = 8 , int radSize1 = 3, int thetaSize1=36, int radSize2=4, int thetaSize2=4);

void FeaturePointsRefineHistogram(int& nf, int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2,int distSize = 10, int thetaSize = 90);

void FeaturePointsRefineZoom(int width, int height, std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, std::vector<uchar>& inliers, int thetaSize = 5);

void FeaturePointsRefineZoom(int width, int height, std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int thetaSize = 5);



//KLT mathcing
void KLTFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,int cornerCount = 100, float featureC = 0.05, float minDist = 10);

void FILESURFFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2);

void SURFFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2);

//Refine with reletive flow
void RelFlowRefine(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, int& ankorId, float threshold = 1.2);

//Refine with reletive flow
void RelFlowRefine(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>&features0, float threshold = 1.2);

//Locate minimum and maximum value in a vector
template<typename T>
void minMaxLoc(const std::vector<T>& vec,T& maxValue,T& minValue, int& maxId, int& minId)
{
	maxValue = vec[0];
	minValue = vec[0];
	maxId = 0;
	minId = 0;
	for(int i=1; i<vec.size(); i++)
	{
		if(vec[i] > maxValue)
		{
			maxValue = vec[i];
			maxId = i;
		}
		if (vec[i] < minValue)
		{
			minValue = vec[i];
			minId = i;
		}
	}
}

//Block based homography estimation
//@width,height: Image Size
//@quadWidth(N): Image is divided to N*N blocks
//@features1 features0: Feature Points
//@out blkweights: weights of each block
//@out sfeatures1,sfeatures0: Feature points can't be used to calc homography(insufficient) 
int BlockDltHomography(int width, int height, int quadWidth, std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features0, 
	std::vector<cv::Mat>& homographies,std::vector<float>& blkWeights,
	std::vector<cv::Point2f>& sfeatures1,std::vector<cv::Point2f>& sfeatures0);

//Show feature refine result
//@title, if title is a file name with extension, the result will be saved to that file, 
//otherwise the result  will be shown in a window with the title
void ShowFeatureRefine(cv::Mat& img1, std::vector<cv::Point2f>& features1, cv::Mat& img0, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, std::string title, bool line = false);

void ShowFeatureRefine(cv::Mat& img1, std::vector<cv::Point2f>& features1, cv::Mat& img0, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, std::string title, int anchorId);

void ShowFeatureRefineSingle(cv::Mat& img1, std::vector<cv::Point2f>& features1, cv::Mat& img0, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, std::string title);

class BlockRelFlowRefine
{
	typedef std::vector<cv::Point2f> Points;
	struct Cell
	{
		std::vector<int> featureIds;
		int idx;
	};
public:
	BlockRelFlowRefine(int width, int height, int quad, float threshold = 1.0f) :_width(width), _height(height), _quad(quad), _threshold(threshold)
	{
		Init();
	}
	void Init();
	void Refine(int id, Points& features1, Points& features0, std::vector<uchar>& inliers, int& aId);
	void Refine(Points& features1, Points& features0, std::vector<uchar>& inliers);
protected:
	int _width, _height, _quad;
	int _blkWidth, _blkHeight;
	std::vector<Cell> _cells;
	float _threshold;
	Points _f1, _f0;
};

void FeatureFlowColor(cv::Mat& img, std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2);