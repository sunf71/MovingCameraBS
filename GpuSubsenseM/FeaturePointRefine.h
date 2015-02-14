#pragma once
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\features2d.hpp>
void MatchingResult(const cv::Mat& simg, const cv::Mat& timg, const std::vector<cv::Point2f>& features1, const std::vector<cv::Point2f>& features2,cv::Mat& matchingRst);
void FeaturePointsRefineRANSAC(std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography,float threshold = 0.1);
void FeaturePointsRefineRANSAC(int& nf, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography,float threshold = 0.1);
void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16, float thetaMin = 0, float thetaMax = 360);
void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, std::vector<float>& rads, std::vector<float>& thetas,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16, float thetaMin = 0, float thetaMax = 360);
float FeedbackOpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, std::vector<float>& rads, std::vector<float>& thetas,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int& DistSize, int thetaSize = 16, float thetaMin = 0, float thetaMax = 360);
void OpticalFlowHistogram(std::vector<float>& dist, std::vector<float>& theta,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16, float thetaMin = 0, float thetaMax = 360);
void OpticalFlowHistogram(const cv::Mat& flow, std::vector<float>& histogram,
	std::vector<float>&avgDx, std::vector<float>& avgDy,
	std::vector<std::vector<int>>& ids, cv::Mat& flowIdx,int DistSize = 16,int thetaSize = 16);
void FeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2,int distSize = 10, int thetaSize = 90);
void C2FFeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2, int radSize1=16, int thetaSize1=36, int radSize2=4, int thetaSize2=4);
//分块特征点直方图求精
void BC2FFeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2,  std::vector<float>& blkWeight,int quadWidth = 8 , int radSize1 = 3, int thetaSize1=36, int radSize2=4, int thetaSize2=4);
void FeaturePointsRefineHistogram(int& nf, int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2);
void KLTFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,int cornerCount = 100, float featureC = 0.05, float minDist = 10);
void FILESURFFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2);
void SURFFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2);

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

//估算分块Homography
//@width height, 图像大小
//@quadWidth 分块大小N，将图像分成N*N
//@features1 features0 对应特征点
//@out blkweights, 各分块的权值，@sfeatures1,sfeatures0因为点数不够无法计算homography的特征点集合
int BlockDltHomography(int width, int height, int quadWidth, std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features0, 
	std::vector<cv::Mat>& homographies,std::vector<float>& blkWeights,
	std::vector<cv::Point2f>& sfeatures1,std::vector<cv::Point2f>& sfeatures0);