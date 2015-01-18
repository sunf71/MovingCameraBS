#pragma once
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\features2d.hpp>
void MatchingResult(const cv::Mat& simg, const cv::Mat& timg, const std::vector<cv::Point2f>& features1, const std::vector<cv::Point2f>& features2,cv::Mat& matchingRst);
void FeaturePointsRefineRANSAC(std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2,cv::Mat& homography);
void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16);
void OpticalFlowHistogram(const cv::Mat& flow, std::vector<float>& histogram,
	std::vector<float>&avgDx, std::vector<float>& avgDy,
	std::vector<std::vector<int>>& ids, cv::Mat& flowIdx,int DistSize = 16,int thetaSize = 16);
void FeaturePointsRefineHistogram(int width, int height,std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2);
void KLTFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2);
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


