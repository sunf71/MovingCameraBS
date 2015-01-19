#pragma once
#include <opencv\cv.h>
#include "CudaSuperpixel.h"

void getFlowField(const cv::Mat& u, const cv::Mat& v, cv::Mat& flowField);
class DenseOpticalFlowProvier
{
public:
	virtual 	void DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow) = 0;
};
class GpuDenseOptialFlow:public DenseOpticalFlowProvier
{
public:
	virtual 	void  DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow);
};
class SFDenseOptialFlow:public DenseOpticalFlowProvier
{
public:
	virtual 	void  DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow);
};
class FarnebackDenseOptialFlow:public DenseOpticalFlowProvier
{
public:
	virtual 	void  DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow);
};
class EPPMDenseOptialFlow:public DenseOpticalFlowProvier
{
public:
	virtual 	void  DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow);
};
float L1Dist(const SLICClusterCenter& center, const float2& pos, const uchar* rgb);
void SuperpixelFlow(const cv::Mat& sgray, const cv::Mat& tgray,int step, int spSize, const SLICClusterCenter* centers, 
	std::vector<cv::Point2f>& features0, std::vector<cv::Point2f>& features1, cv::Mat& flow);
void SuperpixelMatching(const int* labels0, const SLICClusterCenter* centers0, const cv::Mat& img0, int* labels1, const SLICClusterCenter* centers1, const cv::Mat& img1, int spSize, int spStep, int width, int height,
	const cv::Mat& spFlow, std::vector<int>& matchedId);
