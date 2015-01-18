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

void SuperpixelFlow(const cv::Mat& sgray, const cv::Mat& tgray,int step, int spSize, const SLICClusterCenter* centers, cv::Mat& flow);