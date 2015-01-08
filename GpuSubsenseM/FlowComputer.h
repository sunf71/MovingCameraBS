#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
using namespace cv;
void getFlowField(const Mat& u, const Mat& v, Mat& flowField);
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