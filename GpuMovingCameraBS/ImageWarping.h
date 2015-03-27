#pragma once
#undef min
#undef max
#include <opencv\cv.h>
#include <opencv2\gpu\gpu.hpp>
class ImageWarping
{
public:
	ImageWarping(){};
	virtual ~ImageWarping(){};
	virtual void getFlow(cv::Mat& flow) = 0;
	virtual void Warp(const cv::Mat& img, cv::Mat& warpedImg) = 0;	
	virtual void GpuWarp(const cv::gpu::GpuMat& dimg, cv::gpu::GpuMat& dwimg) = 0;
	cv::gpu::GpuMat& getDInvMapX()
	{
		return _dIMapXY[0];
	}
	cv::gpu::GpuMat& getDInvMapY()
	{
		return _dIMapXY[1];
	}
	cv::gpu::GpuMat& getDMapX()
	{
		return _dMapXY[0];

	}
	cv::gpu::GpuMat& getDMapY()
	{
		return _dMapXY[1];
	}
	cv::gpu::GpuMat& getDMapXY()
	{
		return _dMap;
	}
	cv::gpu::GpuMat& getDIMapXY()
	{
		return _dIMap;
	}
	cv::Mat& getInvMapXY()
	{
		return _invMap;
	}

	cv::Mat& getMapXY()
	{
		return _map;
	}
	cv::Mat& getMapX()
	{
		return _mapXY[0];
	}
	cv::Mat& getMapY()
	{
		return _mapXY[1];
	}
	cv::Mat& getInvMapX()
	{
		return _invMapXY[0];
	}
	cv::Mat& getInvMapY()
	{
		return _invMapXY[1];
	}
protected:
	cv::gpu::GpuMat _dImg, _dMap, _dIMap;
	cv::gpu::GpuMat _dMapXY[2];
	cv::gpu::GpuMat _dIMapXY[2];
	cv::Mat _mapXY[2];
	cv::Mat _invMapXY[2];
	cv::Mat _map, _invMap;
};

