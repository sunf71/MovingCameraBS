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
	virtual void WarpPt(const cv::Point2f& input, cv::Point2f& output) = 0;
	virtual void SetFeaturePoints(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2) = 0;
	virtual void GetOutMask(cv::Mat& mask)
	{
		size_t _imgHeight = _map.rows;
		size_t _imgWidth = _map.cols;
		mask.create(_imgHeight, _imgWidth, CV_8U);
		mask = cv::Scalar(0);
		size_t border = 1;
		for (int i = 0; i < _imgHeight; i++)
		{
			cv::Vec2f* ptr = _map.ptr<cv::Vec2f>(i);
			uchar* mPtr = mask.ptr<uchar>(i);
			for (int j = 0; j < _imgWidth; j++)
			{
				int x = ptr[j][0];
				int y = ptr[j][1];
				if (x < border || x >= _imgWidth - border || y < border || y >= _imgHeight-border)
				{
					mPtr[j] = 0xff;
				}
			}
		}
	}
	virtual void GetInvOutMask(cv::Mat& mask)
	{
		size_t _imgHeight = _map.rows;
		size_t _imgWidth = _map.cols;
		mask.create(_imgHeight, _imgWidth, CV_8U);
		mask = cv::Scalar(0);
		size_t border = 1;
		for (int i = 0; i < _imgHeight; i++)
		{
			cv::Vec2f* ptr = _invMap.ptr<cv::Vec2f>(i);
			uchar* mPtr = mask.ptr<uchar>(i);
			for (int j = 0; j < _imgWidth; j++)
			{
				int x = ptr[j][0];
				int y = ptr[j][1];
				if (x < border || x >= _imgWidth - border || y < border || y >= _imgHeight - border)
				{
					mPtr[j] = 0xff;
				}
			}
		}
	}
	virtual void Solve() = 0;
	virtual void Reset(){}
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

