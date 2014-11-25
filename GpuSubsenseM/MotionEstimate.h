#pragma once
#include <opencv2\opencv.hpp>
#include "GpuSuperpixel.h"
#include <vector>
using namespace cv;
class MotionEstimate
{
public:
	MotionEstimate(int width, int height, int step):_width(width),_height(height),_step(step)
	{
		_gs = new GpuSuperpixel(_width,_height,_step);
		_nPixels = width*height;
		_labels0 = new int[_nPixels];
		_labels1 = new int[_nPixels];
		_imgData0 = new uchar4[_nPixels];
		_imgData1 = new uchar4[_nPixels];
		_maxCorners = _nPixels;
		_dataQuality = 0.05;
		_minDist = 1.f;
		_nSuperPixels =( _width+_step-1)/_step * (_height+_step-1)/_step;
	}
	~MotionEstimate()
	{
		delete _gs;
		delete[] _labels0;
		delete[] _labels1;
		delete[] _imgData0;
		delete[] _imgData1;
	}
	void EstimateMotion( Mat& curImg,  Mat& prevImg, Mat& transM, Mat& mask);
private:
	GpuSuperpixel* _gs;
	uchar4* _imgData0, *_imgData1;
	int * _labels0, *_labels1;
	int _width;
	int _height;
	int _nPixels;
	int _nSuperPixels;
	int _step;
	std::vector<cv::Point2f> _features0, _features1;
	std::vector<cv::Point2f> _matched0, _matched1;
	//used for KLT
	int _maxCorners;
	float _dataQuality;
	float _minDist;
	std::vector<uchar> _status;
	std::vector<float> _err;
};