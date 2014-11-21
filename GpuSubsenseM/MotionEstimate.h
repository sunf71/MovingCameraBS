#pragma once
#include <opencv2\opencv.hpp>
#include "GpuSuperpixel.h"
using namespace cv;
class MotionEstimate
{
public:
	MotionEstimate(int width, int height, int step):_width(width),_height(height),_step(step)
	{
		_gs = new GpuSuperpixel(_width,_height,_step);
		_labels0 = new int[_nPixels];
		_labels1 = new int[_nPixels];
	}
	~MotionEstimate()
	{
		delete _gs;
		delete[] _labels0;
		delete[] _labels1;
	}
	void EstimateMotion(const Mat& curImg, const Mat& prevImg, Mat& transM);
private:
	GpuSuperpixel* _gs;
	uchar4* _imgData;
	int * _labels0, *_labels1;
	int _width;
	int _height;
	int _nPixels;
	int _step;
};