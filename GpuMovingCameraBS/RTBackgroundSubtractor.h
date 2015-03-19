#pragma once
#include "Common.h"
#include <opencv\cv.h>
#include <opencv2/video/background_segm.hpp>
#include "SuperpixelComputer.h"
#include "ASAPWarping.h"
#include "videoprocessor.h"
#include "timer.h"


class RTBackgroundSubtractor: public cv::BackgroundSubtractor 
{
public:
	RTBackgroundSubtractor(float mthreshold = 1.0):_mThreshold(mthreshold),_frameIdx(0),_spStep(40),_hogBins(36),_colorBins(12)
	{
		_totalColorBins = _colorBins*_colorBins*_colorBins;
		_hogStep = 360.0/_hogBins;

		/*
		//lab color space
		_colorSteps[0] = 100.0/_colorBins;
		_colorSteps[1] = _colorSteps[2] = 255.0/_colorBins;
		_colorMins[0] = 0;
		_colorMins[1] = _colorMins[2] = -127;*/

		//rgb color space
		_colorSteps[0] = _colorSteps[1] = _colorSteps[2] = 255.0/_colorBins;
		_colorMins[0] = _colorMins[1] = _colorMins[2] = 0;
		
	};
	//! default destructor
	virtual ~RTBackgroundSubtractor()
	{
		safe_delete(_SPComputer);
		safe_delete_array(_segment);
		safe_delete_array(_visited);
	}
	void Initialize(cv::InputArray image);
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=0);
	//! unused, always returns nullptr
	virtual cv::AlgorithmInfo* info() const
	{
		//not implemented
		return NULL;
	}
	//! returns a copy of the latest reconstructed background descriptors image
	virtual void getBackgroundImage(cv::OutputArray backgroundDescImage) const{};

	void GetRegionMap(cv::Mat& regions); 
protected:
	void EstimateCameraMotion(){};
	void MovingSaliency(cv::Mat& fgMask);
	void BuildHistogram(const int* labels, const SLICClusterCenter* centers);
	void RegionMergingFast(const int*  labels, const SLICClusterCenter* centers);
	void SaliencyMap();
private:
	int _spStep;
	bool _initialized;
	cv::Mat _img, _labImg, _rgbaImg,  _fImg, _dxImg, _dyImg, _magImg, _angImg;
	cv::Mat _gray, _preGray;
	SuperpixelComputer* _SPComputer;
	std::vector<std::vector<float>> _colorHists, _nColorHists;
	std::vector<std::vector<float>> _HOGs, _nHOGs;
	//每个超像素所包含像素的位置
	std::vector<std::vector<uint2>> _spPoses;
	//每个区域的超像素Id
	std::vector<std::vector<int>> _regIdices;
	std::vector<int> _regSizes;
	char* _visited;
	int * _segment;
	int _width,_height,_imgSize,_spWidth,_spHeight,_spSize;
	int _frameIdx;
	int _hogBins, _colorBins, _totalColorBins;
	float _hogStep;
	float _colorSteps[3], _colorMins[3];
	cv::Mat _fgMask, _preFgMask;
	//运动比对门限
	float _mThreshold;
	//每个超像素的mask
	std::vector<cv::Mat> _spMasks;
};

