#pragma once
#undef min
#undef max
#include "Common.h"
#include <opencv\cv.h>
#include <opencv2/video/background_segm.hpp>
#include "SuperpixelComputer.h"
#include "ASAPWarping.h"
#include "BlockWarping.h"
#include "videoprocessor.h"
#include "timer.h"
#include "HistComparer.h"
#include <opencv2\gpu\gpu.hpp>

class RTBackgroundSubtractor: public cv::BackgroundSubtractor 
{
public:
	RTBackgroundSubtractor(float mthreshold = 1.0, float spAlpha = 0.9, int warpId = 1):_mThreshold(mthreshold),_frameIdx(0),_spStep(40),_warpId(warpId),_spAlpha(spAlpha),_hogBins(36),_colorBins(12)
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
		safe_delete(_dFeatureDetector);
		safe_delete(_imgWarper);
		safe_delete(_rgbHComp);
		safe_delete(_gradHComp);		
		//safe_delete(m_glbWarping);
	}
	void SetPreGray(cv::Mat& img)
	{
		_preGray = img.clone();
	};
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
	void GetSuperpixelMap(cv::Mat& sp);
	void GetRegionMap(cv::Mat& regions); 
	void GetSaliencyMap(cv::Mat& saliency);
protected:
	void EstimateCameraMotion(){};
	void MovingSaliency(cv::Mat& fgMask);
	void BuildHistogram(const int* labels, const SLICClusterCenter* centers);
	void RegionMergingFast(const int*  labels, const SLICClusterCenter* centers);
	void RegionMergingFastQ(const int*  labels, const SLICClusterCenter* centers);
	void SaliencyMap();
	void upload(std::vector<cv::Point2f>& vec, cv::gpu::GpuMat& d_mat)
	{
		cv::Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
		d_mat.upload(mat);
	}
	void download(const cv::gpu::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
 	{
 	    vec.resize(d_mat.cols);
 	    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
 	    d_mat.download(mat);
 	}
	void download(const cv::gpu::GpuMat& d_mat, std::vector<uchar>& vec)
 	{
 	    vec.resize(d_mat.cols);
 	    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
 	    d_mat.download(mat);
 	}
	void calcCameraMotion(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>&f0);
private:
	int _warpId;
	float _spAlpha;
	int _spStep;
	bool _initialized;
	cv::Mat _img, _preImg, _labImg, _rgbaImg,  _fImg, _dxImg, _dyImg, _magImg, _angImg;
	cv::Mat _gray, _preGray;
	SuperpixelComputer* _SPComputer;
	std::vector<std::vector<float>> _colorHists, _nColorHists;
	std::vector<std::vector<float>> _HOGs, _nHOGs;
	//每个超像素所包含像素的位置
	std::vector<std::vector<uint2>> _spPoses;
	//每个区域的超像素Id
	std::vector<std::vector<int>> _regIdices;
	std::vector<int> _regSizes;
	std::vector<float4> _regColors;
	std::vector<int> _newLabels;
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
	cv::gpu::GpuMat _dGray,_dPreGray;
	cv::gpu::GpuMat _dCurrFrame;
	cv::gpu::GpuMat _dFeatures,_dSPCenters,_dCurrPts,_dPrevPts,d_Status;
	cv::gpu::PyrLKOpticalFlow  d_pyrLk;
	cv::gpu::GoodFeaturesToTrackDetector_GPU* _dFeatureDetector;

	HistComparer* _rgbHComp, *_gradHComp;


	ImageWarping* _imgWarper;
	BlockWarping* _blkWarping;
	NBlockWarping* _nblkWarping;
	ASAPWarping* _ASAP;
	GlobalWarping* _glbWarping;
};

