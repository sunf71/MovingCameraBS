#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <vector_types.h>
#include <vector>
using namespace cv;
using namespace cv::gpu;

void CudaBSOperator(const PtrStepSz<uchar3>& img, std::vector<PtrStep<uchar>>& bmodels,
	std::vector<PtrStep<float>>& fmodels,
	std::vector<PtrStep<uchar3>>& colorModels, 
	std::vector<PtrStep<ushort3>>& descModels, 
	PtrStep<uchar> fgMask);

void CudaRefreshModel(float refreshRate,PtrStepSz<uchar3>& lastImg,std::vector<PtrStep<uchar3>>& colorModels, std::vector<PtrStep<ushort3>>& descModels);