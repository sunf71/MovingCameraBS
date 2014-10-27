#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <vector_types.h>
#include <vector>
using namespace cv;
using namespace cv::gpu;

void CudaBSOperator(const cv::gpu::GpuMat& img,int frameIndex, std::vector<PtrStep<uchar>>& bmodels,
	std::vector<PtrStep<float>>& fmodels,
	std::vector<PtrStep<uchar3>>& colorModels, 
	std::vector<PtrStep<ushort3>>& descModels, 
	cv::gpu::GpuMat& fgMask,float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap);

void CudaRefreshModel(float refreshRate,const cv::gpu::GpuMat& lastImg,std::vector<PtrStep<uchar3>>& colorModels, std::vector<PtrStep<ushort3>>& descModels);

