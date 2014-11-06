#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <vector_types.h>
#include <vector>
using namespace cv;
using namespace cv::gpu;

void CudaBSOperator(const cv::gpu::GpuMat& img,double* homography,int frameIndex, cv::gpu::GpuMat& fgMask,float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap, size_t* m_anLBSPThreshold_8bitLUT);

void InitDeviceModels(std::vector<PtrStep<uchar4>>& colorModels, std::vector<PtrStep<ushort4>>& descModels,
	std::vector<PtrStep<uchar>>& bModels, std::vector<PtrStep<float>>& fModels,std::vector<PtrStep<uchar4>>& wcolorModels, 
	std::vector<PtrStep<ushort4>>& wdescModels,
	std::vector<PtrStep<uchar>>& wbModels,
	std::vector<PtrStep<float>>& wfModels);
void CudaRefreshModel(float refreshRate,const cv::gpu::GpuMat& lastImg, const cv::gpu::GpuMat& lastDescImg,size_t* m_anLBSPThreshold_8bitLUT);
void ReleaseDeviceModels();

void testRandom();