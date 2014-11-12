#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <vector_types.h>
#include <vector>
using namespace cv;
using namespace cv::gpu;
void InitConstantMem();
void CudaBSOperator(const cv::gpu::GpuMat& img,double* homography, int frameIdx, 
PtrStep<uchar4> colorModel,PtrStep<uchar4> wcolorModel,
PtrStep<ushort4> descModel,PtrStep<ushort4> wdescModel,
PtrStep<uchar> bModel,PtrStep<uchar> wbModel,
PtrStep<float> fModel,PtrStep<float> wfModel,
PtrStep<uchar> fgMask,	uchar* outMask, float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap, size_t* m_anLBSPThreshold_8bitLUT);

void DownloadModel(int width,int height, cv::gpu::GpuMat& models, int size, int id, cv::gpu::GpuMat& model);
void DownloadColorModel(int width,int height, cv::gpu::GpuMat& models, int size, int id, cv::gpu::GpuMat& model);
void CudaRefreshModel(float refreshRate,int width, int height, cv::gpu::GpuMat& colorModels, cv::gpu::GpuMat descModels, 
	GpuMat fModel, GpuMat bModel);

void testRandom();

int CountOutPixel(const uchar* d_ptr, size_t size);