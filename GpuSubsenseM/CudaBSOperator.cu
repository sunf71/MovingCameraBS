#include "CudaBSOperator.h"
#include <thrust\device_vector.h>
#include "RandUtils.h"
__global__ void CudaBSOperatorKernel(const PtrStepSz<uchar3> img, PtrStep<uchar>* bmodels,PtrStep<float>* fmodels,PtrStep<uchar3>* colorModels, PtrStep<ushort3>* descModels, PtrStep<uchar> fgMask)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < img.cols && y < img.rows)
    {
        uchar3 v = img(y,x);
        fgMask(y,x) = v.x > 100? 255:0;
    }
}

__global__ void CudaRefreshModelKernel(float refreshRate,PtrStepSz<uchar3> lastImg,PtrStep<uchar3>* colorModels,PtrStep<ushort3>* descModels, int modelSize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	
    if(x < lastImg.cols && y < lastImg.rows)
	{
		int width = lastImg.cols;
		int height = lastImg.rows;
		const size_t nBGSamplesToRefresh = refreshRate<1.0f?(size_t)(refreshRate*modelSize):modelSize;
		const size_t nRefreshStartPos = refreshRate<1.0f?rand()%modelSize:0;
		uchar3 orig_color = lastImg(y,x);
		//ushort3 orig_desc = 
		for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
			int y_sample, x_sample;
			getRandSamplePosition(x_sample,y_sample,x,y,2,width,height);
			colorModels[s](y_sample,x_sample) = orig_color;
		}
	}

}

void CudaBSOperator(const PtrStepSz<uchar3>& img, std::vector<PtrStep<uchar>>& bmodels,
	std::vector<PtrStep<float>>& fmodels,
	std::vector<PtrStep<uchar3>>& colorModels, 
	std::vector<PtrStep<ushort3>>& descModels, 
	PtrStep<uchar> fgMask)
{
	dim3 block(16,16);
    dim3 grid((img.cols + block.x - 1)/block.x,(img.rows + block.y - 1)/block.y);
	thrust::device_vector<PtrStep<uchar>> d_bmodels(bmodels);
	thrust::device_vector<PtrStep<float>> d_fmodels(fmodels);
	thrust::device_vector<PtrStep<uchar3>> d_colorModels(colorModels);
	thrust::device_vector<PtrStep<ushort3>> d_descModels(descModels);
	
    CudaBSOperatorKernel<<<grid,block>>>(img,
		thrust::raw_pointer_cast(&d_bmodels[0]),
		thrust::raw_pointer_cast(&d_fmodels[0]),
		thrust::raw_pointer_cast(&d_colorModels[0]),
		thrust::raw_pointer_cast(&descModels[0]),
		fgMask);
}

void CudaRefreshModel(float refreshRate,PtrStepSz<uchar3>& lastImg,std::vector<PtrStep<uchar3>>& colorModels, std::vector<PtrStep<ushort3>>& descModels)
{
	dim3 block(16,16);
    dim3 grid((lastImg.cols + block.x - 1)/block.x,(lastImg.rows + block.y - 1)/block.y);
	
	thrust::device_vector<PtrStep<uchar3>> d_colorModels(colorModels);
	thrust::device_vector<PtrStep<ushort3>> d_descModels(descModels);
	CudaRefreshModelKernel<<<block,grid>>>(refreshRate,lastImg,thrust::raw_pointer_cast(&d_colorModels[0]),
		thrust::raw_pointer_cast(&descModels[0]),colorModels.size());
}