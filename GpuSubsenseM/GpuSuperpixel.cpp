#include "GpuSuperpixel.h"
#include "CudaSuperpixel.h"
void GpuSuperpixel::Init()
{
	cudaMalloc(&d_rgbaBuffer,m_size);
	cudaMalloc(&d_centers,sizeof(SLICClusterCenter)*m_nPixels);

}
void GpuSuperpixel::Release()
{
	cudaFree(d_rgbaBuffer);
	cudaFree(d_centers);
}
void GpuSuperpixel::Superixel(float4 * rgbaBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* lables)
{
	m_width = width;
	m_height = height;
	m_size = width*height;
	m_step = step;
	m_alpha = alpha;
	
	Init();
	InitClusterCenters(d_rgbaBuffer,width,height,m_step, m_nSuperpixels,d_centers);
}
