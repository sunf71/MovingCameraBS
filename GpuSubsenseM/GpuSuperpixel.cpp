#include "GpuSuperpixel.h"
#include "CudaSuperpixel.h"
void GpuSuperpixel::Init(float4* h_rgbaBuffer)
{
	cudaMalloc(&d_rgbaBuffer,sizeof(float4)*m_size);
	cudaMemcpy(d_rgbaBuffer,h_rgbaBuffer,sizeof(float4)*m_size,cudaMemcpyHostToDevice);
	cudaMalloc(&d_centers,sizeof(SLICClusterCenter)*m_nPixels);
	cudaMalloc(&d_labels,sizeof(int)*m_size);
}
void GpuSuperpixel::Release()
{
	cudaFree(d_rgbaBuffer);
	cudaFree(d_centers);
	cudaFree(d_labels);
}
void GpuSuperpixel::Superixel(float4 * rgbaBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* labels)
{
	m_width = width;
	m_height = height;
	m_size = width*height;
	m_step = step;
	m_alpha = alpha;
	m_nPixels =  ((width+ step-1) / step) * ((height + step -1) / step);
	Init(rgbaBuffer);
	InitClusterCenters(d_rgbaBuffer,d_labels,width,height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);
	while(itrNum < 1)
	{
		UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers,  m_nSuperpixels,m_alpha, m_step/2);
		UpdateClusters(d_rgbaBuffer,m_height,m_width,d_labels,d_centers,m_nSuperpixels);
		itrNum++;
	}
	//UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers,  m_nSuperpixels,m_alpha, m_step/2);
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(rgbaBuffer,d_rgbaBuffer,sizeof(float4)*m_size,cudaMemcpyDeviceToHost);
}
