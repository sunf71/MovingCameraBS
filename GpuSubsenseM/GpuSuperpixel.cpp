#include "GpuSuperpixel.h"
#include "CudaSuperpixel.h"
void GpuSuperpixel::Init(float4* h_rgbaBuffer)
{
	cudaMalloc(&d_rgbaBuffer,sizeof(float4)*m_size);
	cudaMemcpy(d_rgbaBuffer,h_rgbaBuffer,sizeof(float4)*m_size,cudaMemcpyHostToDevice);
	cudaMalloc(&d_centers,sizeof(SLICClusterCenter)*m_nPixels);
	cudaMalloc(&d_labels,sizeof(int)*m_size);
	cudaMalloc(&d_labels_tmp,sizeof(int)*m_size);
	cudaMalloc(&d_centers_in,sizeof(SLICClusterCenter)*m_size);
	InitConstClusterCenters(d_rgbaBuffer,m_height,m_width,d_centers_in);
	cudaMalloc(&d_centers_tmp,sizeof(SLICClusterCenter)*m_size);
	cudaMemcpy(d_centers_tmp,d_centers_in,sizeof(SLICClusterCenter)*m_nPixels,cudaMemcpyDeviceToDevice);
	cudaMalloc(&d_outKeys,sizeof(int)*m_size);
	d_outKeyPtr = thrust::device_pointer_cast(d_outKeys);
	cudaMalloc(&d_outValues,sizeof(SLICClusterCenter)*m_nPixels);
	d_outValuePtr = thrust::device_pointer_cast(d_outValues);
}
void GpuSuperpixel::Release()
{
	cudaFree(d_rgbaBuffer);
	cudaFree(d_centers);
	cudaFree(d_labels);
	cudaFree(d_labels_tmp);
	cudaFree(d_centers_in);
	cudaFree(d_outKeys);
	cudaFree(d_outValues);
	cudaFree(d_centers_tmp);
}
void GpuSuperpixel::Superixel(float4 * rgbaBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* labels)
{
	m_width = width;
	m_height = height;
	m_size = width*height;
	m_step = step;
	m_alpha = alpha;
	m_nSuperpixels = m_nPixels =  ((width+ step-1) / step) * ((height + step -1) / step);
	
	Init(rgbaBuffer);
	InitClusterCenters(d_rgbaBuffer,d_labels,width,height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);
	while(itrNum < 9)
	{
		UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers,m_nPixels,m_alpha, m_step/2);
		cudaMemcpy(d_centers_tmp,d_centers_in,sizeof(SLICClusterCenter)*m_size,cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_labels_tmp,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToDevice);
		UpdateClusters(d_rgbaBuffer,m_height,m_width,d_labels_tmp,d_centers_tmp,d_outKeyPtr,d_outValuePtr,d_centers,m_nSuperpixels,m_nPixels);
		itrNum++;
	}
	UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers,  m_nSuperpixels,m_alpha, m_step/2);
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
	
}
