#include "GpuSuperpixel.h"
#include "CudaSuperpixel.h"
void GpuSuperpixel::Init(unsigned width,unsigned height,unsigned step,float alpha)
{
	m_width = width;
	m_height = height;
	m_size = width*height;
	m_step = step;
	m_radius = m_step/2.0;
	m_alpha = alpha;
	m_nSuperpixels = m_nPixels =  ((width+ step-1) / step) * ((height + step -1) / step);
	cudaMalloc(&d_rgbaBuffer,sizeof(uchar4)*m_size);	
	cudaMalloc(&d_centers,sizeof(SLICClusterCenter)*m_nPixels);
	cudaMalloc(&d_labels,sizeof(int)*m_size);	
}
void GpuSuperpixel::Release()
{
	cudaFree(d_rgbaBuffer);
	cudaFree(d_centers);
	cudaFree(d_labels);	
}

void GpuSuperpixel::DSuperpixel(uchar4* d_rgba,int& num, int* labels,SLICClusterCenter* centers, int iterThreshold)
{
	InitClusterCenters(d_rgba,d_labels,m_width,m_height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);
	while(itrNum < iterThreshold)
	{
		UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers, m_nPixels,m_alpha, m_radius);		
		UpdateClusterCenter(d_rgbaBuffer,m_height,m_width,m_step,d_labels,d_centers,m_nPixels);
		itrNum++;
	}
	num = m_nPixels;
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(centers,d_centers,sizeof(SLICClusterCenter)*m_nPixels,cudaMemcpyDeviceToHost);
}
void GpuSuperpixel::Superpixel(uchar4* rgbaBuffer, int& num,int* labels,SLICClusterCenter* centers,int iterThreshold)
{
	cudaMemcpy(d_rgbaBuffer,rgbaBuffer,sizeof(uchar4)*m_size,cudaMemcpyHostToDevice);
	InitClusterCenters(d_rgbaBuffer,d_labels,m_width,m_height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);
	while(itrNum < iterThreshold)
	{
		UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers, m_nPixels,m_alpha, m_radius);		
		UpdateClusterCenter(d_rgbaBuffer,m_height,m_width,m_step,d_labels,d_centers,m_nPixels);
		itrNum++;
	}
	num = m_nPixels;
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(centers,d_centers,sizeof(SLICClusterCenter)*m_nPixels,cudaMemcpyDeviceToHost);
}
void GpuSuperpixel::SuperpixelLattice(uchar4* rgbaBuffer, int& num,int* labels,SLICClusterCenter* centers,int iterThreshold)
{
	cudaMemcpy(d_rgbaBuffer,rgbaBuffer,sizeof(uchar4)*m_size,cudaMemcpyHostToDevice);
	InitClusterCenters(d_rgbaBuffer,d_labels,m_width,m_height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);
	while(itrNum < iterThreshold)
	{
		UpdateBoundaryLattice(d_rgbaBuffer, m_height, m_width,d_labels, d_centers, m_nPixels,m_alpha, m_radius);		
		UpdateClusterCenter(d_rgbaBuffer,m_height,m_width,m_step,d_labels,d_centers,m_nPixels);
		itrNum++;
	}
	num = m_nPixels;
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(centers,d_centers,sizeof(SLICClusterCenter)*m_nPixels,cudaMemcpyDeviceToHost);
}
void GpuSuperpixel::DSuperpixelB(uchar4* d_rgbaBuffer, int& num, int* lables, SLICClusterCenter* centers, int itrThreshold)
{
	
	InitClusterCenters(d_rgbaBuffer, d_labels, m_width, m_height, m_step, m_nSuperpixels, d_centers);
	int itrNum(0);
	int step = m_step / BIG_CLUSTER_SETP;
	while (itrNum < itrThreshold)
	{
		UpdateBoundary(d_rgbaBuffer, m_height, m_width, d_labels, d_centers, m_nPixels, m_alpha, m_radius);
		//UpdateClusterCenter(d_rgbaBuffer,m_height,m_width,m_step,d_labels,d_centers,m_nPixels);
		UpdateBigClusterCenter(m_height, m_width, step, d_labels, d_centers, m_nPixels);
		itrNum++;
	}
	num = m_nPixels;
	cudaMemcpy(lables, d_labels, sizeof(int)*m_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(centers, d_centers, sizeof(SLICClusterCenter)*m_nPixels, cudaMemcpyDeviceToHost);
}
void GpuSuperpixel::SuperpixelB(const cv::Mat& imgBGRA, int& num, int* labels, SLICClusterCenter* centers, int iterThreshold)
{
	cudaMemcpy(d_rgbaBuffer,imgBGRA.data,sizeof(uchar4)*m_size,cudaMemcpyHostToDevice);
	InitClusterCenters(d_rgbaBuffer,d_labels,m_width,m_height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);
	int step = m_step/BIG_CLUSTER_SETP;
	while(itrNum < iterThreshold)
	{
		UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers, m_nPixels,m_alpha, m_radius);		
		//UpdateClusterCenter(d_rgbaBuffer,m_height,m_width,m_step,d_labels,d_centers,m_nPixels);
		UpdateBigClusterCenter(m_height,m_width,step,d_labels,d_centers,m_nPixels);
		itrNum++;
	}
	num = m_nPixels;
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(centers,d_centers,sizeof(SLICClusterCenter)*m_nPixels,cudaMemcpyDeviceToHost);
}
void GpuSuperpixel::Superpixel(const cv::Mat& imgBGRA, int& num, int* labels, SLICClusterCenter* centers, int iterThreshold)
{
	
	cudaMemcpy(d_rgbaBuffer,imgBGRA.data,sizeof(uchar4)*m_size,cudaMemcpyHostToDevice);
	InitClusterCenters(d_rgbaBuffer,d_labels,m_width,m_height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);
	while(itrNum < iterThreshold)
	{
		UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers, m_nPixels,m_alpha, m_radius);		
		UpdateClusterCenter(d_rgbaBuffer,m_height,m_width,m_step,d_labels,d_centers,m_nPixels);
		itrNum++;
	}
	
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(centers,d_centers,sizeof(SLICClusterCenter)*m_nPixels,cudaMemcpyDeviceToHost);
	num = m_nPixels;
}
void GpuSuperpixel::SuperpixelLattice(uchar4* rgbaBuffer, int& num,int* labels,int iterThreshold)
{
	cudaMemcpy(d_rgbaBuffer,rgbaBuffer,sizeof(uchar4)*m_size,cudaMemcpyHostToDevice);
	InitClusterCenters(d_rgbaBuffer,d_labels,m_width,m_height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);
	while(itrNum < iterThreshold)
	{
		UpdateBoundaryLattice(d_rgbaBuffer, m_height, m_width,d_labels, d_centers, m_nPixels,m_alpha, m_radius);		
		UpdateClusterCenter(d_rgbaBuffer,m_height,m_width,m_step,d_labels,d_centers,m_nPixels);
		itrNum++;
	}
	num = m_nPixels;
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
}
void GpuSuperpixel::Superpixel(uchar4* rgbaBuffer, int& num,int* labels,int iterThreshold)
{
	cudaMemcpy(d_rgbaBuffer,rgbaBuffer,sizeof(uchar4)*m_size,cudaMemcpyHostToDevice);
	InitClusterCenters(d_rgbaBuffer,d_labels,m_width,m_height,m_step, m_nSuperpixels,d_centers);
	int itrNum(0);	
	while(itrNum < iterThreshold)
	{
		UpdateBoundary(d_rgbaBuffer, m_height, m_width,d_labels, d_centers, m_nPixels,m_alpha, m_radius);		
		UpdateClusterCenter(d_rgbaBuffer,m_height,m_width,m_step,d_labels,d_centers,m_nPixels);		
		itrNum++;
	}
	num = m_nPixels;
	cudaMemcpy(labels,d_labels,sizeof(int)*m_size,cudaMemcpyDeviceToHost);
	
}