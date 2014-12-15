#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaSuperpixel.h"
#include <thrust/device_vector.h>

class GpuSuperpixel
{
public:
	GpuSuperpixel(unsigned width,unsigned height,unsigned step,float alpha = 0.9)
	{
		Init(width,height,step, alpha);
	}
	~GpuSuperpixel()
	{
		Release();
	}
	void Superpixel(uchar4* rgbaBuffer,int& num,int* lables,SLICClusterCenter* centers,int iterThreshold = 10);
	void SuperpixelLattice(uchar4* rgbaBuffer,int& num,int* lables,SLICClusterCenter* centers,int iterThreshold = 10);
	void Superpixel(uchar4* rgbaBuffer, int& num,int* labels,int iterThreshold = 10);
	void SuperpixelLattice(uchar4* rgbaBuffer, int& num,int* labels,int iterThreshold = 10);
	void DSuperpixel(uchar4* d_rgbaBuffer,int& num,int* lables,SLICClusterCenter* centers,int iterThreshold = 10);
protected:
	void Init(unsigned width, unsigned height,unsigned step, float alpha);
	void Release();
private:
	unsigned m_height;
	unsigned m_width;
	unsigned m_size;
	
	uchar4* d_rgbaBuffer;
	SLICClusterCenter* d_centers;
	
	int* d_labels;
	float m_alpha;
	float m_radius;
	//超像素理论数量
	unsigned m_nPixels;
	//超像素实际数量
	int m_nSuperpixels;
	double m_step;
	
	
};

