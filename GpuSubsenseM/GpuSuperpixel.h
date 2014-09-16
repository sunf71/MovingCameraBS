#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaSuperpixel.h"


class GpuSuperpixel
{
public:
	~GpuSuperpixel()
	{
		Release();
	}
	void Superixel(float4* rgbaBuffer,unsigned width, unsigned height, int step, float alpha,int& num,int* lables);

protected:
	void Init(float4* h_rgbaBuffer);
	void Release();
private:
	unsigned m_height;
	unsigned m_width;
	unsigned m_size;
	unsigned m_nPixels;
	float4* d_rgbaBuffer;
	SLICClusterCenter* d_centers;
	int* d_labels;
	float m_alpha;
	int m_nSuperpixels;
	double m_step;
};