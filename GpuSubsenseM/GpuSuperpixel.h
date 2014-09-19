#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaSuperpixel.h"
#include <thrust/device_vector.h>
#include "cub\cub.cuh"
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
	
	float4* d_rgbaBuffer;
	SLICClusterCenter* d_centers;
	//m_height*m_width个，用于reduce计算更新超像素中心和平均颜色
	SLICClusterCenter* d_centers_in;
	//d_centers_in的副本
	SLICClusterCenter* d_centers_tmp;
	//d_labels的副本
	int * d_labels_tmp;
	int* d_labels;
	float m_alpha;
	
	//超像素理论数量
	unsigned m_nPixels;
	//超像素实际数量
	int m_nSuperpixels;
	double m_step;
	//used for updating cluster center
	int * d_outKeys;
	thrust::device_ptr<int> d_outKeyPtr;
	SLICClusterCenter* d_outValues;
	thrust::device_ptr<SLICClusterCenter> d_outValuePtr;
};