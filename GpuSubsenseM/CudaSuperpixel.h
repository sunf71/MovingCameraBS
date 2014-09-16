#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct SLICClusterCenter
{
	__host__ __device__ SLICClusterCenter(){}
	__host__ __device__ SLICClusterCenter(float4 _rgb,float2 _xy):rgb(_rgb),xy(_xy){}
	__host__ __device__ SLICClusterCenter operator + (const SLICClusterCenter& a) const
	{
		float4 retF4;
		retF4.x = rgb.x + a.rgb.x;
		retF4.y = rgb.y + a.rgb.y;
		retF4.z = rgb.z + a.rgb.z;
		float2 retF2;
		retF2.x = xy.x + a.xy.x;
		retF2.y = xy.y + a.xy.y;

		return  SLICClusterCenter(retF4,retF2);
	}
	
	__host__ __device__ SLICClusterCenter operator * (const float a) const
	{
		float4 retF4;
		retF4.x = rgb.x*a;
		retF4.y = rgb.y*a;
		retF4.z = rgb.z*a;
		float2 retF2;
		retF2.x = xy.x*a;
		retF2.y = xy.y*a;
		return SLICClusterCenter(retF4,retF2);
	}

	float4 rgb;
	float2 xy;
	int nPoints;

};
__global__ void InitClusterCentersKernel(float4* floatBuffer, int* labels,
									int nWidth, int nHeight, int step,int nSegs,  
									SLICClusterCenter* vSLICCenterList);
__global__ void InitClustersKernel(float4* imgBuffer, int nHeight, int nWidth, SLICClusterCenter* d_ceneters);

__global__ void UpdateBoundaryKernel(float4* imgBuffer, int nHeight, int nWidth,int* labels,SLICClusterCenter* d_ceneters, int nClusters,float alpha, float radius);

__global__ void UpdateClustersKernel(float4* imgBuffer, int nHeight, int nWidth, SLICClusterCenter* d_ceneters, int nClusters);

void InitClusterCenters(float4* d_imgBuffer,int* d_labels, int nWidth, int nHeight, int step, int& nSeg, SLICClusterCenter* d_center);

void UpdateBoundary(float4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_ceneters, int nClusters,float alpha, float radius);

void UpdateClusters(float4* imgBuffer, int nHeight, int nWidth, int* labels, SLICClusterCenter* d_centers, int nClusters);
