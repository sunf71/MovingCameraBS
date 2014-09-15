#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct
{
	float4 rgb;
	float2 xy;
	int nPoints;

}SLICClusterCenter;
__global__ void kInitClusterCentersKernel(float4* floatBuffer, 
									int nWidth, int nHeight, int nSegs,  
									SLICClusterCenter* vSLICCenterList);

__global__ void UpdateBoundaryKernel(float4* imgBuffer, int nHeight, int nWidth, int x, int y, SLICClusterCenter* d_ceneters, int nClusters);

__global__ void UpdateCentersKernel(float4* imgBuffer, int nHeight, int nWidth, SLICClusterCenter* d_ceneters, int nClusters);

void InitClusterCenters(float4* d_imgBuffer, int nWidth, int nHeight, int step, int& nSeg, SLICClusterCenter* d_center);