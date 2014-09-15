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
__global__ void kInitClusterCenters(float4* floatBuffer, 
									int nWidth, int nHeight, int& nSegs,  
									SLICClusterCenter* vSLICCenterList);