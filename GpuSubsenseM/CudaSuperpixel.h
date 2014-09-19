#pragma once
#include <ostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust\device_ptr.h>
struct SLICClusterCenter
{
	__host__ __device__ SLICClusterCenter(){}
	__host__ __device__ SLICClusterCenter(float4 _rgb,float2 _xy,int _points = 1):rgb(_rgb),xy(_xy),nPoints(_points){}
	__host__ __device__ SLICClusterCenter(const SLICClusterCenter& cc):rgb(cc.rgb),xy(cc.xy),nPoints(cc.nPoints){}
	__host__ __device__ SLICClusterCenter operator + (const SLICClusterCenter& a) const
	{
		float4 retF4 = make_float4(rgb.x+a.rgb.x,rgb.y+a.rgb.y,rgb.z+a.rgb.z,rgb.w+a.rgb.w);
		
		float2 retF2 = make_float2(xy.x + a.xy.x,xy.y + a.xy.y);		
		
		return  SLICClusterCenter(retF4,retF2,nPoints + a.nPoints);
	}
	friend std::ostream& operator<<(std::ostream& os, const SLICClusterCenter& u) 
	{ 
		os<<u.xy.x<<" "<<u.xy.y<<" "<<u.nPoints; 
		return os; 
	}; 
	__host__ __device__ SLICClusterCenter operator * (const float a) const
	{
		float4 retF4 = make_float4(rgb.x*a,rgb.y*a,rgb.z*a,rgb.w*a);		
		float2 retF2 = make_float2(xy.x*a,xy.y*a);		
		return SLICClusterCenter(retF4,retF2,nPoints);
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

//调整超像素中心，
//@nClusters实际超像素数量，
//@d_keys,与输入d_inCenters对应的超像素值
//@tNClusters理论总超像素数量（根据step和image size计算得到）
__global__ void UpdateClustersKernel(int nHeight, int nWidth, int* d_keys,SLICClusterCenter* d_inCenters,SLICClusterCenter* d_outCenters, int nClusters,int tNClusters );

void InitClusterCenters(float4* d_imgBuffer,int* d_labels, int nWidth, int nHeight, int step, int& nSeg, SLICClusterCenter* d_center);

void UpdateBoundary(float4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_ceneters, int nClusters,float alpha, float radius);

void UpdateClusters(float4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_centers_in,
	thrust::device_ptr<int>& outKeyPtr,
	thrust::device_ptr<SLICClusterCenter>& outValuePtr,
	SLICClusterCenter* d_centers, int& nClusters,int tNClusters);


void InitConstClusterCenters(float4* imgBuffer, int nHeight, int nWidth,SLICClusterCenter* d_centers_in);