#include "CudaSuperpixel.h"
#include "cub/cub.cuh"
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include "GpuTimer.h"

__global__ void InitClusterCentersKernel( float4* floatBuffer, int* labels, int nWidth, int nHeight,int step, int nSegs, SLICClusterCenter* vSLICCenterList )
{
	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	if (clusterIdx >= nSegs)
		return;
	int offsetBlock = (blockIdx.x *nWidth + threadIdx.x )* step;

	float2 avXY;

	avXY.x=threadIdx.x*step + (float)step/2.0;
	avXY.y=blockIdx.x*step + (float)step/2.0;

	

	float4 tmp;
	tmp.x = 0;
	tmp.y =0; 
	tmp.z = 0;

	for(int i=0; i<step; i++)
	{
		for(int j=0; j<step; j++)
		{
			if ((threadIdx.x )* step + i >= nWidth ||
				(blockIdx.x)*step + j>= nHeight)
				continue;
			int idx = offsetBlock + i + j*nWidth;
			tmp.x += floatBuffer[idx].x;
			tmp.y += floatBuffer[idx].y;
			tmp.z += floatBuffer[idx].z;
			labels[idx] = clusterIdx;
		}
	}

	double sz = step * step;
	tmp.x = tmp.x / sz;
	tmp.y = tmp.y /sz;
	tmp.z = tmp.z/sz;
	

	vSLICCenterList[clusterIdx].rgb= tmp;
	vSLICCenterList[clusterIdx].xy=avXY;
	vSLICCenterList[clusterIdx].nPoints=0;
	
}
__global__ void UpdateClustersKernel(int nHeight, int nWidth, int* keys,SLICClusterCenter* d_inCenters,SLICClusterCenter* d_outCenters, int nClusters,int tNClusters  )
{
	//每个超像素一个线程
	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	if (clusterIdx >= nClusters)
		return;
	d_outCenters[keys[clusterIdx]] = d_inCenters[clusterIdx] * (1.0/d_inCenters[clusterIdx].nPoints);
	
}
__global__ void InitClustersKernel(float4* imgBuffer, int nHeight, int nWidth, SLICClusterCenter* d_ceneters)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = k+j*nWidth;
	if (idx <nHeight*nWidth)
	{
		d_ceneters[idx].xy.x = k;
		d_ceneters[idx].xy.y = j;
		d_ceneters[idx].rgb = imgBuffer[idx];
		d_ceneters[idx].nPoints = 1;
	}
}
__host__ __device__ double distance(int x, int y, float4* imgBuffer,int width, int height, float alpha, float radius, int label, SLICClusterCenter* d_ceneters )
{
	int idx = x + y*width;
	double dr = abs((imgBuffer[idx].x - d_ceneters[label].rgb.x));
	double dg = abs(imgBuffer[idx].y - d_ceneters[label].rgb.y) ;
	double db = abs(imgBuffer[idx].z - d_ceneters[label].rgb.z);
	double d_rgb = sqrt(dr*dr + dg*dg + db*db);
	double dx = abs(x*1.f - d_ceneters[label].xy.x);
	double dy =  abs(y*1.f - d_ceneters[label].xy.y);
	double d_xy = sqrt(dx*dx + dy*dy);
	return (1-alpha)*d_rgb + alpha*d_xy/(radius/2);
}
__global__  void UpdateBoundaryKernel(float4* imgBuffer, int nHeight, int nWidth,int* labels,SLICClusterCenter* d_ceneters, int nClusters,float alpha, float radius)
{
	int dx4[4] = {-1,  0,  1, 0,};
	int dy4[4] = { 0, -1, 0, 1};
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int mainindex = k+j*nWidth;

	int np(0);
	int nl[4];
	for( int i = 0; i < 4; i++ )
	{
		int x = k + dx4[i];
		int y = j + dy4[i];

		if( (x >= 0 && x < nWidth) && (y >= 0 && y < nHeight) )
		{
			int index = y*nWidth + x;
			if( labels[mainindex] != labels[index] ) 
			{
				
				nl[np++] = labels[index];
			}			
		}
	}
	if( np > 0 )//change to 2 or 3 for thinner lines
	{
		double min = distance(k,j,imgBuffer,nWidth,nHeight,alpha,radius,labels[mainindex],d_ceneters);
		int idx = -1;
		for(int i=0; i<np; i++)
		{
			double dis = distance(k,j,imgBuffer,nWidth,nHeight,alpha,radius,nl[i],d_ceneters);
			if (dis < min)
			{
				min = dis;
				idx = i;
			}
		}
		if (idx >=0)
			labels[mainindex] = nl[idx];
	}
	
}


void InitClusterCenters(float4* d_rgbaBuffer, int* d_labels,int width, int height, int step, int &nSeg, SLICClusterCenter* d_centers)
{
	dim3 blockDim = (width+ step-1) / step ;
	dim3 gridDim = (height + step -1) / step;
	nSeg = blockDim.x * gridDim.x;
	
	InitClusterCentersKernel<<<gridDim,blockDim>>>(d_rgbaBuffer,d_labels,width,height,step,nSeg,d_centers);

}

void UpdateBoundary(float4* imgBuffer, int nHeight, int nWidth,int * labels,SLICClusterCenter* d_centers, int nClusters,float alpha, float radius)
{
	GpuTimer timer;
	timer.Start();
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	UpdateBoundaryKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,labels,d_centers,nClusters,alpha,radius);
	timer.Stop();
	std::cout<<"update UpdateBoundary"<<timer.Elapsed()<<std::endl;
}
void InitConstClusterCenters(float4* imgBuffer, int nHeight, int nWidth,SLICClusterCenter* d_centers_in)
{
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	InitClustersKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,d_centers_in);
}

void UpdateClusters(float4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_centers_in,
	thrust::device_ptr<int>& outKeyPtr,
	thrust::device_ptr<SLICClusterCenter>& outValuePtr,
	SLICClusterCenter* d_centers, int& nClusters,int tNClusters)
{
	GpuTimer timer;
	timer.Start();
	int size = nHeight*nWidth;	
	typedef thrust::device_ptr<int>  keyPtr;
	typedef thrust::device_ptr<SLICClusterCenter>  valuePtr;
	
	valuePtr centersPtr(d_centers);
	
	valuePtr d_ptrV(d_centers_in);	
	keyPtr d_ptrK(labels);	
	//std::cout<<"keys in before sort\n";
	//thrust::copy(d_ptrK, d_ptrK+size, std::ostream_iterator<int>(std::cout, "\n"));
	
	//std::cout<<"centers in before sort\n";
	//thrust::copy(d_ptrV, d_ptrV+size , std::ostream_iterator<SLICClusterCenter>(std::cout, "\n"));
	//std::cout<<"sort\n";

	thrust::sort_by_key(d_ptrK,d_ptrK+size,d_ptrV);
	timer.Stop();
	std::cout<<"sort by key "<<timer.Elapsed()<<std::endl;
	//std::cout<<"reduce\n";
	thrust::pair<keyPtr, valuePtr>  new_end;
	//std::cout<<"keys in after sort\n";
	//thrust::copy(d_ptrK, d_ptrK+size, std::ostream_iterator<int>(std::cout, "\n"));
	
	
	timer.Start();
	//std::cout<<"centers in after sort\n";
	//thrust::copy(d_ptrV, d_ptrV+size , std::ostream_iterator<SLICClusterCenter>(std::cout, "\n"));
	new_end = thrust::reduce_by_key(d_ptrK,d_ptrK+size,d_ptrV,outKeyPtr,outValuePtr);
	//std::cout<<"keys out\n";
	//thrust::copy(outKeyPtr, new_end.first , std::ostream_iterator<int>(std::cout, "\n"));
	timer.Stop();
	std::cout<<"reduce by key "<<timer.Elapsed()<<std::endl;

	nClusters = new_end.first - outKeyPtr;

	
	timer.Start();
	SLICClusterCenter* d_centersOut = thrust::raw_pointer_cast(outValuePtr);
	int * d_keysOut = thrust::raw_pointer_cast(outKeyPtr);;
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	UpdateClustersKernel<<<gridDim,blockDim>>>(nHeight, nWidth, d_keysOut,d_centersOut,d_centers,  nClusters, tNClusters);
	//std::cout<<"update\n";
	/*std::cout << "Number of results are: \n" << new_end.first - outKeyVector.begin() << std::endl;*/
	//thrust::device_ptr<SLICClusterCenter> centersoutPtr(d_centers);
	//std::cout<<"out centers\n";
	//thrust::copy(centersoutPtr, centersoutPtr+nClusters , std::ostream_iterator<SLICClusterCenter>(std::cout, "\n"));
	timer.Stop();
	std::cout<<"update cluster kernel"<<timer.Elapsed()<<std::endl;
}