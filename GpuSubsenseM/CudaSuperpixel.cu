#include "CudaSuperpixel.h"
#include "cub/cub.cuh"
#include <iostream>

// CustomAvg functor
struct CustomAvg
{
    template <typename T>
    __forceinline__   __device__ __host__ T operator()(const T &a, const T &b) const {
       return (a + b)*0.5;
    }
};

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
__global__ void UpdateClustersKernel(float4* imgBuffer, int nHeight, int nWidth, SLICClusterCenter* d_ceneters, int nClusters)
{

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
				imgBuffer[index].x = 0xff;
				imgBuffer[index].y = 0;
				imgBuffer[index].z = 0;
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
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	UpdateBoundaryKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,labels,d_centers,nClusters,alpha,radius);

}

void testReduce()
{
	void     *d_temp_storage = NULL;
	int size = 8;
	size_t   temp_storage_bytes = 0;
	int h_labels_in[] = { 1,1,2,2,3,3,4,4};
	int h_labels_out[8];
	float4 fakeColor;
	fakeColor.x = 0;
	fakeColor.y = 1;
	fakeColor.z = 2;
	fakeColor.w = 3;
	SLICClusterCenter h_centers_out[8];
	SLICClusterCenter h_centers_in[8];
	for(int i=0; i<4; i++)
	{
		for(int j=0;j<2;j++)
		{
			int idx = i + j*4;
			h_centers_in[idx].xy.x = i;
			h_centers_in[idx].xy.y = j;
			h_centers_in[idx].rgb = fakeColor;
		}
	}
	
	int* d_labels_in,*d_labels_out;
	SLICClusterCenter* d_centers_in,*d_centers_out;
	
	int * d_num_segments;
	cudaMalloc(&d_num_segments,sizeof(int));
	cudaMalloc(&d_labels_out,sizeof(int)*size);
	cudaMalloc(&d_labels_in,sizeof(int)*size);
	cudaMemcpy(d_centers_in,h_centers_in,sizeof(SLICClusterCenter)*8,cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels_in,h_labels_in,sizeof(int)*8,cudaMemcpyHostToDevice);
	CustomAvg reduction_op;
	cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_labels_in, d_labels_out, d_centers_in, d_centers_out, d_num_segments, reduction_op, size);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run reduce-by-key
	cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_labels_in, d_labels_out, d_centers_in, d_centers_out, d_num_segments, reduction_op, size);

	cudaFree(d_centers_in);
	cudaFree(d_centers_out);
	cudaFree(d_labels_out);
	cudaFree(d_labels_in);
	cudaFree(d_num_segments);
}
void UpdateClusters(float4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_centers, int nClusters)
{
	/*dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	UpdateClustersKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,labels,d_centers,nClusters);*/
	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	int size = nHeight*nWidth;
	size_t   temp_storage_bytes = 0;
	int* d_labels_out;
	SLICClusterCenter* d_centers_in;
	int * d_num_segments;
	cudaMalloc(&d_num_segments,sizeof(int));
	cudaMalloc(&d_labels_out,sizeof(int)*size);
	//cudaMalloc(&d_centers_out,sizeof(SLICClusterCenter)*size);
	cudaMalloc(&d_centers_in,sizeof(SLICClusterCenter)*size);
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	InitClustersKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,d_centers_in);

	/*SLICClusterCenter* h_centers_in = new SLICClusterCenter[size];
	cudaMemcpy(h_centers_in,d_centers_in,sizeof(SLICClusterCenter)*size,cudaMemcpyDeviceToHost);
	for(int i=0; i<size; i++)
	{
		std::cout<<h_centers_in[i].xy.x<<" ,"<<h_centers_in[i].xy.y<<std::endl;
	}*/

	CustomAvg reduction_op;
	cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, labels, d_labels_out, d_centers_in, d_centers, d_num_segments, reduction_op, size);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run reduce-by-key
	cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, labels, d_labels_out, d_centers_in, d_centers, d_num_segments, reduction_op, size);

	int h_num_segments;
	cudaMemcpy(&h_num_segments,d_num_segments,sizeof(int),cudaMemcpyDeviceToHost);
	SLICClusterCenter* h_centers = new SLICClusterCenter[h_num_segments];
	cudaMemcpy(h_centers,d_centers,sizeof(SLICClusterCenter)*h_num_segments,cudaMemcpyDeviceToHost);
	for(int i=0; i<h_num_segments; i++)
	{
		std::cout<<h_centers[i].xy.x<<" ,"<<h_centers[i].xy.y<<std::endl;
	}

	cudaFree(d_centers_in);
	cudaFree(d_labels_out);
	cudaFree(d_num_segments);
}