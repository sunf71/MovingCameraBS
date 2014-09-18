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
__global__ void UpdateClustersKernel(int nHeight, int nWidth, int* keys,SLICClusterCenter* d_inCenters,SLICClusterCenter* d_outCenters, int nClusters,int tNClusters  )
{
	//每个超像素一个线程
	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	if (clusterIdx >= nClusters)
		return;
	d_outCenters[keys[clusterIdx]] = d_inCenters[clusterIdx];

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
// CustomMin functor
struct CustomMin
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};
struct CustomType
{
	__device__ __host__ bool operator < (const CustomType& a) const 
	{
		return rgba< a.rgba;
	}
	__device__ __host__ CustomType()
	{}
	__device__ __host__ CustomType(int v)
	{
		rgba = v;

	}
	float rgba;
	
};
void testReducebykey()
{
	// Declare, allocate, and initialize device pointers for input and output
	int          num_items = 640*480;          // e.g., 8
	int          *d_keys_in;         // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
	CustomType          *d_values_in;       // e.g., [0, 7, 1, 6, 2, 5, 3, 4]
	int          *d_keys_out;        // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
	CustomType          *d_values_out;      // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
	int          *d_num_segments;    // e.g., [ ]
	cudaMalloc(&d_keys_in,sizeof(int)*num_items);
	cudaMalloc(&d_keys_out,sizeof(int)*num_items);
	cudaMalloc(&d_values_out,sizeof(CustomType)*num_items);
	cudaMalloc(&d_values_in,sizeof(CustomType)*num_items);
	cudaMalloc(&d_num_segments,sizeof(int));
	int* h_keys_in = new int[num_items];
	CustomType* h_values_in = new CustomType[num_items];
	for(int i=0; i<num_items; i++)
	{
		h_keys_in[i] = rand()%num_items;
		h_values_in[i] = CustomType(i);
	}
	cudaMemcpy(d_keys_in,h_keys_in,sizeof(CustomType)*num_items,cudaMemcpyHostToDevice);
	cudaMemcpy(d_values_in,h_values_in,sizeof(CustomType)*num_items,cudaMemcpyHostToDevice);
	CustomMin    reduction_op;
	
	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_segments, reduction_op, num_items);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run reduce-by-key
	cudaError_t error = cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_segments, reduction_op, num_items);

	int num;
	cudaMemcpy(&num,d_num_segments,sizeof(int),cudaMemcpyDeviceToHost);
	std::cout<<num;

	cudaFree(d_keys_in);
	cudaFree(d_keys_out);
	cudaFree(d_values_in);
	cudaFree(d_values_out);
	cudaFree(d_num_segments);
	delete[] h_keys_in;
	delete[] h_values_in;
}
void testReduce()
{
	void     *d_temp_storage = NULL;
	int width = 20;
	int height  = 20;
	int size = width*height;
	int nSuperPixels = 50;
	size_t   temp_storage_bytes = 0;
	int* h_labels_in = new int[size];
	for(int i=0; i<size; i++)
		h_labels_in[i] = rand()%nSuperPixels;
	int* h_labels_out = new int[size];
	float4 fakeColor;
	fakeColor.x = 0;
	fakeColor.y = 1;
	fakeColor.z = 2;
	fakeColor.w = 3;
	SLICClusterCenter* h_centers_out = new SLICClusterCenter[size];
	SLICClusterCenter* h_centers_in =  new SLICClusterCenter[size];
	for(int i=0; i<width; i++)
	{
		for(int j=0;j<height;j++)
		{
			int idx = i + j*width;
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
	cudaMalloc(&d_centers_in,sizeof(SLICClusterCenter)*size);
	cudaMalloc(&d_centers_out,sizeof(SLICClusterCenter)*size);
	cudaMemcpy(d_centers_in,h_centers_in,sizeof(SLICClusterCenter)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels_in,h_labels_in,sizeof(int)*size,cudaMemcpyHostToDevice);
	CustomAvg reduction_op;
	cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_labels_in, d_labels_out, d_centers_in, d_centers_out, d_num_segments, reduction_op, size);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run reduce-by-key
	cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_labels_in, d_labels_out, d_centers_in, d_centers_out, d_num_segments, reduction_op, size);
	int num;
	cudaMemcpy(&num,d_num_segments,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_labels_out,d_labels_out,sizeof(int)*num,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_centers_out,d_centers_out,sizeof(SLICClusterCenter)*num,cudaMemcpyDeviceToHost);
	/*for(int i=0; i<num;i++)
		std::cout<<h_labels_out[i]<<std::endl;
	std::cout<<"centers"<<std::endl;
	for(int i=0; i<num; i++)
		std::cout<<h_centers_out[i].xy.x<<","<<h_centers_out[i].xy.y<<std::endl;*/
	cudaFree(d_centers_in);
	cudaFree(d_centers_out);
	cudaFree(d_labels_out);
	cudaFree(d_labels_in);
	cudaFree(d_num_segments);
	delete[] h_labels_in;
	delete[] h_labels_out;
	delete[] h_centers_in;
	delete[] h_centers_out;
}
void UpdateClusters(float4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_centers, int& nClusters)
{
	typedef thrust::device_vector<int>::iterator  dIter;
	typedef thrust::device_vector<SLICClusterCenter>::iterator  vIter;
	CustomAvg op;
	thrust::equal_to<int> binary_pred;

	int size = nHeight*nWidth;

	SLICClusterCenter* d_centers_in;

	
	cudaMalloc(&d_centers_in,sizeof(SLICClusterCenter)*size);
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	InitClustersKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,d_centers_in);
	thrust::device_ptr<SLICClusterCenter> d_ptrV(d_centers_in);
	thrust::device_vector<SLICClusterCenter> valueVector(d_ptrV,d_ptrV+size);
	thrust::device_ptr<int> d_ptrK(labels);
	thrust::device_vector<int> keyVector(d_ptrK,d_ptrK+size);
	thrust::sort(keyVector.begin(),keyVector.end());
	//thrust::sequence(keyVector.begin(), keyVector.end());
	thrust::device_vector<int> outKeyVector(size);
	thrust::device_vector<SLICClusterCenter> outValueVector(size);
	thrust::pair<dIter, vIter>  new_end;
	std::cout<<"keys in\n";
	thrust::copy(keyVector.begin(), keyVector.end(), std::ostream_iterator<int>(std::cout, "\n"));
	new_end = thrust::reduce_by_key(keyVector.begin(),keyVector.end(),valueVector.begin(),outKeyVector.begin(),outValueVector.begin(),binary_pred,op);
	std::cout<<"keys out\n";
	thrust::copy(outKeyVector.begin(), new_end.first , std::ostream_iterator<int>(std::cout, "\n"));
	nClusters = new_end.first - outKeyVector.begin(); 
	std::cout << "Number of results are: \n" << new_end.first - outKeyVector.begin() << std::endl;
	cudaFree(d_centers_in);
	
	
}