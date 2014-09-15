#include "CudaSuperpixel.h"

__global__ void kInitClusterCentersKernel( float4* floatBuffer, int nWidth, int nHeight, int nSegs, SLICClusterCenter* vSLICCenterList )
{


	int blockWidth=nWidth/blockDim.x;
	int blockHeight=nHeight/gridDim.x;

	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	int offsetBlock = blockIdx.x * blockHeight * nWidth + threadIdx.x * blockWidth;

	float2 avXY;

	avXY.x=threadIdx.x*blockWidth + (float)blockWidth/2.0;
	avXY.y=blockIdx.x*blockHeight + (float)blockHeight/2.0;

	//use a single point to init center
	int offset=offsetBlock + blockHeight/2 * nWidth+ blockWidth/2 ;

	float4 fPixel=floatBuffer[offset];
	float4 tmp;
	tmp.x = 0;
	tmp.y =0; 
	tmp.z = 0;

	for(int i=0; i<blockHeight*blockWidth; i++)
	{	
		tmp.x = tmp.x + floatBuffer[offset+i].x;
		tmp.y = tmp.y + floatBuffer[offset+i].y;
		tmp.z = tmp.z + floatBuffer[offset+i].z;
	}

	double sz = blockWidth * blockHeight;
	tmp.x = tmp.x / sz;
	tmp.y = tmp.y /sz;
	tmp.z = tmp.z/sz;
	

	vSLICCenterList[clusterIdx].rgb= tmp;
	vSLICCenterList[clusterIdx].xy=avXY;
	vSLICCenterList[clusterIdx].nPoints=0;
	
}


void InitClusterCenters(float4* d_rgbaBuffer, int width, int height, int step, int &nSeg, SLICClusterCenter* d_centers)
{
	dim3 blockDim = (width+ step-1) / step ;
	dim3 gridDim = (height + step -1) / step;
	nSeg = blockDim.x * gridDim.x;
	
	kInitClusterCentersKernel<<<gridDim,blockDim>>>(d_rgbaBuffer,width,height,nSeg,d_centers);
}