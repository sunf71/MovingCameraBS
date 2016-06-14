#include "CudaSuperpixel.h"
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
#define TILE_W 16
#define TILE_H 16
#define R 1
#define BLOCK_W (TILE_W+(2*R))
#define BLOCK_H (TILE_H + (2*R))
texture<uchar4> ImageTexture;
__global__ void InitClusterCentersKernel(int* labels, int nWidth, int nHeight,int step, int nSegs, SLICClusterCenter* vSLICCenterList )
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
			uchar4 pixel = tex1Dfetch(ImageTexture,idx);
			tmp.x += pixel.x;
			tmp.y += pixel.y;
			tmp.z += pixel.z;
			labels[idx] = clusterIdx;
		}
	}

	double sz = step * step;
	tmp.x = tmp.x / sz;
	tmp.y = tmp.y /sz;
	tmp.z = tmp.z/sz;


	vSLICCenterList[clusterIdx].rgb= tmp;
	vSLICCenterList[clusterIdx].xy=avXY;
	vSLICCenterList[clusterIdx].nPoints= (int)sz;

}
__global__ void UpdateClustersKernel(int nHeight, int nWidth, int* keys,SLICClusterCenter* d_inCenters,SLICClusterCenter* d_outCenters, int nClusters,int tNClusters  )
{
	//每个超像素一个线程
	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	if (clusterIdx >= nClusters)
		return;
	d_outCenters[keys[clusterIdx]] = d_inCenters[clusterIdx] * (1.0/d_inCenters[clusterIdx].nPoints);

}
__global__ void UpdateClusterCenterKernel(int heigth,int width, int step, int* d_labels, SLICClusterCenter* d_inCenters,int nClusters)
{
	const int size = 5*5;
	__shared__ int sLabels[size];
	__shared__ uchar4 sPixels[size];
	__shared__ uchar4 Rgb[size];
	__shared__ float2 XY[size];
	__shared__ int flag[size];
	int clusterIdx = blockIdx.x;
	int cx = d_inCenters[clusterIdx].xy.x;
	int cy = d_inCenters[clusterIdx].xy.y;
}
__global__ void UpdateClusterCenterKernel(uchar4* imgBuffer, int height, int width, int step,int * d_labels, SLICClusterCenter* d_inCenters,int nClusters)
{
	//每个超像素一个线程
	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	if (clusterIdx >= nClusters)
		return;
	int k = d_inCenters[clusterIdx].xy.x;
	int j = d_inCenters[clusterIdx].xy.y;
	float4 crgb = make_float4(0,0,0,0);
	float2 cxy = make_float2(0,0);
	int n = 0;
	//以原来的中心点为中心，step +１　为半径进行更新
	int radius = step;
	for (int x = k- radius; x<= k+radius; x++)
	{
		for(int y = j - radius; y<= j+radius; y++)
		{
			if  (x<0 || x>width-1 || y<0 || y> height-1)
				continue;
			int idx = x+y*width;
			if (d_labels[idx] == clusterIdx)
			{
				uchar4 pixel = tex1Dfetch(ImageTexture,idx);
				crgb.x += pixel.x;
				crgb.y += pixel.y;
				crgb.z += pixel.z;
				cxy.x += x;
				cxy.y += y;
				n++;
			}
		}
	}
	d_inCenters[clusterIdx].rgb = make_float4(crgb.x/n,crgb.y/n,crgb.z/n,0);
	d_inCenters[clusterIdx].xy = make_float2(cxy.x/n,cxy.y/n);
	d_inCenters[clusterIdx].nPoints = n;
}
__global__ void InitClustersKernel(uchar4* imgBuffer, int nHeight, int nWidth, SLICClusterCenter* d_ceneters)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = k+j*nWidth;
	if (idx <nHeight*nWidth)
	{
		d_ceneters[idx].xy.x = k;
		d_ceneters[idx].xy.y = j;
		d_ceneters[idx].rgb = make_float4(imgBuffer[idx].x,imgBuffer[idx].y,imgBuffer[idx].z,imgBuffer[idx].w);
		d_ceneters[idx].nPoints = 1;
	}
}
__device__ double distance(int x, int y, uchar4& pixel,int width, int height, float alpha, float radius, int label, SLICClusterCenter* d_ceneters)
{
	int idx = x + y*width;
	double dr = (pixel.x - d_ceneters[label].rgb.x);
	double dg = (pixel.y - d_ceneters[label].rgb.y) ;
	double db = (pixel.z - d_ceneters[label].rgb.z);
	double d_rgb = sqrt(dr*dr + dg*dg + db*db);
	double dx = (x*1.f - d_ceneters[label].xy.x);
	double dy =  (y*1.f - d_ceneters[label].xy.y);
	double d_xy = (dx*dx + dy*dy);
	return (1-alpha)*d_rgb + alpha*d_xy/(radius);
}
__device__ double distance(int x, int y, uchar4* imgBuffer,int width, int height, float alpha, float radius, int label, SLICClusterCenter* d_ceneters )
{
	int idx = x + y*width;
	uchar4 pixel = tex1Dfetch(ImageTexture,idx);
	double dr = (pixel.x - d_ceneters[label].rgb.x);
	double dg = (pixel.y - d_ceneters[label].rgb.y) ;
	double db = (pixel.z - d_ceneters[label].rgb.z);
	double d_rgb = sqrt(dr*dr + dg*dg + db*db);
	double dx = (x*1.f - d_ceneters[label].xy.x);
	double dy =  (y*1.f - d_ceneters[label].xy.y);
	double d_xy = (dx*dx + dy*dy);
	return (1-alpha)*d_rgb + alpha*d_xy/(radius);
}
__global__  void UpdateBoundaryKernel(uchar4* imgBuffer, int nHeight, int nWidth,int* labels,SLICClusterCenter* d_ceneters, int nClusters,float alpha, float radius)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int mainindex = k+j*nWidth;
	if (mainindex >= nHeight*nWidth)
		return;
	int dx4[4] = {-1,  0,  1, 0,};
	int dy4[4] = { 0, -1, 0, 1};


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
//更新时检查是否满足lattice条件
__global__  void UpdateBoundaryLatticeKernel(uchar4* imgBuffer, int nHeight, int nWidth,int* labels,SLICClusterCenter* d_ceneters, int nClusters,float alpha, float radius)
{

	int dx4[4] = {-1,  0,  1, 0,};
	int dy4[4] = { 0, -1, 0, 1};
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	for(int k = threadIdx.x + blockIdx.x * blockDim.x;
		k < nWidth; k+= blockDim.x*gridDim.x)
	{
		for(int j = threadIdx.y + blockIdx.y * blockDim.y;
			j<nHeight; j+= blockDim.y*gridDim.y)
		{
			
			int mainindex = k+j*nWidth;
			
			int np(0);
			int nl[4];
			int nlx[4];
			int nly[4];
			int nx[8],ny[8];
			int nnx[8],nny[8];
			for( int i = 0; i < 4; i++ )
			{
				int x = k + dx4[i];
				int y = j + dy4[i];

				if( (x >= 0 && x < nWidth) && (y >= 0 && y < nHeight) )
				{
					int index = y*nWidth + x;
					if( labels[mainindex] != labels[index] ) 
					{	
						nlx[np] = x;
						nly[np] = y;
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
				{
					//检查约束1，p所在8邻域内与p的label相同的像素必须是联通的
					bool pass1(true);

					int nc(0);
					int nnc(0);
					for(int i=0; i<8; i++)
					{
						int x = k + dx8[i];
						int y = j + dy8[i];
						if ( (x >= 0 && x < nWidth) && (y >= 0 && y < nHeight) )
						{
							nnx[nnc] = x;
							nny[nnc] = y;
							nnc++;
							int index = y*nWidth + x;
							if( labels[mainindex] == labels[index] ) 
							{				
								nx[nc] = x;
								ny[nc] = y;
								nc++;
							}
						}
					}
					if (nc>1)
					{
						bool xflag(true),yflag(true);
						for(int i=0; i<nc; i++)
						{
							if (nx[i] == k)
								xflag = false;
							if (ny[i] == j)
								yflag  = false;
							bool connect(false);
							for(int j=0; j<nc; j++)
							{
								if (j!=i && (abs(nx[i]-nx[j])<2 && abs(ny[i] - ny[j]) <2))
								{
									connect = true;
									break;
								}

							}
							if (!connect)
							{
								pass1 = false;
								break;
							}
						}
						if (pass1 && (xflag || yflag))
							pass1 = false;
					}

					//约束2，对于p四邻域内的邻居label n，在p的8邻域内必须存在除p之外的点是n的邻居
					bool pass2(true);
					for(int i=0; i<np; i++)
					{
						if (i!=idx)
						{
							bool connect(false);
							int x = nnx[i];
							int y = nny[i];
							for(int j=0; j<nnc; j++)
							{
								if ((abs(nnx[j] - x) < 2) && abs(nny[j] -y) <2)
								{
									connect = true;
									break;
								}
							}
							if (!connect)
							{
								pass2 = false;
								break;
							}
						}
					}
					if (pass2 && pass1 )
						labels[mainindex] = nl[idx];

				}
			}
		}
	}	
}

__global__ void AvgClusterCenterKernel(SLICClusterCenter* d_cenetersIn, int nClusters)
{
	//每个超像素一个线程
	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	if (clusterIdx >= nClusters || (d_cenetersIn[clusterIdx].nPoints ==0))
		return;
	int n = d_cenetersIn[clusterIdx].nPoints;
	d_cenetersIn[clusterIdx].rgb.x /= n;
	d_cenetersIn[clusterIdx].rgb.y /= n;
	d_cenetersIn[clusterIdx].rgb.z /= n;
	d_cenetersIn[clusterIdx].xy.x /= n;
	d_cenetersIn[clusterIdx].xy.y /= n;
}
__global__  void UpdateBoundaryKernel(uchar4* imgBuffer, int nHeight, int nWidth,int* labels,SLICClusterCenter* d_ceneters, SLICClusterCenter* d_centersOut, int nClusters,float alpha, float radius)
{
	int dx4[4] = {-1,  0,  1, 0,};
	int dy4[4] = { 0, -1, 0, 1};
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int mainindex = k+j*nWidth;
	if (mainindex > nHeight*nWidth-1)
		return;
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

	/*d_centersOut[labels[mainindex]].nPoints++;
	d_centersOut[labels[mainindex]].rgb.x +=imgBuffer[mainindex].x;
	d_centersOut[labels[mainindex]].rgb.y +=imgBuffer[mainindex].y;
	d_centersOut[labels[mainindex]].rgb.z +=imgBuffer[mainindex].z;
	d_centersOut[labels[mainindex]].xy.x +=k;
	d_centersOut[labels[mainindex]].xy.y +=j;*/


}
//shared memory version, this version is SLOWER, because there is no data reuse between different threads in a block
__global__  void SUpdateBoundaryKernel(uchar4* imgBuffer, int nHeight, int nWidth,int* labels,SLICClusterCenter* d_ceneters, int nClusters,float alpha, float radius)
{
	__shared__ uchar4 s_color[BLOCK_W*BLOCK_H];
	__shared__ int s_label[BLOCK_W*BLOCK_H];
	// First batch loading
	int dest = threadIdx.y * TILE_W + threadIdx.x,
		destY = dest / BLOCK_W, destX = dest % BLOCK_W,
		srcY = blockIdx.y * TILE_W + destY - R,
		srcX = blockIdx.x * TILE_W + destX - R,
		src = (srcY * nWidth + srcX);
	srcX = max(0,srcX);
	srcX = min(srcX,nWidth-1);
	srcY = max(srcY,0);
	srcY = min(srcY,nHeight-1);
	s_color[dest] = tex1Dfetch(ImageTexture,src);
	s_label[dest] = labels[src];
	//second batch loading
	dest = threadIdx.y * TILE_W + threadIdx.x + TILE_W * TILE_W;
	destY = dest / BLOCK_W, destX = dest % BLOCK_W;
	srcY = blockIdx.y * TILE_W + destY - R;
	srcX = blockIdx.x * TILE_W + destX - R;
	src = (srcY * nWidth + srcX);
	if (destY < BLOCK_W)
	{
		srcX = max(0,srcX);	 
		srcX = min(srcX,nWidth-1);
		srcY = max(srcY,0);
		srcY = min(srcY,nHeight-1);
		s_color[dest] = tex1Dfetch(ImageTexture,src);
		s_label[dest] = labels[src];
	}
	__syncthreads();

	int dx4[4] = {-1,  0,  1, 0,};
	int dy4[4] = { 0, -1, 0, 1};
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int mainindex = k+j*nWidth;
	if (mainindex > nHeight*nWidth-1)
		return;
	//unsigned idx = (threadIdx.y+R+y_sample)*BLOCK_W + threadIdx.x+R+x_sample;
	int label = s_label[(threadIdx.y+R)*BLOCK_W + threadIdx.x+R];

	int np(0);
	int nl[4];
	for( int i = 0; i < 4; i++ )
	{
		int x = k + dx4[i];
		int y = j + dy4[i];

		if( (x >= 0 && x < nWidth) && (y >= 0 && y < nHeight) )
		{
			int index = y*nWidth + x;
			int nlabel = s_label[(threadIdx.y+R+dy4[i])*BLOCK_W + threadIdx.x+R+dx4[i]];
			if( label != nlabel ) 
			{

				nl[np++] = nlabel;
			}			
		}
	}
	if( np > 0 )//change to 2 or 3 for thinner lines
	{
		uchar4 color = s_color[(threadIdx.y+R)*BLOCK_W + threadIdx.x+R];
		double min = distance(k,j,color,nWidth,nHeight,alpha,radius,label,d_ceneters);
		int idx = -1;
		for(int i=0; i<np; i++)
		{
			double dis = distance(k,j,color,nWidth,nHeight,alpha,radius,nl[i],d_ceneters);
			if (dis < min)
			{
				min = dis;
				idx = i;
			}
		}
		if (idx >=0)
			labels[mainindex] = nl[idx];
	}

	/*d_centersOut[labels[mainindex]].nPoints++;
	d_centersOut[labels[mainindex]].rgb.x +=imgBuffer[mainindex].x;
	d_centersOut[labels[mainindex]].rgb.y +=imgBuffer[mainindex].y;
	d_centersOut[labels[mainindex]].rgb.z +=imgBuffer[mainindex].z;
	d_centersOut[labels[mainindex]].xy.x +=k;
	d_centersOut[labels[mainindex]].xy.y +=j;*/


}
void UpdateClusterCenter(uchar4* imgBuffer, int height, int width, int step, int * d_labels, SLICClusterCenter* d_inCenters, int nClusters)
{
	//GpuTimer timer;
	//timer.Start();
	dim3 blockDim(128);
	dim3 gridDim((nClusters+127)/128);
	UpdateClusterCenterKernel<<<gridDim,blockDim>>>(imgBuffer, height,width,step,d_labels, d_inCenters,nClusters);
	//timer.Stop();
	//std::cout<<"UpdateClusterCenter "<<timer.Elapsed()<<std::endl;
}
void InitClusterCenters(uchar4* d_rgbaBuffer, int* d_labels,int width, int height, int step, int &nSeg, SLICClusterCenter* d_centers)
{
	/*GpuTimer timer;
	timer.Start();*/
	dim3 blockDim = (width+ step-1) / step ;
	dim3 gridDim = (height + step -1) / step;
	nSeg = blockDim.x * gridDim.x;
	cudaBindTexture( NULL, ImageTexture,
		d_rgbaBuffer,
		sizeof(uchar4)*width*height );
	InitClusterCentersKernel<<<gridDim,blockDim>>>(d_labels,width,height,step,nSeg,d_centers);
	/*timer.Stop();
	std::cout<<" InitClusterCentersKernel"<<timer.Elapsed()<<std::endl;*/
}
//图像已经在纹理内存中
void InitClusterCenters(int* d_labels,int width, int height, int step, int &nSeg, SLICClusterCenter* d_centers)
{
	/*GpuTimer timer;
	timer.Start();*/
	dim3 blockDim = (width+ step-1) / step ;
	dim3 gridDim = (height + step -1) / step;
	nSeg = blockDim.x * gridDim.x;
	InitClusterCentersKernel<<<gridDim,blockDim>>>(d_labels,width,height,step,nSeg,d_centers);
	/*timer.Stop();
	std::cout<<" InitClusterCentersKernel"<<timer.Elapsed()<<std::endl;*/
}
void UpdateBoundary(uchar4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_cenetersIn, SLICClusterCenter* d_centersOut, int nClusters,float alpha, float radius)
{

	GpuTimer timer;
	timer.Start();
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	UpdateBoundaryKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,labels,d_cenetersIn, d_centersOut,nClusters,alpha,radius);
	timer.Stop();
	std::cout<<"update UpdateBoundary"<<timer.Elapsed()<<std::endl;

}
void AvgClusterCenter(SLICClusterCenter* d_cenetersIn, int nClusters)
{
	GpuTimer timer;
	timer.Start();
	dim3 blockDim(128);
	dim3 gridDim((nClusters+127)/128);
	AvgClusterCenterKernel<<<gridDim,blockDim>>>(d_cenetersIn, nClusters);
	timer.Stop();
	std::cout<<"avg cluster center "<<timer.Elapsed()<<std::endl;
}
void UpdateBoundary(uchar4* imgBuffer, int nHeight, int nWidth,int * labels,SLICClusterCenter* d_centers, int nClusters,float alpha, float radius)
{
	//GpuTimer timer;
	//timer.Start();
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	UpdateBoundaryKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,labels,d_centers,nClusters,alpha,radius);
	//timer.Stop();
	//std::cout<<"update UpdateBoundary "<<timer.Elapsed()<<std::endl;
}
void UpdateBoundaryLattice(uchar4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_centers, int nClusters,float alpha, float radius)
{
	//GpuTimer timer;
	//timer.Start();
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	/*dim3 blockDim(1,1);
	dim3 gridDim(1,1);*/
	UpdateBoundaryLatticeKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,labels,d_centers,nClusters,alpha,radius);
	//timer.Stop();
	//std::cout<<"update UpdateBoundary "<<timer.Elapsed()<<std::endl;
}
void InitConstClusterCenters(uchar4* imgBuffer, int nHeight, int nWidth,SLICClusterCenter* d_centers_in)
{
	dim3 blockDim(16,16);
	dim3 gridDim((nWidth+15)/16,(nHeight+15)/16);
	InitClustersKernel<<<gridDim,blockDim>>>(imgBuffer,nHeight,nWidth,d_centers_in);
}

void UpdateClusters(uchar4* imgBuffer, int nHeight, int nWidth,int* labels, SLICClusterCenter* d_centers_in,
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