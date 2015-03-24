#include "CudaBSOperator.h"
#include <thrust\device_vector.h>
#include <thrust\count.h>
#include "RandUtils.h"
#define BMSIZE 50
int width = 0;
int height = 0;
__constant__ uchar cpopcount_LUT8[256];
__constant__ size_t LBSPThres[256];
__constant__ ushort c_anSamplesInitPattern[7][7];
//curandState* devStates;
texture<uchar4> ImageTexture;
texture<uchar4> WarpedImageTexture;
texture<uchar> FGMaskLastTexture;
texture<uchar4> ColorModelTexture;
texture<ushort4> DescModelTexture;
texture<uchar4> LastColorTexture;
texture<ushort4> LastDescTexture;

#define TILE_W 16
#define TILE_H 16
#define R 2
#define BLOCK_W (TILE_W+(2*R))
#define BLOCK_H (TILE_H + (2*R))

__global__ void setup_kernel ( int size, curandState * state, unsigned long seed )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id<size)
	{
		curand_init ( seed+id, 0 , 0,&state[id] );
	}
} 

__device__  size_t getRandom(curandState* devStates, int ind)
{
    curandState localState = devStates[ind];
    size_t RANDOM = curand( &localState );
    devStates[ind] = localState; 
	return RANDOM;
}
__device__ inline void getRandNeighborPosition(curandState* devStates,int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border, const int width, const int height) {
	// simple 8-connected (3x3) neighbors pattern
	const int s_anNeighborPatternSize_3x3 = 8;
	const int s_anNeighborPattern_3x3[8][2] = {
		{-1, 1},  { 0, 1},  { 1, 1},
		{-1, 0},            { 1, 0},
		{-1,-1},  { 0,-1},  { 1,-1},
	};
	int ind = y_orig*width+x_orig;
    curandState localState = devStates[ind];
    size_t RANDOM = curand( &localState );
    devStates[ind] = localState; 
	int r = RANDOM%s_anNeighborPatternSize_3x3;
	x_neighbor = x_orig+s_anNeighborPattern_3x3[r][0];
	y_neighbor = y_orig+s_anNeighborPattern_3x3[r][1];
	if(x_neighbor<border)
		x_neighbor = border;
	else if(x_neighbor>=width-border)
		x_neighbor = width-border-1;
	if(y_neighbor<border)
		y_neighbor = border;
	else if(y_neighbor>=height-border)
		y_neighbor = height-border-1;
}
//! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
__device__  static inline void getRandSamplePosition(curandState* randState, int& x_sample, int& y_sample, const int x_orig, const int y_orig, const int border, const int width, const int height) {
	/*const int s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
	{0,     0,     4,     7,     4,     0,     0,},
	{0,    11,    53,    88,    53,    11,     0,},
	{4,    53,   240,   399,   240,    53,     4,},
	{7,    88,   399,   660,   399,    88,     7,},
	{4,    53,   240,   399,   240,    53,     4,},
	{0,    11,    53,    88,    53,    11,     0,},
	{0,     0,     4,     7,     4,     0,     0,},
};*/
	int ind = y_orig*width+x_orig;
    curandState localState = randState[ind];
    size_t RANDOM = curand( &localState );
    randState[ind] = localState; 
	int r = 1+RANDOM%s_nSamplesInitPatternTot;
	for(x_sample=0; x_sample<s_nSamplesInitPatternWidth; ++x_sample) {
		for(y_sample=0; y_sample<s_nSamplesInitPatternHeight; ++y_sample) {
			r -= c_anSamplesInitPattern[y_sample][x_sample];
			if(r<=0)
				goto stop;
		}
	}
	stop:
	x_sample += x_orig-s_nSamplesInitPatternWidth/2;
	y_sample += y_orig-s_nSamplesInitPatternHeight/2;
	if(x_sample<border)
		x_sample = border;
	else if(x_sample>=width-border)
		x_sample = width-border-1;
	if(y_sample<border)
		y_sample = border;
	else if(y_sample>=height-border)
		y_sample = height-border-1;
}
//! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
__device__  static inline void getRandSamplePosition(curandState* randState, ushort randomPattern[7][7],  int& x_sample, int& y_sample, const int x_orig, const int y_orig, const int border, const int width, const int height) {
	/*const int s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
	{0,     0,     4,     7,     4,     0,     0,},
	{0,    11,    53,    88,    53,    11,     0,},
	{4,    53,   240,   399,   240,    53,     4,},
	{7,    88,   399,   660,   399,    88,     7,},
	{4,    53,   240,   399,   240,    53,     4,},
	{0,    11,    53,    88,    53,    11,     0,},
	{0,     0,     4,     7,     4,     0,     0,},
};*/
	int ind = y_orig*width+x_orig;
    curandState localState = randState[ind];
    size_t RANDOM = curand( &localState );
    randState[ind] = localState; 
	int r = 1+RANDOM%s_nSamplesInitPatternTot;
	for(x_sample=0; x_sample<s_nSamplesInitPatternWidth; ++x_sample) {
		for(y_sample=0; y_sample<s_nSamplesInitPatternHeight; ++y_sample) {
			r -= randomPattern[y_sample][x_sample];
			if(r<=0)
				goto stop;
		}
	}
	stop:
	x_sample += x_orig-s_nSamplesInitPatternWidth/2;
	y_sample += y_orig-s_nSamplesInitPatternHeight/2;
	if(x_sample<border)
		x_sample = border;
	else if(x_sample>=width-border)
		x_sample = width-border-1;
	if(y_sample<border)
		y_sample = border;
	else if(y_sample>=height-border)
		y_sample = height-border-1;
}
//取(x,y)像素第id个值，每个像素有BMSIZE个数据，按照图像像素行顺序排列, 
template<typename T>
 __device__ T& GetRefFromBigMatrix(PtrStep<T> mat, int width, int height, int id, int x, int y,int bmSize = BMSIZE)
{
	int col = ((y*width)+x)*bmSize+id;
	return mat(0,col);
}
 template<typename T>
 __device__ T* GetPointerFromBigMatrix(PtrStep<T> mat, int width, int height, int id, int x, int y,int bmSize = BMSIZE)
{
	int col = ((y*width)+x)*bmSize+id;
	return  (mat.data+col);
}

template<typename T>
__device__ T   GetValueFromBigMatrix(const PtrStep<T> mat, int width, int height, int id, int x, int y,int bmSize = BMSIZE)
{
	int col = ((y*width)+x)*bmSize+id;
	return mat(0,col);
}
template<typename T>
 __device__  void SetValueToBigMatrix(PtrStep<T> mat, int width, int height, int id, int x, int y,const T& v, int bmSize = BMSIZE)
{
	int col = ((y*width)+x)*bmSize+id;
	mat(0,col) = v;
}

__device__ size_t L1dist_uchar(const uchar4& a, const uchar4& b)
{
	return abs(a.x-b.x) + abs(a.y-b.y) + abs(a.z-b.z);
}
__device__ size_t absdiff_uchar(const uchar&a , const uchar& b)
{
	return abs((int)a - (int)b);
}
//! computes the population count of a 16bit vector using an 8bit popcount LUT (min=0, max=48)
__device__ uchar popcount_ushort_8bitsLUT(ushort x) {
	//! popcount LUT for 8bit vectors
	
	return cpopcount_LUT8[(uchar)x] + cpopcount_LUT8[(uchar)(x>>8)];
}
__device__ size_t hdist_ushort_8bitLUT(const ushort& a, const ushort& b)
{

	return popcount_ushort_8bitsLUT(a^b);
}
__device__ size_t hdist_ushort_8bitLUT(const ushort4& a, const ushort4& b)
{

	return popcount_ushort_8bitsLUT(a.x^b.x)+popcount_ushort_8bitsLUT(a.y^b.y)+popcount_ushort_8bitsLUT(a.z^b.z);
}

void InitRandState(int width, int height,curandState* devStates)
{
	
	size_t N = width*height;
    
    // setup seeds
    setup_kernel <<< (N+127)/128,128>>> (N, devStates, time(NULL) );
}

void InitConstantMem(size_t* h_LBSPThres)
{
		const uchar hpopcount_LUT8[256] = {
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
	};
	const ushort s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
	{0,     0,     4,     7,     4,     0,     0,},
	{0,    11,    53,    88,    53,    11,     0,},
	{4,    53,   240,   399,   240,    53,     4,},
	{7,    88,   399,   660,   399,    88,     7,},
	{4,    53,   240,   399,   240,    53,     4,},
	{0,    11,    53,    88,    53,    11,     0,},
	{0,     0,     4,     7,     4,     0,     0,},
};
	cudaMemcpyToSymbol(cpopcount_LUT8,hpopcount_LUT8,sizeof(uchar)*256);
	cudaMemcpyToSymbol(LBSPThres,h_LBSPThres,sizeof(size_t)*256);
	cudaMemcpyToSymbol(c_anSamplesInitPattern,s_anSamplesInitPattern,sizeof(short)*49);
}

__global__ void TestRandNeighbourKernel(curandState* devStates,int width, int height, int* rand)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x<width && y<height)
	{
		int x_n,y_n;
		getRandNeighborPosition(devStates,x_n, y_n, x, y, 2, width, height);
		//getRandNeighborPosition_3x3(x_n, y_n, x, y, 2, width, height);
		rand[(x+y*width)*2] = abs(x_n-x);
		rand[(x+y*width)*2+1] = abs(y-y_n);
	}
}
void TestRandNeighbour(int width, int height, int* rand)
{
	dim3 block(16,16);
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	size_t N = width*height;
	curandState* devStates;
    cudaMalloc ( &devStates, N*sizeof( curandState ) );
    
    // setup seeds
    setup_kernel <<< N+127/128,128>>> (N, devStates, time(NULL) );


	TestRandNeighbourKernel<<<grid,block>>>(devStates,width,height,rand);
	cudaFree(devStates);
}

__device__ uchar4 GetPixelFromImgTexture(const int x, const int y, const int width)
{
	return tex1Dfetch(ImageTexture,y*width+x);
}

__device__ void LBSP(const uchar4& color, const int x, const int y, const int width, const size_t* const t, ushort4& out)
{
	uchar4 p0 = GetPixelFromImgTexture(x-1,1+y,width);
	uchar4 p1 = GetPixelFromImgTexture(x+1,y-1,width);
	uchar4 p2 = GetPixelFromImgTexture(x+1,y+1, width);
	uchar4 p3 = GetPixelFromImgTexture(x-1,y-1,width);
	uchar4 p4 = GetPixelFromImgTexture( x+1,y, width);
	uchar4 p5 = GetPixelFromImgTexture( x,y-1,width);
	uchar4 p6 = GetPixelFromImgTexture(x-1,y, width);
	uchar4 p7 = GetPixelFromImgTexture(  x,y+1,width);
	uchar4 p8 = GetPixelFromImgTexture(x-2,y-2,width);
	uchar4 p9 = GetPixelFromImgTexture( x+2,y+2, width);
	uchar4 p10 = GetPixelFromImgTexture( x+2,y-2,width);
	uchar4 p11 = GetPixelFromImgTexture(x- 2,y+2,width);
	uchar4 p12 = GetPixelFromImgTexture(  x,y+2,width);
	uchar4 p13 = GetPixelFromImgTexture( x,y-2,width);
	uchar4 p14 = GetPixelFromImgTexture( x+2,y, width);
	uchar4 p15 = GetPixelFromImgTexture(x-2,y, width);
	out.x = ((absdiff_uchar(p0.x,color.x) > t[0]) << 15)
		+ ((absdiff_uchar(p1.x,color.x) > t[0]) << 14)
		+ ((absdiff_uchar(p2.x,color.x) > t[0]) << 13)
		+ ((absdiff_uchar(p3.x,color.x) > t[0]) << 12)
		+ ((absdiff_uchar(p4.x,color.x) > t[0]) << 11)
		+ ((absdiff_uchar(p5.x,color.x) > t[0]) << 10)
		+ ((absdiff_uchar(p6.x,color.x) > t[0]) << 9)
		+ ((absdiff_uchar(p7.x,color.x) > t[0]) << 8)
		+ ((absdiff_uchar(p8.x,color.x) > t[0]) << 7)
		+ ((absdiff_uchar(p9.x,color.x) > t[0]) << 6)
		+ ((absdiff_uchar(p10.x,color.x) > t[0]) << 5)
		+ ((absdiff_uchar(p11.x,color.x) > t[0]) << 4)
		+ ((absdiff_uchar(p12.x,color.x) > t[0]) << 3)
		+ ((absdiff_uchar(p13.x,color.x) > t[0]) << 2)
		+ ((absdiff_uchar(p14.x,color.x) > t[0]) << 1)
		+ ((absdiff_uchar(p15.x,color.x) > t[0]));
	out.y = ((absdiff_uchar(p0.y,color.y) > t[1]) << 15)
		+ ((absdiff_uchar(p1.y,color.y) > t[1]) << 14)
		+ ((absdiff_uchar(p2.y,color.y) > t[1]) << 13)
		+ ((absdiff_uchar(p3.y,color.y) > t[1]) << 12)
		+ ((absdiff_uchar(p4.y,color.y) > t[1]) << 11)
		+ ((absdiff_uchar(p5.y,color.y) > t[1]) << 10)
		+ ((absdiff_uchar(p6.y,color.y) > t[1]) << 9)
		+ ((absdiff_uchar(p7.y,color.y) > t[1]) << 8)
		+ ((absdiff_uchar(p8.y,color.y) > t[1]) << 7)
		+ ((absdiff_uchar(p9.y,color.y) > t[1]) << 6)
		+ ((absdiff_uchar(p10.y,color.y) > t[1]) << 5)
		+ ((absdiff_uchar(p11.y,color.y) > t[1]) << 4)
		+ ((absdiff_uchar(p12.y,color.y) > t[1]) << 3)
		+ ((absdiff_uchar(p13.y,color.y) > t[1]) << 2)
		+ ((absdiff_uchar(p14.y,color.y) > t[1]) << 1)
		+ ((absdiff_uchar(p15.y,color.y) > t[1]));
	out.z = ((absdiff_uchar(p0.z,color.z) > t[2]) << 15)
		+ ((absdiff_uchar(p1.z,color.z) > t[2]) << 14)
		+ ((absdiff_uchar(p2.z,color.z) > t[2]) << 13)
		+ ((absdiff_uchar(p3.z,color.z) > t[2]) << 12)
		+ ((absdiff_uchar(p4.z,color.z) > t[2]) << 11)
		+ ((absdiff_uchar(p5.z,color.z) > t[2]) << 10)
		+ ((absdiff_uchar(p6.z,color.z) > t[2]) << 9)
		+ ((absdiff_uchar(p7.z,color.z) > t[2]) << 8)
		+ ((absdiff_uchar(p8.z,color.z) > t[2]) << 7)
		+ ((absdiff_uchar(p9.z,color.z) > t[2]) << 6)
		+ ((absdiff_uchar(p10.z,color.z) > t[2]) << 5)
		+ ((absdiff_uchar(p11.z,color.z) > t[2]) << 4)
		+ ((absdiff_uchar(p12.z,color.z) > t[2]) << 3)
		+ ((absdiff_uchar(p13.z,color.z) > t[2]) << 2)
		+ ((absdiff_uchar(p14.z,color.z) > t[2]) << 1)
		+ ((absdiff_uchar(p15.z,color.z) > t[2]));

}

__device__ void LBSP(const PtrStep<uchar4>& img, const uchar4& color, const int x, const int y, const size_t* const t, ushort4& out)
{
	uchar4 p0 = img(1+y,x-1);
	uchar4 p1 = img(y-1,x+1);
	uchar4 p2 = img(y+1, x+1);
	uchar4 p3 = img(y-1,x-1);
	uchar4 p4 = img( y, x+1);
	uchar4 p5 = img( y-1,x);
	uchar4 p6 = img(y, x-1);
	uchar4 p7 = img( y+1, x);
	uchar4 p8 = img(y-2,x-2);
	uchar4 p9 = img( y+2, x+2);
	uchar4 p10 = img( y-2,x+2);
	uchar4 p11 = img(y+2,x- 2);
	uchar4 p12 = img( y+2, x);
	uchar4 p13 = img( y-2,x);
	uchar4 p14 = img( y, x+2);
	uchar4 p15 = img(y, x-2);
	out.x = ((absdiff_uchar(p0.x,color.x) > t[0]) << 15)
		+ ((absdiff_uchar(p1.x,color.x) > t[0]) << 14)
		+ ((absdiff_uchar(p2.x,color.x) > t[0]) << 13)
		+ ((absdiff_uchar(p3.x,color.x) > t[0]) << 12)
		+ ((absdiff_uchar(p4.x,color.x) > t[0]) << 11)
		+ ((absdiff_uchar(p5.x,color.x) > t[0]) << 10)
		+ ((absdiff_uchar(p6.x,color.x) > t[0]) << 9)
		+ ((absdiff_uchar(p7.x,color.x) > t[0]) << 8)
		+ ((absdiff_uchar(p8.x,color.x) > t[0]) << 7)
		+ ((absdiff_uchar(p9.x,color.x) > t[0]) << 6)
		+ ((absdiff_uchar(p10.x,color.x) > t[0]) << 5)
		+ ((absdiff_uchar(p11.x,color.x) > t[0]) << 4)
		+ ((absdiff_uchar(p12.x,color.x) > t[0]) << 3)
		+ ((absdiff_uchar(p13.x,color.x) > t[0]) << 2)
		+ ((absdiff_uchar(p14.x,color.x) > t[0]) << 1)
		+ ((absdiff_uchar(p15.x,color.x) > t[0]));
	out.y = ((absdiff_uchar(p0.y,color.y) > t[1]) << 15)
		+ ((absdiff_uchar(p1.y,color.y) > t[1]) << 14)
		+ ((absdiff_uchar(p2.y,color.y) > t[1]) << 13)
		+ ((absdiff_uchar(p3.y,color.y) > t[1]) << 12)
		+ ((absdiff_uchar(p4.y,color.y) > t[1]) << 11)
		+ ((absdiff_uchar(p5.y,color.y) > t[1]) << 10)
		+ ((absdiff_uchar(p6.y,color.y) > t[1]) << 9)
		+ ((absdiff_uchar(p7.y,color.y) > t[1]) << 8)
		+ ((absdiff_uchar(p8.y,color.y) > t[1]) << 7)
		+ ((absdiff_uchar(p9.y,color.y) > t[1]) << 6)
		+ ((absdiff_uchar(p10.y,color.y) > t[1]) << 5)
		+ ((absdiff_uchar(p11.y,color.y) > t[1]) << 4)
		+ ((absdiff_uchar(p12.y,color.y) > t[1]) << 3)
		+ ((absdiff_uchar(p13.y,color.y) > t[1]) << 2)
		+ ((absdiff_uchar(p14.y,color.y) > t[1]) << 1)
		+ ((absdiff_uchar(p15.y,color.y) > t[1]));
	out.z = ((absdiff_uchar(p0.z,color.z) > t[2]) << 15)
		+ ((absdiff_uchar(p1.z,color.z) > t[2]) << 14)
		+ ((absdiff_uchar(p2.z,color.z) > t[2]) << 13)
		+ ((absdiff_uchar(p3.z,color.z) > t[2]) << 12)
		+ ((absdiff_uchar(p4.z,color.z) > t[2]) << 11)
		+ ((absdiff_uchar(p5.z,color.z) > t[2]) << 10)
		+ ((absdiff_uchar(p6.z,color.z) > t[2]) << 9)
		+ ((absdiff_uchar(p7.z,color.z) > t[2]) << 8)
		+ ((absdiff_uchar(p8.z,color.z) > t[2]) << 7)
		+ ((absdiff_uchar(p9.z,color.z) > t[2]) << 6)
		+ ((absdiff_uchar(p10.z,color.z) > t[2]) << 5)
		+ ((absdiff_uchar(p11.z,color.z) > t[2]) << 4)
		+ ((absdiff_uchar(p12.z,color.z) > t[2]) << 3)
		+ ((absdiff_uchar(p13.z,color.z) > t[2]) << 2)
		+ ((absdiff_uchar(p14.z,color.z) > t[2]) << 1)
		+ ((absdiff_uchar(p15.z,color.z) > t[2]));
}
__device__ void LBSP(const uchar4* blockColor, const uchar4& color, const int x, const int y, int width,const size_t* const t, ushort4& out)
{
	int idx = (y+R)*width + x +R;
	uchar4 p0 = blockColor[idx+width-1];
	uchar4 p1 = blockColor[idx-width+1];
	uchar4 p2 = blockColor[idx+1+width];
	uchar4 p3 = blockColor[idx-1-width];
	uchar4 p4 = blockColor[ idx+1];
	uchar4 p5 = blockColor[idx-width];
	uchar4 p6 = blockColor[idx-1];
	uchar4 p7 = blockColor[idx+width];
	uchar4 p8 = blockColor[idx-2*width-2];
	uchar4 p9 = blockColor[idx+2*width+2];
	uchar4 p10 =blockColor[idx-2*width+2];
	uchar4 p11 = blockColor[idx+2*width-2];
	uchar4 p12 =blockColor[idx+2*width];
	uchar4 p13 =blockColor[idx-2*width];
	uchar4 p14 =blockColor[idx+2];
	uchar4 p15 = blockColor[idx-2];
	out.x = ((absdiff_uchar(p0.x,color.x) > t[0]) << 15)
		+ ((absdiff_uchar(p1.x,color.x) > t[0]) << 14)
		+ ((absdiff_uchar(p2.x,color.x) > t[0]) << 13)
		+ ((absdiff_uchar(p3.x,color.x) > t[0]) << 12)
		+ ((absdiff_uchar(p4.x,color.x) > t[0]) << 11)
		+ ((absdiff_uchar(p5.x,color.x) > t[0]) << 10)
		+ ((absdiff_uchar(p6.x,color.x) > t[0]) << 9)
		+ ((absdiff_uchar(p7.x,color.x) > t[0]) << 8)
		+ ((absdiff_uchar(p8.x,color.x) > t[0]) << 7)
		+ ((absdiff_uchar(p9.x,color.x) > t[0]) << 6)
		+ ((absdiff_uchar(p10.x,color.x) > t[0]) << 5)
		+ ((absdiff_uchar(p11.x,color.x) > t[0]) << 4)
		+ ((absdiff_uchar(p12.x,color.x) > t[0]) << 3)
		+ ((absdiff_uchar(p13.x,color.x) > t[0]) << 2)
		+ ((absdiff_uchar(p14.x,color.x) > t[0]) << 1)
		+ ((absdiff_uchar(p15.x,color.x) > t[0]));
	out.y = ((absdiff_uchar(p0.y,color.y) > t[1]) << 15)
		+ ((absdiff_uchar(p1.y,color.y) > t[1]) << 14)
		+ ((absdiff_uchar(p2.y,color.y) > t[1]) << 13)
		+ ((absdiff_uchar(p3.y,color.y) > t[1]) << 12)
		+ ((absdiff_uchar(p4.y,color.y) > t[1]) << 11)
		+ ((absdiff_uchar(p5.y,color.y) > t[1]) << 10)
		+ ((absdiff_uchar(p6.y,color.y) > t[1]) << 9)
		+ ((absdiff_uchar(p7.y,color.y) > t[1]) << 8)
		+ ((absdiff_uchar(p8.y,color.y) > t[1]) << 7)
		+ ((absdiff_uchar(p9.y,color.y) > t[1]) << 6)
		+ ((absdiff_uchar(p10.y,color.y) > t[1]) << 5)
		+ ((absdiff_uchar(p11.y,color.y) > t[1]) << 4)
		+ ((absdiff_uchar(p12.y,color.y) > t[1]) << 3)
		+ ((absdiff_uchar(p13.y,color.y) > t[1]) << 2)
		+ ((absdiff_uchar(p14.y,color.y) > t[1]) << 1)
		+ ((absdiff_uchar(p15.y,color.y) > t[1]));
	out.z = ((absdiff_uchar(p0.z,color.z) > t[2]) << 15)
		+ ((absdiff_uchar(p1.z,color.z) > t[2]) << 14)
		+ ((absdiff_uchar(p2.z,color.z) > t[2]) << 13)
		+ ((absdiff_uchar(p3.z,color.z) > t[2]) << 12)
		+ ((absdiff_uchar(p4.z,color.z) > t[2]) << 11)
		+ ((absdiff_uchar(p5.z,color.z) > t[2]) << 10)
		+ ((absdiff_uchar(p6.z,color.z) > t[2]) << 9)
		+ ((absdiff_uchar(p7.z,color.z) > t[2]) << 8)
		+ ((absdiff_uchar(p8.z,color.z) > t[2]) << 7)
		+ ((absdiff_uchar(p9.z,color.z) > t[2]) << 6)
		+ ((absdiff_uchar(p10.z,color.z) > t[2]) << 5)
		+ ((absdiff_uchar(p11.z,color.z) > t[2]) << 4)
		+ ((absdiff_uchar(p12.z,color.z) > t[2]) << 3)
		+ ((absdiff_uchar(p13.z,color.z) > t[2]) << 2)
		+ ((absdiff_uchar(p14.z,color.z) > t[2]) << 1)
		+ ((absdiff_uchar(p15.z,color.z) > t[2]));
}
__global__ void WarpCudaBSOperatorKernel(const PtrStepSz<uchar4> img, const PtrStepSz<uchar4> warpedImg, curandState* randStates,  const PtrStepSz<float2> map, const PtrStepSz<float2> invMap,int frameIndex,
PtrStep<uchar4> colorModel,PtrStep<uchar4> wcolorModel,
PtrStep<ushort4> descModel,PtrStep<ushort4> wdescModel,
PtrStep<uchar> bModel,PtrStep<uchar> wbModel,
PtrStep<float> fModel,PtrStep<float> wfModel,
PtrStep<uchar> fgMask,	 PtrStep<uchar> lastFgMask, uchar* outMask,float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap)
{
	
	__shared__ uchar4 scolor[BLOCK_W*BLOCK_H];
	__shared__ size_t sLBSPThres[256];
	int width = img.cols;
	int height = img.rows;
	// First batch loading
	int dest = threadIdx.y * TILE_W + threadIdx.x,
		destY = dest / BLOCK_W, destX = dest % BLOCK_W,
		srcY = blockIdx.y * TILE_W + destY - R,
		srcX = blockIdx.x * TILE_W + destX - R;
	srcX = max(0,srcX);
	srcX = min(srcX,width-1);
	srcY = max(srcY,0);
	srcY = min(srcY,height-1);
	//scolor[dest] = img(srcY,srcX);
	scolor[dest] = tex1Dfetch(WarpedImageTexture,srcY*width+srcX);
	if (dest	< 256)
		sLBSPThres[dest] = LBSPThres[dest];
	//second batch loading
	dest = threadIdx.y * TILE_W + threadIdx.x + TILE_W * TILE_W;
	destY = dest / BLOCK_W, destX = dest % BLOCK_W;
	srcY = blockIdx.y * TILE_W + destY - R;
	srcX = blockIdx.x * TILE_W + destX - R;


	if (destY < BLOCK_W)
	{
		srcX = max(0,srcX);	 
		srcX = min(srcX,width-1);
		srcY = max(srcY,0);
		srcY = min(srcY,height-1);
		//scolor[destX + destY * BLOCK_W] = img(srcY,srcX);
		scolor[dest] = tex1Dfetch(WarpedImageTexture,srcY*width+srcX);
	}
	__syncthreads();
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x < img.cols-2 && x>=2 && y>=2 && y < img.rows-2)
	{
		int idx_uchar = x + y*width;
		float2 wxy = map(y,x);
		int wx = (int)(wxy.x+0.5);
		int wy = (int)(wxy.y+0.5);
		int widx_uchar = wx + wy*width;
		float* fptr = GetPointerFromBigMatrix(fModel,width,height,0,x,y,10);
		float* pfCurrLearningRate = fptr;
		float* pfCurrDistThresholdFactor = fptr +1;
		float* pfCurrVariationFactor = fptr +2;
		float* pfCurrMeanLastDist =fptr + 3;
		float* pfCurrMeanMinDist_LT =fptr +4;
		float* pfCurrMeanMinDist_ST =fptr +5;
		float* pfCurrMeanRawSegmRes_LT = fptr + 6;
		float* pfCurrMeanRawSegmRes_ST = fptr + 7;
		float* pfCurrMeanFinalSegmRes_LT =fptr + 8;
		float* pfCurrMeanFinalSegmRes_ST =fptr + 9;
		uchar& pbUnstableRegionMask = GetRefFromBigMatrix(bModel,width,height,0,x,y,2);
		int tidx = BMSIZE*width*height;
		ushort4* anLastIntraDesc = descModel.data + tidx + idx_uchar;//desc model = BMSIZE desc model + lastdesc
		uchar4* anLastColor =colorModel.data + tidx + idx_uchar;//desc model = BMSIZE desc model + lastdesc//color model=BMSIZE bgmodel +  lastcolor
		
		fptr = GetPointerFromBigMatrix(wfModel,width,height,0,wx,wy,10);
		float* wpfCurrLearningRate = fptr;
		float* wpfCurrDistThresholdFactor = fptr +1;
		float* wpfCurrVariationFactor = fptr +2;
		float* wpfCurrMeanLastDist =fptr + 3;
		float* wpfCurrMeanMinDist_LT =fptr +4;
		float* wpfCurrMeanMinDist_ST =fptr +5;
		float* wpfCurrMeanRawSegmRes_LT = fptr + 6;
		float* wpfCurrMeanRawSegmRes_ST = fptr + 7;
		float* wpfCurrMeanFinalSegmRes_LT =fptr + 8;
		float* wpfCurrMeanFinalSegmRes_ST =fptr + 9;
		uchar& wpbUnstableRegionMask = GetRefFromBigMatrix(wbModel,width,height,0,wx,wy,2);
		ushort4* wanLastIntraDesc = wdescModel.data + tidx + idx_uchar;//desc model = BMSIZE desc model + lastdesc
		uchar4* wanLastColor =wcolorModel.data + tidx + idx_uchar;//desc model = BMSIZE desc model + lastdesc//color model=BMSIZE bgmodel +  lastcolor
		
		ushort4 wCurrIntraDesc;
		*wanLastColor = tex1Dfetch(ImageTexture,idx_uchar);
		size_t thresholds[3] = {sLBSPThres[wanLastColor->x],sLBSPThres[wanLastColor->y],sLBSPThres[wanLastColor->z]};
		LBSP(*wanLastColor,x,y,width,thresholds,wCurrIntraDesc);
		*wanLastIntraDesc = wCurrIntraDesc;
		

		ushort4* wBGIntraDescPtr = GetPointerFromBigMatrix(wdescModel,width,height,0,wx,wy);
		uchar4* wBGColorPtr = GetPointerFromBigMatrix(wcolorModel,width,height,0,wx,wy);
		/*ushort4*  BGIntraDescPtr = GetPointerFromBigMatrix(descModel,width,height,0,x,y);
		uchar4*  BGColorPtr= GetPointerFromBigMatrix(colorModel,width,height,0,x,y);*/
		
		
		
		unsigned idx = (threadIdx.y+R)*BLOCK_W + threadIdx.x+R;
		const uchar4 CurrColor = scolor[idx];
		//const uchar4 CurrColor = img(y,x);
		uchar anCurrColor[3] = {CurrColor.x,CurrColor.y,CurrColor.z};
		ushort4 CurrInterDesc, CurrIntraDesc;
		//const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[CurrColor.x],m_anLBSPThreshold_8bitLUT[CurrColor.y],m_anLBSPThreshold_8bitLUT[CurrColor.z]};
		const size_t anCurrIntraLBSPThresholds[3] = {sLBSPThres[CurrColor.x],sLBSPThres[CurrColor.y],sLBSPThres[CurrColor.z]};
		//LBSP(img,CurrColor,x,y,anCurrIntraLBSPThresholds,CurrIntraDesc);
		LBSP(scolor,CurrColor,threadIdx.x,threadIdx.y,BLOCK_W,anCurrIntraLBSPThresholds,CurrIntraDesc);
		//LBSP(CurrColor,x,y,width,anCurrIntraLBSPThresholds,CurrIntraDesc);
		
		outMask[idx_uchar] = 0;
		//std::cout<<x<<","<<y<<std::endl;
		if (wx<2 || wx>= width-2 || wy<2 || wy>=height-2)
		{					
			//m_features.data[oidx_uchar] = 0xff;
			//m_nOutPixels ++;
			fgMask(y,x) = 0;
			outMask[idx_uchar] = 0xff;
			/*size_t s_rand = getRandom(randStates,idx_uchar)%BMSIZE;
			while(s_rand<BMSIZE){
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
				s_rand++;
			}*/
			return;
		}
		else
		{
			//反变换
			float2 fxy = invMap(y,x);
			//std::cout<<x<<","<<y<<std::endl;
			if (fxy.x<2 || fxy.y>= width-2 || fxy.y<2 || fxy.y>=height-2)
			{
				outMask[idx_uchar] = 0xff;
				fgMask(y,x) = 0;
				return;
			}
			
		}
		
		/**anLastColor = *wanLastColor;
		*anLastIntraDesc = *wanLastIntraDesc;*/
		
		const float fRollAvgFactor_LT = 1.0f/min(frameIndex,100);
		const float fRollAvgFactor_ST = 1.0f/min(frameIndex,25);
		
		
		size_t nMinTotDescDist=48;
		size_t nMinTotSumDist=765;
		const size_t s_nColorMaxDataRange_3ch = 765;
		const size_t s_nDescMaxDataRange_3ch = 48;
		
		const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*30)-((!pbUnstableRegionMask)*6));
		size_t m_nDescDistThreshold = 3;
		const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(pbUnstableRegionMask*m_nDescDistThreshold);
		const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
		const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
		const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
		

		ushort anCurrIntraDesc[3] = {CurrIntraDesc.x ,CurrIntraDesc.y, CurrIntraDesc.z};
		pbUnstableRegionMask = ((*pfCurrDistThresholdFactor)>3.0 || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>0.1 || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>0.1)?1:0;
		size_t nGoodSamplesCount=0, nSampleIdx=0;

		int modelOffset = idx_uchar*BMSIZE;
		while(nGoodSamplesCount<2 && nSampleIdx<BMSIZE) {
			/*const ushort4 const BGIntraDesc = BGIntraDescPtr[nSampleIdx];
			const uchar4 const BGColor = BGColorPtr[nSampleIdx];*/
			int  sampleIdx =  modelOffset + nSampleIdx;
			const ushort4 BGIntraDesc = tex1Dfetch(DescModelTexture,sampleIdx);
			const uchar4 BGColor = tex1Dfetch(ColorModelTexture,sampleIdx);
			wBGIntraDescPtr[nSampleIdx] = BGIntraDesc;
			wBGColorPtr[nSampleIdx] = BGColor;
			uchar anBGColor[3] = {BGColor.x,BGColor.y,BGColor.z};
			ushort anBGIntraDesc[3] = {BGIntraDesc.x,BGIntraDesc.y,BGIntraDesc.z};
			//const size_t anCurrInterLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[BGColor.x],m_anLBSPThreshold_8bitLUT[BGColor.y],m_anLBSPThreshold_8bitLUT[BGColor.z]};
			const size_t anCurrInterLBSPThresholds[3] = {sLBSPThres[BGColor.x],sLBSPThres[BGColor.y],sLBSPThres[BGColor.z]};
			//const size_t anCurrInterLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[0],m_anLBSPThreshold_8bitLUT[0],m_anLBSPThreshold_8bitLUT[0]};
			
			LBSP(scolor,BGColor,threadIdx.x,threadIdx.y,BLOCK_W,anCurrInterLBSPThresholds,CurrInterDesc);
			ushort anCurrInterDesc[3] ={CurrInterDesc.x,CurrInterDesc.y, CurrInterDesc.z};
			
			size_t nTotDescDist = 0;
			size_t nTotSumDist = 0;
			for(size_t c=0;c<3; ++c) {
				const size_t nColorDist = abs(anCurrColor[c]-anBGColor[c]);
				
				if(nColorDist>nCurrSCColorDistThreshold)
					goto failedcheck3ch;
				size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
				size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
				const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
				const size_t nSumDist = min((int)((nDescDist/2)*15+nColorDist),255);
				if(nSumDist>nCurrSCColorDistThreshold)
					goto failedcheck3ch;
				nTotDescDist += nDescDist;
				nTotSumDist += nSumDist;				
			}
			if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
				goto failedcheck3ch;

			if(nMinTotDescDist>nTotDescDist)
				nMinTotDescDist = nTotDescDist;
			if(nMinTotSumDist>nTotSumDist)
				nMinTotSumDist = nTotSumDist;
			nGoodSamplesCount++;
failedcheck3ch:
			nSampleIdx++;
		}
		//const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
		const float fNormalizedLastDist = ((float)L1dist_uchar(*anLastColor,CurrColor)/765 +(float)hdist_ushort_8bitLUT(*anLastIntraDesc,CurrIntraDesc)/48)/2;		
		*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
		if(nGoodSamplesCount<2) {
			// == foreground
			const float fNormalizedMinDist = min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(2-nGoodSamplesCount)/2);
			//const float fNormalizedMinDist = min(1.0f,((float)nMinTotSumDist/765) + (float)(2-nGoodSamplesCount)/2);
			*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
			*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
			fgMask(y,x) = UCHAR_MAX;
			/*if((getRandom(randStates,idx_uchar)%(size_t)2)==0) {
				const size_t s_rand = getRandom(randStates,idx_uchar)%BMSIZE;
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
			}*/

			
		}
		else {
			// == background
			fgMask(y,x) = 0;
			const float fNormalizedMinDist = ((float)nMinTotSumDist/765+(float)nMinTotDescDist/48)/2;
			//const float fNormalizedMinDist = ((float)nMinTotSumDist/765);
			*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
			*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
			
		}
		float UNSTABLE_REG_RATIO_MIN = 0.1;
		float FEEDBACK_T_INCR = 0.5;
		float FEEDBACK_T_DECR = 0.25;
		float FEEDBACK_V_INCR(1.f);
		float FEEDBACK_V_DECR(0.1f);
		float FEEDBACK_R_VAR(0.01f);
		/*if(lastFgMask(y,x) || (min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && fgMask(y,x))) {
			if((*pfCurrLearningRate)<fCurrLearningRateUpperCap)
				*pfCurrLearningRate += FEEDBACK_T_INCR/(max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
		}
		else if((*pfCurrLearningRate)>fCurrLearningRateLowerCap)
			*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
		if((*pfCurrLearningRate)< fCurrLearningRateLowerCap)
			*pfCurrLearningRate = fCurrLearningRateLowerCap;
		else if((*pfCurrLearningRate)>fCurrLearningRateUpperCap)
			*pfCurrLearningRate = fCurrLearningRateUpperCap;*/
		//uchar lastFG = tex1Dfetch(FGMaskLastTexture,idx_uchar);
		//if(max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && lastFG && fgMask(y,x))
		//	(*pfCurrVariationFactor) += FEEDBACK_V_INCR;
		//else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
		//	
		//	(*pfCurrVariationFactor) -= lastFG ? FEEDBACK_V_DECR/4:pbUnstableRegionMask?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
		//	if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
		//		(*pfCurrVariationFactor) = FEEDBACK_V_DECR;
		//}
		if((*pfCurrDistThresholdFactor)<pow(1.0f+min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
			(*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
		else {
			(*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
			if((*pfCurrDistThresholdFactor)<1.0f)
				(*pfCurrDistThresholdFactor) = 1.0f;
		}
		/*if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>=4)
		++nNonZeroDescCount;*/
		
		
		*wpfCurrDistThresholdFactor =  *pfCurrDistThresholdFactor;
		*wpfCurrVariationFactor = *pfCurrVariationFactor;
		*wpfCurrLearningRate = *pfCurrLearningRate;
		*wpfCurrMeanLastDist = *pfCurrMeanLastDist;
		*wpfCurrMeanMinDist_LT = *pfCurrMeanMinDist_LT;
		*wpfCurrMeanMinDist_ST = *pfCurrMeanMinDist_ST;
		*wpfCurrMeanRawSegmRes_LT = *pfCurrMeanRawSegmRes_LT; 
		*wpfCurrMeanRawSegmRes_ST = *pfCurrMeanRawSegmRes_ST;
		*wpfCurrMeanFinalSegmRes_LT = *pfCurrMeanFinalSegmRes_LT;
		*wpfCurrMeanFinalSegmRes_ST = *pfCurrMeanFinalSegmRes_ST;
		wpbUnstableRegionMask = pbUnstableRegionMask;

		
		for(int i=nSampleIdx+1; i<BMSIZE; i++)
		{		

			wBGIntraDescPtr[i] = tex1Dfetch(DescModelTexture,i+modelOffset);
			wBGColorPtr[i] = tex1Dfetch(ColorModelTexture,i+modelOffset);
		}
	}
	else if(x<width && y<height)
	{
		fgMask(y,x) = 0;
		outMask[y*width+x] =0;
	}
}
__global__ void CudaBSOperatorKernel(const PtrStepSz<uchar4> img,curandState* randStates,  double* homography, int frameIndex,
PtrStep<uchar4> colorModel,PtrStep<uchar4> wcolorModel,
PtrStep<ushort4> descModel,PtrStep<ushort4> wdescModel,
PtrStep<uchar> bModel,PtrStep<uchar> wbModel,
PtrStep<float> fModel,PtrStep<float> wfModel,
PtrStep<uchar> fgMask,	 PtrStep<uchar> lastFgMask, uchar* outMask, float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap, size_t* m_anLBSPThreshold_8bitLUT)
{
	
	__shared__ uchar4 scolor[BLOCK_W*BLOCK_H];
	int width = img.cols;
	int height = img.rows;
	// First batch loading
	int dest = threadIdx.y * TILE_W + threadIdx.x,
		destY = dest / BLOCK_W, destX = dest % BLOCK_W,
		srcY = blockIdx.y * TILE_W + destY - R,
		srcX = blockIdx.x * TILE_W + destX - R,
		src = (srcY * width + srcX);
	srcX = max(0,srcX);
	srcX = min(srcX,width-1);
	srcY = max(srcY,0);
	srcY = min(srcY,height-1);
	//scolor[dest] = img(srcY,srcX);
	scolor[dest] = tex1Dfetch(ImageTexture,srcY*width+srcX);

	//second batch loading
	dest = threadIdx.y * TILE_W + threadIdx.x + TILE_W * TILE_W;
	destY = dest / BLOCK_W, destX = dest % BLOCK_W;
	srcY = blockIdx.y * TILE_W + destY - R;
	srcX = blockIdx.x * TILE_W + destX - R;


	if (destY < BLOCK_W)
	{
		srcX = max(0,srcX);	 
		srcX = min(srcX,width-1);
		srcY = max(srcY,0);
		srcY = min(srcY,height-1);
		//scolor[destX + destY * BLOCK_W] = img(srcY,srcX);
		scolor[dest] = tex1Dfetch(ImageTexture,srcY*width+srcX);
	}
	__syncthreads();

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x < img.cols-2 && x>=2 && y>=2 && y < img.rows-2)
	{
		int idx_uchar = x + y*width;
		double* ptr = homography;
		float fx,fy,fw;
		fx = x*ptr[0] + y*ptr[1] + ptr[2];
		fy = x*ptr[3] + y*ptr[4] + ptr[5];
		fw = x*ptr[6] + y*ptr[7] + ptr[8];
		fx /=fw;
		fy/=fw;
		int wx = (int)(fx+0.5);
		int wy = (int)(fy+0.5);
		
		float* fptr = GetPointerFromBigMatrix(fModel,width,height,0,x,y,10);
		float* pfCurrLearningRate = fptr;
		float* pfCurrDistThresholdFactor = fptr +1;
		float* pfCurrVariationFactor = fptr +2;
		float* pfCurrMeanLastDist =fptr + 3;
		float* pfCurrMeanMinDist_LT =fptr +4;
		float* pfCurrMeanMinDist_ST =fptr +5;
		float* pfCurrMeanRawSegmRes_LT = fptr + 6;
		float* pfCurrMeanRawSegmRes_ST = fptr + 7;
		float* pfCurrMeanFinalSegmRes_LT =fptr + 8;
		float* pfCurrMeanFinalSegmRes_ST =fptr + 9;
		uchar& pbUnstableRegionMask = GetRefFromBigMatrix(bModel,width,height,0,x,y,2);
		ushort4* anLastIntraDesc = descModel.data + BMSIZE*width*height + y*width + x;//desc model = BMSIZE desc model + lastdesc
		uchar4* anLastColor =colorModel.data + BMSIZE*width*height + y*width + x;//desc model = BMSIZE desc model + lastdesc//color model=BMSIZE bgmodel +  lastcolor
		
		fptr = GetPointerFromBigMatrix(wfModel,width,height,0,wx,wy,10);
		float* wpfCurrLearningRate = fptr;
		float* wpfCurrDistThresholdFactor = fptr +1;
		float* wpfCurrVariationFactor = fptr +2;
		float* wpfCurrMeanLastDist =fptr + 3;
		float* wpfCurrMeanMinDist_LT =fptr +4;
		float* wpfCurrMeanMinDist_ST =fptr +5;
		float* wpfCurrMeanRawSegmRes_LT = fptr + 6;
		float* wpfCurrMeanRawSegmRes_ST = fptr + 7;
		float* wpfCurrMeanFinalSegmRes_LT =fptr + 8;
		float* wpfCurrMeanFinalSegmRes_ST =fptr + 9;
		uchar& wpbUnstableRegionMask = GetRefFromBigMatrix(wbModel,width,height,0,wx,wy,2);
		ushort4* wanLastIntraDesc = wdescModel.data + BMSIZE*width*height + wy*width + wx;//desc model = BMSIZE desc model + lastdesc
		uchar4* wanLastColor =wcolorModel.data + BMSIZE*width*height+ wy*width +wx;//desc model = BMSIZE desc model + lastdesc//color model=BMSIZE bgmodel +  lastcolor
		

		

		ushort4* wBGIntraDescPtr = GetPointerFromBigMatrix(wdescModel,width,height,0,wx,wy);
		uchar4* wBGColorPtr = GetPointerFromBigMatrix(wcolorModel,width,height,0,wx,wy);
		ushort4*  BGIntraDescPtr = GetPointerFromBigMatrix(descModel,width,height,0,x,y);
		uchar4*  BGColorPtr= GetPointerFromBigMatrix(colorModel,width,height,0,x,y);
		

		
		unsigned idx = (threadIdx.y+R)*BLOCK_W + threadIdx.x+R;
		const uchar4 CurrColor = scolor[idx];
		//const uchar4 CurrColor = img(y,x);
		uchar anCurrColor[3] = {CurrColor.x,CurrColor.y,CurrColor.z};
		ushort4 CurrInterDesc, CurrIntraDesc;
		//const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[CurrColor.x],m_anLBSPThreshold_8bitLUT[CurrColor.y],m_anLBSPThreshold_8bitLUT[CurrColor.z]};
		const size_t anCurrIntraLBSPThresholds[3] = {LBSPThres[CurrColor.x],LBSPThres[CurrColor.y],LBSPThres[CurrColor.z]};
		//LBSP(img,CurrColor,x,y,anCurrIntraLBSPThresholds,CurrIntraDesc);
		LBSP(scolor,CurrColor,threadIdx.x,threadIdx.y,BLOCK_W,anCurrIntraLBSPThresholds,CurrIntraDesc);

		outMask[y*width+x] = 0;
		//std::cout<<x<<","<<y<<std::endl;
		if (wx<2 || wx>= width-2 || wy<2 || wy>=height-2)
		{					
			//m_features.data[oidx_uchar] = 0xff;
			//m_nOutPixels ++;
			fgMask(y,x) = 0;
			outMask[y*width+x] = 0xff;
			size_t s_rand = getRandom(randStates,idx_uchar)%BMSIZE;
			while(s_rand<BMSIZE){
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
				s_rand++;
			}
			return;
		}
		else
		{
			//反变换
			ptr += 9;
			fx = x*ptr[0] + y*ptr[1] + ptr[2];
			fy = x*ptr[3] + y*ptr[4] + ptr[5];
			fw = x*ptr[6] + y*ptr[7] + ptr[8];
			fx /=fw;
			fy/=fw;
			//std::cout<<x<<","<<y<<std::endl;
			if (fx<2 || fx>= width-2 || fy<2 || fy>=height-2)
			{
				outMask[y*width+x] = 0xff;
			}
		}
		*pfCurrDistThresholdFactor =  *wpfCurrDistThresholdFactor;
		*pfCurrVariationFactor = *wpfCurrVariationFactor;
		*pfCurrLearningRate = *wpfCurrLearningRate;
		*pfCurrMeanLastDist = *wpfCurrMeanLastDist;
		*pfCurrMeanMinDist_LT = *wpfCurrMeanMinDist_LT;
		*pfCurrMeanMinDist_ST = *wpfCurrMeanMinDist_ST;
		*pfCurrMeanRawSegmRes_LT = *wpfCurrMeanRawSegmRes_LT; 
		*pfCurrMeanRawSegmRes_ST = *wpfCurrMeanRawSegmRes_ST;
		*pfCurrMeanFinalSegmRes_LT = *wpfCurrMeanFinalSegmRes_LT;
		*pfCurrMeanFinalSegmRes_ST = *wpfCurrMeanFinalSegmRes_ST;
		pbUnstableRegionMask = wpbUnstableRegionMask;


		for(int i=0; i<BMSIZE; i++)
		{
			BGIntraDescPtr[i] = wBGIntraDescPtr[i];
			BGColorPtr[i] = wBGColorPtr[i];
		}
		/**anLastColor = *wanLastColor;
		*anLastIntraDesc = *wanLastIntraDesc;*/
		
		const float fRollAvgFactor_LT = 1.0f/min(frameIndex,25*4);
		const float fRollAvgFactor_ST = 1.0f/min(frameIndex,25);
		
		
		size_t nMinTotDescDist=48;
		size_t nMinTotSumDist=765;
		const size_t s_nColorMaxDataRange_3ch = 255*3;
		const size_t s_nDescMaxDataRange_3ch = 16*3;
		
		const size_t nCurrColorDistThreshold = (size_t)(((*wpfCurrDistThresholdFactor)*30)-((!wpbUnstableRegionMask)*6));
		size_t m_nDescDistThreshold = 3;
		const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*wpfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(wpbUnstableRegionMask*m_nDescDistThreshold);
		const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
		const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
		const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;

		ushort anCurrIntraDesc[3] = {CurrIntraDesc.x ,CurrIntraDesc.y, CurrIntraDesc.z};
		pbUnstableRegionMask = ((*wpfCurrDistThresholdFactor)>3.0 || (*wpfCurrMeanRawSegmRes_LT-*wpfCurrMeanFinalSegmRes_LT)>0.1 || (*wpfCurrMeanRawSegmRes_ST-*wpfCurrMeanFinalSegmRes_ST)>0.1)?1:0;
		size_t nGoodSamplesCount=0, nSampleIdx=0;

		
		while(nGoodSamplesCount<2 && nSampleIdx<BMSIZE) {
			const ushort4 const BGIntraDesc = BGIntraDescPtr[nSampleIdx];
			const uchar4 const BGColor = BGColorPtr[nSampleIdx];
			
			uchar anBGColor[3] = {BGColor.x,BGColor.y,BGColor.z};
			ushort anBGIntraDesc[3] = {BGIntraDesc.x,BGIntraDesc.y,BGIntraDesc.z};
			//const size_t anCurrInterLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[BGColor.x],m_anLBSPThreshold_8bitLUT[BGColor.y],m_anLBSPThreshold_8bitLUT[BGColor.z]};
			const size_t anCurrInterLBSPThresholds[3] = {LBSPThres[BGColor.x],LBSPThres[BGColor.y],LBSPThres[BGColor.z]};
			//const size_t anCurrInterLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[0],m_anLBSPThreshold_8bitLUT[0],m_anLBSPThreshold_8bitLUT[0]};
			
			LBSP(scolor,BGColor,threadIdx.x,threadIdx.y,BLOCK_W,anCurrInterLBSPThresholds,CurrInterDesc);
			ushort anCurrInterDesc[3] ={CurrInterDesc.x,CurrInterDesc.y, CurrInterDesc.z};
			
			size_t nTotDescDist = 0;
			size_t nTotSumDist = 0;
			for(size_t c=0;c<3; ++c) {
				const size_t nColorDist = abs(anCurrColor[c]-anBGColor[c]);
				
				if(nColorDist>nCurrSCColorDistThreshold)
					goto failedcheck3ch;
				size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
				size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
				const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
				const size_t nSumDist = min((int)((nDescDist/2)*15+nColorDist),255);
				if(nSumDist>nCurrSCColorDistThreshold)
					goto failedcheck3ch;
				nTotDescDist += nDescDist;
				nTotSumDist += nSumDist;
				//nTotSumDist += nColorDist;
			}
			if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
				goto failedcheck3ch;

			if(nMinTotDescDist>nTotDescDist)
				nMinTotDescDist = nTotDescDist;
			if(nMinTotSumDist>nTotSumDist)
				nMinTotSumDist = nTotSumDist;
			nGoodSamplesCount++;
failedcheck3ch:
			nSampleIdx++;
		}
		//const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
		const float fNormalizedLastDist = ((float)L1dist_uchar(*anLastColor,CurrColor)/765 +(float)hdist_ushort_8bitLUT(*anLastIntraDesc,CurrIntraDesc)/48)/2;		
		*pfCurrMeanLastDist = (*wpfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
		if(nGoodSamplesCount<2) {
			// == foreground
			const float fNormalizedMinDist = min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(2-nGoodSamplesCount)/2);
			//const float fNormalizedMinDist = min(1.0f,((float)nMinTotSumDist/765) + (float)(2-nGoodSamplesCount)/2);
			*pfCurrMeanMinDist_LT = (*wpfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			*pfCurrMeanMinDist_ST = (*wpfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			*pfCurrMeanRawSegmRes_LT = (*wpfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
			*pfCurrMeanRawSegmRes_ST = (*wpfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
			fgMask(y,x) = UCHAR_MAX;
			if((getRandom(randStates,idx_uchar)%(size_t)2)==0) {
				const size_t s_rand = getRandom(randStates,idx_uchar)%BMSIZE;
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
			}
		}
		else {
			// == background
			fgMask(y,x) = 0;
			const float fNormalizedMinDist = ((float)nMinTotSumDist/765+(float)nMinTotDescDist/48)/2;
			//const float fNormalizedMinDist = ((float)nMinTotSumDist/765);
			*pfCurrMeanMinDist_LT = (*wpfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			*pfCurrMeanMinDist_ST = (*wpfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			*pfCurrMeanRawSegmRes_LT = (*wpfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
			*pfCurrMeanRawSegmRes_ST = (*wpfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
			const size_t nLearningRate =(size_t)ceil(*wpfCurrLearningRate);
			if(getRandom(randStates,idx_uchar)%nLearningRate==0) {
				const size_t s_rand =getRandom(randStates,idx_uchar)%BMSIZE;
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
			}
			int x_rand,y_rand;
			const bool bCurrUsing3x3Spread = !pbUnstableRegionMask;
			if(bCurrUsing3x3Spread)
			{
				getRandNeighborPosition(randStates,x_rand,y_rand,wx,wy,2,img.cols,img.rows);

				const size_t n_rand = getRandom(randStates,idx_uchar);
				const float fRandMeanLastDist = GetValueFromBigMatrix(wfModel,width,height,3,x_rand,y_rand,10);
				const float fRandMeanRawSegmRes = GetValueFromBigMatrix(wfModel,width,height,8,x_rand,y_rand,10);
				const size_t s_rand =getRandom(randStates,idx_uchar)%BMSIZE;
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>0.995 && fRandMeanLastDist<0.01 && (n_rand%((size_t)fCurrLearningRateLowerCap))==0)) {
						SetValueToBigMatrix(colorModel,width,height,s_rand,x_rand,y_rand,CurrColor);
						SetValueToBigMatrix(descModel,width,height,s_rand,x_rand,y_rand,CurrIntraDesc);
				}
			}
		}
		float UNSTABLE_REG_RATIO_MIN = 0.1;
		float FEEDBACK_T_INCR = 0.5;
		float FEEDBACK_T_DECR = 0.1;
		float FEEDBACK_V_INCR(1.f);
		float FEEDBACK_V_DECR(0.1f);
		float FEEDBACK_R_VAR(0.01f);
		if(lastFgMask(wy,wx) || (min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && fgMask(y,x))) {
			if((*pfCurrLearningRate)<fCurrLearningRateUpperCap)
				*pfCurrLearningRate += FEEDBACK_T_INCR/(max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
		}
		else if((*pfCurrLearningRate)>fCurrLearningRateLowerCap)
			*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
		if((*pfCurrLearningRate)< fCurrLearningRateLowerCap)
			*pfCurrLearningRate = fCurrLearningRateLowerCap;
		else if((*pfCurrLearningRate)>fCurrLearningRateUpperCap)
			*pfCurrLearningRate = fCurrLearningRateUpperCap;
		if(max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && GetValueFromBigMatrix(wbModel,width,height,1,wx,wy))
			(*pfCurrVariationFactor) += FEEDBACK_V_INCR;
		else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
			(*pfCurrVariationFactor) -= pbUnstableRegionMask?FEEDBACK_V_DECR/4:pbUnstableRegionMask?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
			if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
				(*pfCurrVariationFactor) = FEEDBACK_V_DECR;
		}
		if((*pfCurrDistThresholdFactor)<pow(1.0f+min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
			(*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
		else {
			(*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
			if((*pfCurrDistThresholdFactor)<1.0f)
				(*pfCurrDistThresholdFactor) = 1.0f;
		}
		/*if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>=4)
		++nNonZeroDescCount;*/
		*anLastColor = CurrColor;
		*anLastIntraDesc = CurrIntraDesc;

	}
	else if(x<width && y<height)
	{
		fgMask(y,x) = 0;
		outMask[y*width+x] =0;
	}
}
__global__ void CudaBSOperatorKernel(const PtrStepSz<uchar4> img,const PtrStep<uchar> mask, curandState* randStates,  double* homography, int frameIndex,
PtrStep<uchar4> colorModel,PtrStep<uchar4> wcolorModel,
PtrStep<ushort4> descModel,PtrStep<ushort4> wdescModel,
PtrStep<uchar> bModel,PtrStep<uchar> wbModel,
PtrStep<float> fModel,PtrStep<float> wfModel,
PtrStep<uchar> fgMask,	 PtrStep<uchar> lastFgMask, uchar* outMask, float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap)
{
	
	__shared__ uchar4 scolor[BLOCK_W*BLOCK_H];
	

	int width = img.cols;
	int height = img.rows;
	// First batch loading
	int dest = threadIdx.y * TILE_W + threadIdx.x,
		destY = dest / BLOCK_W, destX = dest % BLOCK_W,
		srcY = blockIdx.y * TILE_W + destY - R,
		srcX = blockIdx.x * TILE_W + destX - R;
	
	srcX = max(0,srcX);
	srcX = min(srcX,width-1);
	srcY = max(srcY,0);
	srcY = min(srcY,height-1);
	int src = (srcY * width + srcX);
	//scolor[dest] = img(srcY,srcX);
	scolor[dest] = tex1Dfetch(ImageTexture,src);

	//second batch loading
	dest = threadIdx.y * TILE_W + threadIdx.x + TILE_W * TILE_W;
	destY = dest / BLOCK_W, destX = dest % BLOCK_W;
	srcY = blockIdx.y * TILE_W + destY - R;
	srcX = blockIdx.x * TILE_W + destX - R;


	if (destY < BLOCK_W)
	{
		srcX = max(0,srcX);	 
		srcX = min(srcX,width-1);
		srcY = max(srcY,0);
		srcY = min(srcY,height-1);
		//scolor[destX + destY * BLOCK_W] = img(srcY,srcX);
		scolor[dest] = tex1Dfetch(ImageTexture,srcY*width+srcX);
	}
	__syncthreads();

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x < img.cols-2 && x>=2 && y>=2 && y < img.rows-2)
	{
		int idx_uchar = x + y*width;
		
		double* ptr = homography;
		float fx,fy,fw;
		fx = x*ptr[0] + y*ptr[1] + ptr[2];
		fy = x*ptr[3] + y*ptr[4] + ptr[5];
		fw = x*ptr[6] + y*ptr[7] + ptr[8];
		fx /=fw;
		fy/=fw;
		int wx = (int)(fx+0.5);
		int wy = (int)(fy+0.5);
		
		float* fptr = GetPointerFromBigMatrix(fModel,width,height,0,x,y,10);
		float* pfCurrLearningRate = fptr;
		float* pfCurrDistThresholdFactor = fptr +1;
		float* pfCurrVariationFactor = fptr +2;
		float* pfCurrMeanLastDist =fptr + 3;
		float* pfCurrMeanMinDist_LT =fptr +4;
		float* pfCurrMeanMinDist_ST =fptr +5;
		float* pfCurrMeanRawSegmRes_LT = fptr + 6;
		float* pfCurrMeanRawSegmRes_ST = fptr + 7;
		float* pfCurrMeanFinalSegmRes_LT =fptr + 8;
		float* pfCurrMeanFinalSegmRes_ST =fptr + 9;
		uchar& pbUnstableRegionMask = GetRefFromBigMatrix(bModel,width,height,0,x,y,2);
		ushort4* anLastIntraDesc = descModel.data + BMSIZE*width*height + y*width + x;//desc model = BMSIZE desc model + lastdesc
		uchar4* anLastColor =colorModel.data + BMSIZE*width*height + y*width + x;//desc model = BMSIZE desc model + lastdesc//color model=BMSIZE bgmodel +  lastcolor
		
		fptr = GetPointerFromBigMatrix(wfModel,width,height,0,wx,wy,10);
		float* wpfCurrLearningRate = fptr;
		float* wpfCurrDistThresholdFactor = fptr +1;
		float* wpfCurrVariationFactor = fptr +2;
		float* wpfCurrMeanLastDist =fptr + 3;
		float* wpfCurrMeanMinDist_LT =fptr +4;
		float* wpfCurrMeanMinDist_ST =fptr +5;
		float* wpfCurrMeanRawSegmRes_LT = fptr + 6;
		float* wpfCurrMeanRawSegmRes_ST = fptr + 7;
		float* wpfCurrMeanFinalSegmRes_LT =fptr + 8;
		float* wpfCurrMeanFinalSegmRes_ST =fptr + 9;
		uchar& wpbUnstableRegionMask = GetRefFromBigMatrix(wbModel,width,height,0,wx,wy,2);
		ushort4* wanLastIntraDesc = wdescModel.data + BMSIZE*width*height + wy*width + wx;//desc model = BMSIZE desc model + lastdesc
		uchar4* wanLastColor =wcolorModel.data + BMSIZE*width*height+ wy*width +wx;//desc model = BMSIZE desc model + lastdesc//color model=BMSIZE bgmodel +  lastcolor
		

		

		ushort4* wBGIntraDescPtr = GetPointerFromBigMatrix(wdescModel,width,height,0,wx,wy);
		uchar4* wBGColorPtr = GetPointerFromBigMatrix(wcolorModel,width,height,0,wx,wy);
		ushort4*  BGIntraDescPtr = GetPointerFromBigMatrix(descModel,width,height,0,x,y);
		uchar4*  BGColorPtr= GetPointerFromBigMatrix(colorModel,width,height,0,x,y);
		

		
		unsigned idx = (threadIdx.y+R)*BLOCK_W + threadIdx.x+R;
		const uchar4 CurrColor = scolor[idx];
		//const uchar4 CurrColor = img(y,x);
		uchar anCurrColor[3] = {CurrColor.x,CurrColor.y,CurrColor.z};
		ushort4 CurrInterDesc, CurrIntraDesc;
		//const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[CurrColor.x],m_anLBSPThreshold_8bitLUT[CurrColor.y],m_anLBSPThreshold_8bitLUT[CurrColor.z]};
		const size_t anCurrIntraLBSPThresholds[3] = {LBSPThres[CurrColor.x],LBSPThres[CurrColor.y],LBSPThres[CurrColor.z]};
		//LBSP(img,CurrColor,x,y,anCurrIntraLBSPThresholds,CurrIntraDesc);
		LBSP(scolor,CurrColor,threadIdx.x,threadIdx.y,BLOCK_W,anCurrIntraLBSPThresholds,CurrIntraDesc);

		outMask[y*width+x] = 0;
		//std::cout<<x<<","<<y<<std::endl;
		if (wx<2 || wx>= width-2 || wy<2 || wy>=height-2)
		{					
			//m_features.data[oidx_uchar] = 0xff;
			//m_nOutPixels ++;
			fgMask(y,x) = 0;
			outMask[y*width+x] = 0xff;
			size_t s_rand = getRandom(randStates,idx_uchar)%BMSIZE;
			while(s_rand<BMSIZE){
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
				s_rand++;
			}
			return;
		}
		else
		{
			//反变换
			ptr += 9;
			fx = x*ptr[0] + y*ptr[1] + ptr[2];
			fy = x*ptr[3] + y*ptr[4] + ptr[5];
			fw = x*ptr[6] + y*ptr[7] + ptr[8];
			fx /=fw;
			fy/=fw;
			//std::cout<<x<<","<<y<<std::endl;
			if (fx<2 || fx>= width-2 || fy<2 || fy>=height-2)
			{
				outMask[y*width+x] = 0xff;
			}
		}
		*pfCurrDistThresholdFactor =  *wpfCurrDistThresholdFactor;
		*pfCurrVariationFactor = *wpfCurrVariationFactor;
		*pfCurrLearningRate = *wpfCurrLearningRate;
		*pfCurrMeanLastDist = *wpfCurrMeanLastDist;
		*pfCurrMeanMinDist_LT = *wpfCurrMeanMinDist_LT;
		*pfCurrMeanMinDist_ST = *wpfCurrMeanMinDist_ST;
		*pfCurrMeanRawSegmRes_LT = *wpfCurrMeanRawSegmRes_LT; 
		*pfCurrMeanRawSegmRes_ST = *wpfCurrMeanRawSegmRes_ST;
		*pfCurrMeanFinalSegmRes_LT = *wpfCurrMeanFinalSegmRes_LT;
		*pfCurrMeanFinalSegmRes_ST = *wpfCurrMeanFinalSegmRes_ST;
		pbUnstableRegionMask = wpbUnstableRegionMask;


		for(int i=0; i<BMSIZE; i++)
		{
			BGIntraDescPtr[i] = wBGIntraDescPtr[i];
			BGColorPtr[i] = wBGColorPtr[i];
		}
		/**anLastColor = *wanLastColor;
		*anLastIntraDesc = *wanLastIntraDesc;*/
		
		const float fRollAvgFactor_LT = 1.0f/min(frameIndex,25*4);
		const float fRollAvgFactor_ST = 1.0f/min(frameIndex,25);
		
		
		size_t nMinTotDescDist=48;
		size_t nMinTotSumDist=765;
		const size_t s_nColorMaxDataRange_3ch = 255*3;
		const size_t s_nDescMaxDataRange_3ch = 16*3;
		
		const size_t nCurrColorDistThreshold = (size_t)(((*wpfCurrDistThresholdFactor)*30)-((!wpbUnstableRegionMask)*6));
		size_t m_nDescDistThreshold = 3;
		const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*wpfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(wpbUnstableRegionMask*m_nDescDistThreshold);
		const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
		const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
		const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;

		ushort anCurrIntraDesc[3] = {CurrIntraDesc.x ,CurrIntraDesc.y, CurrIntraDesc.z};
		pbUnstableRegionMask = ((*wpfCurrDistThresholdFactor)>3.0 || (*wpfCurrMeanRawSegmRes_LT-*wpfCurrMeanFinalSegmRes_LT)>0.1 || (*wpfCurrMeanRawSegmRes_ST-*wpfCurrMeanFinalSegmRes_ST)>0.1)?1:0;
		size_t nGoodSamplesCount=0, nSampleIdx=0;

		if (mask(y,x) != 0xff)
		{
			while(nGoodSamplesCount<2 && nSampleIdx<BMSIZE) {
			const ushort4 const BGIntraDesc = BGIntraDescPtr[nSampleIdx];
			const uchar4 const BGColor = BGColorPtr[nSampleIdx];
			
			uchar anBGColor[3] = {BGColor.x,BGColor.y,BGColor.z};
			ushort anBGIntraDesc[3] = {BGIntraDesc.x,BGIntraDesc.y,BGIntraDesc.z};
			//const size_t anCurrInterLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[BGColor.x],m_anLBSPThreshold_8bitLUT[BGColor.y],m_anLBSPThreshold_8bitLUT[BGColor.z]};
			const size_t anCurrInterLBSPThresholds[3] = {LBSPThres[BGColor.x],LBSPThres[BGColor.y],LBSPThres[BGColor.z]};
			//const size_t anCurrInterLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[0],m_anLBSPThreshold_8bitLUT[0],m_anLBSPThreshold_8bitLUT[0]};
			
			LBSP(scolor,BGColor,threadIdx.x,threadIdx.y,BLOCK_W,anCurrInterLBSPThresholds,CurrInterDesc);
			ushort anCurrInterDesc[3] ={CurrInterDesc.x,CurrInterDesc.y, CurrInterDesc.z};
			
			size_t nTotDescDist = 0;
			size_t nTotSumDist = 0;
			for(size_t c=0;c<3; ++c) {
				const size_t nColorDist = abs(anCurrColor[c]-anBGColor[c]);
				
				if(nColorDist>nCurrSCColorDistThreshold)
					goto failedcheck3ch;
				size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
				size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
				const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
				const size_t nSumDist = min((int)((nDescDist/2)*15+nColorDist),255);
				if(nSumDist>nCurrSCColorDistThreshold)
					goto failedcheck3ch;
				nTotDescDist += nDescDist;
				nTotSumDist += nSumDist;
				//nTotSumDist += nColorDist;
			}
			if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
				goto failedcheck3ch;

			if(nMinTotDescDist>nTotDescDist)
				nMinTotDescDist = nTotDescDist;
			if(nMinTotSumDist>nTotSumDist)
				nMinTotSumDist = nTotSumDist;
			nGoodSamplesCount++;
failedcheck3ch:
			nSampleIdx++;
		}
		//const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
		const float fNormalizedLastDist = ((float)L1dist_uchar(*anLastColor,CurrColor)/765 +(float)hdist_ushort_8bitLUT(*anLastIntraDesc,CurrIntraDesc)/48)/2;		
		*pfCurrMeanLastDist = (*wpfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
		if(nGoodSamplesCount<2) {
			// == foreground
			const float fNormalizedMinDist = min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(2-nGoodSamplesCount)/2);
			//const float fNormalizedMinDist = min(1.0f,((float)nMinTotSumDist/765) + (float)(2-nGoodSamplesCount)/2);
			*pfCurrMeanMinDist_LT = (*wpfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			*pfCurrMeanMinDist_ST = (*wpfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			*pfCurrMeanRawSegmRes_LT = (*wpfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
			*pfCurrMeanRawSegmRes_ST = (*wpfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
			fgMask(y,x) = UCHAR_MAX;
			if((getRandom(randStates,idx_uchar)%(size_t)2)==0) {
				const size_t s_rand = getRandom(randStates,idx_uchar)%BMSIZE;
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
			}
		}
		else {
			// == background
			fgMask(y,x) = 0;
			const float fNormalizedMinDist = ((float)nMinTotSumDist/765+(float)nMinTotDescDist/48)/2;
			//const float fNormalizedMinDist = ((float)nMinTotSumDist/765);
			*pfCurrMeanMinDist_LT = (*wpfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			*pfCurrMeanMinDist_ST = (*wpfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			*pfCurrMeanRawSegmRes_LT = (*wpfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
			*pfCurrMeanRawSegmRes_ST = (*wpfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
			const size_t nLearningRate =(size_t)ceil(*wpfCurrLearningRate);
			if(getRandom(randStates,idx_uchar)%nLearningRate==0) {
				const size_t s_rand =getRandom(randStates,idx_uchar)%BMSIZE;
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
			}
			int x_rand,y_rand;
			const bool bCurrUsing3x3Spread = !pbUnstableRegionMask;
			if(bCurrUsing3x3Spread)
			{
				getRandNeighborPosition(randStates,x_rand,y_rand,wx,wy,2,img.cols,img.rows);

				const size_t n_rand = getRandom(randStates,idx_uchar);
				const float fRandMeanLastDist = GetValueFromBigMatrix(wfModel,width,height,3,x_rand,y_rand,10);
				const float fRandMeanRawSegmRes = GetValueFromBigMatrix(wfModel,width,height,8,x_rand,y_rand,10);
				const size_t s_rand =getRandom(randStates,idx_uchar)%BMSIZE;
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>0.995 && fRandMeanLastDist<0.01 && (n_rand%((size_t)fCurrLearningRateLowerCap))==0)) {
						SetValueToBigMatrix(colorModel,width,height,s_rand,x_rand,y_rand,CurrColor);
						SetValueToBigMatrix(descModel,width,height,s_rand,x_rand,y_rand,CurrIntraDesc);
				}
			}
		}

		}
		else
		{
			fgMask(y,x) = 0;
			//const float fNormalizedMinDist = ((float)nMinTotSumDist/765+(float)nMinTotDescDist/48)/2;
			////const float fNormalizedMinDist = ((float)nMinTotSumDist/765);
			//*pfCurrMeanMinDist_LT = (*wpfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			//*pfCurrMeanMinDist_ST = (*wpfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			//*pfCurrMeanRawSegmRes_LT = (*wpfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
			//*pfCurrMeanRawSegmRes_ST = (*wpfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
			const size_t nLearningRate =(size_t)ceil(*wpfCurrLearningRate);
			if(getRandom(randStates,idx_uchar)%nLearningRate==0) {
				const size_t s_rand =getRandom(randStates,idx_uchar)%BMSIZE;
				BGIntraDescPtr[s_rand] = CurrIntraDesc;
				BGColorPtr[s_rand] = CurrColor;
			}
			int x_rand,y_rand;
			const bool bCurrUsing3x3Spread = !pbUnstableRegionMask;
			if(bCurrUsing3x3Spread)
			{
				getRandNeighborPosition(randStates,x_rand,y_rand,wx,wy,2,img.cols,img.rows);

				const size_t n_rand = getRandom(randStates,idx_uchar);
				const float fRandMeanLastDist = GetValueFromBigMatrix(wfModel,width,height,3,x_rand,y_rand,10);
				const float fRandMeanRawSegmRes = GetValueFromBigMatrix(wfModel,width,height,8,x_rand,y_rand,10);
				const size_t s_rand =getRandom(randStates,idx_uchar)%BMSIZE;
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes>0.995 && fRandMeanLastDist<0.01 && (n_rand%((size_t)fCurrLearningRateLowerCap))==0)) {
						SetValueToBigMatrix(colorModel,width,height,s_rand,x_rand,y_rand,CurrColor);
						SetValueToBigMatrix(descModel,width,height,s_rand,x_rand,y_rand,CurrIntraDesc);
				}
			}
		}
		
		float UNSTABLE_REG_RATIO_MIN = 0.1;
		float FEEDBACK_T_INCR = 0.5;
		float FEEDBACK_T_DECR = 0.1;
		float FEEDBACK_V_INCR(1.f);
		float FEEDBACK_V_DECR(0.1f);
		float FEEDBACK_R_VAR(0.01f);
		if(lastFgMask(wy,wx) || (min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && fgMask(y,x))) {
			if((*pfCurrLearningRate)<fCurrLearningRateUpperCap)
				*pfCurrLearningRate += FEEDBACK_T_INCR/(max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
		}
		else if((*pfCurrLearningRate)>fCurrLearningRateLowerCap)
			*pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
		if((*pfCurrLearningRate)< fCurrLearningRateLowerCap)
			*pfCurrLearningRate = fCurrLearningRateLowerCap;
		else if((*pfCurrLearningRate)>fCurrLearningRateUpperCap)
			*pfCurrLearningRate = fCurrLearningRateUpperCap;
		if(max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && GetValueFromBigMatrix(wbModel,width,height,1,wx,wy))
			(*pfCurrVariationFactor) += FEEDBACK_V_INCR;
		else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
			(*pfCurrVariationFactor) -= pbUnstableRegionMask?FEEDBACK_V_DECR/4:pbUnstableRegionMask?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
			if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
				(*pfCurrVariationFactor) = FEEDBACK_V_DECR;
		}
		if((*pfCurrDistThresholdFactor)<pow(1.0f+min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
			(*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
		else {
			(*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
			if((*pfCurrDistThresholdFactor)<1.0f)
				(*pfCurrDistThresholdFactor) = 1.0f;
		}
		/*if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>=4)
		++nNonZeroDescCount;*/
		*anLastColor = tex1Dfetch(ImageTexture,idx_uchar);
		*anLastIntraDesc = CurrIntraDesc;

	}
	else if(x<width && y<height)
	{
		fgMask(y,x) = 0;
		outMask[y*width+x] =0;
	}
}


__global__ void CudaRefreshModelKernel(curandState* devStates,float refreshRate, int width ,int height,PtrStep<uchar> mask, PtrStep<uchar4> colorModels,PtrStep<ushort4> descModels, int modelSize,
	cv::gpu::PtrStep<float> fModel, cv::gpu::PtrStep<uchar> bModel)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x < width-2 && x>=2 && y>=2 && y <height-2 && mask(y,x) == 0xff)
	{
		int ind = x +y*width;
		curandState localState = devStates[ind];
		size_t RANDOM = curand( &localState );
		devStates[ind] = localState; 
		const size_t nBGSamplesToRefresh = refreshRate<1.0f?(size_t)(refreshRate*modelSize):modelSize;
		const size_t nRefreshStartPos = refreshRate<1.0f?RANDOM%modelSize:0;
		size_t offset = width*height*modelSize;
		uchar4* colorPtr = colorModels.data + offset;
		ushort4* descPtr= descModels.data + offset;
		for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {

			int y_sample, x_sample;
			getRandSamplePosition(devStates,x_sample,y_sample,x,y,2,width,height);
			//getRandNeighborPosition(devStates,x_sample,y_sample,x,y,2,width,height);
			int idx =  y_sample*width+ x_sample;
			uchar4 value =  colorPtr[idx];
			ushort4 svalue = descPtr[idx];
			int pos = s%modelSize;
			SetValueToBigMatrix(colorModels,width,height,pos,x,y,value);
			SetValueToBigMatrix(descModels,width,height,pos,x,y,svalue);
		}
		const int fSize =10;
		const int bSize = 2;
		float* fptr = GetPointerFromBigMatrix(fModel,width,height,0,x,y,fSize);
		uchar* bptr = GetPointerFromBigMatrix(bModel,width,height,0,x,y,bSize);
		fptr[0] = 2.f;
		fptr[1] = 1.f;
		fptr[2] = 10.f;
		for( int i=3; i<fSize; i++)
		{
			fptr[i] = 0;
		}
		for(int i=0; i<bSize; i++)
		{
			bptr[i] = 0;
		}
	}

}

__global__ void CudaRefreshModelKernel(curandState* randStates,float refreshRate, int width ,int height,PtrStep<uchar4> colorModels,PtrStep<ushort4> descModels, int modelSize,
	cv::gpu::PtrStep<float> fModel, cv::gpu::PtrStep<uchar> bModel)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x < width-2 && x>=2 && y>=2 && y <height-2)
	{
		size_t idx_uchar = width*y + x;
		const size_t nBGSamplesToRefresh = refreshRate<1.0f?(size_t)(refreshRate*modelSize):modelSize;
		const size_t nRefreshStartPos = refreshRate<1.0f?getRandom(randStates,idx_uchar)%modelSize:0;
		size_t offset = width*height*modelSize;
		uchar4* colorPtr = colorModels.data + offset;
		ushort4* descPtr= descModels.data + offset;
		for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {

			int y_sample, x_sample;
			getRandSamplePosition(randStates,x_sample,y_sample,x,y,2,width,height);
			int idx =  y_sample*width+ x_sample;
			uchar4 value =  colorPtr[idx];
			ushort4 svalue = descPtr[idx];
			int pos = s%modelSize;
			SetValueToBigMatrix(colorModels,width,height,pos,x,y,value);
			SetValueToBigMatrix(descModels,width,height,pos,x,y,svalue);
		}
		const int fSize =10;
		const int bSize = 2;
		float* fptr = GetPointerFromBigMatrix(fModel,width,height,0,x,y,fSize);
		uchar* bptr = GetPointerFromBigMatrix(bModel,width,height,0,x,y,bSize);
		fptr[0] = 2.f;
		fptr[1] = 1.f;
		fptr[2] = 10.f;
		for( int i=3; i<fSize; i++)
		{
			fptr[i] = 0;
		}
		for(int i=0; i<bSize; i++)
		{
			bptr[i] = 0;
		}
	}

}

__global__ void SCudaRefreshModelKernel(float refreshRate,const PtrStepSz<uchar4> lastImg,const PtrStepSz<ushort4> lastDescImg,PtrStep<uchar4>* colorModels,PtrStep<ushort4>* descModels, int modelSize)
{
	__shared__ uchar4 scolor[BLOCK_W*BLOCK_H];
	__shared__ ushort4 sdesc[BLOCK_W*BLOCK_H];
	int width = lastImg.cols;
	int height = lastImg.rows;
	// First batch loading
	int dest = threadIdx.y * TILE_W + threadIdx.x,
		destY = dest / BLOCK_W, destX = dest % BLOCK_W,
		srcY = blockIdx.y * TILE_W + destY - R,
		srcX = blockIdx.x * TILE_W + destX - R,
		src = (srcY * width + srcX);
	srcX = max(0,srcX);
	srcX = min(srcX,width-1);
	srcY = max(srcY,0);
	srcY = min(srcY,height-1);
	scolor[dest] = lastImg(srcY,srcX);
	sdesc[dest] = lastDescImg(srcY,srcX);
	//second batch loading
	dest = threadIdx.y * TILE_W + threadIdx.x + TILE_W * TILE_W;
	destY = dest / BLOCK_W, destX = dest % BLOCK_W;
	srcY = blockIdx.y * TILE_W + destY - R;
	srcX = blockIdx.x * TILE_W + destX - R;


	if (destY < BLOCK_W)
	{
		srcX = max(0,srcX);	 
		srcX = min(srcX,width-1);
		srcY = max(srcY,0);
		srcY = min(srcY,height-1);
		scolor[destX + destY * BLOCK_W] = lastImg(srcY,srcX);
		sdesc[destX + destY * BLOCK_W] = lastDescImg(srcY,srcX);
	}

	__syncthreads();

	int y = blockIdx.y * TILE_W + threadIdx.y;
	int  x = blockIdx.x * TILE_W + threadIdx.x;
	if(x < lastImg.cols && y < lastImg.rows)
	{

		curandState state;
		curand_init(threadIdx.y,0,0,&state);

		const size_t nBGSamplesToRefresh = refreshRate<1.0f?(size_t)(refreshRate*modelSize):modelSize;
		const size_t nRefreshStartPos = refreshRate<1.0f?curand(&state)%modelSize:0;
		for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {

			int y_sample, x_sample;
			getRandSamplePosition(&state,x_sample,y_sample,x,y,2,width,height);
			y_sample -= y;
			x_sample -= x;
			//unsigned idx = bindex + x_sample + (y_sample*blockDim.x);
			unsigned idx = (threadIdx.y+R+y_sample)*BLOCK_W + threadIdx.x+R+x_sample;
			colorModels[s%modelSize](y,x) = scolor[idx];
			descModels[s%modelSize](y,x) = sdesc[idx];
		}
	}
	//__syncthreads();
}


__global__ void DownloadKernel(int width, int height, int id, const PtrStep<ushort4> models, int modelSize, PtrStep<ushort4> model)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x < width-2 && x>=2 && y>=2 && y <height-2)
	{
		
		model(y,x) = GetValueFromBigMatrix(models,width,height,id,x,y,modelSize);
		
	}
}
__global__ void DownloadColorKernel(int width, int height, int id, const PtrStep<uchar4> models, int modelSize, PtrStep<uchar4> model)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x < width-2 && x>=2 && y>=2 && y <height-2)
	{
		model(y,x) = GetValueFromBigMatrix(models,width,height,id,x,y,modelSize);
	}
}
//基于共享内存
__global__ void SCudaUpdateModelKernel(curandState* devStates,const PtrStepSz<uchar4> img ,int width, int height,PtrStep<uchar> fgmask,PtrStep<uchar4> colorModel,PtrStep<ushort4> descModel)
{
	__shared__ uchar4 scolor[BLOCK_W*BLOCK_H];
	__shared__ ushort4 sdesc[TILE_W*TILE_W];
	// First batch loading
	int dest = threadIdx.y * TILE_W + threadIdx.x,
		destY = dest / BLOCK_W, destX = dest % BLOCK_W,
		srcY = blockIdx.y * TILE_W + destY - R,
		srcX = blockIdx.x * TILE_W + destX - R,
		src = (srcY * width + srcX);
	srcX = max(0,srcX);
	srcX = min(srcX,width-1);
	srcY = max(srcY,0);
	srcY = min(srcY,height-1);
	//scolor[dest] = img(srcY,srcX);
	scolor[dest] = tex1Dfetch(ImageTexture,srcY*width+srcX);

	//second batch loading
	dest = threadIdx.y * TILE_W + threadIdx.x + TILE_W * TILE_W;
	destY = dest / BLOCK_W, destX = dest % BLOCK_W;
	srcY = blockIdx.y * TILE_W + destY - R;
	srcX = blockIdx.x * TILE_W + destX - R;


	if (destY < BLOCK_W)
	{
		srcX = max(0,srcX);	 
		srcX = min(srcX,width-1);
		srcY = max(srcY,0);
		srcY = min(srcY,height-1);
		//scolor[destX + destY * BLOCK_W] = img(srcY,srcX);
		scolor[dest] = tex1Dfetch(ImageTexture,srcY*width+srcX);
	}
	
	__syncthreads();
	unsigned shidx = (threadIdx.y+R)*BLOCK_W + threadIdx.x+R;
	uchar4 value = scolor[shidx];
	const size_t Thresholds[3] = {LBSPThres[value.x],LBSPThres[value.y],LBSPThres[value.z]};
	ushort4 svalue;
	LBSP(scolor,value,threadIdx.x,threadIdx.y,BLOCK_W,Thresholds,svalue);
	int idx = threadIdx.x + threadIdx.y*TILE_W;
	sdesc[idx] = svalue;
	__syncthreads();
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x <width-2 && x>=2 && y>=2 && y < height)
	{
		int idx = x+y*width;
		unsigned shidx = (threadIdx.y+R)*BLOCK_W + threadIdx.x+R;
		if (fgmask(y,x) == 0xff)
		{
			if (getRandom(devStates,idx) % 2 == 0)
			{
				const size_t s_rand = getRandom(devStates,idx)%BMSIZE;
				//求idx处的颜色和desc
		
				//uchar4 value = tex1Dfetch(ImageTexture,idx);
				uchar4 value = scolor[shidx];
				ushort4 svalue;
				const size_t Thresholds[3] = {LBSPThres[value.x],LBSPThres[value.y],LBSPThres[value.z]};
				//LBSP(value,x,y,width,Thresholds,svalue);
				LBSP(scolor,value,threadIdx.x,threadIdx.y,BLOCK_W,Thresholds,svalue);
				/*uchar4 value =  colorPtr[idx];
				ushort4 svalue = descPtr[idx];*/
				
				SetValueToBigMatrix(colorModel,width,height,s_rand,x,y,value);
				SetValueToBigMatrix(descModel,width,height,s_rand,x,y,svalue);
			}
		}
		else
		{
			
			float fSamplesRefreshFrac = 0.1;
			const size_t nBGSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*BMSIZE):BMSIZE;
			const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?getRandom(devStates,idx) %BMSIZE:0;
			const size_t nLearningRate = 1;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				//getRandNeighborPosition(devStates,x_sample,y_sample,x,y,2,width,height);
				getRandSamplePosition(devStates,x_sample,y_sample,x,y,2,width,height);
				int sidx =  y_sample*width+ x_sample;
				int dx = x_sample-x;
				int dy = y_sample - y;
				int shidx =  (threadIdx.y+dy + R)*BLOCK_W + threadIdx.x + dx+R;
				//求sidx处的颜色和desc
				uchar4 value = tex1Dfetch(ImageTexture,sidx);
				//uchar4 value = scolor[shidx];
				int tidx = threadIdx.x+dx+(threadIdx.y+dy)*TILE_W;
				if (tidx >0 && tidx < TILE_W*TILE_W)
					ushort4 svalue = sdesc[tidx];
				else
				{
					const size_t Thresholds[3] = {LBSPThres[value.x],LBSPThres[value.y],LBSPThres[value.z]};
					LBSP(value,x_sample,y_sample,width,Thresholds,svalue);					
				}
				int pos = s%BMSIZE;
				
				SetValueToBigMatrix(colorModel,width,height,pos,x,y,value);
				SetValueToBigMatrix(descModel,width,height,pos,x,y,svalue);
				

			}
		}
	}
}
//update the whole  model with current image, 
__global__ void CudaUpdateModelKernel(curandState* devStates,const PtrStepSz<uchar4> img ,int width, int height, PtrStep<uchar4> colorModel,PtrStep<ushort4> descModel)
{
	__shared__ ushort s_randomPattern[7][7];
	if (threadIdx.x < 7 && threadIdx.y < 7)
		s_randomPattern[threadIdx.y][threadIdx.x] = c_anSamplesInitPattern[threadIdx.y][threadIdx.x];
	__syncthreads();
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	const int offset = BMSIZE*width*height;
	if(x <width-2 && x>=2 && y>=2 && y < height)
	{
		int idx = x+y*width;


		float fSamplesRefreshFrac = 1.0;
		const size_t nBGSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*BMSIZE):BMSIZE;
		const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?getRandom(devStates,idx) %BMSIZE:0;
		const size_t nLearningRate = 1;
		for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
			int y_sample, x_sample;
			//getRandNeighborPosition(devStates,x_sample,y_sample,x,y,2,width,height);
			getRandSamplePosition(devStates,s_randomPattern,x_sample,y_sample,x,y,2,width,height);
			int sidx =  y_sample*width+ x_sample;
			/*uchar4 value = tex1Dfetch(LastColorTexture,sidx);
			ushort4 svalue = tex1Dfetch(LastDescTexture,sidx);
			uchar4* ptr = colorModel.data + offset;
			ushort4* descPtr = descModel.data + offset;
			uchar4 value = ptr[sidx];
			ushort4 svalue = descPtr[sidx];*/
			uchar4 value = tex1Dfetch(ImageTexture,sidx);
			const size_t Thresholds[3] = {LBSPThres[value.x],LBSPThres[value.y],LBSPThres[value.z]};
			ushort4 svalue;
			LBSP(value,x_sample,y_sample,width,Thresholds,svalue);		

			int pos = s%BMSIZE;

			SetValueToBigMatrix(colorModel,width,height,pos,x,y,value);
			SetValueToBigMatrix(descModel,width,height,pos,x,y,svalue);
				

			
		}
	}
}
//update with last frame
__global__ void CudaUpdateModelKernel(curandState* devStates,int width, int height,PtrStep<uchar> fgmask,PtrStep<uchar4> colorModel,PtrStep<ushort4> descModel)
{
	__shared__ ushort s_randomPattern[7][7];
	if (threadIdx.x < 7 && threadIdx.y < 7)
		s_randomPattern[threadIdx.y][threadIdx.x] = c_anSamplesInitPattern[threadIdx.y][threadIdx.x];
	__syncthreads();
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	const int offset = BMSIZE*width*height;
	if(x <width-2 && x>=2 && y>=2 && y < height)
	{
		int idx = x+y*width;
	
		if (fgmask(y,x) == 0xff)
		{
			if (getRandom(devStates,idx) % 2 == 0)
			{
				const size_t s_rand = getRandom(devStates,idx)%BMSIZE;
				//求idx处的颜色和desc
				/*uchar4 value = tex1Dfetch(LastColorTexture,idx);
				ushort4 svalue = tex1Dfetch(LastDescTexture,idx);*/
				uchar4* ptr = colorModel.data + offset;
				ushort4* descPtr = descModel.data + offset;
				uchar4 value = ptr[idx];
				ushort4 svalue = descPtr[idx];
				/*uchar4 value = tex1Dfetch(ImageTexture,idx);
				
				const size_t Thresholds[3] = {LBSPThres[value.x],LBSPThres[value.y],LBSPThres[value.z]};
				ushort4 svalue;
				LBSP(value,x,y,width,Thresholds,svalue);		*/	
				SetValueToBigMatrix(colorModel,width,height,s_rand,x,y,value);
				SetValueToBigMatrix(descModel,width,height,s_rand,x,y,svalue);
			}
		}
		else
		{
			
			float fSamplesRefreshFrac = 0.1;
			const size_t nBGSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*BMSIZE):BMSIZE;
			const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?getRandom(devStates,idx) %BMSIZE:0;
			const size_t nLearningRate = 1;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				//getRandNeighborPosition(devStates,x_sample,y_sample,x,y,2,width,height);
				getRandSamplePosition(devStates,s_randomPattern,x_sample,y_sample,x,y,2,width,height);
				int sidx =  y_sample*width+ x_sample;
				/*uchar4 value = tex1Dfetch(LastColorTexture,sidx);
				ushort4 svalue = tex1Dfetch(LastDescTexture,sidx);*/
				uchar4* ptr = colorModel.data + offset;
				ushort4* descPtr = descModel.data + offset;
				uchar4 value = ptr[sidx];
				ushort4 svalue = descPtr[sidx];
				/*uchar4 value = tex1Dfetch(ImageTexture,sidx);
				const size_t Thresholds[3] = {LBSPThres[value.x],LBSPThres[value.y],LBSPThres[value.z]};
				ushort4 svalue;
				LBSP(value,x_sample,y_sample,width,Thresholds,svalue);*/		
			
				int pos = s%BMSIZE;
				
				SetValueToBigMatrix(colorModel,width,height,pos,x,y,value);
				SetValueToBigMatrix(descModel,width,height,pos,x,y,svalue);
				

			}
		}
	}
}

void DownloadColorModel(int width,int height, cv::gpu::GpuMat& models, int size, int id, cv::gpu::GpuMat& model)
{
	dim3 block(16,16);
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	DownloadColorKernel<<<grid,block>>>(width,height,id,models,size,model);
}
void DownloadModel(int width,int height,cv::gpu::GpuMat& models, int size, int id, cv::gpu::GpuMat& model)
{
	dim3 block(16,16);
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	DownloadKernel<<<grid,block>>>(width,height,id,models,size,model);
}
void CudaBSOperator(const cv::gpu::GpuMat& img,curandState* randStates, double* homography, int frameIdx, 
PtrStep<uchar4> colorModel,PtrStep<uchar4> wcolorModel,
PtrStep<ushort4> descModel,PtrStep<ushort4> wdescModel,
PtrStep<uchar> bModel,PtrStep<uchar> wbModel,
PtrStep<float> fModel,PtrStep<float> wfModel,
PtrStep<uchar> fgMask, PtrStep<uchar> lastFgMask,	uchar* outMask, float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap, size_t* m_anLBSPThreshold_8bitLUT)
{
	dim3 block(16,16);
	dim3 grid((img.cols + block.x - 1)/block.x,(img.rows + block.y - 1)/block.y);
	cudaBindTexture( NULL, ImageTexture,
		img.ptr<uchar4>(),	sizeof(uchar4)*img.cols*img.rows );
	CudaBSOperatorKernel<<<grid,block>>>(img,randStates,homography,frameIdx,colorModel,
		wcolorModel,descModel, wdescModel,
		bModel,wbModel,fModel,wfModel,
		fgMask, lastFgMask, outMask,fCurrLearningRateLowerCap, fCurrLearningRateUpperCap,  m_anLBSPThreshold_8bitLUT);
}
void WarpCudaBSOperator(const cv::gpu::GpuMat& img, const cv::gpu::GpuMat& warpedImg,curandState* randStates, const cv::gpu::GpuMat& map, const cv::gpu::GpuMat& invMap, int frameIdx, 
PtrStep<uchar4> colorModel,PtrStep<uchar4> wcolorModel,
PtrStep<ushort4> descModel,PtrStep<ushort4> wdescModel,
PtrStep<uchar> bModel,PtrStep<uchar> wbModel,
PtrStep<float> fModel,PtrStep<float> wfModel,
PtrStep<uchar> fgMask,	PtrStep<uchar> lastFgMask, uchar* outMask,
float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap)
{
	dim3 block(16,16);
	dim3 grid((img.cols + block.x - 1)/block.x,(img.rows + block.y - 1)/block.y);
	int imgSize = img.cols*img.rows;
	cudaBindTexture( NULL, ImageTexture,
		img.ptr<uchar4>(),	imgSize*sizeof(uchar4) );
	cudaBindTexture(NULL,WarpedImageTexture,
		warpedImg.ptr<uchar4>(),imgSize*sizeof(uchar4));
	cudaBindTexture(NULL,FGMaskLastTexture,
		lastFgMask.data,imgSize);
	cudaBindTexture(NULL,ColorModelTexture,colorModel.data,imgSize*sizeof(uchar4)*50);

	cudaBindTexture(NULL,DescModelTexture,descModel.data,imgSize*sizeof(ushort4)*50);


	WarpCudaBSOperatorKernel<<<grid,block>>>(img,warpedImg,randStates,map,invMap,frameIdx,colorModel,
		wcolorModel,descModel, wdescModel,
		bModel,wbModel,fModel,wfModel,
		fgMask, lastFgMask, outMask,fCurrLearningRateLowerCap, fCurrLearningRateUpperCap);

	
}
void CudaBSOperator(const cv::gpu::GpuMat& img, const cv::gpu::GpuMat& mask, curandState* randStates, double* homography, int frameIdx, 
PtrStep<uchar4> colorModel,PtrStep<uchar4> wcolorModel,
PtrStep<ushort4> descModel,PtrStep<ushort4> wdescModel,
PtrStep<uchar> bModel,PtrStep<uchar> wbModel,
PtrStep<float> fModel,PtrStep<float> wfModel,
PtrStep<uchar> fgMask,	PtrStep<uchar> lastFgMask, uchar* outMask,
float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap)
{
	dim3 block(16,16);
	dim3 grid((img.cols + block.x - 1)/block.x,(img.rows + block.y - 1)/block.y);
	cudaBindTexture( NULL, ImageTexture,
		img.ptr<uchar4>(),	sizeof(uchar4)*img.cols*img.rows );
	CudaBSOperatorKernel<<<grid,block>>>(img,mask,randStates,homography,frameIdx,colorModel,
		wcolorModel,descModel, wdescModel,
		bModel,wbModel,fModel,wfModel,
		fgMask, lastFgMask, outMask,fCurrLearningRateLowerCap, fCurrLearningRateUpperCap);
}
void CudaRefreshModel(curandState* randStates,float refreshRate,int width, int height,cv::gpu::GpuMat& mask, cv::gpu::GpuMat& colorModels, cv::gpu::GpuMat& descModels, 
	GpuMat& fModel, GpuMat& bModel)
{
	dim3 block(16,16);
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	//colorModels还包含 downsample 和lastcolor
	//CudaRefreshModelKernel<<<grid,block>>>(refreshRate,lastImg,lastDescImg,ptr_colorModel,ptr_descModel,d_colorModels.size()-2);
	//colorModels还包含 downsample 和lastcolor
	CudaRefreshModelKernel<<<grid,block>>>(randStates,refreshRate,width,height,mask,colorModels,descModels,BMSIZE,fModel,bModel);
}
void CudaRefreshModel(curandState* randStates,float refreshRate,int width, int height, cv::gpu::GpuMat& colorModels, cv::gpu::GpuMat& descModels, 
	GpuMat& fModel, GpuMat& bModel)
{
	dim3 block(16,16);
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	//colorModels还包含 downsample 和lastcolor
	//CudaRefreshModelKernel<<<grid,block>>>(refreshRate,lastImg,lastDescImg,ptr_colorModel,ptr_descModel,d_colorModels.size()-2);
	//colorModels还包含 downsample 和lastcolor
	CudaRefreshModelKernel<<<grid,block>>>(randStates,refreshRate,width,height,colorModels,descModels,BMSIZE,fModel,bModel);


}
bool equalToFF(uchar a)
{
	return a==0xff;
}
int CountOutPixel(const uchar* outMask,size_t size)
{

	
	return thrust::count(outMask,outMask+size,0);
}
void CudaUpdateModel(curandState* devStates, int width, int height, PtrStep<uchar> fgmask,PtrStep<uchar4> colorModel,PtrStep<ushort4> descModel)
{
	dim3 block(16,16);
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	int size = width*height;
	int offset = BMSIZE*size;
	cudaBindTexture(NULL,LastColorTexture,colorModel.data+offset,sizeof(uchar4)*size);
	cudaBindTexture(NULL,LastDescTexture,descModel.data+offset,sizeof(ushort4)*size);
	CudaUpdateModelKernel<<<grid,block>>>(devStates,width,height,fgmask,colorModel,descModel);

}
void CudaUpdateModel(curandState* devStates,const cv::gpu::GpuMat& img ,int width, int height,PtrStep<uchar4> colorModel,PtrStep<ushort4> descModel)
{
	dim3 block(16,16);
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	int size = width*height;
	int offset = BMSIZE*size;
	
	CudaUpdateModelKernel<<<grid,block>>>(devStates,img ,width,height,colorModel,descModel);
}
void CudaBindImgTexture(const cv::gpu::GpuMat& img)
{
	cudaBindTexture( NULL, ImageTexture,
		img.ptr<uchar4>(),	sizeof(uchar4)*img.cols*img.rows );

	
}
void CudaBindWarpedImgTexture(const cv::gpu::GpuMat& img)
{
	cudaBindTexture( NULL, WarpedImageTexture,
		img.ptr<uchar4>(),	sizeof(uchar4)*img.cols*img.rows );
}
//__global__ void testRandomKernel(int n, int* d_in, int* d_out)
//{
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//	if (idx >= n/2)
//		return;
//	int x = d_in[idx*2];
//	int y = d_in[idx*2+1];
//	int x_sample,y_sample;
//	getRandSamplePosition(x_sample,y_sample,x,y,2,100,100);
//	d_out[idx*2] =x_sample;
//	d_out[idx*2+1] =y_sample;
//
//}
//
//void testRandom()
//{
//	const int n = 100;
//	int *d_in,*d_out;
//	cudaMalloc(&d_in,sizeof(int)*n);
//	cudaMalloc(&d_out, sizeof(int)*n);
//	
//	int h_in[n],h_out[n];
//	for(int i=0; i<n; i+=2)
//	{
//		h_in[i] = i;
//		h_in[i+1]=i+1;
//	}
//	cudaMemcpy(d_in,h_in,sizeof(int)*n,cudaMemcpyHostToDevice);
//	testRandomKernel<<<n/2+127/128,128>>>(n,d_in,d_out);
//	cudaMemcpy(h_out,d_out,sizeof(int)*n,cudaMemcpyDeviceToHost);
//	for(int i=0; i<n; i+=2)
//		std::cout<<h_out[i]<<","<<h_out[i+1]<<std::endl;
//
//	for(int i=0; i<n; i+=2)
//	{
//		getRandSamplePosition(h_out[i],h_out[i+1],h_in[i],h_in[i+1],2,cv::Size(100,100));
//	}
//	std::cout<<"----------------------\n";
//	for(int i=0; i<n; i+=2)
//		std::cout<<h_out[i]<<","<<h_out[i+1]<<std::endl;
//}
__global__ void CudaWarpKernel(int width, int height, int blkStep, double* homos, double*  invHomo,
	PtrStep<float> mapX, PtrStep<float> mapY, 
	PtrStep<float> imapX, PtrStep<float> imapY)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x<width && y<height)
	{
		int blkWidth = width / blkStep;
		int blkHeight = height/ blkStep;
		int idx = x/ blkWidth;
		int idy = y/ blkHeight;
		int bidx = idx + idy*blkStep;
		int hidx = bidx*blkStep;
		double* ptr = homos + hidx;
		double* iptr = invHomo + hidx;
		float wx = ptr[0]*x + ptr[1]*y + ptr[2];
		float wy = ptr[3]*x + ptr[4]*y + ptr[5];
		float w = ptr[6]*x + ptr[7]*y + 1;
		wx/=w;
		wy/=w;
		mapX(y,x) = wx;
		mapY(y,x) = wy;

		wx = iptr[0]*x + iptr[1]*y + iptr[2];
		wy = iptr[3]*x + iptr[4]*y + iptr[5];
		w = iptr[6]*x + iptr[7]*y + 1;
		wx /=w;
		wy/=w;
		imapX(y,x) = wx;
		imapY(y,x) = wy;


		 
	}
}
void CudaWarp(const cv::gpu::GpuMat& img, int blkWidth, double* homo, double*  invHomo,
	cv::gpu::GpuMat& mapX, cv::gpu::GpuMat& mapY, 
	cv::gpu::GpuMat& imapX, cv::gpu::GpuMat& imapY,
	cv::Mat& warped)
{
	dim3 block(16,16);
	int width = img.cols;
	int height = img.rows;
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	CudaWarpKernel<<<grid,block>>>(width,height,blkWidth,homo,invHomo,mapX,mapY,imapX,imapY);
	cv::gpu::GpuMat dwarp;
	cv::gpu::remap(img,dwarp,mapX,mapY,CV_INTER_CUBIC);
	dwarp.download(warped);
}

void CudaWarp(const cv::gpu::GpuMat& img, int blkWidth, double* homo, double*  invHomo,
	cv::gpu::GpuMat& mapX, cv::gpu::GpuMat& mapY, 
	cv::gpu::GpuMat& imapX, cv::gpu::GpuMat& imapY,
	cv::gpu::GpuMat& wimg)
{
	dim3 block(16,16);
	int width = img.cols;
	int height = img.rows;
	dim3 grid((width + block.x - 1)/block.x,(height + block.y - 1)/block.y);
	CudaWarpKernel<<<grid,block>>>(width,height,blkWidth,homo,invHomo,mapX,mapY,imapX,imapY);
	
	cv::gpu::remap(img,wimg,mapX,mapY,CV_INTER_CUBIC);
	
}