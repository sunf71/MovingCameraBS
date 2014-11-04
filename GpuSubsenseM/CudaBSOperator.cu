#include "CudaBSOperator.h"
#include <thrust\device_vector.h>
#include "RandUtils.h"
thrust::device_vector<PtrStep<uchar4>> d_colorModels;
thrust::device_vector<PtrStep<ushort4>> d_descModels;
thrust::device_vector<PtrStep<uchar>> d_bModels;
thrust::device_vector<PtrStep<float>> d_fModels;
PtrStep<uchar4>* ptr_colorModel;
PtrStep<ushort4>* ptr_descModel;
PtrStep<uchar>* ptr_bModel;
PtrStep<float>* ptr_fModel;
#define TILE_W 16
#define TILE_H 16
#define R 2
#define BLOCK_W (TILE_W+(2*R))
#define BLOCK_H (TILE_H + (2*R))
void ReleaseDeviceModels()
{
	d_colorModels.clear();
	d_descModels.clear();
	d_bModels.clear();
	d_fModels.clear();
}
void InitDeviceModels(std::vector<PtrStep<uchar4>>& colorModels, std::vector<PtrStep<ushort4>>& descModels,
	std::vector<PtrStep<uchar>>& bModels, std::vector<PtrStep<float>>& fModels)
{
	d_colorModels = colorModels;
	d_descModels = descModels;
	d_bModels = bModels;
	d_fModels = fModels;
	ptr_colorModel = thrust::raw_pointer_cast(&d_colorModels[0]);
	ptr_descModel = thrust::raw_pointer_cast(&d_descModels[0]);	
	ptr_bModel = thrust::raw_pointer_cast(&d_bModels[0]);	
	ptr_fModel = thrust::raw_pointer_cast(&d_fModels[0]);
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
	const uchar popcount_LUT8[256] = {
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
	return popcount_LUT8[(uchar)x] + popcount_LUT8[(uchar)(x>>8)];
}
__device__ size_t hdist_ushort_8bitLUT(const ushort& a, const ushort& b)
{

	return popcount_ushort_8bitsLUT(a^b);
}
__device__ size_t hdist_ushort_8bitLUT(const ushort4& a, const ushort4& b)
{

	return popcount_ushort_8bitsLUT(a.x^b.x)+popcount_ushort_8bitsLUT(a.y^b.y)+popcount_ushort_8bitsLUT(a.z^b.z);
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
__global__ void CudaBSOperatorKernel(const PtrStepSz<uchar4> img, int frameIndex,PtrStep<uchar>* bmodels,PtrStep<float>* fmodels,PtrStep<uchar4>* colorModels, PtrStep<ushort4>* descModels, PtrStep<uchar> fgMask,
	float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap, size_t* m_anLBSPThreshold_8bitLUT)
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
	scolor[dest] = img(srcY,srcX);

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
		scolor[destX + destY * BLOCK_W] = img(srcY,srcX);
	}

	__syncthreads();

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x < img.cols-2 && x>=2 && y>=2 && y < img.rows-2)
	{
		curandState state;
		curand_init(threadIdx.x,0,0,&state);

		const float fRollAvgFactor_LT = 1.0f/min(frameIndex,25*4);
		const float fRollAvgFactor_ST = 1.0f/min(frameIndex,25);
		unsigned idx = (threadIdx.y+R)*BLOCK_W + threadIdx.x+R;
		const uchar4 CurrColor = scolor[idx];
		uchar anCurrColor[3] = {CurrColor.x,CurrColor.y,CurrColor.z};
		size_t nMinTotDescDist=48;
		size_t nMinTotSumDist=765;
		float& pfCurrDistThresholdFactor = fmodels[1](y,x);
		float& pfCurrVariationFactor = fmodels[2](y,x);
		float& pfCurrLearningRate = fmodels[0](y,x);
		float& pfCurrMeanLastDist = fmodels[3](y,x);
		float& pfCurrMeanMinDist_LT = fmodels[4](y,x);
		float& pfCurrMeanMinDist_ST = fmodels[5](y,x);
		float& pfCurrMeanRawSegmRes_LT = fmodels[8](y,x);
		float& pfCurrMeanRawSegmRes_ST =fmodels[9](y,x);
		float& pfCurrMeanFinalSegmRes_LT = fmodels[10](y,x);
		float& pfCurrMeanFinalSegmRes_ST = fmodels[11](y,x);
		uchar& pbUnstableRegionMask = bmodels[0](y,x);
		ushort4& anLastIntraDesc = descModels[50](y,x);//desc model = 50 desc model + lastdesc
		uchar4& anLastColor = colorModels[51](y,x);//color model=50 bgmodel + downsample + lastcolor
		const size_t nCurrColorDistThreshold = (size_t)(((pfCurrDistThresholdFactor)*30)-((!pbUnstableRegionMask)*6));
		size_t m_nDescDistThreshold = 3;
		const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(pbUnstableRegionMask*m_nDescDistThreshold);
		const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
		const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
		const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;


		ushort4 CurrInterDesc, CurrIntraDesc;
		const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[CurrColor.x],m_anLBSPThreshold_8bitLUT[CurrColor.y],m_anLBSPThreshold_8bitLUT[CurrColor.z]};
		//LBSP(img,CurrColor,x,y,anCurrIntraLBSPThresholds,CurrIntraDesc);
		LBSP(scolor,CurrColor,threadIdx.x,threadIdx.y,BLOCK_W,anCurrIntraLBSPThresholds,CurrIntraDesc);
		ushort anCurrIntraDesc[3] = {CurrIntraDesc.x ,CurrIntraDesc.y, CurrIntraDesc.z};
		pbUnstableRegionMask = ((pfCurrDistThresholdFactor)>3.0 || (pfCurrMeanRawSegmRes_LT-pfCurrMeanFinalSegmRes_LT)>0.1 || (pfCurrMeanRawSegmRes_ST-pfCurrMeanFinalSegmRes_ST)>0.1)?1:0;
		size_t nGoodSamplesCount=0, nSampleIdx=0;

		while(nGoodSamplesCount<2 && nSampleIdx<50) {
			const ushort4 const BGIntraDesc = descModels[nSampleIdx](y,x);
			const uchar4 const BGColor = colorModels[nSampleIdx](y,x);
			uchar anBGColor[3] = {BGColor.x,BGColor.y,BGColor.z};
			ushort anBGIntraDesc[3] = {BGIntraDesc.x,BGIntraDesc.y,BGIntraDesc.z};
			const size_t anCurrInterLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[BGColor.x],m_anLBSPThreshold_8bitLUT[BGColor.y],m_anLBSPThreshold_8bitLUT[BGColor.z]};
			LBSP(scolor,BGColor,threadIdx.x,threadIdx.y,BLOCK_W,anCurrInterLBSPThresholds,CurrInterDesc);
			ushort anCurrInterDesc[3] ={CurrInterDesc.x,CurrInterDesc.y, CurrInterDesc.z};

			size_t nTotDescDist = 0;
			size_t nTotSumDist = 0;
			for(size_t c=0;c<3; ++c) {
				const size_t nColorDist = abs(anCurrColor[c]-anBGColor[c]);
				if(nColorDist>nCurrSCColorDistThreshold)
					goto failedcheck3ch;
				size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
				size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
				const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
				const size_t nSumDist = (nDescDist/2)*15+nColorDist;
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
		//const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,CurrColor)/765 +(float)hdist_ushort_8bitLUT(anLastIntraDesc,CurrIntraDesc)/48)/2;
		const float fNormalizedLastDist = (float)L1dist_uchar(anLastColor,CurrColor)/765;
		pfCurrMeanLastDist = (pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
		if(nGoodSamplesCount<2) {
			// == foreground
			//const float fNormalizedMinDist = std::min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
			const float fNormalizedMinDist = min(1.0f,((float)nMinTotSumDist/765) + (float)(2-nGoodSamplesCount)/2);
			pfCurrMeanMinDist_LT = (pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			pfCurrMeanMinDist_ST = (pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			pfCurrMeanRawSegmRes_LT = (pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
			pfCurrMeanRawSegmRes_ST = (pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
			fgMask(y,x) = UCHAR_MAX;
			if((curand(&state)%(size_t)2)==0) {
				const size_t s_rand = curand(&state)%50;
				colorModels[curand(&state)%50](y,x) = CurrColor;
				descModels[curand(&state)%50](y,x) = CurrIntraDesc;
			}
		}
		else {
			// == background
			fgMask(y,x) = 0;
			const float fNormalizedMinDist = ((float)nMinTotSumDist/765+(float)nMinTotDescDist/48)/2;
			//const float fNormalizedMinDist = ((float)nMinTotSumDist/765);
			pfCurrMeanMinDist_LT = (pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			pfCurrMeanMinDist_ST = (pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
			pfCurrMeanRawSegmRes_LT = (pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
			pfCurrMeanRawSegmRes_ST = (pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
			const size_t nLearningRate =(size_t)ceil(pfCurrLearningRate);
			if(curand(&state)%nLearningRate==0) {
				const size_t s_rand =curand(&state)%50;
				colorModels[s_rand](y,x) = CurrColor;
				descModels[s_rand](y,x) = CurrIntraDesc;
			}
			int x_rand,y_rand;
			const bool bCurrUsing3x3Spread = !pbUnstableRegionMask;
			if(bCurrUsing3x3Spread)
				getRandNeighborPosition_3x3(x_rand,y_rand,x,y,5/2,img.cols,img.rows);
			/*else
			getRandNeighborPosition_5x5(x_rand,y_rand,x,y,5/2,width,height);*/
			const size_t n_rand = curand(&state);
			const float fRandMeanLastDist = fmodels[3](y_rand,x_rand);
			const float fRandMeanRawSegmRes = fmodels[8](y_rand,x_rand);
			if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
				|| (fRandMeanRawSegmRes>0.995 && fRandMeanLastDist<0.01 && (n_rand%((size_t)fCurrLearningRateLowerCap))==0)) {
					colorModels[curand(&state)%50](y_rand,x_rand) = CurrColor;
					descModels[curand(&state)%50](y_rand,x_rand) = CurrIntraDesc;
			}
		}
		float UNSTABLE_REG_RATIO_MIN = 0.1;
		float FEEDBACK_T_INCR = 0.5;
		float FEEDBACK_T_DECR = 0.1;
		float FEEDBACK_V_INCR(1.f);
		float FEEDBACK_V_DECR(0.1f);
		float FEEDBACK_R_VAR(0.01f);
		if(bmodels[3](y,x) || (min(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && fgMask(y,x))) {
			if((pfCurrLearningRate)<fCurrLearningRateUpperCap)
				pfCurrLearningRate += FEEDBACK_T_INCR/(max(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST)*(pfCurrVariationFactor));
		}
		else if((pfCurrLearningRate)>fCurrLearningRateLowerCap)
			pfCurrLearningRate -= FEEDBACK_T_DECR*(pfCurrVariationFactor)/max(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST);
		if((pfCurrLearningRate)< fCurrLearningRateLowerCap)
			pfCurrLearningRate = fCurrLearningRateLowerCap;
		else if((pfCurrLearningRate)>fCurrLearningRateUpperCap)
			pfCurrLearningRate = fCurrLearningRateUpperCap;
		if(max(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && bmodels[1](y,x))
			(pfCurrVariationFactor) += FEEDBACK_V_INCR;
		else if((pfCurrVariationFactor)>FEEDBACK_V_DECR) {
			(pfCurrVariationFactor) -= bmodels[3](y,x)?FEEDBACK_V_DECR/4:pbUnstableRegionMask?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
			if((pfCurrVariationFactor)<FEEDBACK_V_DECR)
				(pfCurrVariationFactor) = FEEDBACK_V_DECR;
		}
		if((pfCurrDistThresholdFactor)<pow(1.0f+min(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST)*2,2))
			(pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(pfCurrVariationFactor-FEEDBACK_V_DECR);
		else {
			(pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(pfCurrVariationFactor);
			if((pfCurrDistThresholdFactor)<1.0f)
				(pfCurrDistThresholdFactor) = 1.0f;
		}
		/*if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>=4)
		++nNonZeroDescCount;*/
		anLastColor = CurrColor;
		anLastIntraDesc = CurrIntraDesc;

	}
}

__global__ void CudaRefreshModelKernel(float refreshRate,const PtrStepSz<uchar4> lastImg,const PtrStepSz<ushort4> lastDescImg,PtrStep<uchar4>* colorModels,PtrStep<ushort4>* descModels, int modelSize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x < lastImg.cols-2 && x>=2 && y>=2 && y < lastImg.rows-2)
	{
		curandState state;
		curand_init(threadIdx.x,0,0,&state);
		int width = lastImg.cols;
		int height = lastImg.rows;
		const size_t nBGSamplesToRefresh = refreshRate<1.0f?(size_t)(refreshRate*modelSize):modelSize;
		const size_t nRefreshStartPos = refreshRate<1.0f?curand(&state)%modelSize:0;
		for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {

			int y_sample, x_sample;
			getRandSamplePosition(s,x_sample,y_sample,x,y,2,width,height);
			colorModels[s%modelSize](y,x) = lastImg(y_sample,x_sample);
			descModels[s%modelSize](y,x) = lastDescImg(y_sample,x_sample);
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
		curand_init(threadIdx.x,0,0,&state);

		const size_t nBGSamplesToRefresh = refreshRate<1.0f?(size_t)(refreshRate*modelSize):modelSize;
		const size_t nRefreshStartPos = refreshRate<1.0f?curand(&state)%modelSize:0;
		for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {

			int y_sample, x_sample;
			getRandSamplePosition(s,x_sample,y_sample,x,y,2,width,height);
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


void CudaBSOperator(const cv::gpu::GpuMat& img, int frameIdx, 
	cv::gpu::GpuMat& fgMask,
	float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap, size_t* m_anLBSPThreshold_8bitLUT)
{
	dim3 block(16,16);
	dim3 grid((img.cols + block.x - 1)/block.x,(img.rows + block.y - 1)/block.y);
	CudaBSOperatorKernel<<<grid,block>>>(img,frameIdx,
		ptr_bModel,
		ptr_fModel,
		ptr_colorModel,
		ptr_descModel,
		fgMask, fCurrLearningRateLowerCap, fCurrLearningRateUpperCap,  m_anLBSPThreshold_8bitLUT);
}

void CudaRefreshModel(float refreshRate,const cv::gpu::GpuMat& lastImg, const cv::gpu::GpuMat& lastDescImg,size_t* m_anLBSPThreshold_8bitLUT)
{
	dim3 block(16,16);
	dim3 grid((lastImg.cols + block.x - 1)/block.x,(lastImg.rows + block.y - 1)/block.y);
	//colorModels还包含 downsample 和lastcolor
	//CudaRefreshModelKernel<<<grid,block>>>(refreshRate,lastImg,lastDescImg,ptr_colorModel,ptr_descModel,d_colorModels.size()-2);
	//colorModels还包含 downsample 和lastcolor
	CudaRefreshModelKernel<<<grid,block>>>(refreshRate,lastImg,lastDescImg,ptr_colorModel,ptr_descModel,d_colorModels.size()-2);


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