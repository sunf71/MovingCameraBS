#include "CudaBSOperator.h"
#include <thrust\device_vector.h>
#include "RandUtils.h"

__device__ size_t L1dist_uchar(const uchar3& a, const uchar3& b)
{
	return abs(a.x-b.x) + abs(a.y-b.y) + abs(a.z-b.z);
}

__global__ void CudaBSOperatorKernel(const PtrStepSz<uchar3> img, int frameIndex,PtrStep<uchar>* bmodels,PtrStep<float>* fmodels,PtrStep<uchar3>* colorModels, PtrStep<ushort3>* descModels, PtrStep<uchar> fgMask,
	float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < img.cols-2 && x>=2 && y>=2 && y < img.rows-2)
    {
		curandState state;
		curand_init(threadIdx.x,0,0,&state);
        			
		const float fRollAvgFactor_LT = 1.0f/min(frameIndex,25*4);
		const float fRollAvgFactor_ST = 1.0f/min(frameIndex,25);
			const uchar3 CurrColor = img(y,x);
			uchar anCurrColor[3] = {CurrColor.x,CurrColor.y,CurrColor.z};
			//size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
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
			//ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar3& anLastColor = colorModels[51](y,x);//color model=50 bgmodel + downsample + lastcolor
			const size_t nCurrColorDistThreshold = (size_t)(((pfCurrDistThresholdFactor)*30)-((!pbUnstableRegionMask)*6));
			//const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
			//const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
			const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
			
			//ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			//const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			//LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			pbUnstableRegionMask = ((pfCurrDistThresholdFactor)>3.0 || (pfCurrMeanRawSegmRes_LT-pfCurrMeanFinalSegmRes_LT)>0.1 || (pfCurrMeanRawSegmRes_ST-pfCurrMeanFinalSegmRes_ST)>0.1)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			
			while(nGoodSamplesCount<2 && nSampleIdx<50) {
				//const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar3 const BGColor = colorModels[nSampleIdx](y,x);
				uchar anBGColor[3] = {BGColor.x,BGColor.y,BGColor.z};
				
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = abs(anCurrColor[c]-anBGColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					//size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
					//LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
					//size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
					//const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					//const size_t nSumDist = std::min((nDescDist/2)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					//if(nSumDist>nCurrSCColorDistThreshold)
					//	goto failedcheck3ch;
					//nTotDescDist += nDescDist;
					//nTotSumDist += nSumDist;
					nTotSumDist += nColorDist;
				}
				//if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
				if(nTotSumDist>nCurrTotColorDistThreshold)
					goto failedcheck3ch;
				/*if(nMinTotDescDist>nTotDescDist)
					nMinTotDescDist = nTotDescDist;*/
				if(nMinTotSumDist>nTotSumDist)
					nMinTotSumDist = nTotSumDist;
				nGoodSamplesCount++;
				failedcheck3ch:
				nSampleIdx++;
			}
			
			
			//const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
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
				}
			}
			else {
				// == background
				//const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2;
				const float fNormalizedMinDist = ((float)nMinTotSumDist/765);
				pfCurrMeanMinDist_LT = (pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				pfCurrMeanMinDist_ST = (pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
				pfCurrMeanRawSegmRes_LT = (pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				pfCurrMeanRawSegmRes_ST = (pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate =(size_t)ceil(pfCurrLearningRate);
				if(curand(&state)%nLearningRate==0) {
					const size_t s_rand =curand(&state)%50;
					colorModels[s_rand](y,x) = CurrColor;
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
				}
			}
			//float UNSTABLE_REG_RATIO_MIN = 0.1;
			//float FEEDBACK_T_INCR = 0.5;
			//float FEEDBACK_T_DECR = 0.1;
			//float FEEDBACK_V_INCR(1.f);
			//float FEEDBACK_V_DECR(0.1f);
			//float FEEDBACK_R_VAR(0.01f);
			//if(bmodels[3](y,x) || (min(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && fgMask(y,x))) {
			//	if((pfCurrLearningRate)<fCurrLearningRateUpperCap)
			//		pfCurrLearningRate += FEEDBACK_T_INCR/(max(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST)*(pfCurrVariationFactor));
			//}
			//else if((pfCurrLearningRate)>fCurrLearningRateLowerCap)
			//	pfCurrLearningRate -= FEEDBACK_T_DECR*(pfCurrVariationFactor)/max(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST);
			//if((pfCurrLearningRate)< fCurrLearningRateLowerCap)
			//	pfCurrLearningRate = fCurrLearningRateLowerCap;
			//else if((pfCurrLearningRate)>fCurrLearningRateUpperCap)
			//	pfCurrLearningRate = fCurrLearningRateUpperCap;
			//if(max(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && bmodels[1](y,x))
			//	(pfCurrVariationFactor) += FEEDBACK_V_INCR;
			//else if((pfCurrVariationFactor)>FEEDBACK_V_DECR) {
			//	(pfCurrVariationFactor) -= bmodels[3](y,x)?FEEDBACK_V_DECR/4:pbUnstableRegionMask?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
			//	if((pfCurrVariationFactor)<FEEDBACK_V_DECR)
			//		(pfCurrVariationFactor) = FEEDBACK_V_DECR;
			//}
			//if((pfCurrDistThresholdFactor)<pow(1.0f+min(pfCurrMeanMinDist_LT,pfCurrMeanMinDist_ST)*2,2))
			//	(pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(pfCurrVariationFactor-FEEDBACK_V_DECR);
			//else {
			//	(pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(pfCurrVariationFactor);
			//	if((pfCurrDistThresholdFactor)<1.0f)
			//		(pfCurrDistThresholdFactor) = 1.0f;
			//}
			
    }
}

__global__ void CudaRefreshModelKernel(float refreshRate,const PtrStepSz<uchar3> lastImg,PtrStep<uchar3>* colorModels,PtrStep<ushort3>* descModels, int modelSize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	
    if(x < lastImg.cols && y < lastImg.rows)
	{
		curandState state;
		curand_init(threadIdx.x,0,0,&state);
		int width = lastImg.cols;
		int height = lastImg.rows;
		const size_t nBGSamplesToRefresh = refreshRate<1.0f?(size_t)(refreshRate*modelSize):modelSize;
		const size_t nRefreshStartPos = refreshRate<1.0f?curand(&state)%modelSize:0;
		for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
			int y_sample, x_sample;
			getRandSamplePosition(x_sample,y_sample,x,y,2,width,height);
			colorModels[s%modelSize](y,x) = lastImg(y_sample,x_sample);
		}
	}

}



void CudaBSOperator(const cv::gpu::GpuMat& img, int frameIdx, std::vector<PtrStep<uchar>>& bmodels,
	std::vector<PtrStep<float>>& fmodels,
	std::vector<PtrStep<uchar3>>& colorModels, 
	std::vector<PtrStep<ushort3>>& descModels, 
	cv::gpu::GpuMat& fgMask,
	float fCurrLearningRateLowerCap,float fCurrLearningRateUpperCap)
{
	dim3 block(16,16);
    dim3 grid((img.cols + block.x - 1)/block.x,(img.rows + block.y - 1)/block.y);
	thrust::device_vector<PtrStep<uchar>> d_bmodels(bmodels);
	thrust::device_vector<PtrStep<float>> d_fmodels(fmodels);
	thrust::device_vector<PtrStep<uchar3>> d_colorModels(colorModels);
	thrust::device_vector<PtrStep<ushort3>> d_descModels(descModels);
	
    CudaBSOperatorKernel<<<grid,block>>>(img,frameIdx,
		thrust::raw_pointer_cast(&d_bmodels[0]),
		thrust::raw_pointer_cast(&d_fmodels[0]),
		thrust::raw_pointer_cast(&d_colorModels[0]),
		thrust::raw_pointer_cast(&descModels[0]),
		fgMask, fCurrLearningRateLowerCap, fCurrLearningRateUpperCap);
}

void CudaRefreshModel(float refreshRate,const cv::gpu::GpuMat& lastImg,std::vector<PtrStep<uchar3>>& colorModels, std::vector<PtrStep<ushort3>>& descModels)
{
	dim3 block(16,16);
    dim3 grid((lastImg.cols + block.x - 1)/block.x,(lastImg.rows + block.y - 1)/block.y);
	
	thrust::device_vector<PtrStep<uchar3>> d_colorModels(colorModels);
	thrust::device_vector<PtrStep<ushort3>> d_descModels(descModels);
	
	PtrStep<uchar3>* p1 = thrust::raw_pointer_cast(&d_colorModels[0]);
	PtrStep<ushort3>* p2 = thrust::raw_pointer_cast(&d_descModels[0]);	
	//colorModels»¹°üº¬ downsample ºÍlastcolor
	CudaRefreshModelKernel<<<block,grid>>>(refreshRate,lastImg,p1,p2,colorModels.size()-2);
}