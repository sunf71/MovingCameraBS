#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <queue>
#include <vector>
#include "CudaSuperpixel.h"
#include "HistComparer.h"

struct RegInfo
{
	RegInfo(){}
	RegInfo(int l, int _x, int _y, float d):dist(d),label(l),x(_x),y(_y){} 
	float dist;
	int label;
	int x,y;
};
//结构体的比较方法 改写operator()  
struct RegInfoCmp  
{  
    bool operator()(const RegInfo &na, const RegInfo &nb)  
    {  
		return na.dist > nb.dist;
    }  
};
void inline SaveSegment(int width, int height, int* segmented, char* name)
{
	cv::Mat img(height,width,CV_32S,segmented);
	double max,min;
	cv::minMaxLoc(img,&min,&max);
	cv::imwrite(name,img);

}
typedef std::priority_queue<RegInfo,std::vector<RegInfo>,RegInfoCmp> RegInfos;

void SuperPixelRegionMerging(int width, int height, int step,const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented, 
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors,float confidence = 0.6);

void SuperPixelRegionMerging(int width, int height, int step, const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	HistComparer* histComp1,
	HistComparer* histComp2,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented,
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors, float confidence = 0.6);

void SuperPixelRegionMergingFast(int width, int height, int step,const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented, 
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors,float confidence = 0.6);


void SuperPixelRegionMergingFast(int width, int height, int step, const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	HistComparer* histComp1,
	HistComparer* histComp2,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented,
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors, float confidence = 0.6);