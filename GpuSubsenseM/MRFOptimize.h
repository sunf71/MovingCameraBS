#pragma once
#include <math.h>
#include <hash_map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "GpuSuperpixel.h"
#include "GridCut/GridGraph_2D_4C.h"
#include "SuperpixelComputer.h"
using namespace std;
typedef std::pair<int,int> Point2i;

bool compare(const Point2i& p1, const Point2i& p2);
void SuperPixelRegionGrowing(int width, int height, int step,std::vector<int>& spLabels, const int*  labels, const SLICClusterCenter* centers, cv::Mat& result,int threshold);
struct SuperPixel
{
	int idx;
	int lable;
	bool operator < (const SuperPixel& a)
	{
		return lable < a.lable;
	}
	std::vector<Point2i> pixels;
	std::vector<SuperPixel*> neighbors;
	float avgColor;
	float ps;
	float distance;
	float histDist;
	int temporalNeighbor;
	int n[8];
};

typedef std::hash_map<int,SuperPixel*>SuperPixelMap;

class MRFOptimize
{
public:
	MRFOptimize(int width, int height,int step,float mdlConfidence = 0.8, float tcConfidence = 0.25,float scConfidence = 0.35):m_width(width),
		m_height(height),
		m_theta(0.35),
		m_lmd1(0.3),
		m_lmd2(scConfidence),
		m_step(step),
		m_modelConfidence(mdlConfidence),
		m_tcConfidence(tcConfidence)
	{
		Init();
	}
	~MRFOptimize()
	{
		Release();
	}
	void Init();
	void Release();
	void SetConfidence(float mdlConfidence, float tcConfidence)
	{
		m_modelConfidence = mdlConfidence;
		m_tcConfidence = tcConfidence;
	}

	void GetSegment2DArray(SuperPixel *& superpixels, size_t & spSize, const int* lables, const int width,const int height);


	void ComputeAvgColor(SuperPixel* superpixels, size_t spSize, const int width, const int height,  const unsigned int* imgData, const unsigned char* maskData);
	void MaxFlowOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result);
	void GridCutOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result);	
	void TCMaxFlowOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result);
	void TCMaxFlowOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result, float mdlConfidence, float tcConfidence);
	void GraphCutOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result);
	void Optimize(GpuSuperpixel* GS, const string& originalImgName, const string& maskImgName, const string& resultImgName);
	void Optimize(GpuSuperpixel* GS, const string& originalImgName, const string& maskImgName,  const string& featuremaskImgName,const string& resultImgName);
	void Optimize(GpuSuperpixel* GS, cv::Mat& origImg, cv::Mat& maskImg, cv::Mat& featureImg, cv::Mat& resultImg);
	void Optimize(GpuSuperpixel* GS, uchar4* d_rbga,cv::Mat& maskImg, cv::Mat& featureImg, cv::Mat& resultImg);
	void Optimize(GpuSuperpixel* GS, uchar4* d_rbga,cv::Mat& maskImg, cv::Mat& featureImg, float* distance, cv::Mat& resultImg);
	void Optimize(const cv::Mat& maskImg, const  cv::Mat& featureImg, cv::Mat& resultImg);
	void Optimize(GpuSuperpixel* GS, const cv::Mat& origImg, const cv::Mat& maskImg,const cv::Mat& lastMaskImg, const cv::Mat& flow, const cv::Mat& homography,cv::Mat& resultImg);
	void Optimize(GpuSuperpixel* GS, const cv::Mat& origImg, const cv::Mat& maskImg, const cv::Mat& flow, const cv::Mat& wflow,cv::Mat& resultImg);
	void Optimize(GpuSuperpixel* GS, const cv::Mat& origImg, const cv::Mat& maskImg, const cv::Mat& wflow,cv::Mat& resultImg);
	void Optimize(SuperpixelComputer* spComputer,const cv::Mat& maskImg,const cv::Mat& featureImg,const std::vector<int>& matchedId, cv::Mat& resultImg);
	void Optimize(SuperpixelComputer* spComputer,const cv::Mat& maskImg,const std::vector<int>& matchedId, cv::Mat& resultImg);
	void Optimize(SuperpixelComputer* spComputer,const cv::Mat& maskImg,const std::vector<int>& matchedId, cv::Mat& resultImg,float mdlConfidence, float tcConfidence);
	void ComuteSuperpixel(GpuSuperpixel* GS, uchar4* d_rgba);
	void ComputeSuperpixel(GpuSuperpixel* gs, cv::Mat& rgbaImg);
	//mask:前景
	void GetSuperpixels(const unsigned char* mask);
	//mask:前景, bgLabels:通过superpixel region growing得到的背景label
	void GetSuperpixels(const unsigned char* mask,int* bgLabels);
	//mask:前景， features：特征点跟踪情况的mask
	void GetSuperpixels(const unsigned char* fgMask, const unsigned char* featuresMask);
	//mask:前景， features：特征点跟踪情况的mask, distance: homography*pt - klttracted Position
	void GetSuperpixels(const unsigned char* mask, const uchar* featureMask,const float* distanceMask);
	//mask 前景，lastmask: 前一帧的前景， flow:光流， homography:单应性矩阵
	void GetSuperpixels(const unsigned char* mask, const uchar* lastMask, const cv::Mat& flow, const cv::Mat& homography);
	void GetSuperpixelResult(int& nPixels,int*& labels, SLICClusterCenter*& centers, float& avgE)
	{
		labels = m_labels;
		centers = m_centers;
		avgE = m_avgE;
		nPixels = m_nPixel;
	}
private:
	SuperPixel* m_spPtr;
	SLICClusterCenter* m_centers;
	SLICClusterCenter* m_preCenters;
	int m_nPixel;
	int m_width,m_gWidth;
	int m_height,m_gHeight;
	bool * m_visited;
	Point2i *m_stack;
	int* m_labels;
	int* m_preLabels;
	int* m_result;
	int* m_preResult;
	int m_step;
	int* m_data;
	int* m_smooth;
	unsigned char* m_resultImgData;
	float m_theta;
	float m_lmd1;
	float m_lmd2;
	size_t m_QSIZE;	
	uchar4* m_imgData;
	unsigned int* m_idata;
	cv::Mat m_gray,m_preGray;
	std::vector<std::vector<int>> m_neighbor;
	float m_avgE;
	float m_avgD;
	float m_variance;
	float m_modelConfidence;
	float m_tcConfidence;
};