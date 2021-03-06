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

using namespace std;
typedef std::pair<int,int> Point2i;

bool compare(const Point2i& p1, const Point2i& p2);

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
	int n[8];
};

typedef std::hash_map<int,SuperPixel*>SuperPixelMap;

class MRFOptimize
{
public:
	MRFOptimize(int width, int height,int step):m_width(width),
		m_height(height),
		m_theta(0.35),
		m_lmd1(0.3),
		m_lmd2(3),
		m_step(step)
	{
		Init();
	}
	~MRFOptimize()
	{
		Release();
	}
	void Init();
	void Release();
	void GetSegment2DArray(SuperPixel *& superpixels, size_t & spSize, const int* lables, const int width,const int height);

	void ComputeAvgColor(SuperPixel* superpixels, size_t spSize, const int width, const int height,  const unsigned int* imgData, const unsigned char* maskData);
	void MaxFlowOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result);
	void GraphCutOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result);
	void Optimize(GpuSuperpixel* GS, const string& originalImgName, const string& maskImgName, const string& resultImgName);
	void Optimize(GpuSuperpixel* GS, const string& originalImgName, const string& maskImgName,  const string& featuremaskImgName,const string& resultImgName);
	//mask:前景
	void GetSuperpixels(const unsigned char* mask);
	//mask:前景， features：特征点跟踪情况的mask
	void GetSuperpixels(const unsigned char* fgMask, const unsigned char* featuresMask);
private:
	SuperPixel* m_spPtr;
	SLICClusterCenter* m_centers;
	int m_nPixel;
	int m_width;
	int m_height;
	bool * m_visited;
	Point2i *m_stack;
	int* m_labels;
	int* m_result;
	int m_step;
	int* m_data;
	int* m_smooth;
	unsigned char* m_resultImgData;
	float m_theta;
	float m_lmd1;
	float m_lmd2;
	size_t m_QSIZE;	
	float4* m_imgData;
	unsigned int* m_idata;
	std::vector<std::vector<int>> m_neighbor;
};