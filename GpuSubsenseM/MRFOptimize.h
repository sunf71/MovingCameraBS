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
	std::vector<Point2i> pixels;
	std::vector<SuperPixel*> neighbors;
	float avgColor;
	float ps;
};

typedef std::hash_map<int,SuperPixel*>SuperPixelMap;

const float theta(0.35);
const float lmd1(0.3);
const float lmd2(3.0);

void GetSegment2DArray(SuperPixel *& superpixels, size_t & spSize, const int* lables, const int width,const int height);

void ComputeAvgColor(SuperPixel* superpixels, size_t spSize, const int width, const int height,  const unsigned int* imgData, const unsigned char* maskData);
void MaxFlowOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result);
void GraphCutOptimize(SuperPixel* spPtr, int num_pixels,float beta, int num_labels,const int width, const int height,int *result);
void MRFOptimize(GpuSuperpixel* GS, const string& originalImgName, const string& maskImgName, const string& resultImgName);