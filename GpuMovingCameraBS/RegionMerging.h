#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <queue>
#include <vector>
#include "CudaSuperpixel.h"
#include "HistComparer.h"
#include "SuperpixelComputer.h"
#include <math.h>
const int HoleSize = 10;
const int HoleNeighborsNum = 2;
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
inline unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}
inline unsigned int SeparateBy4(unsigned int x) {
	x &= 0x0000007f;                  // x = ---- ---- ---- ---- ---- ---- -654 3210
	x = (x ^ (x << 16)) & 0x0070000F; // x = ---- ---- -654 ---- ---- ---- ---- 3210
	x = (x ^ (x << 8)) & 0x40300C03; // x = -6-- ---- --54 ---- ---- 32-- ---- --10
	x = (x ^ (x << 4)) & 0x42108421; // x = -6-- --5- ---4 ---- 3--- -2-- --1- ---0
	return x;
}
// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
inline unsigned int morton3D(float x, float y, float z)
{
	using namespace std;
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}
inline unsigned int morton5D(float x, float y, float r, float g, float b)
{
	using namespace std;
	x = min(max(x * 256.f, 0.0f), 256.f);
	y = min(max(y * 256.f, 0.0f), 256.f);
	r = min(max(r * 256.f, 0.0f), 256.f);
	g = min(max(g * 256.f, 0.0f), 256.f);
	b = min(max(b * 256.f, 0.0f), 256.f);
	
	return SeparateBy4(x) | (SeparateBy4(y) << 1) | (SeparateBy4(r) << 2) | (SeparateBy4(g) << 3) | (SeparateBy4(b) << 4);
}

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
struct SPRegion
{
	SPRegion(){}
	SPRegion(int l, int _x, int _y, float d) :dist(d), id(l), cX(_x), cY(_y){}
	int id;
	int size;
	//区域边缘中是图像边缘的超像素数量
	float edgeSpNum;
	//区域的边缘中是图像边缘的像素数量
	float edgePixNum;
	//区域的边缘中是图像边缘的像素
	std::vector<cv::Point> borderEdgePixels;
	//区域像素数量
	float pixels;
	//区域周长（像素数）
	float regCircum;
	//区域超像素距离图像中心的平均距离
	float2 ad2c;
	//区域超像素距离区域中心的平均距离
	float2 rad2c;
	float4 color;
	////邻居区域Id
	std::vector<int> neighbors;
	//与每个邻居区域的边界超像素数
	std::vector<int> borders;
	//与每个邻居的边界超像素Id
	std::vector<std::vector<int>> borderSpIndices;
	//与每个邻居区域的边界像素数
	std::vector<int> borderPixelNum;
	//与每个邻居的边界像素
	std::vector<std::vector<uint2>> borderPixels;
	//与每个邻居的边界像素中是边缘的像素个数
	std::vector<float> edgeness;
	//区域中所有超像素的Id
	std::vector<int> spIndices;
	std::vector<float> colorHist;
	std::vector<float> hog;
	std::vector<float> lbpHist;
	float cX;
	float cY;
	float dist;
	//一阶中心距
	float moment;
};
//结构体的比较方法 改写operator()  
struct RegionSizeCmp
{
	bool operator()(const SPRegion &na, const SPRegion &nb)
	{
		return na.size > nb.size;
	}
};
//结构体的比较方法 改写operator()  
struct RegionWSizeCmp
{
	bool operator()(const SPRegion &na, const SPRegion &nb)
	{
		return na.edgeSpNum > nb.edgeSpNum;
	}
};
struct RegionWSizeDescCmp
{
	bool operator()(const SPRegion &na, const SPRegion &nb)
	{
		return na.edgeSpNum < nb.edgeSpNum;
	}
};
struct RegionSizeZero
{
	bool operator()(const SPRegion &na)
	{
		return na.size == 0;
	}
};
struct RegionColorCmp
{
	bool operator()(const SPRegion &na, const SPRegion &nb)
	{
		unsigned int ca = morton3D(na.color.x/255, na.color.y/255, na.color.z/255);
		unsigned int cb = morton3D(nb.color.x/255, nb.color.y/255, nb.color.z/255);
		return ca > cb;
	}
}; 

struct RegionDistCmp
{
	bool operator()(const SPRegion &na, const SPRegion &nb)
	{
		return na.dist > nb.dist;
	}
};
typedef std::priority_queue<SPRegion, std::vector<SPRegion>, RegionDistCmp> SPRegionPQ;

struct RegDist
{
	friend std::ostream &  operator << (std::ostream & os, RegDist& rd)
	{
		os << rd.sRid << "," << rd.bRid << "\n";
		os << "\t" << rd.colorDist << " " << rd.sizeDist << " " << rd.edgeness << "\n";
		return os;
	}
	RegDist()
	{
		edgeness = colorDist = hogDist = sizeDist = lbpDist = 0;
	}
	int sRid;
	int bRid;
	double colorDist;
	double hogDist;
	double sizeDist;
	double lbpDist;
	double edgeness;
};

struct RegDistDescComparer
{
	RegDistDescComparer()
	{
		colorW = hogW = sizeW = 1.0 / 3;
	}
	RegDistDescComparer(double cw, double hw, double sw) :colorW(cw), hogW(hw), sizeW(sw){};
	double colorW, hogW, sizeW;
	bool operator()(const RegDist& rd1, const RegDist& rd2)
	{
		double d1 = colorW*rd1.colorDist + hogW*rd1.edgeness + sizeW*rd1.sizeDist;
		double d2 = colorW*rd2.colorDist + hogW*rd2.edgeness + sizeW*rd2.sizeDist;
		
		return d1 < d2;
	}
};
inline float4 operator * (float4& a, float n)
{
	return make_float4(a.x*n, a.y*n, a.z*n, a.w*n);
}
inline float4 operator+(float4& a, float4& b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

void inline SaveSegment(int width, int height, int* segmented, const char* name)
{
	cv::Mat img(height,width,CV_32S,segmented);
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




void SuperPixelRegionMergingFast(int width, int height, SuperpixelComputer* computer,
	std::vector<std::vector<uint2>>& _spPoses,
	std::vector<std::vector<float>>& _colorHists,
	std::vector<std::vector<float>>& _HOGs,
	std::vector<std::vector<int>>& _regIdices,
	std::vector<int>& newLabels,
	std::vector<std::vector<float>>& _nColorHists,
	std::vector<std::vector<float>>& _nHOGs,
	float threshold, int*& segmented,
	std::vector<int>& _regSizes, std::vector<float4>& _regColors, float confidence=0.6);


void SuperPixelRegionMergingFast(int width, int height, SuperpixelComputer* computer,
	std::vector<std::vector<uint2>>& _spPoses,
	std::vector<std::vector<float>>& _colorHists,
	std::vector<std::vector<float>>& _HOGs,
	std::vector<int>& newLabels,
	std::vector<SPRegion>& regions,
	int*& segmented,
	float threshold, float confidence = 0.6
	);
void GetRegionMap(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, std::vector<uint2>& regParis, cv::Mat& mask);
void GetRegionMap(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, cv::Mat& mask, int flag = 0);
void GetRegionMap(int widht, int height, SuperpixelComputer* computer, int* segmented, std::vector<SPRegion>& regions, cv::Mat& mask, int flag = 0);
void GetRegionMap(int widht, int height, SuperpixelComputer* computer, int* segmented, std::vector<float4>& regColors, cv::Mat& mask);
void GetRegionMap(int widht, int height, SuperpixelComputer* computer, int* segmented, std::vector<int>& regions, std::vector<float4>& regColors, cv::Mat& mask);
void GetSaliencyMap(int widht, int height, SuperpixelComputer* computer, int* segmented, std::vector<int>& regSizes, std::vector<std::vector<int>>& regIndices, std::vector<int>& newLabels, std::vector<std::vector<uint2>>& spPoses, std::vector<SPRegion>& regions, cv::Mat& mask);
void GetSaliencyMap(int widht, int height, SuperpixelComputer* computer, std::vector<int>& newLabels, std::vector<std::vector<uint2>>& spPoses, std::vector<SPRegion>& regions, cv::Mat& mask);

void RegionAnalysis(int width, int height, SuperpixelComputer* computer, int* segmented,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<int>& regSizes,
	std::vector<std::vector<int>>& regIndices,
	std::vector<float4>& regColors,
	std::vector<SPRegion>& regions);

//handle holes and occlusions
void RegionAnalysis(int width, int height, SuperpixelComputer* computer, int* segmented,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions,
	std::vector<std::vector<int>>& regNeighbors);

//handle hole
int HandleHole(int i, std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions,
	std::vector<std::vector<int>>& regNeighbors);

//handle hole
int HandleHoles(int i, std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions);

int HandleHoleDemo(int width, int height, int i, SuperpixelComputer* computer, std::vector<std::vector<uint2>>& spPoses, std::vector<int>& nLabels, std::vector<SPRegion>& regions);

//merge region i to region nRegId
void MergeRegion(int i, int nRegId,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions,
	std::vector<std::vector<int>>& regNeighbors);

//merge region i to region nRegId
void MergeRegion(int i, int nRegId,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<int>& regSizes,
	std::vector<std::vector<int>>& regIndices,
	std::vector<float4>& regColors,
	std::vector<SPRegion>& regions,
	std::vector<std::vector<int>>& regNeighbors);

void MergeRegion(int i, int nRegId,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions);

void MergeRegions(int i, int nRegId,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions);

void GetContrastMap(int widht, int height, SuperpixelComputer* computer, std::vector<int>& newLabels, std::vector<std::vector<uint2>>& spPoses, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors,cv::Mat& mask);

typedef std::vector<float> HISTOGRAM;
typedef std::vector<HISTOGRAM> HISTOGRAMS;
void BuildHistogram(const cv::Mat& img, SuperpixelComputer* computer, HISTOGRAMS& colorHist, HISTOGRAMS& gradHist, int colorSpace = 0);
void BuildHistogram(const cv::Mat& img, SuperpixelComputer* computer, HISTOGRAMS& colorHist, HISTOGRAMS& gradHist, HISTOGRAMS& lbpHist, int colorSpace = 0);
void BuildQHistorgram(const cv::Mat& idxImg, int colorNum, SuperpixelComputer* computer, HISTOGRAMS& colorHist);

//计算mask区域以及其他区域的直方图对比度，如果mask是正确的显著性对象，这个对比度应该更大
float RegionContrast(const cv::Mat&img, const cv::Mat& mask, int colorSpace);

inline double RegionDist(const SPRegion& ra, const SPRegion& rb)
{
	double colorDist =  cv::compareHist(ra.colorHist, rb.colorHist, CV_COMP_BHATTACHARYYA);
	double gradDist = cv::compareHist(ra.hog, rb.hog, CV_COMP_BHATTACHARYYA);	
	
	return colorDist/*+gradDist*/;
}

inline double RegionDist(const SPRegion& ra, const SPRegion& rb, cv::Mat1f& colorDist)
{
	double d(0);
	for (int i = 0; i < ra.colorHist.size(); i++)
	{
		for (int j = 0; j < rb.colorHist.size(); j++)
		{
			float dist = colorDist[i][j];
			d += ra.colorHist[i] * rb.colorHist[j] * (dist);
		}
	}

	return d;
}

void IterativeRegionGrowing(const cv::Mat& img, const cv::Mat& edgeMap, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, cv::Mat& saliencyRst, int regThreshold = 15, bool debug = false);

void IterativeRegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, int regThreshold = 15);

void IterativeRegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, int regThreshold = 15);

void RegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF);

void RegionGrowing(const cv::Mat& img, const char* outPath, std::vector<float>& spSaliency, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF);

void RegionGrowing(const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug = false);

void AllRegionGrowing(const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug = false);

void RegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF);

void RegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, cv::Mat1f& colorDist, float thresholdF);

void RegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, cv::Mat1f& colorDist, float thresholdF);

template<class T, int D> inline T vecSqrDist(const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2) { T s = 0; for (int i = 0; i < D; i++) s += sqrt((v1[i] - v2[i])*(v1[i] - v2[i])*1.0f); return s; } // out of range risk for T = byte, ...
template<class T, int D> inline T    vecDist(const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); } // out of range risk for T = byte, ...
int Quantize(cv::Mat& img3f, cv::Mat &idx1i, cv::Mat &_color3f, cv::Mat &_colorNum, double ratio, const int clrNums[3]);

void GetRegionSegment(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, cv::Mat& segmet);

void GetRegionSegment(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, int* segmet);

void PickSaliencyRegion(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& salMap, float ratio);

void PickSaliencyRegion(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& salMap);

float PickMostSaliencyRegions(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& salMap, cv::Mat& dbgMap);

//更新区域的边界等信息
void UpdateRegionInfo(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int * segment);