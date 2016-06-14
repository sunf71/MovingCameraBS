#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <queue>
#include <vector>
#include "../gpumovingcameraBS/CudaSuperpixel.h"
#include "../gpumovingcameraBS/HistComparer.h"
#include "../gpumovingcameraBS/SuperpixelComputer.h"
#include <math.h>
const int HoleSize = 5;
const int HoleNeighborsNum = 2;
const double ZERO = 1e-6;

const float cw = 0.5;
const float hw = 0.1;
const float shw = 0.2;
const float siw = 0.2;
typedef std::vector<float> HISTOGRAM;
typedef std::vector<HISTOGRAM> HISTOGRAMS;
//global color dist used for  quantitized histogram distance
extern cv::Mat1f gColorDist;
extern double gMaxDist;
static double sqr(double a)
{
	return a*a;
}
static double HistogramVariance(const HISTOGRAM&  hist)
{
	double avg(0);
	for (size_t i = 0; i < hist.size(); i++)
	{
		avg += hist[i];
	}
	avg /= hist.size();
	double Variance(0);
	for (size_t i = 0; i < hist.size(); i++)
	{
		Variance += sqr(hist[i] - avg);
	}
	//Variance /= hist.size();
	return Variance;
}


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
	RegInfo(int l, int _x, int _y, float d) :dist(d), label(l), x(_x), y(_y){}
	float dist;
	int label;
	int x, y;
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
	SPRegion()
	{
		size = 0;
		regFlag = false;
		filleness = 0;
		regSalScore = 0;
		edgeSpNum = edgePixNum = cX = cY = pixels = 0;
		color = make_float4(0, 0, 0, 0);
	}
	SPRegion(int l, int _x, int _y, float d) :dist(d), id(l), cX(_x), cY(_y)
	{ 
		size = 0; 
		regFlag = false; 
		regSalScore = 0;
		size = 0;
		regFlag = false;
		filleness = 0;
		regSalScore = 0;
		edgeSpNum = edgePixNum = cX = cY = pixels = 0;
		color = make_float4(0, 0, 0, 0);
	}
	int id;
	int size;
	//区域边缘中是图像边缘的超像素数
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
	//外边界超像素Id
	std::vector<int> outBorderSPs;
	//与每个邻居的边界像素
	std::vector<std::vector<cv::Point>> borderPixels;
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
	//形状紧凑min(width,height)/max(width,height)
	float compactness;
	//包围盒
	cv::Rect Bbox;
	cv::Rect spBbox;
	float focusness;
	//区域是否是合并遮挡时得到的(两个不相邻区域合并而来)
	bool regFlag;
	float filleness;
	//颜色直方图方差
	float colorHistV;
	float regSalScore;
	//与背景的对比度
	float regContrast;
};
inline double RegionDist(const HISTOGRAM& rah, const HISTOGRAM& rbh, cv::Mat1f& colorDist)
{
	double d(0);
	for (int i = 0; i < rah.size(); i++)
	{
		for (int j = 0; j < rbh.size(); j++)
		{
			float dist = colorDist[i][j];
			d += rah[i] * rbh[j] * (dist);
		}
	}

	return d;
}
inline double RegionDist(const SPRegion& ra, const SPRegion& rb, cv::Mat1f& colorDist)
{
	double d(0);
#pragma omp parallel for
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

double RegionColorDist(const HISTOGRAM& h1, const HISTOGRAM& h2, float4 avgc1 = make_float4(0, 0, 0, 0), float4 avgc2 = make_float4(0, 0, 0, 0));

double RegionColorDist(const SPRegion& reg0, const SPRegion& reg1);

static inline bool isNeighbor(std::vector<SPRegion>& regions, int i, int j)
{
	assert(i < regions.size() && j < regions.size() && i >= 0 && j >= 0);
	return std::find(regions[i].neighbors.begin(), regions[i].neighbors.end(), j) != regions[i].neighbors.end();
}
struct RegionPartition
{
	std::vector<SPRegion> regions;
	std::vector<int> bkgRegIds;
	std::vector<std::vector<double>> minDistances;
};
struct RegionSalInfo
{

	RegionSalInfo(){ wp = 2, wc = 1, ws = 2; };
	RegionSalInfo(float w1, float w2, float w3) :wc(w1), wp(w2), ws(w3){};
	//区域Id
	int id;
	//区域距离图像中心的距离
	float ad2c;
	//区域的相对大小
	float relSize;
	//区域的对比度(与背景区域的对比度global)
	float contrast;
	//局部对比度，与邻居区域的对比度
	float localContrast;
	//区域中包含的图像边缘占所有图像边缘的比例
	float borderRatio;
	//区域包含的邻居数占总区域数的比例
	float neighRatio;
	//区域的形状（区域的长宽比值min(len,width)/max(len,width)）
	float compactness;
	//区域的填充度，区域面积除以轮廓多边形的面积
	float fillness;
	//各项权值
	float wp, ws, wc;
	std::vector<int> neighbors;
	float RegionSaliency(float wc, float wp, float ws)
	{
		return wc * contrast + wp*((1 - borderRatio) + (1 - ad2c)) / 2 + ws*(compactness + fillness) / 2;
	}
	float RegionSaliency() const
	{
		//区域显著性越大越好
		//return contrast + (1 - borderRatio) + (1 - ad2c) + compactness + fillness + neighRatio;
		//return wc * (contrast + contrast) / 2 + wp*((1 - borderRatio) + (1 - ad2c)) / 2 + ws*(compactness + fillness) / 2;
		return wc * contrast + wp*((1 - borderRatio) + (1 - ad2c)) / 2 + ws*(compactness + fillness) / 2;
	}
	friend std::ostream &  operator << (std::ostream & os, RegionSalInfo& rd)
	{
		os << "id " << rd.id << " b " << 1 - rd.borderRatio << ",c " << rd.contrast << ",a " << 1 - rd.ad2c << ",co " << rd.compactness << ",f " << rd.fillness << " n " << rd.neighRatio << " lc " << rd.localContrast << "\n";
		os << rd.RegionSaliency() << "\n";
		return os;
	}
	float Saliency() const
	{
		return ad2c + borderRatio;
	}

	bool IsSaliency() const
	{
		float minContrast(0.7);
		float maxAd2c(0.6);
		float maxBorder(0.3);
		float minCompactness(0.45);
		return borderRatio < maxBorder;
		/*return compactness > minCompactness &&
		ad2c < maxAd2c &&
		borderRatio < maxBorder &&
		contrast > minContrast;*/

	}
};
struct RegionSalDescCmp
{
	bool operator()(const RegionSalInfo &na, const RegionSalInfo &nb)
	{

		return na.RegionSaliency() > nb.RegionSaliency();
	}
};

struct WRegionSalDescCmp
{
	WRegionSalDescCmp(float wc, float wp, float ws) :_wc(wc), _wp(wp), _ws(ws){};
	float _wc, _wp, _ws;
	bool operator()(RegionSalInfo &na, RegionSalInfo &nb)
	{

		return na.RegionSaliency(_wc, _wp, _ws) > nb.RegionSaliency(_wc, _wp, _ws);
	}
};

struct RegionSalBorderCmp
{
	bool operator()(const RegionSalInfo &na, const RegionSalInfo &nb)
	{

		return na.borderRatio < nb.borderRatio;
	}
};
struct RegionSalCmp
{
	bool operator()(const RegionSalInfo &na, const RegionSalInfo &nb)
	{

		return na.Saliency() < nb.Saliency();
	}
};
struct RegionSalscoreCmp
{
	bool operator()(const SPRegion &na, const SPRegion &nb)
	{

		return na.regSalScore > nb.regSalScore;
	}
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
struct RegionIdLocate
{
	int _id;
	RegionIdLocate(int id) :_id(id){};
	bool operator()(const SPRegion &na)
	{
		return na.id == _id;
	}
};
struct RegionSizeSmall
{
	float threshold;
	const static int minNeighbors = 3;
	const static int maxHoleSize = 5;
	RegionSizeSmall(float _threshold) :threshold(_threshold){};
	RegionSizeSmall()
	{
		threshold = 0;
	}
	bool operator()(const SPRegion &na)
	{
		return (na.size < threshold && na.size>0) || (na.neighbors.size() < minNeighbors && na.size > 0 && na.size < maxHoleSize);
	}
};
struct RegionColorCmp
{
	bool operator()(const SPRegion &na, const SPRegion &nb)
	{
		unsigned int ca = morton3D(na.color.x / 255, na.color.y / 255, na.color.z / 255);
		unsigned int cb = morton3D(nb.color.x / 255, nb.color.y / 255, nb.color.z / 255);
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
		os << "\t" << rd.colorDist << ", " << rd.sizeDist << ", " << rd.edgeness << ", " << rd.shapeDist << "\n";
		return os;
	}
	friend bool operator< (const RegDist& rd1, const RegDist& rd2)
	{
		const static double colorW = 0.25;
		const static double hogW = 0.25;
		const static double sizeW = 0.25;
		const static double shapeW = 0.25;
		double d1 = colorW*rd1.colorDist + hogW*rd1.edgeness + sizeW*rd1.sizeDist + shapeW* rd1.shapeDist;
		double d2 = colorW*rd2.colorDist + hogW*rd2.edgeness + sizeW*rd2.sizeDist + shapeW * rd2.shapeDist;

		return d1 < d2;
	}


	RegDist()
	{
		edgeness = colorDist = hogDist = sizeDist = lbpDist = 0;
	}
	int id;
	int sRid;
	int bRid;
	double colorDist;
	double hogDist;
	double sizeDist;
	double lbpDist;
	double shapeDist;
	double edgeness;
	double oColorDist;
	//Region age, how many times region has been merged
	int sRidAge;
	int bRidAge;
};
struct RegDistAcsdComparer
{
	RegDistAcsdComparer()
	{
		colorW = hogW = sizeW = shapeW = 1.0 / 4;
	}
	RegDistAcsdComparer(double cw, double hw, double shw, double siw) :colorW(cw), hogW(hw), sizeW(siw), shapeW(shw){};
	double colorW, hogW, sizeW, shapeW;
	bool operator()(const RegDist& rd1, const RegDist& rd2)
	{
		double d1 = colorW*rd1.colorDist + hogW*rd1.edgeness + sizeW*rd1.sizeDist + shapeW* rd1.shapeDist;
		double d2 = colorW*rd2.colorDist + hogW*rd2.edgeness + sizeW*rd2.sizeDist + shapeW * rd2.shapeDist;

		return d1 > d2;
	}
};
struct RegDistDescComparer
{
	RegDistDescComparer()
	{
		colorW = hogW = sizeW = shapeW = 1.0 / 4;
	}
	RegDistDescComparer(double cw, double hw, double shw, double siw) :colorW(cw), hogW(hw), sizeW(siw), shapeW(shw){};
	double colorW, hogW, sizeW, shapeW;
	bool operator()(const RegDist& rd1, const RegDist& rd2)
	{
		double d1 = colorW*rd1.colorDist + hogW*rd1.edgeness + sizeW*rd1.sizeDist + shapeW* rd1.shapeDist;
		double d2 = colorW*rd2.colorDist + hogW*rd2.edgeness + sizeW*rd2.sizeDist + shapeW * rd2.shapeDist;

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
	cv::Mat img(height, width, CV_32S, segmented);
	cv::imwrite(name, img);
}
typedef std::priority_queue<RegInfo, std::vector<RegInfo>, RegInfoCmp> RegInfos;

void SuperPixelRegionMerging(int width, int height, int step, const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented,
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors, float confidence = 0.6);

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
	std::vector<int>& _regSizes, std::vector<float4>& _regColors, float confidence = 0.6);


void SuperPixelRegionMergingFast(int width, int height, SuperpixelComputer* computer,
	std::vector<std::vector<uint2>>& _spPoses,
	std::vector<std::vector<float>>& _colorHists,
	std::vector<std::vector<float>>& _HOGs,
	std::vector<int>& newLabels,
	std::vector<SPRegion>& regions,
	int*& segmented,
	float threshold, float confidence = 0.6
	);
void GetRegionMap(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, std::vector<uint2>& regParis, cv::Mat& mask, int flag = 0);
void GetRegionMap(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, std::vector<int>& flagSPs, cv::Mat& mask);
void GetRegionSaliencyMap(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, int candiRegSize, cv::Mat& mask);
void GetRegionMap(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, cv::Mat& mask, int flag = 0, bool textflag = true);

void GetRegionMap(int widht, int height, SuperpixelComputer* computer, int* segmented, std::vector<SPRegion>& regions, cv::Mat& mask, int flag = 0, bool txtflag = true);
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

void GetContrastMap(int widht, int height, SuperpixelComputer* computer, std::vector<int>& newLabels, std::vector<std::vector<uint2>>& spPoses, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, cv::Mat& mask);


void BuildHistogram(const cv::Mat& img, SuperpixelComputer* computer, HISTOGRAMS& colorHist, HISTOGRAMS& gradHist, int colorSpace = 0);
void BuildHistogram(const cv::Mat& img, SuperpixelComputer* computer, HISTOGRAMS& colorHist, HISTOGRAMS& gradHist, HISTOGRAMS& lbpHist, int colorSpace = 0);
void BuildQHistorgram(const cv::Mat& idxImg, int colorNum, SuperpixelComputer* computer, HISTOGRAMS& colorHist);

//计算mask区域以及其他区域的直方图对比度，如果mask是正确的显著性对象，这个对比度应该更大
float RegionContrast(const cv::Mat&img, const cv::Mat& mask, int colorSpace);

inline double RegionDist(const SPRegion& ra, const SPRegion& rb)
{
	double colorDist = cv::compareHist(ra.colorHist, rb.colorHist, CV_COMP_BHATTACHARYYA);
	double gradDist = cv::compareHist(ra.hog, rb.hog, CV_COMP_BHATTACHARYYA);

	return colorDist/*+gradDist*/;
}


float SLICRegionBoxLocalContrast(SuperpixelComputer& computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int rid, HISTOGRAMS& colorHist, float theta = 0.5);
float RegionBoxLocalContrast(SuperpixelComputer& computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int rid, HISTOGRAMS& colorHist, float theta = 0.5);

void SaliencyGuidedRegionGrowing(const char* workingPath, const char* imgFolder, const char* rstFolder, const char* imgName, const cv::Mat& img, const cv::Mat& edgeMap, SuperpixelComputer& computer, cv::Mat& salMap, int regThreshold = 15, bool debug = false);

void SLICSaliencyGuidedRegionGrowing(const char* workingPath, const char* imgFolder, const char* rstFolder, const char* imgName, const cv::Mat& img, const cv::Mat& edgeMap, SuperpixelComputer& computer, cv::Mat& salMap, int regThreshold = 15, bool debug = false);

void IterativeRegionGrowing(const cv::Mat& img, const cv::Mat& edgeMap, const char* imgName, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, cv::Mat& saliencyRst, int regThreshold = 15, bool debug = false);

void IterativeRegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, int regThreshold = 15);

void IterativeRegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, int regThreshold = 15);

void RegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF);

void RegionGrowing(const cv::Mat& img, const char* outPath, std::vector<float>& spSaliency, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF);

void RegionGrowing(int iter, const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug = false);


typedef std::priority_queue<RegDist, std::vector<RegDist>, RegDistAcsdComparer> Queue;
void PrepareForRegionGrowing(int spSize, std::vector<SPRegion>& regions, Queue& RegNParis, std::vector<float>& regAges);

void FastRegionGrowing(int iter, const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, Queue& RegNPairs, std::vector<float>& regAges, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug = false);

int RegionGrowingN(int iter, const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug = false);

void AllRegionGrowing(const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug = false);

void SalGuidedRegMergion(const cv::Mat& img, const char* outPath, std::vector<RegionSalInfo>& regSalInfos, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, bool debug = false);

void SalGuidedRegMergion(const cv::Mat& img, const char* outPath, std::vector<RegionSalInfo>& regSalInfos, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, RegionPartition& rp, bool debug = false);

void SalGuidedRegMergion2(const cv::Mat& img, const char* path, std::vector<RegionSalInfo>& regSalInfos, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, bool debug = false);

void RegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF);

void RegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, cv::Mat1f& colorDist, float thresholdF);

void RegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, cv::Mat1f& colorDist, float thresholdF);

template<class T, int D> inline T vecSqrDist(const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2) { T s = 0; for (int i = 0; i < D; i++) s += sqr(v1[i] - v2[i]); return s; } // out of range risk for T = byte, ...
template<class T, int D> inline T    vecDist(const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); } // out of range risk for T = byte, ...
int Quantize(cv::Mat& img3f, cv::Mat &idx1i, cv::Mat &_color3f, cv::Mat &_colorNum, double ratio, const int clrNums[3]);

void GetRegionSegment(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, cv::Mat& segmet);

void GetRegionSegment(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, int* segmet);

void PickSaliencyRegion(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& salMap, float ratio);

void PickSaliencyRegion(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& salMap);

float PickMostSaliencyRegions(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& salMap, cv::Mat& dbgMap);

void RegionSaliencyL(int width, int height, HISTOGRAMS& colorHists, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfo, cv::Mat& salMap, bool debug = false);
void RegionSaliency(int width, int height, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfo, cv::Mat& salMap, bool debug = false);
void RegionSaliency(int width, int height, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfo, bool debug = false);
void RegionSaliency(int width, int height, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, RegionPartition & bkgRegions, std::vector<RegionSalInfo>& regInfo);
bool RegionSaliency(int width, int height, HISTOGRAMS& colorHists, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfo, bool debug = false);
//更新区域的边界等信息
void UpdateRegionInfo(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, const  cv::Mat& edgeMap, std::vector<SPRegion>& regions);

void UpdateRegionInfo(int width, int height, SuperpixelComputer* computer, const cv::Mat& gradMap, const cv::Mat& scaleMap, const cv::Mat& edgemap, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int * segment);

void InitRegions(const cv::Mat& img, HISTOGRAMS& colorHists, SuperpixelComputer* computer, const  cv::Mat& edgeMap, std::vector<SPRegion>& regions);
void GetRegionEdgeness(const cv::Mat& edgeMap, std::vector<SPRegion>& regions);


static inline cv::Rect MergeBox(cv::Rect& boxi, cv::Rect& boxj)
{
	cv::Rect box;
	box.x = std::min(boxi.x, boxj.x);
	box.y = std::min(boxi.y, boxj.y);
	int maxX = std::max(boxi.x + boxi.width, boxj.x + boxj.width);
	int maxY = std::max(boxi.y + boxi.height, boxj.y + boxj.height);
	box.width = maxX - box.x;
	box.height = maxY - box.y;
	return box;
}

//merge reg1 to reg2
void MergeRegionHist(const SPRegion& reg1, SPRegion& reg2);


void GetRegionBorder(SPRegion& reg, std::vector<cv::Point>& borders);

void ShowRegionBorder(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, int regId, cv::Mat rmask);

void ShowRegionBorder(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, cv::Mat rmask);


//http://blog.bigbing.org/?p=661
static inline cv::Mat guidedFilter(cv::Mat I, cv::Mat p, int r, double eps)
{
	/*
	% GUIDEDFILTER   O(1) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, CV_64FC1);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, CV_64FC1);
	p = _p;

	//[hei, wid] = size(I);
	int hei = I.rows;
	int wid = I.cols;

	//N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
	cv::Mat N;
	cv::boxFilter(cv::Mat::ones(hei, wid, I.type()), N, CV_64FC1, cv::Size(r, r));

	//mean_I = boxfilter(I, r) ./ N;
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;	
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
	mean_a = mean_a / N;

	//mean_b = boxfilter(b, r) ./ N;
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
	mean_b = mean_b / N;

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
	cv::Mat q = mean_a.mul(I) + mean_b;

	return q;
}