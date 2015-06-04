#pragma once
#include "GpuSuperpixel.h"

class SuperpixelComputer
{
public:
	SuperpixelComputer(const int width, const int height, const int step, const float alpha = 0.9):
	  _width(width),_height(height),_step(step),_alpha(alpha)
	  {
		  init();
	  };
	  float ComputAvgColorDistance()
	  {
		  //计算超像素之间的平均颜色差
		  float avgE= 0;
		  size_t count = 0;	
		  for(int i=0; i<_nPixels; i++)
		  {
			  float4 color = _centers[i].rgb;
			  for (int j=0; j< _neighbors[i].size(); j++)
			  {
				  if (_centers[i].nPoints > 0 && _centers[_neighbors[i][j]].nPoints >0 )
				  {				
					  float4 ncolor =  _centers[_neighbors[i][j]].rgb;
					  avgE += (abs(color.x-ncolor.x) + abs(color.y - ncolor.y) + abs(color.z-ncolor.z))/3;
					  count++;
				  }
			  }		
		  }
		  avgE /= count;
		  return avgE;
	  }
	  int GetSuperpixelSize()
	  {
		  return _nPixels;
	  }
	  void ComputeSuperpixel(const cv::Mat& img);
	  void ComputeSLICSuperpixel(const cv::Mat& img);
	  void ComputeBigSuperpixel(uchar4* d_rgbaBuffer);
	  void ComputeBigSuperpixel(const cv::Mat& img);
	  void ComputeSuperpixel(const cv::Mat& Img, int& num, int*& labels, SLICClusterCenter*& centers);
	  void ComputeSuperpixel(uchar4* d_rgbaBuffer, int& num, int*& labels, SLICClusterCenter*& centers);
	  void GetPreSuperpixelResult(int& num, int*& preLabels, SLICClusterCenter*& preCenters)
	  {
		  num = _nPixels;
		  preLabels = _preLabels;
		  preCenters = _preCenters;
	  }
	  void GetSuperpixelRegionGrowingResult(int*& bgLabels)
	  {
		  bgLabels = _bgLabels;
	  }
	  void GetSuperpixelResult(int& num,int*& labels, SLICClusterCenter*& centers)
	  {
		  num = _nPixels;
		  labels = _labels;
		  centers = _centers;
	  }
	  void RegionGrowing(const std::vector<int>& seedLabels, float threshold,int*& resultLabel);
	  void RegionGrowingFast(const std::vector<int>& seedLabels, float threshold,int*& resultLabel);
	  void GetRegionGrowingImg(cv::Mat& rstImg);
	  void GetRegionGrowingSeedImg(const std::vector<int>& seedLabels, cv::Mat& rstImg);
	  //基于超像素的降采样
	  void GetSuperpixelDownSampleImg(cv::Mat& rstImg);
	  void GetSuperpixelDownSampleGrayImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& src, cv::Mat &dstImg);
	  //基于超像素的升采样
	  void GetSuperpixelUpSampleImg(const int * labels, const SLICClusterCenter* centers, const cv::Mat& src, cv::Mat& dstImg);
	  //超像素划分示意图
	  void GetVisualResult(const cv::Mat& img, cv::Mat& rstMat);
	  //将输入图像按照超像素划分进行降采样
	  void GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, cv::Mat& dstColorImg);
	  void GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, const cv::Mat& srcMapXImg, const cv::Mat& srcMapYImg, cv::Mat& dstColorImg, cv::Mat& dstMapXImg, cv::Mat& dstMapYImg);
	  void GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, const cv::Mat& srcMapXImg, const cv::Mat& srcMapYImg, const cv::Mat& srcInvMapXImg, const cv::Mat& srcInvMapYImg, 
		  cv::Mat& dstColorImg, cv::Mat& dstMapXImg, cv::Mat& dstMapYImg, cv::Mat& dstInvMapXImg,  cv::Mat& dstInvMapYImg );

	  void GetSuperpixelPoses(std::vector<std::vector<uint2>>& poses);
	  ~SuperpixelComputer()
	  {
		  release();
	  }
	  int GetSPWidth()
	  {
		  return _spWidth;
	  }
	  int GetSPHeight()
	  {
		  return _spHeight;
	  }
	  std::vector<int>& GetNeighbors(int i)
	  {
		  if (i < _neighbors.size())
		  {
			  return _neighbors[i];
		  }
	  }
protected:
	void release();
	void init();
	
private:
	int _spWidth;
	int _spHeight;
	int _width;
	int _height;
	int _step;
	int _imgSize;
	float _alpha;
	int _nPixels;
	int * _labels;
	SLICClusterCenter* _centers;
	int* _preLabels;
	SLICClusterCenter* _preCenters;
	GpuSuperpixel* _gs;
	std::vector<std::vector<int>> _neighbors;
	//用于superpixel region growing
	//背景label数组大小为_nPixels若为1表示是背景
	int * _bgLabels;
	char* _visited;
	char* _segmented;
};