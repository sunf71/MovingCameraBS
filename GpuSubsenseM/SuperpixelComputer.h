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
	  void ComputeSuperpixel(const cv::Mat& Img, int& num, int*& labels, SLICClusterCenter*& centers);
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
	  void GetRegionGrowingImg(cv::Mat& rstImg);
	  ~SuperpixelComputer()
	  {
		  release();
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