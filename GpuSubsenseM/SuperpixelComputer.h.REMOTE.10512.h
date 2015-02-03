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
		  //���㳬����֮���ƽ����ɫ��
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
	  void GetRGSeedsImg(const std::vector<int>& seedLabels, cv::Mat& rstImg);
	  //���ڳ����صĽ�����
	  void GetSuperpixelDownSampleImg(cv::Mat& rstImg);
	  void GetSuperpixelDownSampleGrayImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& src, cv::Mat &dstImg);
	  //���ڳ����ص�������
	  void GetSuperpixelUpSampleImg(const int * labels, const SLICClusterCenter* centers, const cv::Mat& src, cv::Mat& dstImg);
	  
	  //������ͼ���ճ����ػ��ֽ��н�����
	  void GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, cv::Mat& dstColorImg);
	  void GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, const cv::Mat& srcMapXImg, const cv::Mat& srcMapYImg, cv::Mat& dstColorImg, cv::Mat& dstMapXImg, cv::Mat& dstMapYImg);
	  void GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, const cv::Mat& srcMapXImg, const cv::Mat& srcMapYImg, const cv::Mat& srcInvMapXImg, const cv::Mat& srcInvMapYImg, 
		  cv::Mat& dstColorImg, cv::Mat& dstMapXImg, cv::Mat& dstMapYImg, cv::Mat& dstInvMapXImg,  cv::Mat& dstInvMapYImg );
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
	
	//����superpixel region growing
	//����label�����СΪ_nPixels��Ϊ1��ʾ�Ǳ���
	int * _bgLabels;
	char* _visited;
	char* _segmented;
};