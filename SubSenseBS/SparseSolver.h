#pragma once

#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
#include <algorithm>
#include "umfpack.h"

template <typename T>
struct MyAcsdRCComp 
{
	MyAcsdRCComp(std::vector<T>* colVec, std::vector<T>* rowVec)
	{
		_colVec = colVec;
		_rowVec = rowVec;
	}
	std::vector<T>* _colVec, *_rowVec;
  bool operator() (int i,int j) 
  { 
	  if ( abs( (*_colVec)[i] - (*_colVec)[j]) < 1e-6)
		  return ((*_rowVec)[i]<(*_rowVec)[j]);
	  else
	  return ((*_colVec)[i]<(*_colVec)[j]);
  }
};

//cmat ��3��n�о���ÿһ����ǰ���б�ʾ�����������к��У����һ�б�ʾ���ݵ�ֵ
void SolveSparse(const cv::Mat& cmat, std::vector<double>& rhs, std::vector<double>& result);