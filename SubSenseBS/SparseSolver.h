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

//cmat 是3行n列矩阵，每一行中前两列表示数据所处的行和列，最后一列表示数据的值
void SolveSparse(const cv::Mat& cmat, std::vector<double>& rhs, std::vector<double>& result);