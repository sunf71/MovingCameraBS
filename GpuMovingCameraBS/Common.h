#pragma once
#include <io.h>
#include <direct.h>
#include <vector>
#include <string>

static inline int CreateDir(char *pszDir)
{
	int i = 0;
	int iRet;
	int iLen = strlen(pszDir);
	//在末尾加/
	if (pszDir[iLen - 1] != '\\' && pszDir[iLen - 1] != '/')
	{
		pszDir[iLen] = '/';
		pszDir[iLen + 1] = '\0';
	}

	// 创建目录
	for (i = 0;i < iLen;i ++)
	{
		if (pszDir[i] == '\\' || pszDir[i] == '/')
		{ 
			pszDir[i] = '\0';

			//如果不存在,创建
			iRet = _access(pszDir,0);
			if (iRet != 0)
			{
				iRet = _mkdir(pszDir);
				if (iRet != 0)
				{
					return -1;
				} 
			}
			//支持linux,将所有\换成/
			pszDir[i] = '/';
		} 
	}

	return 0;
}
template<typename T>
void mySwap(T*& a, T*& b)
{
	T* tmp = b;
	b = a;
	a = tmp;
}
template<typename T> void safe_delete_array(T*& a) 
{

	if (a!=NULL)
	{
		delete[] a;
		a = NULL;
	}
}

template<typename T> void safe_delete(T*& a) 
{

	if (a!=NULL)
	{
		delete a;
		a = NULL;
	}
}

template <typename T>
void DrawHistogram(std::vector<T>& histogram, int size, const std::string name)
{
	T max = histogram[0];
	int idx = 0;
	for (int i = 1; i<size; i++)
	{
		if (histogram[i] > max)
		{
			max = histogram[i];
			idx = i;
		}

	}
	cv::Mat img(400, 600, CV_8UC3);
	img = cv::Scalar(0);
	float step = (img.cols - 100 + size - 1) / size;

	cv::Scalar color(255, 255, 0);
	for (int i = 0; i<size; i++)
	{
		cv::Point2i pt1, pt2;
		pt1.x = i*step;
		pt1.y = img.rows - (histogram[i] / max*img.rows);
		pt2.x = pt1.x + step;
		pt2.y = img.rows;
		cv::rectangle(img, cv::Rect(pt1, pt2), color);
	}
	cv::imshow(name, img);
	//cv::waitKey();
}


