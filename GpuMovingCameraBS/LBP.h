#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
static uchar riu2Mapping[] =
{0,1,1,2,1,9,2,3,1,9,9,9,2,9,3,4,1,9,9,9,9
,9,9,9,2,9,9,9,3,9,4,5,1,9,9,9,9,9,9,9,9,9,
9,9,9,9,9,9,2,9,9,9,9,9,9,9,3,9,9,9,4,9,5,
6,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
9,9,9,9,9,9,9,9,9,9,9,9,2,9,9,9,9,9,9,9,9,
9,9,9,9,9,9,9,3,9,9,9,9,9,9,9,4,9,9,9,5,9,
6,7,1,2,9,3,9,9,9,4,9,9,9,9,9,9,9,5,9,9,9,
9,9,9,9,9,9,9,9,9,9,9,9,6,9,9,9,9,9,9,9,9,
9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
9,9,7,2,3,9,4,9,9,9,5,9,9,9,9,9,9,9,6,9,9,
9,9,9,9,9,9,9,9,9,9,9,9,9,7,3,4,9,5,9,9,9,
6,9,9,9,9,9,9,9,7,4,5,9,6,9,9,9,7,5,6,9,7,
6,7,7,8};
static uchar u2Mapping[] = 
{0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,
11,58,58,58,58,58,58,58,12,58,58,58,
13,58,14,15,16,58,58,58,58,58,58,58,
58,58,58,58,58,58,58,58,17,58,58,58,
58,58,58,58,18,58,58,58,19,58,20,21,
22,58,58,58,58,58,58,58,58,58,58,58,
58,58,58,58,58,58,58,58,58,58,58,58,
58,58,58,58,58,58,58,58,23,58,58,58,
58,58,58,58,58,58,58,58,58,58,58,58,
24,58,58,58,58,58,58,58,25,58,58,58,
26,58,27,28,29,30,58,31,58,58,58,32,
58,58,58,58,58,58,58,33,58,58,58,58,
58,58,58,58,58,58,58,58,58,58,58,34,
58,58,58,58,58,58,58,58,58,58,58,58,
58,58,58,58,58,58,58,58,58,58,58,58,
58,58,58,58,58,58,58,35,36,37,58,38,
58,58,58,39,58,58,58,58,58,58,58,40,
58,58,58,58,58,58,58,58,58,58,58,58,
58,58,58,41,42,43,58,44,58,58,58,45,
58,58,58,58,58,58,58,46,47,48,58,49,
58,58,58,50,51,52,58,53,54,55,56,57};

template<typename T>
static void BilinearInterpolation(int width, int height, const T* data, float x, float y,T* out, int step = 3,int channel = 3)
{

	if ( x >=0 && x <width && y>=0&& y< height)
	{
		int sx = (int)x;
		int sy = (int)y;
		int bx = min(sx +1,width-1);
		int by = min(sy +1,height-1);
		float tx = x - sx;
		float ty = y - sy;
		size_t idx_rgb_lu = (sx+sy*width)*step;
		size_t idx_rgb_ru = (bx+sy*width)*step;
		size_t idx_rgb_ld =(sx+by*width)*step;
		size_t idx_rgb_rd = (bx+by*width)*step;
		for(int c=0; c<channel; c++)
		{
			out[c] =(1- ty)*((1-tx)*data[idx_rgb_lu+c]+tx*data[idx_rgb_ru+c]) + ty*((1-tx)*data[idx_rgb_ld+c] + tx*data[idx_rgb_rd+c]);
		}

	}


}
inline void LBPRGB(const cv::Mat& img, int r, int neighbors, cv::Mat& lbpImg)
{
	//目前只支持neighbors = 8
	lbpImg.create(img.size(),CV_8UC3);
	lbpImg = cv::Scalar(0);
	std::vector<cv::Point2f> spoints(neighbors);
	double angleStep = 2*M_PI/neighbors;
	for(int i=0; i<neighbors; i++)
	{
		spoints[i].y = -r*sin(i*angleStep);
		spoints[i].x = r*cos(i*angleStep);
	}
	std::vector<cv::Vec3b> values(neighbors);
	for(int i=r; i<img.rows-r; i++)
	{
		cv::Vec3b* ptr = lbpImg.ptr<cv::Vec3b>(i);
		const cv::Vec3b * cPtr = img.ptr<cv::Vec3b>(i);
		for(int j=r; j<img.cols-r; j++)
		{
			cv::Vec3b center = cPtr[j];
			for(int n=0; n<neighbors; n++)
				BilinearInterpolation(img.cols,img.rows,img.data,j+spoints[n].x,i+spoints[n].y,(uchar*)&values[n]);

			for(int c=0; c<3; c++)
			{
				int cc = 1;
				
				for(int n=0; n<neighbors; n++)
				{
					ptr[j][c] += (values[n][c]>center[c])* cc;
					cc *= 2;
				}

			}

		}
	}
}


inline void LBPRGB(const cv::Mat& img, cv::Mat& lbpImg)
{
	//目前只支持neighbors = 8
	lbpImg.create(img.size(),CV_8UC3);
	lbpImg = cv::Scalar(0);
	int ny[] = {-1,-1,-1,0,0,1,1,1};
	int nx[] = {-1,0,1,-1,1,-1,0,1};

	int r = 1;
	int neighbors(8);
	std::vector<cv::Vec3b> values(neighbors);
	for(int i=r; i<img.rows-r; i++)
	{
		cv::Vec3b* ptr = lbpImg.ptr<cv::Vec3b>(i);
		const cv::Vec3b * cPtr = img.ptr<cv::Vec3b>(i);
		for(int j=r; j<img.cols-r; j++)
		{
			cv::Vec3b center = cPtr[j];
			for(int n=0; n<neighbors; n++)
			{
				int idx = (i+ny[n])*img.cols+j+nx[n];
				uchar* dptr = (uchar*)(img.data+idx*3);
				for(int c=0; c<3; c++)
					values[n][c] = dptr[c];
			}

			for(int c=0; c<3; c++)
			{
				int cc = 1;
				
				for(int n=0; n<neighbors; n++)
				{
					ptr[j][c] += (values[n][c]>=center[c])* cc;
					cc *= 2;
				}
				ptr[j][c] = u2Mapping[ptr[j][c]];
			}

		}
	}
}

inline void LBPGRAY(const cv::Mat& img, cv::Mat& lbpImg)
{
	//目前只支持neighbors = 8
	lbpImg.create(img.size(),CV_8U);
	lbpImg = cv::Scalar(0);
	int ny[] = {-1,-1,-1,0,0,1,1,1};
	int nx[] = {-1,0,1,-1,1,-1,0,1};

	int r = 1;
	int neighbors(8);
	std::vector<uchar> values(neighbors);
	for(int i=r; i<img.rows-r; i++)
	{
		uchar* ptr = lbpImg.ptr<uchar>(i);
		const uchar * cPtr = img.ptr<uchar>(i);
		for(int j=r; j<img.cols-r; j++)
		{
			uchar center = cPtr[j];
			for(int n=0; n<neighbors; n++)
			{
				int idx = (i+ny[n])*img.cols+j+nx[n];
				uchar* dptr = (uchar*)(img.data+idx);
				values[n] = *dptr;
			}

			
			{
				int cc = 1;
				
				for(int n=0; n<neighbors; n++)
				{
					ptr[j] += (values[n]>=center)* cc;
					cc *= 2;
				}
				ptr[j] = u2Mapping[ptr[j]];
			}

		}
	}
}