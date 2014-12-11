/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 10 of the cookbook:  
Computer Vision Programming using the OpenCV Library. 
by Robert Laganiere, Packt Publishing, 2011.

This program is free software; permission is hereby granted to use, copy, modify, 
and distribute this source code, or portions thereof, for any purpose, without fee, 
subject to the restriction that the copyright notice may not be removed 
or altered from any source or altered source distribution. 
The software is released on an as-is basis and without any warranties of any kind. 
In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
The author disclaims all warranties with regard to this software, any use, 
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2010-2011 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "Affine2D.h"
#include "motiontracker.h"
#include <algorithm>
#include "LBSP.h"
#include <fstream>
#include <math.h>
#include "LBP.h"
#include <direct.h>
#include <io.h>
#include <list>
#include <bitset>
//#include "libAnn.h"
//#include "mclmcr.h"
//#include "matrix.h"
//#include "mclcppclass.h"

int CreateDir(char *pszDir)
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
//统计类，用于计算标准差，协方差等统计数据
class STAT
{
public:

	STAT(void)
	{
	}

	~STAT(void)
	{
	}

	//计算均值 
	template  <class T>
	static double Mean(vector<T>& v)
	{
		double sum = 0;
		for(int i=0; i<v.size(); i++)
		{
			sum += v[i];
		}
		return sum/v.size();
	}
	//计算方差
	template <class T>
	static double Variance(vector<T>& v)
	{
		double ret = 0;
		double mean = Mean(v);
		for(int i=0; i<v.size(); i++)
		{
			ret += (v[i] - mean)*(v[i] - mean);
		}
		return ret/v.size();
	}
	template <class T>
	static double Variance(vector<T>& v,double mean)
	{
		double ret = 0;		
		for(int i=0; i<v.size(); i++)
		{
			ret += (v[i] - mean)*(v[i] - mean);
		}
		return ret/v.size();
	}
	//计算标准差
	template <class T>
	static double STD(vector<T>& v)
	{
		return sqrt(Variance(v));
	}

	//计算协方差
	template <class T>
	static double Cov(vector<T>& a, vector<T>& b)
	{
		assert(a.size() == b.size());

		//cov(a,b) = E(a*b) - E(a)*E(b)
		double ret = 0;
		double suma = 0;
		double sumb = 0;
		for(int i=0; i<a.size(); i++)
		{
			ret += a[i]*b[i];
			suma += a[i];
			sumb += b[i];		
		}
		
		return (ret - suma*sumb/a.size())/a.size();
	}

	template <class T>
	static double Cov(vector<T>& a, vector<T>&b, double meanA, double meanB)
	{
		assert(a.size() == b.size());
		double ret = 0;
		
		for(int i=0; i<a.size(); i++)
		{
			ret += a[i]*b[i];
			
		}
		return ret/a.size() - meanA*meanB;
	}

	template<class T>
	static double BhatBinDistance(const vector<T>& a, const vector<T>& b)
	{
		assert(a.size() == b.size());
		float sumXY = 0;
		float sumX = 0;
		float sumY = 0;
		for(int i=0; i<a.size(); i++)
		{
			sumXY += sqrt(a[i]*b[i]);
			sumX += a[i];
			sumY += b[i];
		}

		return 1 - sumXY/sqrt(sumX*sumY);
	}

	//计算直方图距离
	template <class T>
	static double BinDistance(const vector<T>& a, const vector<T>& b)
	{
		return 1- abs(CorDistance(a,b));
	/*	return BhatBinDistance(a,b);*/
	}

	//计算直方图距离
	template <class T>
	static double BinDistance(const vector<vector<T>>& a, const vector<vector<T>>& b)
	{
		
		double Max = 0;		
		int c = 0;
	    vector<T> sumA(a[0].size(),0),sumB(a[0].size(),0);

		for(int i=0; i<3; i++)
		{
		/*	double max = *max_element(a[i].begin(),a[i].end());
			double ratio = max/accumulate(a[i].begin(),a[i].end(),0);
			max *= ratio;
			if (max > Max)
			{
				Max = max;
				c = i;
			}*/
			for(int j=0; j<a[0].size(); j++)
			{
				sumA[j] += a[i][j];
				sumB[j] += b[i][j];
			}
		}	
		return BinDistance(sumA,sumB);
		//return BinDistance(a[c],b[c]);		
	}
	//计算直方图距离
	template <class T>
	static double CorDistance(const vector<T>& a, const vector<T>& b)
	{	
		//distance = (cov(a,b) + c)/(std(a)*std(b) + c) c是一个防止除0的常数 0.00001
		const double c = 0.00001;
		assert(a.size() == b.size());

		//cov(a,b) = E(a*b) - E(a)*E(b)
		double cov = 0;
		double suma = 0;
		double sumb = 0;
		for(int i=0; i<a.size(); i++)
		{
			cov += a[i]*b[i];
			suma += a[i];
			sumb += b[i];		
		}
		
		double meanA = suma/a.size();
		double meanB = sumb/b.size();

		double stdA = 0;
		double stdB = 0;
		for(int i=0; i<a.size(); i++)
		{
			stdA += (a[i] - meanA)*(a[i] - meanA);
			stdB += (b[i] - meanB)*(b[i] - meanB);

		}
		stdA = sqrt(stdA/a.size());
		stdB = sqrt(stdB/a.size());

		cov /= a.size();
		cov -= meanA * meanB;
		
		return (cov + c)/(stdA*stdB +c);	
	
	}
};
void TestAffine()
{
	using namespace cv;
	/*Mat img1 = imread("..//PTZ//input0//in000007.jpg");
	Mat img2 = imread("..//PTZ//input0//in000008.jpg");*/
	Mat img1 = imread("..//PTZ//input3//in000204.jpg");
	Mat img2 = imread("..//PTZ//input3//in000211.jpg");
	Mat gray1,gray2;
	cvtColor(img1, gray1, CV_BGR2GRAY); 
	cvtColor(img2, gray2, CV_BGR2GRAY);
	imwrite("i1.jpg",gray1);
	imwrite("i2.jpg",gray2);
	std::vector<cv::Point2f> features1,features2;  // detected features

	int max_count = 50;	  // maximum number of features to detect
	double qlevel = 0.01;    // quality level for feature detection
	double minDist = 10;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	// detect the features
	cv::goodFeaturesToTrack(gray1, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray1, gray2, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	Mat affine = estimateRigidTransform(features1,features2,true);
	Mat result;
	warpAffine(gray1,result,affine,cv::Size(gray1.cols,gray1.rows));
	imwrite("i3.jpg",result);
	std::cout<<"affine"<<std::endl;
	std::cout<<affine;
	Mat A(2,2,affine.type());
	for(int i=0; i<2; i++)
		for(int j=0;j<2;j++)
			A.at<double>(i,j) = affine.at<double>(i,j);
	A = A.inv();
	Mat invAffine(2,3,affine.type());
	for(int i=0; i<2; i++)
		for(int j=0;j<2;j++)
			invAffine.at<double>(i,j) = A.at<double>(i,j);
	invAffine.at<double>(0,2) = affine.at<double>(0,2)*-1;
	invAffine.at<double>(1,2) = affine.at<double>(1,2)*-1;
	Mat grayInv;
	cout<<invAffine;
	warpAffine(result,grayInv,invAffine,cv::Size(grayInv.cols,grayInv.rows));
	imshow("inv affine",grayInv);
	Mat out;
	std::vector<uchar> in(features1.size(),0);
	estimateAffine2D(features1,features2,out,in);
	std::cout<<out;
	Mat rresult;
	warpAffine(gray1,rresult,out,cv::Size(gray1.cols,gray1.rows));

	std::vector<uchar> inliers(features1.size(),0);
	cv::Mat homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		1.); // max distance to reprojection point

	// Warp image 1 to image 2
	cv::Mat presult;
	cv::warpPerspective(gray1, // input image
		presult,			// output image
		homography,		// homography
		cv::Size(gray1.cols,gray1.rows),INTER_LINEAR | WARP_INVERSE_MAP); // size of output image

	Mat ErrImg = abs(gray2 - result);
	ErrImg.convertTo(ErrImg,CV_8U);
	Mat pErrImg = abs(gray2 - presult);
	Mat rErrImg = abs(gray2 - rresult);
	// Display the warp image
	cv::namedWindow("After perspective warping");
	cv::imshow("After perspective warping",presult);

	cv::namedWindow("After affine warping");
	cv::imshow("After affine warping",result);

	cv::namedWindow("After ransac affine warping");
	cv::imshow("After ransac affine warping",rresult);

	cv::namedWindow("img1");
	cv::imshow("img1",gray1);
	cv::namedWindow("img2");
	cv::imshow("img2",gray2);

	cv::namedWindow("affine error");

	imwrite("i2-i3.jpg",ErrImg);
	cv::imshow("affine error",ErrImg);
	/*cv::Mat ErrImgRead = imread("i2-i3.bmp");
	cv::imshow("affine error read",ErrImgRead);*/
	cv::namedWindow("ransac affine error");
	cv::imshow("ransac affine error",rErrImg);
	cv::namedWindow("perror");
	cv::imshow("perror",pErrImg);

	cv::waitKey();
}
template<typename T>
T LinearInterData(int width, int height, T*data, float x, float y)
{
	if ( x >=0 && x <width && y>=0&& y< height)
	{
		int sx = (int)x;
		int sy = (int)y;
		int bx = sx +1;
		int by = sy +1;
		float tx = x - sx;
		float ty = y - sy;
		return (1-ty)*((1-tx)*data[sx+sy*width]+tx*data[bx+sy*width]) + ty*((1-tx)*data[sx+by*width] + tx*data[bx+by*width]);

	}
	else
		return 0;
}
//void computeGrayscaleDescriptor(const cv::Mat& dxImg, const cv::Mat& dyImg,const int _x, const int _y, ushort& res)
//{
//	
//	const size_t patchSize = 5;
//	const int halfPSize = patchSize/2;
//	int width = dxImg.cols;
//	int height = dxImg.rows;
//	CV_DbgAssert(!dxImg.empty());
//	CV_DbgAssert(dxImg.type()==CV_16SC1);
//	if ( _x - halfPSize< 0 || _x+halfPSize > width-1 || _y-halfPSize <0 || _y+halfPSize > height-1)
//		return;
//	const int _step0 = dxImg.step.p[0];
//	const int _step1 = dxImg.step.p[1];
//	const uchar* const _dxData = dxImg.data;
//	const uchar* const _dyData = dyImg.data;
//	double lmd[patchSize*patchSize];
//	double ori[patchSize*patchSize];
//	for(int i=-halfPSize; i<=halfPSize; i++)
//	{
//		for(int j=-halfPSize; j<= halfPSize; j++)
//		{
//			int x = i+_x;
//			int y = j+_y;
//			int idx = y*_step0 +x*_step1;
//			short dx = *((short*)(_dxData +idx ));
//			short dy = *((short*)(_dyData +idx));
//			double j11 = dx*dx;
//			double j12 = dx*dy;
//			double j22 = dy*dy;
//			double tmp = sqrt(1.0*(j22-j11)*(j22-j11)+4*j12*j12);
//			double lmdMax = 0.5*(j11+j22+tmp);
//			double lmdMin =  0.5*(j11+j22-tmp);
//			double orientation(0);		
//			if (abs((j22-j11+tmp)< 1e-6))
//			{
//				orientation = 90.0;
//			}
//			else
//			{
//				orientation = atan(2*j12/(j22-j11+tmp))/M_PI*180.0;
//
//			}
//			if (orientation < 0)
//				orientation += 180;
//			
//			lmd[(i+halfPSize)+(j+halfPSize)*patchSize] = lmdMax;
//			ori[(i+halfPSize)+(j+halfPSize)*patchSize] = orientation;
//		}
//	}
//
//	_res = ((absdiff_uchar(_val(-1, 1, n),_ref[n]) > _t[n]) << 15)
//			+ ((absdiff_uchar(_val( 1,-1, n),_ref[n]) > _t[n]) << 14)
//			+ ((absdiff_uchar(_val( 1, 1, n),_ref[n]) > _t[n]) << 13)
//			+ ((absdiff_uchar(_val(-1,-1, n),_ref[n]) > _t[n]) << 12)
//			+ ((absdiff_uchar(_val( 1, 0, n),_ref[n]) > _t[n]) << 11)
//			+ ((absdiff_uchar(_val( 0,-1, n),_ref[n]) > _t[n]) << 10)
//			+ ((absdiff_uchar(_val(-1, 0, n),_ref[n]) > _t[n]) << 9)
//			+ ((absdiff_uchar(_val( 0, 1, n),_ref[n]) > _t[n]) << 8)
//			+ ((absdiff_uchar(_val(-2,-2, n),_ref[n]) > _t[n]) << 7)
//			+ ((absdiff_uchar(_val( 2, 2, n),_ref[n]) > _t[n]) << 6)
//			+ ((absdiff_uchar(_val( 2,-2, n),_ref[n]) > _t[n]) << 5)
//			+ ((absdiff_uchar(_val(-2, 2, n),_ref[n]) > _t[n]) << 4)
//			+ ((absdiff_uchar(_val( 0, 2, n),_ref[n]) > _t[n]) << 3)
//			+ ((absdiff_uchar(_val( 0,-2, n),_ref[n]) > _t[n]) << 2)
//			+ ((absdiff_uchar(_val( 2, 0, n),_ref[n]) > _t[n]) << 1)
//			+ ((absdiff_uchar(_val(-2, 0, n),_ref[n]) > _t[n]));
//
//}
void computeRGBDescriptor(const cv::Mat& dxImg, const cv::Mat& dyImg, const int _x, const int _y, std::vector<std::vector<float>>&histogram,ushort* binPattern)
{
	const size_t binSize = 16 ;
	histogram.resize(3);
	
	const size_t patchSize = 5;
	const int halfPSize = patchSize/2;
	int width = dxImg.cols;
	int height = dxImg.rows;
	CV_DbgAssert(!dxImg.empty());
	CV_DbgAssert(dxImg.type()==CV_16SC3);
	if ( _x - halfPSize< 0 || _x+halfPSize > width-1 || _y-halfPSize <0 || _y+halfPSize > height-1)
		return;
	const int _step0 = dxImg.step.p[0];
	const int _step1 = dxImg.step.p[1];
	const uchar* const _dxData = dxImg.data;
	const uchar* const _dyData = dyImg.data;
	float bin_step = (180+binSize-1)/binSize;
	for(int c=0; c<3; c++)
	{
		histogram[c].resize(binSize);
		memset(&histogram[c][0],0,sizeof(float)*binSize);
		for(int i=-halfPSize; i<=halfPSize; i++)
		{
			for(int j=-halfPSize; j<= halfPSize; j++)
			{
				int x = i+_x;
				int y = j+_y;
				int idx = y*_step0 +x*_step1;
				short dx = *((short*)(_dxData +idx)+c);
				short dy = *((short*)(_dyData +idx)+c);
				double j11 = dx*dx;
				double j12 = dx*dy;
				double j22 = dy*dy;
				double tmp = sqrt(1.0*(j22-j11)*(j22-j11)+4*j12*j12);
				double lmdMax = 0.5*(j11+j22+tmp);
				double lmdMin =  0.5*(j11+j22-tmp);
				double orientation(0);		
				if (abs((j22-j11+tmp)< 1e-6))
				{
					orientation = 90.0;
				}
				else
				{
					orientation = atan(2*j12/(j22-j11+tmp))/M_PI*180.0;

				}
				if (orientation < 0)
					orientation += 180;
				/*float ang = atan(dy/(dx+1e-6))/M_PI*180;
				if (ang<0)
				ang+=180;
				histogram[ang/bin_step] += (abs(dx)+abs(dy));*/
				histogram[c][(int)(orientation/bin_step)] += lmdMax;

			}
		}
		double max = histogram[c][0];
		int idx(0);
		for(int i=1; i<binSize; i++)
		{
			if (histogram[c][i] > max)
			{
				max = histogram[c][i];
				idx = i;
			}
		}
		binPattern[c] = 0;
		//std::cout<<max<<std::endl;
		for(int i=0; i<binSize; i++)
		{
			if (histogram[c][i] > max*0.2)
			{
				binPattern[c] |= 1 << (binSize-i-1);
			}
		}
	}
}
void computeGrayscaleDescriptor(const cv::Mat& dxImg, const cv::Mat& dyImg,const int _x, const int _y, std::vector<float>& histogram,ushort& binPattern)
{
	const size_t binSize = 16 ;
	histogram.resize(binSize);
	memset(&histogram[0],0,sizeof(float)*binSize);
	const size_t patchSize = 5;
	const int halfPSize = patchSize/2;
	int width = dxImg.cols;
	int height = dxImg.rows;
	CV_DbgAssert(!dxImg.empty());
	CV_DbgAssert(dxImg.type()==CV_16SC1);
	if ( _x - halfPSize< 0 || _x+halfPSize > width-1 || _y-halfPSize <0 || _y+halfPSize > height-1)
		return;
	const int _step0 = dxImg.step.p[0];
	const int _step1 = dxImg.step.p[1];
	const uchar* const _dxData = dxImg.data;
	const uchar* const _dyData = dyImg.data;
	float bin_step = (180+binSize-1)/binSize;
	for(int i=-halfPSize; i<=halfPSize; i++)
	{
		for(int j=-halfPSize; j<= halfPSize; j++)
		{
			int x = i+_x;
			int y = j+_y;
			int idx = y*_step0 +x*_step1;
			short dx = *((short*)(_dxData +idx ));
			short dy = *((short*)(_dyData +idx));
			double j11 = dx*dx;
			double j12 = dx*dy;
			double j22 = dy*dy;
			double tmp = sqrt(1.0*(j22-j11)*(j22-j11)+4*j12*j12);
			double lmdMax = 0.5*(j11+j22+tmp);
			double lmdMin =  0.5*(j11+j22-tmp);
			double orientation(0);		
			if (abs((j22-j11+tmp)< 1e-6))
			{
				orientation = 90.0;
			}
			else
			{
				orientation = atan(2*j12/(j22-j11+tmp))/M_PI*180.0;

			}
			if (orientation < 0)
				orientation += 180;
			/*float ang = atan(dy/(dx+1e-6))/M_PI*180;
			if (ang<0)
				ang+=180;
			histogram[ang/bin_step] += (abs(dx)+abs(dy));*/
			histogram[(int)(orientation/bin_step)] += lmdMax;
			
		}
	}
	double max = histogram[0];
	int idx(0);
	for(int i=1; i<binSize; i++)
	{
		if (histogram[i] > max)
		{
			max = histogram[i];
			idx = i;
		}
	}
	binPattern = 0;
	for(int i=0; i<binSize; i++)
	{
		if (histogram[i] > max*0.5)
		{
			binPattern |= 1 << (binSize-i-1);
		}
	}

}
//计算LBP
uchar LBP(uchar* imgData,int width, int height, int c,int x, int y, int r, int p)
{
	std::ofstream file("out.txt");
	if ( x-r<0 || x+r>width-1 || y-r<0 || y+r>height-1)
		return 0;
	uchar ret = 0;
	size_t c_idx = x + y*width;
	std::vector<double> dx,dy,tx,ty;
	dx.resize(p);
	dy.resize(p);
	tx.resize(p);
	ty.resize(p);

	for(int i=0; i<p; i++)
	{
		dx[i] = r*cos(2*M_PI*i/p);
		tx[i] = dx[i] - (int)dx[i];
		dy[i] = -r*sin(2*M_PI*i/p);
		ty[i] = dy[i] - (int)dy[i];
	}
	file<<"txty\n";
	for(int i=0; i<p; i++)
	{
		file<<(1-ty[i])*(1-tx[i])<<","<<tx[i]*(1-ty[i])<<","<<ty[i]*(1-tx[i])<<","<<ty[i]*tx[i]<<","<<std::endl;

	}
	file<<"\n";
	file<<"dxdy=\n";
	for(int i=0; i<p; i++)
	{
		file<<dx[i]<<","<<dy[i]<<","<<std::endl;

	}
	file<<"\n";

	file.close();
	std::vector<uchar> value(p);
	for(int i=0; i<p; i++)
	{
		value[i] = LinearInterData(width,height,imgData,dx[i]+x,dy[i]+y);
		std::cout<<(int)value[i]<<" ";
	}
	uchar pixels[16];
	for(int i=0; i<p; i++)
		pixels[i] = LinearInterData(width,height,imgData,dx[i]+x,dy[i]+y);
	for(int i=0; i<p/2; i++)
	{

		ret |= (pixels[i]-pixels[i+p/2]>=0) << p-1-i;

	}
	for(int i=0; i<p; i+=2)
	{

		ret |= (pixels[i]-pixels[i+1]>=0) << p/2-1-i;


		std::cout<<std::endl;

		for(int i=0; i<p/2; i++)
		{

			ret |= (value[i]-value[i+p/2]>=0) << p/2-1-i;

		}
		return ret;
	}
}

void TestLBP()
{
	using namespace cv;
	Mat img1 = imread("..//PTZ//input3//in000289.jpg");
	Mat img2 = imread("..//PTZ//input3//in000290.jpg");


	Mat gray1,gray2;
	cvtColor(img1, gray1, CV_BGR2GRAY); 
	cvtColor(img2, gray2, CV_BGR2GRAY);

	cv::GaussianBlur(gray1,gray1,cv::Size(3,3),0.1);
	cv::GaussianBlur(gray2,gray2,cv::Size(3,3),0.1);

	std::vector<cv::Point2f> features1,features2;  // detected features

	int max_count = 500;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 10;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	// detect the features
	cv::goodFeaturesToTrack(gray1, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray1, gray2, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	std::vector<uchar> inliers(features1.size(),0);
	cv::Mat homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.5); // max distance to reprojection point
	double* ptr = (double*)homography.data;
	cv::Mat lbp1(gray1.size(),CV_8U);
	lbp1 = cv::Scalar(0);
	cv::Mat lbp2;
	lbp2 = lbp1.clone();
	cv::Mat diff = lbp1.clone();
	cv::Mat diff2 = diff.clone();
	for(int i=2; i<gray1.rows-2; i++)
	{
		for(int j=2; j<gray1.cols-2; j++)
		{

			float x,y,w;
			x = j*ptr[0] + i*ptr[1] + ptr[2];
			y = j*ptr[3] + i*ptr[4] + ptr[5];
			w = j*ptr[6] + i*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			int wx = int(x+0.5);
			int wy = int(y+0.5);
			size_t idx = j+i*gray1.cols;
			size_t widx = j+i*gray1.cols;
			ushort res1,res2;
			if (wx >= 2 && wx < gray1.cols-2 && wy >=2 && wy <gray1.rows-2)
			{
				LBSP::computeGrayscaleDescriptor(gray1,gray1.data[idx],j,i,0,res1);
				LBSP::computeGrayscaleDescriptor(gray2,gray2.data[widx],wx,wy,0,res2);
				diff2.data[idx] = hdist_ushort_8bitLUT(res1,res2)/16.0*255;

				lbp1.data[idx] = LBP(gray1.data,gray1.cols,gray1.rows,1,j,i,2,16);
				//std::cout<<"lbp1: "<<(int)lbp1<<std::endl;
				lbp2.data[widx] = LBP(gray2.data,gray1.cols,gray1.rows,1,wx,wy,2,16);
				//std::cout<<"lbp2: "<<(int)lbp2<<std::endl;
				diff.data[idx] = popcount_LUT8[lbp1.data[idx]^lbp2.data[widx]]/8.0*255;
			}

		}

	}
	cv::imshow("lbp1",lbp1);
	cv::imshow("lbp2",lbp2);
	cv::imshow("diff",diff);
	cv::imshow("diff2",diff2);
	cv::waitKey(0);	
}
cv::Point2f warpPoint(cv::Point2f&p, double* ptr)
{
	float x,y,w;
	float j = p.x;
	float i = p.y;
	x = j*ptr[0] + i*ptr[1] + ptr[2];
	y = j*ptr[3] + i*ptr[4] + ptr[5];
	w = j*ptr[6] + i*ptr[7] + ptr[8];
	x /=w;
	y/=w;
	return cv::Point2f(x,y);
}
void OpticalFlowHistogram(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,
	std::vector<float>& histogram, std::vector<std::vector<int>>& ids, int DistSize = 16,int thetaSize = 16)
{
	//直方图共256个bin，其中根据光流强度分16个bin，每个bin根据光流方向分16个bin
	int binSize = DistSize * thetaSize;
	histogram.resize(binSize);
	ids.resize(binSize);
	for(int i=0; i<binSize; i++)
		ids[i].clear();
	memset(&histogram[0],0,sizeof(float)*binSize);
	float max = -9999;
	float min = -max;
	std::vector<float> thetas(f1.size());
	std::vector<float> rads(f1.size());
	for(int i =0; i<f1.size(); i++)
	{
		float dx = f1[i].x - f2[i].x;
		float dy = f1[i].y - f2[i].y;
		float theta = atan(dy/(dx+1e-6))/M_PI*180;
		if (theta<0)
			theta+=90;
		thetas[i] = theta;
		rads[i] = sqrt(dx*dx + dy*dy);
	
		max = rads[i] >max? rads[i] : max;
		min = rads[i]<min ? rads[i]: min;

	}
	float stepR = (max-min+1e-6)/DistSize;
	float stepT = 180/thetaSize;
	for(int i=0; i<f1.size(); i++)
	{
		int r = (int)((rads[i] - min)/stepR);
		int t = (int)(thetas[i]/stepT);
		r = r>DistSize? DistSize:r;
		t = t>thetaSize? thetaSize:t;
		int idx = t*DistSize+r;
		//std::cout<<idx<<std::endl;
		histogram[idx]++;
		ids[idx].push_back(i);
	
	}
}
void DrawHistogram(std::vector<float>& histogram, int size, const std::string name = "histogram")
{
	float max = histogram[0];
	int idx = 0;
	for(int i=1; i<size; i++)
	{
		if (histogram[i] > max)
		{
			max = histogram[i];
			idx = i;
		}

	}
	cv::Mat img(400,300,CV_8UC3);
	img = cv::Scalar(0);
	int step = (img.cols+size-1)/size;
	cv::Scalar color(255,255,0);
	for(int i=0; i<size; i++)
	{
		cv::Point2i pt1,pt2;
		pt1.x = i*step;
		pt1.y = img.rows - (histogram[i]/max*img.rows);
		pt2.x = pt1.x + step;
		pt2.y = img.rows;
		cv::rectangle(img,cv::Rect(pt1,pt2),color);
	}
	cv::imshow(name,img);
	//cv::waitKey();
}
void findHomographyDLT(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,cv::Mat& homography)
{
	homography.create(3,3,CV_64F);
	double* homoData = (double*)homography.data;
	cv::Mat dataM(2*f1.size(),9,CV_64F);
	double* ptr = (double*)dataM.data;
	int rowStep = dataM.step.p[0];
	for(int i=0; i<f1.size(); i++)
	{
		ptr[0] = ptr[1] = ptr[2] =0;
		ptr[3] = -1*f1[i].x;
		ptr[4] = -1*f1[i].y;
		ptr[5] = -1;
		ptr[6] = f2[i].y*f1[i].x;
		ptr[7] = f2[i].y*f1[i].y;
		ptr[8] = f2[i].y;
		ptr += 9;
		ptr[0] = f1[i].x;
		ptr[1] = f1[i].y;
		ptr[2] = 1;
		ptr[3] = ptr[4] = ptr[5] = 0;
		ptr[6] = -f2[i].x * f1[i].x;
		ptr[7] = -f2[i].x * f1[i].y;
		ptr[8] = -f2[i].x;
		ptr += 9;
	}
	cv::Mat w,u,vt;
	//std::cout<<"A = "<<dataM<<std::endl;
	cv::SVDecomp(dataM,w,u,vt);
	//std::cout<<"vt = "<<vt<<std::endl;
	ptr = (double*)(vt.data + (vt.rows-1)*vt.step.p[0]);
	for(int i=0; i<9; i++)
		homoData[i] = ptr[i]/ptr[8];


}
void TestOpticalFlowHistogram()
{
	int start = 1; 
	int end = 19;
	char outPathName[100];
	char fileName[100];
	sprintf(outPathName,".\\histogram\\moseg\\cars1\\features\\");
	CreateDir(outPathName);
	cv::Mat img,preImg,gray,preGray;
	std::vector<cv::Point2f> features1,features2;  // detected features
	int max_count = 5000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 10;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	for(int i=start; i<=end;i++)
	{
		features1.clear();
		features2.clear();
		sprintf(fileName,"..\\moseg\\cars1\\in%06d.jpg",i);
		img = cv::imread(fileName);
		cv::cvtColor(img,gray,CV_BGR2GRAY);
		if (preImg.empty())
		{
			preImg = img.clone();
			preGray = gray.clone();
		}
		// detect the features
		cv::goodFeaturesToTrack(gray, // the image 
			features1,   // the output detected features
			max_count,  // the maximum number of features 
			qlevel,     // quality level
			minDist);   // min distance between two features

		// 2. track features
		cv::calcOpticalFlowPyrLK(gray, preGray, // 2 consecutive images
			features1, // input point position in first image
			features2, // output point postion in the second image
			status,    // tracking success
			err);      // tracking error

		int k=0;
		for(int i=0; i<features1.size(); i++)
		{
			if (status[i] == 1)
			{
				features2[k] = features2[i];
				features1[k] = features1[i];
				k++;
			}
		}
		features1.resize(k);
		features2.resize(k);

		cv::Mat histImg = img.clone();

		std::vector<uchar> inliers(features1.size(),0);
		cv::Mat homography= cv::findHomography(
			cv::Mat(features1), // corresponding
			cv::Mat(features2), // points
			inliers, // outputted inliers matches
			CV_RANSAC, // RANSAC method
			0.1); // max distance to reprojection point
		for(int i=0; i<inliers.size(); i++)
		{
			if (inliers[i] == 1)
			{
				cv::circle(histImg,features1[i],3,cv::Scalar(0,255,0));
			}
		}

		std::vector<float> histogram;
		std::vector<std::vector<int>> ids;
		OpticalFlowHistogram(features1,features2,histogram,ids);

		//最大bin
		int max =ids[0].size(); 
		int idx(0);
		for(int i=1; i<256; i++)
		{
			if (ids[i].size() > max)
			{
				max = ids[i].size();
				idx = i;
			}
		}
		
		for(int i=0; i<ids[idx].size(); i++)
		{
			cv::circle(histImg,features1[ids[idx][i]],3,cv::Scalar(255,0,0));
		}
		sprintf(fileName,"%sin%06d.jpg",outPathName,i);
		cv::imwrite(fileName,histImg);
		cv::swap(img,preImg);
		cv::swap(gray,preGray);
	}
}
double TestHomoAccuracy(const cv::Mat& gray1, const cv::Mat& gray2, const cv::Mat& homography)
{
	double colorErr(0);
	cv::Mat mask = Mat::zeros(gray1.size(),gray1.type());
	double* ptr = (double*)homography.data;
	std::ofstream fout("inAccuracy.txt");
	for(int i=0; i<gray1.rows; i++)
	{
		for(int j=0; j<gray1.cols; j++)
		{
			uchar color = gray1.data[j + i*gray1.cols];
			float x,y,w;
			x = j*ptr[0] + i*ptr[1] + ptr[2];
			y = j*ptr[3] + i*ptr[4] + ptr[5];
			w = j*ptr[6] + i*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			int wx = int(x+0.5);
			int wy = int(y+0.5);
			/*uchar ucolor = LinearInterData(gray1.cols,gray1.rows,gray2.data,x,y);*/
			if (wx >= 0 && wx<gray1.cols && wy >=0 && wy<gray1.rows)
			{
				colorErr +=abs(color-gray2.data[wx+wy*gray1.cols]);
				if ( abs(color-gray2.data[wx+wy*gray1.cols]) > 30)
				{
					mask.data[j+i*gray1.cols] = 0xff;
					fout<<j<<","<<i<<" = "<<(int)color<<" "<<wx<<","<<wy<<" = "<<(int)gray2.data[wx+wy*gray1.cols]<<std::endl;
				}
			}
			
		}
	}
	cv::imshow("homo accuracy mask",mask);
	fout.close();
	return colorErr;
}

void TestPatchStructralSimilarity()
{
	using namespace cv;
	Mat img1 = imread("..//ptz//input0//in000087.jpg");
	Mat img2 = imread("..//ptz//input0//in000086.jpg");

	GaussianBlur( img1, img1, Size(3,3), 0, 0, BORDER_DEFAULT );
	GaussianBlur( img2, img2, Size(3,3), 0, 0, BORDER_DEFAULT );
	Mat gray1,gray2;
	cvtColor(img1, gray1, CV_BGR2GRAY); 
	cvtColor(img2, gray2, CV_BGR2GRAY);
	
	Mat grad1x,grad2x,grad1y,grad2y;
	Scharr(img1,grad1x,CV_16S,1,0);
	Scharr(img1,grad2x,CV_16S,1,0);
	Scharr(img2,grad1y,CV_16S,0,1);
	Scharr(img2,grad2y,CV_16S,0,1);

	std::vector<cv::Point2f> features1,features2;  // detected features

	int max_count = 5000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 10;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	// detect the features
	cv::goodFeaturesToTrack(gray1, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray1, gray2, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	int k=0;
	for(int i=0; i<features1.size(); i++)
	{
		if (status[i] == 1 /*&& 
			 abs(gray1.data[(int)features1[i].x+(int)features1[i].y*gray1.cols] - gray2.data[(int)features2[i].x+(int)(features2[i].y)*gray1.cols]) < 30*/)
		{
			features2[k] = features2[i];
			features1[k] = features1[i];
			k++;
		}
	}
	features1.resize(k);
	features2.resize(k);
	std::vector<uchar> inliers(features1.size(),0);
	cv::Mat homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
	std::cout<<"homography\n"<<homography<<std::endl;
	double* ptr = (double*)homography.data;
	size_t patchSize = 5;
	size_t hPSize = patchSize/2;
	cv::Mat mask(gray1.size(),gray1.type());
	mask = cv::Scalar(0);
	std::vector<std::vector<float>> hist1,hist2;
	
	ushort p1[3],p2[3];
	//int x(309),y(3);
	//float wx =x*ptr[0] + y*ptr[1] + ptr[2];
	//float wy = x*ptr[3] + y*ptr[4] + ptr[5];
	//float w = x*ptr[6] + y*ptr[7] + ptr[8];
	//wx /=w;
	//wy/=w;
	//
	//computeRGBDescriptor(grad1x,grad1y,x,y,hist1,p1);
	//computeRGBDescriptor(grad2x,grad2y,(int)(wx+0.5),(int)(wy+0.5),hist2,p2);
	//char name[20];
	//for(int c=0; c<3; c++)
	//{
	//	sprintf(name,"hist1_%d",c);
	//	DrawHistogram(hist1[c],hist1[c].size(),name);
	//	sprintf(name,"hist2_%d",c);
	//	DrawHistogram(hist2[c],hist2[c].size(),name);
	//	std::cout<<"hist distance "<<STAT::BinDistance(hist1[c],hist2[c])<<
	//		" p distance "<<hdist_ushort_8bitLUT(p1[c],p2[c])<<std::endl;
	//	std::cout<<"p1["<<c<<"] "<<std::bitset<16>(p1[c])<<std::endl;
	//	std::cout<<"p2["<<c<<"] "<<std::bitset<16>(p2[c])<<std::endl;
	//}
	//cv::circle(img1,cv::Point2f(x,y),3,cv::Scalar(255,0,0));
	//cv::circle(img2,cv::Point2f(wx,wy),3,cv::Scalar(255,0,0));


	for(int i=hPSize; i<gray1.rows-hPSize; i++)
	{
		for(int j=hPSize; j<gray1.cols-hPSize; j++)
		{
			uchar color = gray1.data[j + i*gray1.cols];
			float x,y,w;
			x = j*ptr[0] + i*ptr[1] + ptr[2];
			y = j*ptr[3] + i*ptr[4] + ptr[5];
			w = j*ptr[6] + i*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			int wx = int(x+0.5);
			int wy = int(y+0.5);			
			if (wx >= hPSize && wx<gray1.cols-hPSize && wy >=hPSize && wy<gray1.rows-hPSize)
			{
				
				if (abs(color-gray2.data[wx+wy*gray1.cols])>20)
				{
					ushort p1[3],p2[3];
					computeRGBDescriptor(grad1x,grad1y,j,i,hist1,p1);
					computeRGBDescriptor(grad2x,grad2y,wx,wy,hist2,p2);
					double dist = STAT::BinDistance(hist1,hist2);
					size_t pdist = hdist_ushort_8bitLUT(p1,p2);
					if ( dist < 0.5/*pdist < 12*/ )
					{
						
						cv::circle(img1,cv::Point2f(j,i),3,cv::Scalar(255,0,0));
						cv::circle(img2,cv::Point2f(wx,wy),3,cv::Scalar(255,0,0));
					}
					else
					{
						mask.data[j+i*gray1.cols] = 0xff;
						cv::circle(img1,cv::Point2f(j,i),3,cv::Scalar(0,255,0));
						cv::circle(img2,cv::Point2f(wx,wy),3,cv::Scalar(0,255,0));
					}
					
				}			
			}			
		}
	}
	cv::imshow("img1",img1);
	cv::imshow("img2",img2);
	cv::imshow("mask",mask);
	cv::waitKey();
}
void TestHomographyEstimate()
{
	using namespace cv;
	Mat img1 = imread("..//ptz//input0//in000013.jpg");
	Mat img2 = imread("..//ptz//input0//in000012.jpg");


	Mat gray1,gray2;
	cvtColor(img1, gray1, CV_BGR2GRAY); 
	cvtColor(img2, gray2, CV_BGR2GRAY);
	cv::Mat img3(img1.rows,img2.cols*2,CV_8UC3);
	cv::Mat img4(img1.rows,img2.cols*2,CV_8UC3);
	
	std::vector<cv::Point2f> features1,features2;  // detected features

	int max_count = 5000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 10;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	// detect the features
	cv::goodFeaturesToTrack(gray1, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray1, gray2, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	int k=0;
	for(int i=0; i<features1.size(); i++)
	{
		if (status[i] == 1 /*&& 
			 abs(gray1.data[(int)features1[i].x+(int)features1[i].y*gray1.cols] - gray2.data[(int)features2[i].x+(int)(features2[i].y)*gray1.cols]) < 30*/)
		{
			features2[k] = features2[i];
			features1[k] = features1[i];
			k++;
		}
	}
	features1.resize(k);
	features2.resize(k);
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;
	OpticalFlowHistogram(features1,features2,histogram,ids);
	
	DrawHistogram(histogram,256);
	//最大bin
	int max =ids[0].size(); 
	int idx(0);
	for(int i=1; i<256; i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	cv::Mat histImg = img1.clone();

	//用直方图统计最大的特征点求变换矩阵
	std::vector<cv::Point2f> f1,f2;
	for(int i=0; i<ids[idx].size(); i++)
	{
		cv::circle(histImg,features1[ids[idx][i]],3,cv::Scalar(255,0,0));
		f1.push_back(features1[ids[idx][i]]);
		f2.push_back(features2[ids[idx][i]]);
	}
	cv::Mat dHomo;
	findHomographyDLT(f1,f2,dHomo);
	std::cout<<"dhomo \n"<<dHomo<<std::endl;
	std::cout<<"dhomo error "<<TestHomoAccuracy(gray1,gray2,dHomo)/gray1.rows/gray1.cols<<std::endl;
	cv::imshow("hist max bin",histImg);
	
	
	//{
	//	FILE* file = fopen("data.txt","r");
	//	features1.clear();
	//	features2.clear();
	//	int x,y,wx,wy;
	//	while(fscanf(file,"%d %d %d %d ",&x,&y,&wx,&wy)!=EOF )
	//	{
	//		uchar diff = abs(gray1.data[x+y*gray1.cols] - gray2.data[wx+wy*gray1.cols]);
	//		if (diff>50)
	//			std::cout<<(int)diff<<std::endl;
	//		else
	//		{
	//			features1.push_back(cv::Point2f(x,y));
	//			features2.push_back(cv::Point2f(wx,wy));
	//		}
	//	}
	//	

	//}
	cv::Mat fMatrix,h1,h2;
	fMatrix = cv::findFundamentalMat(features1,features2);
	cv::stereoRectifyUncalibrated(features1,features2,fMatrix,img1.size(),h1,h2);
	cv::Mat rec1,rec2;
	cv::warpPerspective(img1,rec1,h1,img1.size());
	cv::warpPerspective(img2,rec2,h2,img2.size());
	cv::imshow("rec1",rec1);
	cv::imshow("rec2",rec2);
	for(int i=0; i<img1.rows; i++)
	{
		for(int j=0; j<img1.cols; j++)
		{
			uchar* ptr3 = img3.data+ i*img3.step.p[0]+j*img3.step.p[1];
			uchar* ptr4 = img4.data+ i*img4.step.p[0]+j*img4.step.p[1];
			uchar* ptr1 = rec1.data+ i*rec1.step.p[0]+j*rec1.step.p[1];
			uchar* ptr2 = rec2.data+ i*rec2.step.p[0]+j*rec2.step.p[1];
			uchar* iptr1 = img1.data+ i*img1.step.p[0]+j*img1.step.p[1];
			uchar* iptr2 = img2.data+ i*img2.step.p[0]+j*img2.step.p[1];
			for(int c =0; c<3; c++)
			{
				ptr3[c] = ptr1[c];
				ptr4[c] = iptr1[c];
			}
			ptr3 = img3.data+ i*img3.step.p[0]+(img2.cols+j)*img3.step.p[1];
			ptr4 = img4.data+ i*img4.step.p[0]+(img2.cols+j)*img4.step.p[1];
			for(int c =0; c<3; c++)
			{
				ptr3[c] = ptr2[c];
				ptr4[c] = iptr2[c];
			}
			
	
		}
	}
	std::vector<uchar> inliers(features1.size(),0);
	cv::Mat homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
	std::cout<<"homography\n"<<homography<<std::endl;
	double* ptr = (double*)homography.data;
	cv::Mat diffMask(gray1.rows,gray1.cols,CV_8U);
	diffMask = cv::Scalar(0);
	cv::Mat interDiffMask(gray1.rows,gray1.cols,CV_8U);
	interDiffMask = cv::Scalar(0);
	cv::Mat gradDiffMask(gray1.rows,gray1.cols,CV_16U);
	cv::Mat warp,warpDiff;
	cv::Mat featureMask(gray1.rows,gray1.cols,CV_8U);
	featureMask = cv::Scalar(0);
	cv::warpPerspective(gray1,warp,homography,gray1.size());
	cv::absdiff(warp,gray2,warpDiff);
	cv::imshow("warp diff",warpDiff);
	double colorErr(0), interColorErr(0);
	for(int i=0; i<gray1.rows; i++)
	{
		for(int j=0; j<gray1.cols; j++)
		{
			uchar color = gray1.data[j + i*gray1.cols];
			float x,y,w;
			x = j*ptr[0] + i*ptr[1] + ptr[2];
			y = j*ptr[3] + i*ptr[4] + ptr[5];
			w = j*ptr[6] + i*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			int wx = int(x+0.5);
			int wy = int(y+0.5);
			uchar ucolor = LinearInterData(gray1.cols,gray1.rows,gray2.data,x,y);
			if (wx >= 0 && wx<gray1.cols && wy >=0 && wy<gray1.rows)
			{
				colorErr +=abs(color-gray2.data[wx+wy*gray1.cols]);
			/*	if (abs(color-gray2.data[wx+wy*gray1.cols]) > 50)
					std::cout<<j<<" , "<<i<<std::endl;*/
				diffMask.data[j+i*gray1.cols] = abs(color-gray2.data[wx+wy*gray1.cols]);
				interDiffMask.data[j+i*gray1.cols] = abs(color-ucolor);
				interColorErr +=abs(color-ucolor);
			}
			
		}
	}
	cv::imshow("diffMask",diffMask);
	cv::imshow("interDiffMask",interDiffMask);
	std::cout<<"avg color error total "<<colorErr/img1.rows/img1.cols<<std::endl;
	std::cout<<"avg interpolation color error total "<<interColorErr/img1.rows/img1.cols<<std::endl;
	k = 0;
	double avgerr = 0;
	int c = 0;
	for(int i=0; i<inliers.size(); i++)
	{
		cv::Point2f features3 = warpPoint(features2[i],(double*)h2.data);
		features3.x = features3.x+img2.cols;
		cv::Point2f wf1 = warpPoint(features1[i],(double*)h1.data);
		if (wf1.x<img1.cols && features3.x > img1.cols)
		{
			cv::circle(img3,features3,3,cv::Scalar(255,0,0));
			cv::circle(img3,wf1,3,cv::Scalar(255,0,0));
			cv::line(img3,wf1,features3,cv::Scalar(0,255,0));
		}
		features3 = cv::Point2f(features2[i].x+gray1.cols,features2[i].y);
		cv::circle(img4,features3,3,cv::Scalar(255,0,0));
		cv::circle(img4,features1[i],3,cv::Scalar(255,0,0));
		cv::line(img4,features1[i],features3,cv::Scalar(0,255,0));
		if (inliers[i]==1)
		{
			
			cv::circle(img1,features1[i],3,cv::Scalar(255,0,0));
			cv::circle(img2,features2[i],3,cv::Scalar(255,0,0));
			
			
			float x,y,w;
			float fx,fy;
			fx = features1[i].x;
			fy = features1[i].y;
			x = fx*ptr[0] + fy*ptr[1] + ptr[2];
			y = fx*ptr[3] + fy*ptr[4] + ptr[5];
			w = fx*ptr[6] + fy*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			double dx = x-features2[i].x;
			double dy = y- features2[i].y;
			uchar icolor = LinearInterData(gray2.cols,gray2.rows,gray2.data,features2[i].x,features2[i].y);
			//uchar diff = abs(gray1.data[(int)fx+(int)fy*gray1.cols] - gray2.data[(int)features2[i].x+(int)(features2[i].y)*gray1.cols]);
			uchar diff = abs(gray1.data[(int)fx+(int)fy*gray1.cols] - icolor);
			/*if (diff > 50)
			{
				
			}*/
			featureMask.data[(int)fx+(int)fy*gray1.cols] = diff < 35 ? diff : 255;
			avgerr += sqrt(dx*dx+dy*dy);
			c++;
		}
		else
		{
			features1[k] = features1[i];
			features2[k] = features2[i];
			k++;
			/*cv::circle(img3,features1[i],3,cv::Scalar(255,0,0));
			cv::circle(img3,features3,3,cv::Scalar(255,0,0));
			cv::line(img3,features1[i],features3,cv::Scalar(255,255,255));*/
		}
	}
	std::cout<<"avg distance  err "<<avgerr/c<<std::endl;
	cv::dilate(featureMask,featureMask,cv::Mat(),cv::Point(-1,-1),2);
	cv::imshow("featureMask",featureMask);
	features1.resize(k);
	features2.resize(k);
	inliers.resize(k);
	homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
	
	k=0;
	for(int i=0; i<inliers.size(); i++)
	{
		if (inliers[i]==1)
		{
			cv::circle(img1,features1[i],3,cv::Scalar(0,0,255));
			cv::circle(img2,features2[i],3,cv::Scalar(0,0,255));
		}
		else
		{
			features1[k] = features1[i];
			features2[k] = features2[i];
			k++;
		}
	}
	diffMask = cv::Scalar(0);
	ptr = (double*)homography.data;
	
	for(int i=0; i<gray1.rows; i++)
	{
		for(int j=0; j<gray1.cols; j++)
		{
			uchar color = gray1.data[j + i*gray1.cols];
			float x,y,w;
			x = j*ptr[0] + i*ptr[1] + ptr[2];
			y = j*ptr[3] + i*ptr[4] + ptr[5];
			w = j*ptr[6] + i*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			int wx = int(x+0.5);
			int wy = int(y+0.5);

			if (wx >= 0 && wx<gray1.cols && wy >=0 && wy<gray1.rows)
			{
				diffMask.data[j+i*gray1.cols] = abs(color-gray2.data[wx+wy*gray1.cols]);
			}
			
		}
	}
	cv::imshow("diffMask2",diffMask);

	
	cv::namedWindow("img1");
	cv::imshow("img1",img1);
	cv::namedWindow("img2");
	cv::imshow("img2",img2);

	cv::imshow("img3",img3);
	cv::imshow("img4",img4);
	cv::waitKey();
}
void TestPerspective()
{
	using namespace cv;
	Mat img1 = imread("..//PTZ//input3//in000289.jpg");
	Mat img2 = imread("..//PTZ//input3//in000290.jpg");


	Mat gray1,gray2;
	cvtColor(img1, gray1, CV_BGR2GRAY); 
	cvtColor(img2, gray2, CV_BGR2GRAY);

	cv::GaussianBlur(gray1,gray1,cv::Size(3,3),0.1);
	cv::GaussianBlur(gray2,gray2,cv::Size(3,3),0.1);

	std::vector<cv::Point2f> features1,features2;  // detected features

	int max_count = 5000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 10;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	// detect the features
	cv::goodFeaturesToTrack(gray1, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray1, gray2, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	std::vector<uchar> inliers(features1.size(),0);
	cv::Mat homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.5); // max distance to reprojection point
	double* ptr = (double*)homography.data;
	cv::Mat diffMask(gray1.rows,gray1.cols,CV_8U);
	diffMask = cv::Scalar(0);
	cv::Mat diffMinMask;
	diffMask.copyTo(diffMinMask);
	cv::Mat gradient1,gradient2;
	cv::Sobel(gray1,gradient1,0,1,1);
	cv::Sobel(gray2,gradient2,0,1,1);
	for(int i=0; i<gray1.rows; i++)
	{
		for(int j=0; j<gray1.cols; j++)
		{
			float x,y,w;
			x = j*ptr[0] + i*ptr[1] + ptr[2];
			y = j*ptr[3] + i*ptr[4] + ptr[5];
			w = j*ptr[6] + i*ptr[7] + ptr[8];
			x /=w;
			y/=w;
			int wx = int(x+0.5);
			int wy = int(y+0.5);
			uchar color = gray1.data[j+i*gray1.cols];
			uchar grad = gradient1.data[j+i*gray1.cols];
			uchar gradTh = 100;
			int wwx = wx;
			int wwy = wy;
			if (gradient1.data[j+i*gray1.cols] > gradTh || gradient2.data[wx + wy*gray1.cols] > gradTh)
			{

				//在s*s的区域内搜索与原图像最接近的点
				int s = 2;
				float alpha = 1;
				float min = 16384;

				for(int m=-s; m<=s; m++)
				{
					for(int n=-s; n<=s; n++)
					{
						int mx = m+wx;
						int ny = n+wy;
						if (mx >=0 && mx<gray1.cols && ny>=0 && ny<gray1.rows)
						{
							int idx = mx+ny*gray1.cols;
							float diff = std::abs(gray2.data[idx] - color) + (1-alpha)*std::abs(gradient2.data[idx]-grad);
							if (diff<min)
							{
								min = diff;
								wwx = mx;
								wwy = ny;
							}
						}
					}
				}
			}
			if (wx >= 0 && wx<gray1.cols && wy >=0 && wy<gray1.rows)
			{
				diffMask.data[j+i*gray1.cols] = abs(color-gray2.data[wx+wy*gray1.cols]);
			}
			if (wwx >= 0 && wwx<gray1.cols && wwy >=0 && wwy<gray1.rows)
			{
				diffMinMask.data[j+i*gray1.cols] = abs(color-gray2.data[wwx+wwy*gray1.cols]);
			}
		}
	}
	cv::imshow("diffMask",diffMask);
	cv::imshow("diffMinMask",diffMinMask);
	Mat affine = estimateRigidTransform(features1,features2,true);
	Mat Affineresult;
	warpAffine(gray1,Affineresult,affine,cv::Size(gray1.cols,gray1.rows));


	// Warp image 1 to image 2
	cv::Mat result;
	cv::warpPerspective(gray1, // input image
		result,			// output image
		homography,		// homography
		cv::Size(gray1.cols,gray1.rows)); // size of output image

	double * M = (double*) homography.data;
	int width = gray1.cols;
	int height = gray1.rows;
	cv::Mat warpMask(height,width,CV_8U);
	warpMask = cv::Scalar(0);
	cv::Mat warpEMask;
	warpMask.copyTo(warpEMask);
	double x,y,w;
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			x = M[0]*i + M[1]*j + M[2];
			y  = M[3]*i + M[4]*j + M[5];
			w = M[6]*i + M[7]*j + M[8];
			w = w? 1.0/w : 1;
			x *= w;
			y *= w;
			int ix = int(x+0.5);
			int iy = int(y+0.5);
			uchar interColor = LinearInterData(width,height,gray2.data,x,y);
			if (ix >=0 && ix < width && iy >=0 && iy < height)
			{
				int idx = i+j*width;
				int widx = ix + iy*width;
				warpEMask.data[idx] = abs(gray2.data[widx] - gray1.data[idx]);
				warpMask.data[idx] = abs(interColor - gray1.data[idx]);
			}
		}
	}
	cv::imshow("warpError",warpEMask);
	cv::imshow("warpMask",warpMask);
	Mat ErrImg = gray2 - result;
	Mat ErrAImg = gray2 - Affineresult;
	// Display the warp image
	cv::namedWindow("After warping");
	cv::imshow("After warping",result);

	cv::namedWindow("img1");
	cv::imshow("img1",gray1);
	cv::namedWindow("img2");
	cv::imshow("img2",gray2);

	cv::namedWindow("error");
	cv::imshow("error",ErrImg);
	cv::Mat absDiff;
	cv::absdiff(gray2,result,absDiff);
	std::cout<<"perspective abs diff ="<<cv::sum(absDiff)<<std::endl;
	cv::absdiff(gray2,Affineresult,absDiff);
	std::cout<<"affine abs diff ="<<cv::sum(absDiff)<<std::endl;
	cv::waitKey();
}
struct EdgePoint
{
	EdgePoint(int _x, int _y, float _theta,float _color):x(_x),y(_y),theta(_theta),color(_color)
	{}
	int x;
	int y;
	float color;//3×3邻域的平均颜色
	float theta;//角度，0~180
};
//row,col为中心的size*size区域平均颜色
float AvgColor(const Mat& img, int row, int col, int size=3)
{
	Mat gray = img;
	if (img.channels() == 3)
		cvtColor(img,gray,CV_BGR2GRAY);
	int width = img.cols;
	int height = img.rows;
	int step = size/2;
	size_t avgColor = 0;
	int c = 0;
	for(int i= -step; i<=step; i++)
	{
		for(int j=-step; j<=step; j++)
		{
			int x = col+ i;
			int y = row +j;
			if ( x>=0 && x<width && y>=0 && y<height)
			{
				int idx = y*width + x;
				avgColor += gray.data[idx];
				c++;
			}
		}
	}
	avgColor/=c;
	return avgColor;
}
//提取边缘点
void ExtractEdgePoint(const Mat& img, double tr1,double tr2, const Mat& edge, Mat& edgeThetaMat,std::vector<EdgePoint>& edgePoints)
{
	edgePoints.clear();
	Mat dx,dy;	
	edgeThetaMat = cv::Scalar(0);
	cv::Sobel(img,dx,0,1,0);
	cv::Sobel(img,dy,0,0,1);
	for(int i=0; i< img.rows; i++)
	{
		for(int j=0; j<img.cols; j++)
		{
			int idx = i*img.cols + j;
			if (edge.data[idx] == 0xff)
			{
				float theta = atan(dy.data[idx]*1.0/(dx.data[idx]+1e-6))/M_PI*180;
				/*std::cout<<theta<<std::endl;*/
				float avgColor = AvgColor(img,i,j);
				float* ptr = edgeThetaMat.ptr<float>(i)+2*j;
				*ptr= theta;
				*(ptr+1) = avgColor;
				edgePoints.push_back(EdgePoint(j,i,theta,avgColor));
			}			
		}
	}
}

//边缘点匹配
void MapEdgePoint(const Mat& gray,const std::vector<EdgePoint>& ePoints1, const Mat& edge2,const Mat& edgeThetamat, const  Mat& transform, float deltaTheta, Mat& matchMask)
{
	double * ptr = (double*)transform.data;
	int r = 1;//搜素范围
	int width = edge2.cols;
	int height = edge2.rows;
	matchMask = Mat(edge2.size(),CV_8U);
	matchMask = Scalar(0);
	float thetaDist = 0.5;
	float colorDist = 20;
	 for(int i=0; i<ePoints1.size(); i++)
	 {
		 int ox = ePoints1[i].x;
		 int oy = ePoints1[i].y;
		 float theta = ePoints1[i].theta;
		 float color = ePoints1[i].color;
		 float x,y,w;
		 x = ox*ptr[0] + oy*ptr[1] + ptr[2];
		 y = ox*ptr[3] + oy*ptr[4] + ptr[5];
		 w = ox*ptr[6] + oy*ptr[7] + ptr[8];
		 x /=w;
		 y/=w;
		 int wx = int(x+0.5);
		 int wy = int(y+0.5);
		 for(int m=-r; m<=r; m++)
		 {
			 for(int n=-r; n<=r; n++)
			 {
				 int nx  = wx + m;
				 int ny = wy + n;
				 if (nx>=0 && nx<width && ny >=0 && ny<height)
				 {
					 int id = nx + ny*width;
					 int tid = ny*edgeThetamat.step.p[0]+nx*edgeThetamat.step.p[1];
					 float* angColorPtr = (float*)(edgeThetamat.data+tid);
				
					 if (edge2.data[id]==255  && 
						abs( angColorPtr[0] - theta-deltaTheta) < thetaDist &&
						 abs(AvgColor(gray,oy,ox) - angColorPtr[1]) < colorDist)
					 {
						 //match
						 matchMask.data[ox+oy*width] = UCHAR_MAX;
					 }
				 }
			 }
		 }
	 }
}
//边缘点匹配
void MapEdgePoint(const Mat& gray,const std::vector<EdgePoint>& ePoints1, const Mat& edge2,const Mat& edgeThetamat, const  Mat& transform, float deltaTheta, Mat& matchMask,Mat& unmatchMask)
{
	double * ptr = (double*)transform.data;
	int r = 1;//搜素范围
	int width = edge2.cols;
	int height = edge2.rows;
	matchMask = Mat(edge2.size(),CV_8U);
	matchMask = Scalar(0);
	float thetaDist = 0.5;
	float colorDist = 20;
	 for(int i=0; i<ePoints1.size(); i++)
	 {
		 int ox = ePoints1[i].x;
		 int oy = ePoints1[i].y;
		 float theta = ePoints1[i].theta;
		 float color = ePoints1[i].color;
		 float x,y,w;
		 x = ox*ptr[0] + oy*ptr[1] + ptr[2];
		 y = ox*ptr[3] + oy*ptr[4] + ptr[5];
		 w = ox*ptr[6] + oy*ptr[7] + ptr[8];
		 x /=w;
		 y/=w;
		 int wx = int(x+0.5);
		 int wy = int(y+0.5);
		 bool matched = false;;
		 for(int m=-r; m<=r; m++)
		 {
			 for(int n=-r; n<=r; n++)
			 {
				 int nx  = wx + m;
				 int ny = wy + n;
				 if (nx>=0 && nx<width && ny >=0 && ny<height)
				 {
					 int id = nx + ny*width;
					 int tid = ny*edgeThetamat.step.p[0]+nx*edgeThetamat.step.p[1];
					 float* angColorPtr = (float*)(edgeThetamat.data+tid);
				
					 if (edge2.data[id]==255  && 
						abs( angColorPtr[0] - theta-deltaTheta) < thetaDist &&
						 abs(AvgColor(gray,oy,ox) - angColorPtr[1]) < colorDist)
					 {
						 //match
						 matchMask.data[ox+oy*width] = UCHAR_MAX;
						 matched = true;
					 }					 
				 }
			 }
		 }
		 if (!matched)
			 unmatchMask.data[ox+oy*width] = UCHAR_MAX;
	 }
}
void GetTransformMatrix(const cv::Mat& gray, const cv::Mat pre_gray, std::vector<cv::Point2f>& features1,std::vector<cv::Point2f>& features2,
	cv::Mat& homoM, cv::Mat& affineM, std::vector<uchar>& inliers)
{
	int max_count = 50000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 2;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	// detect the features
	cv::goodFeaturesToTrack(gray, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray, pre_gray, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	int k=0;

	for( int i= 0; i < features1.size(); i++ ) 
	{

		// do we keep this point?
		if (status[i] == 1) 
		{

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
			// keep this point in vector
			features1[k] = features1[i];
			features2[k++] = features2[i];
		}
	}
	features1.resize(k);
	features2.resize(k);

	inliers.resize(k);
	memset(&inliers[0],0,sizeof(char)*k);
	homoM= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point

	affineM = estimateRigidTransform(features1,features2,true);
}
void TestEdgeTracking(cv::Mat& imgC, cv::Mat& imgPre)
{
	Mat affineM,homoM;
	Mat pre_gray,gray,pre_thetaMat,thetaMat,pre_edge,edge;
	cv::cvtColor(imgC,gray,CV_BGR2GRAY);
	cv::cvtColor(imgPre,pre_gray,CV_BGR2GRAY);
	std::vector<EdgePoint> pre_edgePoints,edgePoints;
	std::vector<uchar> inliers;
	double tr1(120.f),tr2(300.f);
	std::vector<cv::Point2f> features1,features2;
	cv::Canny(gray,edge,tr1,tr2);	
	thetaMat = Mat(gray.size(),CV_32FC2);
	pre_thetaMat = Mat(gray.size(),CV_32FC2);
	ExtractEdgePoint(gray,tr1,tr2,edge,thetaMat,edgePoints);
	cv::Canny(pre_gray,pre_edge,tr1,tr2);	
	
	ExtractEdgePoint(pre_gray,tr1,tr2,pre_edge,pre_thetaMat,pre_edgePoints);
	GetTransformMatrix(gray,pre_gray,features1,features2,homoM,affineM,inliers);
	double theta = atan(affineM.at<double>(1,0)/affineM.at<double>(1,1))/M_PI*180;
	std::cout<<"theta = "<<theta<<std::endl;
	cv::imshow("current edge",edge);
	cv::imshow("previous edge", pre_edge);
	
	Mat mask(gray.size(),CV_8U);
	mask = Scalar(0);
	Mat umask(gray.size(),CV_8U);
	umask = Scalar(0);
	MapEdgePoint(gray,edgePoints,pre_edge,pre_thetaMat,homoM,theta, mask,umask);
	cv::cvtColor(mask,mask,CV_GRAY2BGR);
	for(int i=0; i<inliers.size(); i++)
	{
		if (inliers[i] == 1)		
			cv::circle(mask,features1[i],2,cv::Scalar(255,0,0));
	}
	imwrite("edgeMatched.jpg",mask);
	cv::imshow("matched",mask);
	
	cv::imshow("unmatched",umask);
	cv::waitKey();
}
void TestEdgeTracking2Img()
{
	cv::Mat imgC = cv::imread("..//moseg//cars1//in000003.jpg");
	cv::Mat imgPre = cv::imread("..//moseg//cars1//in000001.jpg");
	TestEdgeTracking(imgC,imgPre);
}
void TestDynamicTexture()
{
	int start = 1; 
	int end = 18;
	char outPathName[100];
	char fileName[100];
	sprintf(outPathName,".//DT//moseg//cars1//");
	CreateDir(outPathName);
	cv::Mat img1,img2,gray1,gray2;
	std::vector<cv::Point2f> features1,features2;  // detected features
	int max_count = 5000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 10;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	int x(50), y(50);
	for(int i=start; i<=end;i++)
	{
		sprintf(fileName,"..//moseg//cars1//in%06d.jpg",i);
		img1 = cv::imread(fileName);
		cv::cvtColor(img1,gray1,CV_BGR2GRAY);
		sprintf(fileName,"..//moseg//cars1//in%06d.jpg",i+1);
		img2 = cv::imread(fileName);
		cv::cvtColor(img2,gray2,CV_BGR2GRAY);
		// detect the features
		cv::goodFeaturesToTrack(gray1, // the image 
			features1,   // the output detected features
			max_count,  // the maximum number of features 
			qlevel,     // quality level
			minDist);   // min distance between two features

		// 2. track features
		cv::calcOpticalFlowPyrLK(gray1, gray2, // 2 consecutive images
			features1, // input point position in first image
			features2, // output point postion in the second image
			status,    // tracking success
			err);      // tracking error

		int k=0;
		for(int i=0; i<features1.size(); i++)
		{
			if (status[i] == 1 /*&& 
							   abs(gray1.data[(int)features1[i].x+(int)features1[i].y*gray1.cols] - gray2.data[(int)features2[i].x+(int)(features2[i].y)*gray1.cols]) < 30*/)
			{
				features2[k] = features2[i];
				features1[k] = features1[i];
				k++;
			}
		}
		features1.resize(k);
		features2.resize(k);
		std::vector<uchar> inliers(features1.size(),0);
		cv::Mat homography= cv::findHomography(
			cv::Mat(features1), // corresponding
			cv::Mat(features2), // points
			inliers, // outputted inliers matches
			CV_RANSAC, // RANSAC method
			0.1); // max distance to reprojection point
		//std::cout<<"homography\n"<<homography<<std::endl;
		double* ptr = (double*)homography.data;
		float wx,wy,w;
		wx = x*ptr[0] + y*ptr[1] + ptr[2];
		wy = x*ptr[3] + y*ptr[4] + ptr[5];
		w = x*ptr[6] + y*ptr[7] + ptr[8];
		wx/=w;
		wy/=w;
		Mat img = imread(fileName);
		cv::Mat texture2 = img2(cv::Rect((int)wx,(int)wy,32,32));
		sprintf(fileName,"%stexture%06d.jpg",outPathName,i);
		cv::imwrite(fileName,texture2);
		cv::Mat texture1 = img1(cv::Rect(x,y,32,32));
		sprintf(fileName,"%sotexture%06d.jpg",outPathName,i);
		cv::imwrite(fileName,texture1);
		sprintf(fileName,"%stexture_diff%06d.jpg",outPathName,i);
		cv::Mat diffTexture;
		cv::absdiff(texture1,texture2,diffTexture);
		cv::imwrite(fileName,diffTexture);
	}
}
void GetHomography(const cv::Mat& gray, const cv::Mat& pre_gray, cv::Mat& homography)
{
	int max_count = 50000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 2;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	std::vector<cv::Point2f> features1,features2;
	// detect the features
	cv::goodFeaturesToTrack(gray, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray, pre_gray, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	int k=0;

	for( int i= 0; i < features1.size(); i++ ) 
	{

		// do we keep this point?
		if (status[i] == 1) 
		{

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
			// keep this point in vector
			features1[k] = features1[i];
			features2[k++] = features2[i];
		}
	}
	features1.resize(k);
	features2.resize(k);

	std::vector<uchar> inliers(features1.size());
	homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point

}
void WarpPosition(const int x, const int y, int&wx, int& wy, const cv::Mat& homography)
{
	double* ptr = (double*)homography.data;
	float fx,fy,w;
	fx = x*ptr[0] + y*ptr[1] + ptr[2];
	fy = x*ptr[3] + y*ptr[4] + ptr[5];
	w = x*ptr[6] + y*ptr[7] + ptr[8];
	fx /=w;
	fy/=w;
	wx = int(fx+0.5);
	wy = int(fy+0.5);			

}
void TestVLBP()
{	
	cv::Mat img1,img2,img3;
	img1 = cv::imread("..//ptz//input0//in000086.jpg");
	img2 = cv::imread("..//ptz//input0//in000087.jpg");
	img3 = cv::imread("..//ptz//input0//in000088.jpg");
	cv::Mat gray1,gray2,gray3;
	cv::cvtColor(img1,gray1,CV_BGR2GRAY);
	cv::cvtColor(img2,gray2,CV_BGR2GRAY);
	cv::cvtColor(img3,gray3,CV_BGR2GRAY);
	cv::Mat h1,h2;
	GetHomography(gray2,gray1,h1);
	GetHomography(gray2,gray3,h2);
	int x2(180),y2(131);
	int x1,y1,x3,y3;
	cv::Mat mask(gray1.size(),gray1.type());
	mask = cv::Scalar(0);
	cv::Mat mask2 = mask.clone();
	for(int i=2; i<gray1.cols-2; i++)
	{
		for(int j=2; j<gray1.rows-2; j++)
		{
			x2 = i;
			y2 = j;
			/*WarpPosition(x2,y2,x1,y1,h1);
			WarpPosition(x2,y2,x3,y3,h2);*/
			x1 = x3 = x2;
			y1 = y3 = y2;
			if (x1>2 && x3>2 && y1>2 && y3>2 && x1<gray1.cols-2 &&  x3< gray1.cols-2 && y1<gray1.rows-2 && y3< gray1.rows-2)
			{
				ushort cres1[3],cres2[3];
				VLBP::computeVLBPRGBDescriptor(img2,x2,y2,img1,x1,y1,cres1);
				VLBP::computeVLBPRGBDescriptor(img3,x3,y3,img2,x2,y2,cres2);
				if (hdist_ushort_8bitLUT(cres1,cres2) > 15)
					mask.data[j*gray1.cols+i] = 0xff;
				ushort res1,res2;
				VLBP::computeLBPGrayscaleDescriptor(gray3,x3,y3,res1);
				VLBP::computeLBPGrayscaleDescriptor(gray2,x2,y2,res2);
				if (hdist_ushort_8bitLUT(res1,res2) > 2)
					mask2.data[j*gray1.cols+i] = 0xff;
			}
			/*std::cout<<std::bitset<16>(res1)<<std::endl;
			std::cout<<std::bitset<16>(res2)<<std::endl;*/
		}
	}
	cv::imshow("mask",mask);
	cv::imshow("mask2",mask2);
	cv::waitKey();

}
void TestListEdgeTracking()	
{
	char fileName[150];
	const size_t LIST_SIZE = 3;
	size_t trackDist = 3;
	Mat pre_gray,gray,pre_thetaMat,thetaMat,pre_edge,edge;
	std::vector<EdgePoint> pre_edgePoints,edgePoints;
	std::list<Mat> grayList;
	std::list<Mat> edgeList;
	std::list<Mat> thetaMatList;
	std::list<std::vector<EdgePoint>> edgePointList;
	double tr1(120.f),tr2(300.f);
	
	int start = 1; 
	int end = 19;
	char outPathName[100];
	sprintf(outPathName,"..\\result\\subsensex\\moseg\\cars1\\features\\");
	CreateDir(outPathName);
	for(int i=start; i<=end;i++)
	{
		sprintf(fileName,"..//moseg//cars1//in%06d.jpg",i);
		Mat img = imread(fileName);
		cvtColor(img, gray, CV_BGR2GRAY); 
		cv::Canny(gray,edge,tr1,tr2);
		thetaMat.create(gray.size(),CV_32FC2);	
		ExtractEdgePoint(gray,tr1,tr2,edge,thetaMat,edgePoints);
		if (edgeList.size()< LIST_SIZE)
		{
			grayList.push_back(gray.clone());
			thetaMatList.push_back(thetaMat.clone());
			edgeList.push_back(edge.clone());
			edgePointList.push_back(edgePoints);
		}
		else if(edgeList.size() == LIST_SIZE)
		{
			grayList.pop_front();
			grayList.push_back(gray.clone());
			thetaMatList.pop_front();
			thetaMatList.push_back(thetaMat.clone());
			edgeList.pop_front();
			edgeList.push_back(edge.clone());
			edgePointList.pop_front();
			edgePointList.push_back(edgePoints);

		}
		if(edgeList.size() < trackDist)
		{
			edge.copyTo(pre_edge);
			thetaMat.copyTo(pre_thetaMat);
			pre_edgePoints = edgePoints;
			gray.copyTo(pre_gray);
		}
		else
		{
			size_t id = LIST_SIZE-trackDist;
			std::list<cv::Mat>::iterator itr = std::next(edgeList.begin(), id);			
			pre_edge = *itr;
			itr = std::next(grayList.begin(),id);
			pre_gray = *itr;
			itr = std::next(thetaMatList.begin(),id);
			pre_thetaMat = *(itr);
			std::list<vector<EdgePoint>>::iterator eitr = std::next(edgePointList.begin(),id);
			pre_edgePoints = *(eitr);
		}
		cv::Mat affineM,homoM;
		std::vector<uchar> inliers;
		std::vector<cv::Point2f> features1,features2;
		GetTransformMatrix(gray,pre_gray,features1,features2,homoM,affineM,inliers);
		//std::cout<<affine<<std::endl;
		if (affineM.empty())
		{
			std::cout<<"affine is empty"<<std::endl;
			imwrite("empty_edge.jpg",edge);
			imwrite("empty_preEdge.jpg",pre_edge);
			continue;
		}
		double theta = atan(affineM.at<double>(1,0)/affineM.at<double>(1,1))/M_PI*180;
		
		Mat mask(gray.size(),CV_8U);
		mask = Scalar(0);
		MapEdgePoint(gray,edgePoints,pre_edge,pre_thetaMat,homoM,theta, mask);
		/*if (i == 256)
		{
			imwrite("edge.jpg",edge);
			imwrite("preEdge.jpg",pre_edge);
			imwrite("gray.jpg",gray);
			imwrite("preGray.jpg",pre_gray);
		}*/
		cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1),2);
		sprintf(fileName,"%sfeatures%06d.jpg",outPathName,i);
		imwrite(fileName,mask);
		/*swap(pre_gray,gray);
		swap(pre_edge,edge);
		swap(pre_thetaMat,thetaMat);
		swap(pre_edgePoints,edgePoints);*/
	}
	
	
}

void TestEdgeTracking()
{
	char fileName[150];
	Mat pre_gray,gray,pre_thetaMat,thetaMat,pre_edge,edge;
	std::vector<EdgePoint> pre_edgePoints,edgePoints;
	double tr1(120.f),tr2(300.f);
	std::vector<cv::Point2f> features1,features2;  // detected features
	int max_count = 50;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 10;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	int start = 1; 
	int end = 100;
	char outPathName[100];
	sprintf(outPathName,"edgeTracking\\features\\");
	CreateDir(outPathName);
	for(int i=start; i<=end;i++)
	{
		sprintf(fileName,"..//PTZ//input0//in%06d.jpg",i);
		Mat img = imread(fileName);
		cvtColor(img, gray, CV_BGR2GRAY); 
		cv::Canny(gray,edge,tr1,tr2);
	
		ExtractEdgePoint(gray,tr1,tr2,edge,thetaMat,edgePoints);

		if (pre_thetaMat.empty())
		{
			edge.copyTo(pre_edge);
			thetaMat.copyTo(pre_thetaMat);
			pre_edgePoints = edgePoints;
			gray.copyTo(pre_gray);
		}
		cv::Mat affineM,homoM;
		std::vector<uchar> inliers;
		std::vector<cv::Point2f> features1,features2;
		GetTransformMatrix(gray,pre_gray,features1,features2,homoM,affineM,inliers);
		//std::cout<<affine<<std::endl;
		

		double theta = atan(affineM.at<double>(1,0)/affineM.at<double>(1,1))/M_PI*180;
		//std::cout<<theta<<std::endl;
	
		Mat mask(gray.size(),CV_8U);
		mask = Scalar(0);
		MapEdgePoint(gray,edgePoints,pre_edge,pre_thetaMat,homoM,theta, mask);
		/*if (i == 89)
		{
			imwrite("edge.jpg",edge);
			imwrite("preEdge.jpg",pre_edge);
		}*/
		/*cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1),2);*/
		sprintf(fileName,"%sfeatures%06d.jpg",outPathName,i);
		imwrite(fileName,mask);
		swap(pre_gray,gray);
		swap(pre_edge,edge);
		swap(pre_thetaMat,thetaMat);
		swap(pre_edgePoints,edgePoints);
	}
	
	
}
void postProcess(const Mat& img, Mat& mask, Mat& imgDst, Mat& dst)
{
	cv::Mat m_oFGMask_PreFlood(img.size(),CV_8U);
	cv::Mat m_oFGMask_FloodedHoles(img.size(),CV_8U);
	cv::morphologyEx(mask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_or(mask,m_oFGMask_FloodedHoles,mask);
	cv::bitwise_or(mask,m_oFGMask_PreFlood,mask);
	cv::medianBlur(mask,mask,3);
	
}
void postProcessSegments(const Mat& img, Mat& mask, Mat& imgDst, Mat& dst)
{
	int niters = 3;

	vector<vector<Point> > contours,imgContours;
	vector<Vec4i> hierarchy,imgHierarchy;
	
	Mat temp;
	
	Mat edge(img.size(),CV_8U);	
	cv::Canny(img,edge,100,300);
	dilate(edge, edge, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
	erode(edge, edge, Mat(), Point(-1,-1), niters*2);//腐蚀
	dilate(edge, edge, Mat(), Point(-1,-1), niters);
	findContours( edge, imgContours, imgHierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//找轮廓

	dilate(mask, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
	erode(temp, temp, Mat(), Point(-1,-1), niters*2);//腐蚀
	dilate(temp, temp, Mat(), Point(-1,-1), niters);
	
	findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//找轮廓
	
	dst = Mat::zeros(img.size(), CV_8UC3);
	imgDst = Mat::zeros(img.size(), CV_8UC3);
	if( contours.size() == 0 )
		return;

	
	double minArea = 10*10;
	Scalar color( 255, 255, 255 );
	for( int i = 0; i< contours.size(); i++ )
	{
		const vector<Point>& c = contours[i];
		double area = fabs(contourArea(Mat(c)));
		if( area > minArea )
		{
			drawContours( dst, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );
			
		}
		
	}
	for( int i = 0; i< imgContours.size(); i++ )
	{
		drawContours( imgDst, imgContours, i, color, 1, 8, imgHierarchy, 0, Point() );
	}
	cv::cvtColor(dst, dst, CV_BGR2GRAY); 
	cv::cvtColor(imgDst, imgDst, CV_BGR2GRAY); 
}
void TestPostProcess()
{
	int start = 1; 
	int end = 1130;
	char outPathName[100];
	sprintf(outPathName,"..\\result\\subsensex\\ptz\\input3\\postprocess\\");
	CreateDir(outPathName);
	char fileName[100];
	for(int i=start; i<=end;i++)
	{
		sprintf(fileName,"..//result//subsensex//PTZ//input3//bin%06d.png",i);
		Mat mask = imread(fileName);
		if(mask.channels() ==3)
			cv::cvtColor(mask,mask,CV_BGR2GRAY);
		sprintf(fileName,"..//PTZ//input3//in%06d.jpg",i);
		cv::Mat img = cv::imread(fileName);
		if(img.channels()==3)
			cvtColor(img,img,CV_BGR2GRAY);
		cv::Mat dst,imgDst;
		/*(mask.size(),mask.type());
		cv::Mat imgDst(mask.size(),mask.type());*/
		//postProcessSegments(img,mask,imgDst,dst);
		postProcess(img,mask,imgDst,dst);
		sprintf(fileName,"%sbin%06d.png",outPathName,i);
		imwrite(fileName,mask);
		/*sprintf(fileName,"%spi%06d.jpg",outPathName,i);
		imwrite(fileName,imgDst);*/
	}
}
void TestfindHomographyDLT()
{
	int width(640),height(480);
	int n = 5;
	std::vector<cv::Point2f> f1,f2;
	//for(int i=0; i<5; i++)
	//{
		//f1.push_back(cv::Point2f(rand()%width,rand()%height);

	//}
	f1.push_back(cv::Point2f(139,23));
	f1.push_back(cv::Point2f(434,326));
	f1.push_back(cv::Point2f(599,185));
	f1.push_back(cv::Point2f(332,400));
	f1.push_back(cv::Point2f(22,25));

	f2.push_back(cv::Point2f(282,240));
	f2.push_back(cv::Point2f(357,223));
	f2.push_back(cv::Point2f(607,458));
	f2.push_back(cv::Point2f(272,146));
	f2.push_back(cv::Point2f(33,23));
	cv::Mat homography;
	findHomographyDLT(f1,f2,homography);
	std::cout<<homography;
}
//void TestImageRetification()
//{
//	int start = 2; 
//	int end = 2;
//	char outPathName[100];
//	sprintf(outPathName,"..\\result\\subsensex\\ptz\\input3\\postprocess\\");
//	CreateDir(outPathName);
//	char fileName[100];
//	cv::Mat curImg,prevImg,curGray, prevGray;
//	std::vector<cv::Point2f> features0,features1;
//	for(int i=start; i<=end;i++)
//	{
//		sprintf(fileName,"..//PTZ//input3//bin%06d.png",i);
//		curImg = cv::imread(filename);
//		sprintf(fileName,"..//PTZ//input3//bin%06d.png",i-1);
//		prevImg = cv::imread(filename);
//		cv::cvtColor(curImg,curGray,CV_BGR2GRAY);
//		cv::cvtColor(prevImg,prevGray,CV_BGR2GRAY);
//		cv::goodFeaturesToTrack(curGray, features0, 5000,0.05,10);
//		cv:
//		
//	}
//}
int main()
{
	TestVLBP();
	//TestDynamicTexture();
	//TestPatchStructralSimilarity();
	//TestOpticalFlowHistogram();
	//TestfindHomographyDLT();
	//TestHomographyEstimate();
	//TestPerspective();	
	//TestPostProcess();
	//TestEdgeTracking2Img();
	//TestListEdgeTracking();
	//TestLBP();
	//TestPerspective();	
	//TestAffine();
	return 0;
	// Create video procesor instance
	VideoProcessor processor;

	// Create feature tracker instance
	MotionTracker tracker;
	//MCDBSProcessor tracker;
	//VibeProcessor tracker;
	std::vector<std::string> fileNames;
	for(int i=1; i<=1700;i++)
	{
		char name[50];
		sprintf(name,"..\\PTZ\\input0\\in%06d.jpg",i);
		//sprintf(name,"..\\PTZ\\input4\\drive1_%03d.png",i);
		fileNames.push_back(name);
	}
	// Open video file
	//processor.setInput(fileNames);
	processor.setInput("..\\MCD\\pets_2005_1.avi");
	// set frame processor
	processor.setFrameProcessor(&tracker);

	// Declare a window to display the video
	processor.displayOutput("Tracked Features");

	// Play the video at the original frame rate
	//processor.setDelay(1000./processor.getFrameRate());
	processor.setDelay(1000/25);

	// Start the process
	processor.run();

	cv::waitKey();

}