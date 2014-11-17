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
	EdgePoint(int _x, int _y, float _theta):x(_x),y(_y),theta(_theta)
	{}
	int x;
	int y;
	float theta;//角度，0~180
};
//提取边缘点
void ExtractEdgePoint(const Mat& img, double tr1,double tr2, const Mat& edge, Mat& edgeThetaMat,std::vector<EdgePoint>& edgePoints)
{
	edgePoints.clear();
	Mat dx,dy;
	edgeThetaMat = Mat(img.size(),CV_32FC1);
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
				*(float*)(edgeThetaMat.data+idx*4) = theta;
				edgePoints.push_back(EdgePoint(j,i,theta));
			}
		}
	}

}

//边缘点匹配
void MapEdgePoint(const std::vector<EdgePoint>& ePoints1, const Mat& edge2,const Mat edgeThetamat, const const Mat& transform, float deltaTheta, Mat& matchMask)
{
	double * ptr = (double*)transform.data;
	int r = 1;//搜素范围
	int width = edge2.cols;
	int height = edge2.rows;
	matchMask = Mat(edge2.size(),CV_8U);
	matchMask = Scalar(0);
	float thetaDist = 0.5;
	 for(int i=0; i<ePoints1.size(); i++)
	 {
		 int ox = ePoints1[i].x;
		 int oy = ePoints1[i].y;
		 float theta = ePoints1[i].theta;

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
					 if (edge2.data[id]==255 && abs( *(float*)(edgeThetamat.data+id*4) - theta-deltaTheta) < thetaDist)
					 {
						 //match
						 matchMask.data[ox+oy*width] = UCHAR_MAX;
					 }
				 }
			 }
		 }
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
	sprintf(outPathName,"..\\result\\subsensem\\ptz\\input0\\features\\");
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

		Mat affine = estimateRigidTransform(features1,features2,true);
		//std::cout<<affine<<std::endl;

		double theta = atan(affine.at<double>(1,0)/affine.at<double>(1,1))/M_PI*180;
		//std::cout<<theta<<std::endl;
		std::vector<uchar> inliers(features1.size(),0);
		cv::Mat homography= cv::findHomography(
			cv::Mat(features1), // corresponding
			cv::Mat(features2), // points
			inliers, // outputted inliers matches
			CV_RANSAC, // RANSAC method
			0.5); // max distance to reprojection point

		Mat mask(gray.size(),CV_8U);
		mask = Scalar(0);
		MapEdgePoint(edgePoints,pre_edge,pre_thetaMat,homography,theta, mask);
		if (i == 89)
		{
			imwrite("edge.jpg",edge);
			imwrite("preEdge.jpg",pre_edge);
		}
		cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1),2);
		sprintf(fileName,"%sfeatures%06d.jpg",outPathName,i);
		imwrite(fileName,mask);
		swap(pre_gray,gray);
		swap(pre_edge,edge);
		swap(pre_thetaMat,thetaMat);
		swap(pre_edgePoints,edgePoints);
	}
	
	
}
int main()
{
	TestEdgeTracking();
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