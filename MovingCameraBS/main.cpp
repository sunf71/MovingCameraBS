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
void DrawHistogram(std::vector<float>& histogram, int size)
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
	cv::imshow("histogram",img);
	//cv::waitKey();
}
void TestOpticalFlowHistogram()
{
	int start = 1; 
	int end = 1130;
	char outPathName[100];
	char fileName[100];
	sprintf(outPathName,".\\histogram\\ptz\\input3\\features\\");
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
		sprintf(fileName,"..\\ptz\\input3\\in%06d.jpg",i);
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
		cv::Mat histImg = img.clone();
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
void TestHomographyEstimate()
{
	using namespace cv;
	Mat img1 = imread("..//moseg//people1//in000003.jpg");
	Mat img2 = imread("..//moseg//people1//in000002.jpg");


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
		if (status[i] == 1)
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
	for(int i=0; i<ids[idx].size(); i++)
	{
		cv::circle(histImg,features1[ids[idx][i]],3,cv::Scalar(255,0,0));
	}
	cv::imshow("hist max bin",histImg);

	/*{
		FILE* file = fopen("data.txt","r");
		features1.clear();
		features2.clear();
		int x,y,wx,wy;
		while(fscanf(file,"%d %d %d %d ",&x,&y,&wx,&wy)!=EOF )
		{
			uchar diff = abs(gray1.data[x+y*gray1.cols] - gray2.data[wx+wy*gray1.cols]);
			if (diff>50)
				std::cout<<(int)diff<<std::endl;
			else
			{
				features1.push_back(cv::Point2f(x,y));
				features2.push_back(cv::Point2f(wx,wy));
			}
		}
		

	}*/
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
			uchar diff = abs(gray1.data[(int)fx+(int)fy*gray1.cols] - gray2.data[(int)features2[i].x+(int)(features2[i].y)*gray1.cols]);
			/*if (diff > 50)
			{
				
			}*/
			featureMask.data[(int)fx+(int)fy*gray1.cols] = diff < 255 ? diff : 255;
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
void TestCSHAnn()
{

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
	TestOpticalFlowHistogram();
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