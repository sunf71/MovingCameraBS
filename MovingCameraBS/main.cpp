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

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "Affine2D.h"
#include "motiontracker.h"
#include <algorithm>





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
		return ty*((1-tx)*data[sx+sy*width]+tx*data[bx+sy*width]) + (1-ty)*((1-tx)*data[sx+by*width] + tx*data[bx+by*width]);

	}
	else
		return 0;
}
<<<<<<< HEAD

void TestPerspective()
{
	using namespace cv;
	Mat img1 = imread("..//PTZ//input0//in000089.jpg");
	Mat img2 = imread("..//PTZ//input0//in000090.jpg");
=======
void TestPerspective()
{
	using namespace cv;
	Mat img1 = imread("..//PTZ//input3//in000289.jpg");
	Mat img2 = imread("..//PTZ//input3//in000290.jpg");
>>>>>>> e72b57787f309af8f2698afa49a6f23632be9714

	Mat gray1,gray2;
	cvtColor(img1, gray1, CV_BGR2GRAY); 
	cvtColor(img2, gray2, CV_BGR2GRAY);

	cv::GaussianBlur(gray1,gray1,cv::Size(3,3),0.1);
	cv::GaussianBlur(gray2,gray2,cv::Size(3,3),0.1);

	std::vector<cv::Point2f> features1,features2;  // detected features

	int max_count = 500;	  // maximum number of features to detect
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

	std::vector<uchar> inliers(features1.size(),0);
	cv::Mat homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		1.); // max distance to reprojection point
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

			//在s*s的区域内搜索与原图像最接近的点
			int s = 3;
			float alpha = 1;
			float min = 16384;
			int wwx = wx;
			int wwy = wy;
			for(int m=-s; m<s; m++)
			{
				for(int n=-s; n<s; n++)
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
int main()
{
	TestPerspective();	
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