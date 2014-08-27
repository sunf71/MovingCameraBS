#pragma once

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "videoprocessor.h"
#include "VIBE.h"
#include "prob_model.h"
using namespace cv;
void refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
	int niters = 3;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat temp;

	dilate(mask, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
	erode(temp, temp, Mat(), Point(-1,-1), niters*2);//腐蚀
	dilate(temp, temp, Mat(), Point(-1,-1), niters);

	findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//找轮廓

	dst = Mat::zeros(img.size(), CV_8UC3);

	if( contours.size() == 0 )
		return;

	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	int idx = 0, largestComp = 0;
	double maxArea = 0;

	for( ; idx >= 0; idx = hierarchy[idx][0] )//这句没怎么看懂
	{
		const vector<Point>& c = contours[idx];
		double area = fabs(contourArea(Mat(c)));
		if( area > maxArea )
		{
			maxArea = area;
			largestComp = idx;//找出包含面积最大的轮廓
		}
	}
	Scalar color( 255, 255, 255 );
	drawContours( dst, contours, largestComp, color, CV_FILLED, 8, hierarchy );
	cv::cvtColor(dst, dst, CV_BGR2GRAY); 
}
Mat InvAffineMatrix(const Mat affine)
{
	Mat A(2,2,affine.type());
	for(int i=0; i<2; i++)
		for(int j=0;j<2;j++)
			A.at<double>(i,j) = affine.at<double>(i,j);
	A = A.inv();
	Mat invAffine(2,3,affine.type());
	for(int i=0; i<2; i++)
		for(int j=0;j<2;j++)
			invAffine.at<double>(i,j) = A.at<double>(i,j);
	Mat B(2,1,affine.type());
	B.at<double>(0,0) = affine.at<double>(0,2)*-1;
	B.at<double>(0,1) = affine.at<double>(1,2)*-1;
	B = A*B;
	invAffine.at<double>(0,2) = B.at<double>(0,0);
	invAffine.at<double>(1,2) = B.at<double>(0,1);
	return invAffine;
}
class MotionTracker : public FrameProcessor 
{
private:
	cv::Mat gray;			// current gray-level image
	cv::Mat gray_prev;		// previous gray-level image
	std::vector<cv::Point2f> points[2]; // tracked features from 0->1
	std::vector<cv::Point2f> initial;   // initial position of tracked points
	std::vector<cv::Point2f> features;  // detected features
	int max_count;	  // maximum number of features to detect
	double qlevel;    // quality level for feature detection
	double minDist;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	bool ViBeInitialized;
	ViBe_BGS ViBe1;
	ViBe_BGS ViBe;
	BackgroundSubtractorMOG2 bg_model;
	bool UpdateViBe;
	cv::Mat mask_prev;

public:

	MotionTracker() : max_count(500), qlevel(0.01), minDist(10.), ViBeInitialized(false),UpdateViBe(false) {}

	// processing method
	void process(cv:: Mat &frame, cv:: Mat &output) {

		static int frameNo = 0;

		// convert to gray-level image
		cv::cvtColor(frame, gray, CV_BGR2GRAY); 
		frame.copyTo(output);
		if( !ViBeInitialized)
		{
			ViBe.init(gray);
			ViBe.processFirstFrame(gray);


			ViBe1.init(gray);
			ViBe1.processFirstFrame(gray);
			ViBeInitialized = true;
			mask_prev =  Mat::zeros(cv::Size(gray.cols,gray.rows),CV_8U);
		}

		// 1. if new feature points must be added
		if(addNewPoints())
		{
			UpdateViBe = true;
			// detect feature points
			detectFeaturePoints();
			// add the detected features to the currently tracked features
			points[0].insert(points[0].end(),features.begin(),features.end());
			initial.insert(initial.end(),features.begin(),features.end());
		}

		// for first image of the sequence
		if(gray_prev.empty())
			gray.copyTo(gray_prev);

		// 2. track features
		cv::calcOpticalFlowPyrLK(gray_prev, gray, // 2 consecutive images
			points[0], // input point position in first image
			points[1], // output point postion in the second image
			status,    // tracking success
			err);      // tracking error

		// 2. loop over the tracked points to reject the undesirables
		int k=0;

		for( int i= 0; i < points[1].size(); i++ ) {

			// do we keep this point?
			if (acceptTrackedPoint(i)) {

				// keep this point in vector
				points[0][k] = points[0][i];
				initial[k]= initial[i];
				points[1][k++] = points[1][i];
			}
		}

		// eliminate unsuccesful points
		points[0].resize(k);
		points[1].resize(k);
		initial.resize(k);

		// 3. handle the accepted tracked points
		handleTrackedPoints(frame, output);
		//handleTrackedPointsGMM(frame,output);
		// 4. current points and image become previous ones
		std::swap(points[1], points[0]);
		cv::swap(gray_prev, gray);
		frameNo++;
	}

	// feature point detection
	void detectFeaturePoints() {

		// detect the features
		cv::goodFeaturesToTrack(gray, // the image 
			features,   // the output detected features
			max_count,  // the maximum number of features 
			qlevel,     // quality level
			minDist);   // min distance between two features
	}

	// determine if new points should be added
	bool addNewPoints() {

		// if too few points
		return points[0].size()<=500*0.75;
	}

	// determine which tracked point should be accepted
	bool acceptTrackedPoint(int i) {
		return status[i] == 1;
		//return status[i] &&
		//	// if point has moved
		//	(abs(points[0][i].x-points[1][i].x)+
		//	(abs(points[0][i].y-points[1][i].y))>2);
	}

	// handle the currently tracked points
	void handleTrackedPoints(cv:: Mat &frame, cv:: Mat &output) {
		static int frameNo = 1;
		//vibe
		/*if (ViBeInitialized)
		{
		ViBe.testAndUpdate(gray);
		cv::Mat mask = ViBe.getMask();
		morphologyEx(mask, mask, MORPH_OPEN, Mat());
		imshow("mask", mask);
		char fileName[20];
		sprintf(fileName,"bin%06d.png",frameNo++);
		imwrite(fileName,mask);
		}*/

		//perspective transform
		std::vector<uchar> inliers(points[0].size(),0);
		cv::Mat homography= cv::findHomography(
			cv::Mat(points[1]), // corresponding
			cv::Mat(initial), // points
			inliers, // outputted inliers matches
			CV_RANSAC, // RANSAC method
			1.); // max distance to reprojection point
		// Warp image 1 to image 2
		cv::Mat result;
		cv::warpPerspective(gray, // input image
			result,			// output image
			homography,		// homography
			cv::Size(frame.cols,frame.rows)); // size of output image

		//affine transform
		//Mat affine;
		//std::vector<uchar> in(initial.size(),0);
		//estimateAffine2D(points[1],initial,affine,in);
		/*affine = estimateRigidTransform(points[1],initial,true);
		if (affine.cols != 3 || affine.rows !=2)
		estimateAffine2D(points[1],initial,affine,in);
		Mat result;
		warpAffine(gray,result,affine,cv::Size(gray.cols,gray.rows));*/

		/*char fileName[20];
		sprintf(fileName,"warp%06d.png",frameNo++);
		imwrite(fileName,result);*/

		// Display the warp image
		cv::namedWindow("After warping");
		cv::imshow("After warping",result);

		Mat warpMask(cv::Size(result.cols,result.rows),CV_8U);
		for(int i=0; i<result.rows; i++)
		{
			for(int j=0; j<result.cols; j++)
			{
				if (result.at<uchar>(i,j) == 0)
					warpMask.at<uchar>(i,j) = 0;
				else
					warpMask.at<uchar>(i,j) = 255;

			}
		}

		imshow("warp mask",warpMask);
		//vibe after warping
		if (ViBeInitialized)
		{	
			if (UpdateViBe)
			{
				ViBe1.processFirstFrame(gray);
				UpdateViBe = false;
			}

			ViBe1.testAndUpdate(result);
			cv::Mat mask2 = ViBe1.getMask();
			morphologyEx(mask2, mask2, MORPH_OPEN, Mat());
			/*Mat invAffine= InvAffineMatrix(affine);
			warpAffine(mask2,mask2,invAffine,cv::Size(mask2.cols,mask2.rows));*/
			Mat invHomo = homography.inv();
			warpPerspective(mask2,mask2,invHomo,cv::Size(mask2.cols,mask2.rows));
			//refineSegments(gray,mask2,mask2);
			imshow("maskAffine", mask2);
			char fileName[50];
			sprintf(fileName,".\\perspective\\input0\\bin%06d.png",frameNo++);
			imwrite(fileName,mask2);
			mask_prev = warpMask;
		}


		//// for all tracked points
		//for(int i= 0; i < points[1].size(); i++ ) {

		//	// draw line and circle
		//    cv::line(output, initial[i], points[1][i], cv::Scalar(255,255,255));
		//	cv::circle(output, points[1][i], 3, cv::Scalar(255,255,255),-1);
		//	//cv::circle(output,prev[i],3,cv::Scalar(255,0,0));
		//}
	}


	void handleTrackedPointsGMM(cv::Mat & frame, cv::Mat &output)
	{
		//perspective transform
		std::vector<uchar> inliers(points[0].size(),0);
		cv::Mat homography= cv::findHomography(
			cv::Mat(points[1]), // corresponding
			cv::Mat(initial), // points
			inliers, // outputted inliers matches
			CV_RANSAC, // RANSAC method
			1.); // max distance to reprojection point
		// Warp image 1 to image 2
		cv::Mat result;
		cv::warpPerspective(gray, // input image
			result,			// output image
			homography,		// homography
			cv::Size(frame.cols,frame.rows)); // size of output image

		//affine transform
		//Mat affine;
		//std::vector<uchar> in(initial.size(),0);
		//estimateAffine2D(points[1],initial,affine,in);
		/*affine = estimateRigidTransform(points[1],initial,true);
		if (affine.cols != 3 || affine.rows !=2)
		estimateAffine2D(points[1],initial,affine,in);
		Mat result;
		warpAffine(gray,result,affine,cv::Size(gray.cols,gray.rows));*/

		/*char fileName[20];
		sprintf(fileName,"warp%06d.png",frameNo++);
		imwrite(fileName,result);*/

		// Display the warp image
		cv::namedWindow("After warping");
		cv::imshow("After warping",result);

		Mat warpMask(cv::Size(result.cols,result.rows),CV_8U);
		for(int i=0; i<result.rows; i++)
		{
			for(int j=0; j<result.cols; j++)
			{
				if (result.at<uchar>(i,j) == 0)
					warpMask.at<uchar>(i,j) = 0;
				else
					warpMask.at<uchar>(i,j) = 255;

			}
		}

		imshow("warp mask",warpMask);

		Mat mask;
		bg_model(result,mask);
		Mat invHomo = homography.inv();
		warpPerspective(mask,mask,invHomo,cv::Size(mask.cols,mask.rows));
		imshow("gmm mask",mask);
	}
};
class VibeProcessor : public FrameProcessor
{
private:
	ViBe_BGS _vibe;
	bool _initFlag;
public:
	VibeProcessor():_initFlag(false){}
	void process(cv:: Mat &frame, cv:: Mat &output) 
	{
		cv::Mat gray;
		static int frameNo = 1;
		// convert to gray-level image
		cv::cvtColor(frame, gray, CV_BGR2GRAY); 

		if (!_initFlag)
		{
			_vibe.init(gray);
			_vibe.processFirstFrame(gray);
			_initFlag = true;
		}
		_vibe.testAndUpdate(gray);
		cv::Mat mask = _vibe.getMask();
		char fileName[50];
		sprintf(fileName,"..\\result\\vibe\\input0\\bin%06d.png",frameNo++);
		imwrite(fileName,mask);
		imshow("mask",mask);
		output = frame.clone();
	}
};
class MCDBSProcessor : public FrameProcessor 
{
private:
	CProbModel _model;
	std::vector<cv::Point2f> points[2];
	cv::Mat gray;			// current gray-level image
	cv::Mat gray_prev;		// previous gray-level image
	bool _initFlag;
	double _qlevel;    // quality level for feature detection
	double _minDist;   // minimum distance between two feature points
	std::vector<uchar> _status; // status of tracked features
	std::vector<float> _err;    // error in tracking
	IplImage image;
public:
	MCDBSProcessor():_initFlag(false),_qlevel(0.01), _minDist(10.){}
	void preprocess(cv::Mat& input, cv::Mat& output)
	{
		GaussianBlur(input,output,cvSize(3,3),0.1);
		medianBlur(output,output,3);
	}
	void process(cv:: Mat &frame, cv:: Mat &output) 
	{
		static int frameNo = 1;
		// convert to gray-level image
		cv::cvtColor(frame, gray, CV_BGR2GRAY); 
		preprocess(gray,gray);
		//preprocess guass filtering and median filtering

		const unsigned blockX = 24;
		const unsigned blockY = 32;

		gray.copyTo(output);
		if ( !_initFlag )
		{
			image = gray;
			_model.init(&image);
			unsigned rSize = gray.rows/blockX;
			unsigned cSize = gray.cols/blockY;
			for(int i=0; i<=gray.rows/blockX; i++)
			{
				for(int j=0; j<=gray.cols/blockY; j++)
					points[0].push_back(Point2f(j*blockY,i*blockX));
			}
			_initFlag = true;
		}



		// for first image of the sequence
		if(gray_prev.empty())
			gray.copyTo(gray_prev);

		// 2. track features
		cv::calcOpticalFlowPyrLK(gray_prev, gray, // 2 consecutive images
			points[0], // input point position in first image
			points[1], // output point postion in the second image
			_status,    // tracking success
			_err);      // tracking error
		// 2. loop over the tracked points to reject the undesirables
		int k=0;
		std::vector<cv::Point2f> prev;
		prev.reserve(points[1].size());
		for( int i= 0; i < points[1].size(); i++ ) {

			// do we keep this point?
			if (_status[i] == 1) {

				// keep this point in vector
				prev.push_back(points[0][i]);
				points[1][k++] = points[1][i];
			}
		}

		// eliminate unsuccesful points
		points[1].resize(k);
		std::vector<uchar> inliers(k,0);
		cv::Mat homography= cv::findHomography(
			cv::Mat(points[1]), // corresponding
			cv::Mat(prev), // points
			inliers, // outputted inliers matches
			CV_RANSAC, // RANSAC method
			1.);
		for(int i=0; i<prev.size(); i++)
		{
			cv::circle(output,prev[i],2,cv::Scalar(255,0,0));
			cv::circle(output,points[1][i],2,cv::Scalar(0,255,0));
			cv::line(output, prev[i],points[1][i],cv::Scalar(255,255,255));
		}

		cv::Mat result;
		cv::warpPerspective(gray, // input image
			result,			// output image
			homography,		// homography
			cv::Size(frame.cols,frame.rows)); // size of output image

		cv::namedWindow("After warping");
		cv::imshow("After warping",result);
		imshow("original",frame);
		IplImage iplimage = output;
		/*std::cout<<homography<<std::endl;
		double *ptr = (double*)homography.data;
		for(int i=0; i<9; i++,ptr++)
		std::cout<<*ptr<<std::endl;*/
		_model.motionCompensate((double*)homography.data);
		_model.update(&iplimage);
		char fileName[50];
		sprintf(fileName,"..\\MCD\\result\\input0\\bin%06d.png",frameNo++);
		imwrite(fileName,output);
		cv::swap(gray_prev, gray);


	}
};