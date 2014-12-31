#define _USE_MATH_DEFINES
#include "SubSenseBSProcessor.h"
#include "BGSMovieMaker.h"
#include "timer.h"
#include "ASAPWarping.h"
#include <fstream>
#include <opencv2/features2d/features2d.hpp>
#include "SparseSolver.h"
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <math.h>
using namespace cv;
void FeaturePointsRefineRANSAC(std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2)
{
	std::vector<uchar> inliers(vf1.size());
	cv::Mat homography = cv::findHomography(
		cv::Mat(vf1), // corresponding
		cv::Mat(vf2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
	int k=0;
	for(int i=0; i<vf1.size(); i++)
	{
		if (inliers[i] ==1)
		{
			vf1[k] = vf1[i];
			vf2[k] = vf2[i];
			k++;
		}
	}
	vf1.resize(k);
	vf2.resize(k);
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
void FeaturePointsRefineHistogram(std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>& features2)
{
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
	int k=0;
	for(int i=0; i<ids[idx].size(); i++)
	{
		features1[k] = features1[ids[idx][i]];
		features2[k] = features2[ids[idx][i]];
		k++;
	}
	
	features1.resize(k);
	features2.resize(k);
}
void KLTFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2)
{
	std::vector<uchar>status;
	std::vector<float> err;
	cv::Mat sGray,tGray;
	cv::cvtColor(simg,sGray,CV_BGR2GRAY);
	cv::cvtColor(timg,tGray,CV_BGR2GRAY);
	cv::goodFeaturesToTrack(sGray,vf1,5000,0.05,10);
	cv::calcOpticalFlowPyrLK(sGray,tGray,vf1,vf2,status,err);
	int k=0;
	for(int i=0; i<vf1.size(); i++)
	{
		if(status[i] == 1)
		{
			vf1[k] = vf1[i];
			vf2[k] = vf2[i];
			k++;
		}
	}

	vf1.resize(k);
	vf2.resize(k);
	FeaturePointsRefineHistogram(vf1,vf2);
}
void FILESURFFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2)
{
	FILE* f1 = fopen("f1.txt","r");
	FILE* f2 = fopen("f2.txt","r");	
	float x,y;
	while(fscanf(f1,"%f\t%f",&x,&y) > 0)
	{
		
		vf1.push_back(cv::Point2f(x-1,y-1));
		
	}
	while(fscanf(f2,"%f\t%f",&x,&y)>0)
	{
		
		vf2.push_back(cv::Point2f(x-1,y-1));
		
	}
	fclose(f1);
	fclose(f2);
}
void SURFFeaturesMatching(const cv::Mat& simg, const cv::Mat& timg, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2)
{
	using namespace cv;
	Mat img_1,img_2;
	cvtColor(simg,img_1,CV_BGR2GRAY);
	cvtColor(timg,img_2,CV_BGR2GRAY);

	
	//-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
  
    SurfFeatureDetector detector( minHessian );
  
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
  
    detector.detect( img_1, keypoints_object );
    detector.detect( img_2, keypoints_scene );
  
    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
  
    Mat descriptors_object, descriptors_scene;
  
    extractor.compute( img_1, keypoints_object, descriptors_object );
    extractor.compute( img_2, keypoints_scene, descriptors_scene );
  
    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );
  
    double max_dist = 0; double min_dist = 100;
  
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
  
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
  
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
  
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
       { good_matches.push_back( matches[i]); }
    }
  
   
  
    
    for( int i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      vf1.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      vf2.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
}
void TestASAPWarping()
{
	//cv::Mat img = cv::imread("..\\ptz\\input0\\in000001.jpg");
	//Mesh mesh(img.rows,img.cols,img.cols/8,img.rows/8);
	//cv::Mat nimg;
	//mesh.drawMesh(img,0,nimg);
	//cv::imshow("mesh",nimg);
	//cv::waitKey();
	cv::Mat simg = cv::imread(".//5//s.png");
	cv::Mat timg = cv::imread(".//5//t.png");
	int width = simg.cols;
	int height = simg.rows;
	int quadStep = 8;
	ASAPWarping asap(width,height,quadStep,1.0);
	
	std::vector<cv::Point2f>vf1,vf2;
	KLTFeaturesMatching(simg,timg,vf1,vf2);
	//FILESURFFeaturesMatching(simg,timg,vf1,vf2);
	//SURFFeaturesMatching(simg,timg,vf1,vf2);
	asap.SetControlPts(vf1,vf2);
	asap.Solve();
	cv::Mat wimg;
	asap.Warp(simg,wimg);
	cv::imshow("s",simg);
	cv::imshow("t",timg);
	cv::imshow("warped",wimg);
	cv::imwrite(".//5//twarped.jpg",wimg);
	cv::Mat diff;
	cv::absdiff(wimg,timg,diff);
	cv::imshow("diff",diff);
	cv::imwrite(".//5//diff.jpg",diff);
	std::vector<cv::Mat>& homo = asap.getHomographies();
	std::ofstream of("homographies.txt");
	for(int i=0; i<homo.size(); i++)
	{
		of<<homo[i]<<std::endl;
	}
	of.close();
	cv::waitKey();
}
void TestCVSolve()
{
	cv::Mat mat(19,3,CV_32F);
	float* ptr = mat.ptr<float>(0);
	ptr[0] = 0, ptr[1] = 0, ptr[2] = 2;
	ptr = mat.ptr<float>(1);
	ptr[0] = 0, ptr[1] = 1, ptr[2] = 3;
	ptr = mat.ptr<float>(2);
	ptr[0] = 0, ptr[1] = 2, ptr[2] = 0;
	ptr = mat.ptr<float>(3);
	ptr[0] = 0, ptr[1] = 3, ptr[2] = 0;
	ptr = mat.ptr<float>(4);
	ptr[0] = 0, ptr[1] = 4, ptr[2] = 0;
	ptr = mat.ptr<float>(5);
	ptr[0] = 1, ptr[1] = 0, ptr[2] = 3;
	ptr = mat.ptr<float>(6);
	ptr[0] = 1, ptr[1] = 1, ptr[2] = 0;
	ptr = mat.ptr<float>(7);
	ptr[0] = 1, ptr[1] = 2, ptr[2] = 4;
	ptr = mat.ptr<float>(8);
	ptr[0] = 1, ptr[1] = 3, ptr[2] = 0;
	ptr = mat.ptr<float>(9);
	ptr[0] = 1, ptr[1] = 4, ptr[2] = 6;
	ptr = mat.ptr<float>(10);
	ptr[0] = 2, ptr[1] = 0, ptr[2] = 0;
	ptr = mat.ptr<float>(11);
	ptr[0] = 2, ptr[1] = 1, ptr[2] = -1;
	ptr = mat.ptr<float>(12);
	ptr[0] = 2, ptr[1] = 2, ptr[2] = -3;
	ptr = mat.ptr<float>(13);
	ptr[0] = 2, ptr[1] = 3, ptr[2] = 2;
	ptr = mat.ptr<float>(14);
	ptr[0] = 2, ptr[1] = 4, ptr[2] = 0;
	ptr = mat.ptr<float>(15);
	ptr[0] = 3, ptr[1] = 2, ptr[2] = 1;
	ptr = mat.ptr<float>(16);
	ptr[0] = 4, ptr[1] = 1, ptr[2] = 4;
	ptr = mat.ptr<float>(17);
	ptr[0] = 4, ptr[1] = 2, ptr[2] = 2;
	ptr = mat.ptr<float>(18);
	ptr[0] = 4, ptr[1] =4, ptr[2] = 1;
	double b [ ] = {8., 45., -3., 3., 19.} ;
	std::vector<double> rhs(b,b+5);

	std::vector<double> result;
	SolveSparse(mat,rhs,result);

	for(int i=0; i<result.size(); i++)
		std::cout<<result[i]<<std::endl;
}
int main()
{
	//TestCVSolve();
	TestASAPWarping();
	/*BGSMovieMaker::MakeMovie("..\\result\\subsense\\ptz\\input0","..\\ptz\\input0",cv::Size(704,480),1,1700,"continuousPan_s.avi");*/
	return 0;
	VideoProcessor processor;
	
	// Create feature tracker instance
	SubSenseBSProcessor tracker;
	std::vector<std::string> fileNames;
	int start =180;
	int end = 1130;
	char name[50];
	for(int i=start; i<=end;i++)
	{		
		sprintf(name,"..\\ptz\\input3\\in%06d.jpg",i);
		//sprintf(name,"..\\PTZ\\input4\\drive1_%03d.png",i);
		fileNames.push_back(name);
	}
	// Open video file
	processor.setInput(fileNames);
	//processor.setInput("..\\ptz\\woman.avi");
	// set frame processor
	processor.setFrameProcessor(&tracker);

	processor.dontDisplay();
	// Declare a window to display the video
	//processor.displayOutput("Tracked Features");

	// Play the video at the original frame rate
	//processor.setDelay(1000./processor.getFrameRate());
	processor.setDelay(0);

	nih::Timer timer;
	timer.start();
	// Start the process
	processor.run();
	timer.stop();

	std::cout<<(end-start+1)/timer.seconds()<<" fps"<<std::endl;
	
	//BGSMovieMaker::MakeMovie("..\\result\\subsensem\\ptz\\input3","..\\ptz\\input3",cv::Size(320,240),1,1130,"zoominzoomout.avi");
	cv::waitKey();


	return 0;
}