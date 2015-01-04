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
	FeaturePointsRefineHistogram(vf1,vf2);
	asap.SetControlPts(vf1,vf2);
	asap.Solve();
	cv::Mat wimg;
	asap.Warp(simg,wimg);
	cv::Mat iwimg;
	cv::remap(timg,iwimg,asap.getInvMapX(),asap.getInvMapY(),CV_INTER_CUBIC);
	cv::imshow("inv Warped",iwimg);
	cv::imshow("s",simg);
	cv::imshow("t",timg);
	cv::imshow("warped",wimg);
	cv::imwrite(".//5//twarped.jpg",wimg);
	cv::imwrite(".//5//swarped.jpg",iwimg);
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
	//TestASAPWarping();
	/*BGSMovieMaker::MakeMovie("..\\result\\subsense\\ptz\\input0","..\\ptz\\input0",cv::Size(704,480),1,1700,"continuousPan_s.avi");*/
	//return 0;
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