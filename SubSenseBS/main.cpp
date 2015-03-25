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
	cv::Mat simg = cv::imread("..\\particle\\vperson\\in000121.jpg");
	cv::Mat timg = cv::imread("..\\particle\\vperson\\in000120.jpg");
	int width = simg.cols;
	int height = simg.rows;
	int quadStep = 8;
	ASAPWarping asap(width,height,quadStep,1.0);
	
	std::vector<cv::Point2f>vf1,vf2;
	KLTFeaturesMatching(simg,timg,vf1,vf2);
	//FILESURFFeaturesMatching(simg,timg,vf1,vf2);
	//SURFFeaturesMatching(simg,timg,vf1,vf2);
	/*cv::Mat homography;
	FeaturePointsRefineRANSAC(vf1,vf2,homography);*/
	FeaturePointsRefineHistogram(simg.cols,simg.rows,vf1,vf2);
	cv::Mat mr;
	MatchingResult(simg,timg,vf1,vf2,mr);
	cv::imshow("matching result", mr);
	nih::Timer timer;
	timer.start();
	asap.SetControlPts(vf1,vf2);
	asap.Solve();
	cv::Mat wimg;
	asap.Warp(simg,wimg);
	timer.stop();
	std::cout<<"ASAP Warping "<<timer.seconds()<<" s\n";
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
	float fb [] = {8., 45., -3., 3., 19.} ; 
	std::vector<double> rhs(b,b+5);

	std::vector<double> result;
	SolveSparse(mat,rhs,result);
	std::vector<float> fresult;
	cv::Mat bm(5,1,CV_32F);
	for(int i=0; i<5; i++)
	bm.at<float>(i,0) = fb[i];
	LeastSquareSolve(5,5,mat,bm,fresult);
	for(int i=0; i<result.size(); i++)
		std::cout<<result[i]<<std::endl;
	std::cout<<"cula result\n";
	for(int i=0; i<fresult.size(); i++)
		std::cout<<fresult[i]<<std::endl;
}
void MovieMakeMain(int argc, char* argv[])
{
	if (argc == 8)
	{
		BGSMovieMaker::MakeMovie(argv[1],argv[2],cv::Size(atoi(argv[3]),atoi(argv[4])),atoi(argv[5]),atoi(argv[6]),argv[7]);
	}
	else if(argc ==9)
	{
		BGSMovieMaker::MakeMovie(argv[1],argv[2],cv::Size(atoi(argv[3]),atoi(argv[4])),atoi(argv[5]),atoi(argv[6]),argv[7],atoi(argv[8]));
	}
	else
	{
		printf("const char* maskPath,const char* imgPath,int width, int height, int from, int to, const char* outFileName, int fps = 25\n");
	}

}
int main(int argc, char* argv[])
{
	MovieMakeMain(argc,argv);
	//TestCVSolve();
	//TestASAPWarping();
	//BGSMovieMaker::MakeMovie("..\\result\\subsensex\\ptz\\input3\\warpbaseline","..\\ptz\\input3",cv::Size(320,240),1,1130,"zoominzoomout_s.avi");
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