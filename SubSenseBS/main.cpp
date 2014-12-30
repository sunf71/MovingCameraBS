#include "SubSenseBSProcessor.h"
#include "BGSMovieMaker.h"
#include "timer.h"
#include "ASAPWarping.h"
#include <fstream>
#include "opencv2/features2d/features2d.hpp"
#include "SparseSolver.h"
using namespace cv;
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
	std::vector<uchar> inliers(vf1.size());
	cv::Mat homography = cv::findHomography(
		cv::Mat(vf1), // corresponding
		cv::Mat(vf2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
	k=0;
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
void SURFFeaturesMatching(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::Point2f>& vf1, std::vector<cv::Point2f>& vf2)
{
	using namespace cv;
	//第一步，用SURF算子检测关键点
    int minHessian=400;

    SurfFeatureDetector detector(minHessian);
    std::vector<KeyPoint> keypoints_1,keypoints_2;//构造2个专门由点组成的点向量用来存储特征点

    detector.detect(img_1,keypoints_1);//将img_1图像中检测到的特征点存储起来放在keypoints_1中
    detector.detect(img_2,keypoints_2);//同理

	//计算特征向量
    SurfDescriptorExtractor extractor;//定义描述子对象

    Mat descriptors_1,descriptors_2;//存放特征向量的矩阵

    extractor.compute(img_1,keypoints_1,descriptors_1);
    extractor.compute(img_2,keypoints_2,descriptors_2);

    //用burte force进行匹配特征向量
    BruteForceMatcher<L2<float>>matcher;//定义一个burte force matcher对象
    vector<DMatch>matches;
    matcher.match(descriptors_1,descriptors_2,matches);

}
void TestASAPWarping()
{
	//cv::Mat img = cv::imread("..\\ptz\\input0\\in000001.jpg");
	//Mesh mesh(img.rows,img.cols,img.cols/8,img.rows/8);
	//cv::Mat nimg;
	//mesh.drawMesh(img,0,nimg);
	//cv::imshow("mesh",nimg);
	//cv::waitKey();
	cv::Mat simg = cv::imread(".//4//s.png");
	cv::Mat timg = cv::imread(".//4//t.png");
	int width = simg.cols;
	int height = simg.rows;
	int quadWidth = width/8;
	int quadHeight = height/8;
	ASAPWarping asap(height,width,quadWidth,quadHeight,1.0);
	
	std::vector<cv::Point2f>vf1,vf2;
	//KLTFeaturesMatching(simg,timg,vf1,vf2);
	FILESURFFeaturesMatching(simg,timg,vf1,vf2);
	asap.SetControlPts(vf1,vf2);
	asap.Solve();
	cv::Mat wimg;
	asap.Warp(simg,wimg);
	cv::imshow("s",simg);
	cv::imshow("t",timg);
	cv::imshow("warped",wimg);
	cv::imwrite(".//4//twarped.png",wimg);
	cv::Mat diff;
	cv::absdiff(wimg,timg,diff);
	cv::imshow("diff",diff);
	cv::imwrite(".//diff.jpg",diff);
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