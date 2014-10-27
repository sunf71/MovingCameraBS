#pragma once
#include <direct.h>
#include <io.h>
#include <vector>
#include "GpuBackgroundSubtractor.h"
#include "videoprocessor.h"
#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
int CreatDir(char *pszDir);
class SubSenseBSProcessor : public FrameProcessor
{
private:
	GpuBackgroundSubtractor _bgs;
	//BGSSubsenseM _bgs;
	std::vector<cv::KeyPoint> _voKeyPoints;
	bool _initFlag;
	char fileName[50];
	char pathName[50];
public:
	SubSenseBSProcessor():_initFlag(false)
	{

		sprintf(pathName,"..\\result\\subsensex\\ptz\\input3\\");
		CreatDir(pathName);
	}
	void  process(cv:: Mat &frame, cv:: Mat &output)
	{
		/*cv::Mat gray;
		cv::cvtColor(frame, gray, CV_BGR2GRAY);*/ 
		static int frameNo = 1;
		if (!_initFlag)
		{
			_bgs.initialize(frame,_voKeyPoints);
			_initFlag = true;
		}
		_bgs(frame,output);

		sprintf(fileName,"%sbin%06d.png",pathName,frameNo++);
		imwrite(fileName,output);
		//imshow("input",frame);
		//output = frame;
	}
};
