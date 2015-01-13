#pragma once
#include <direct.h>
#include <io.h>
#include <vector>
#include "GpuBackgroundSubtractor.h"
#include "videoprocessor.h"
#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
int CreateDir(char *pszDir);
class SubSenseBSProcessor : public FrameProcessor
{
private:
	GpuBackgroundSubtractor _bgs;
	//BGSSubsenseM _bgs;
	std::vector<cv::KeyPoint> _voKeyPoints;
	bool _initFlag;
	char fileName[150];
	char pathName[150];
public:
	SubSenseBSProcessor():_initFlag(false)
	{

		sprintf(pathName,"..\\result\\subsensex\\moseg\\cars1\\warpBaseline\\");
		CreateDir(pathName);
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
