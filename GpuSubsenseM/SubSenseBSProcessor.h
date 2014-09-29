#pragma once
#include <vector>
#include "GpuBackgroundSubtractor.h"
#include "videoprocessor.h"
#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
class SubSenseBSProcessor : public FrameProcessor
{
private:
	GpuBackgroundSubtractor _bgs;
	//BGSSubsenseM _bgs;
	std::vector<cv::KeyPoint> _voKeyPoints;
	bool _initFlag;
public:
	SubSenseBSProcessor():_initFlag(false)
	{}
	void  process(cv:: Mat &frame, cv:: Mat &output)
	{
		cv::Mat gray;
		cv::cvtColor(frame, gray, CV_BGR2GRAY); 
		static int frameNo = 1;
		if (!_initFlag)
		{
			_bgs.initialize(gray,_voKeyPoints);
			_initFlag = true;
		}
		_bgs(gray,output);
		char fileName[50];
		sprintf(fileName,"..\\result\\subsensex\\baseline\\input0\\bin%06d.png",frameNo++);
		imwrite(fileName,output);
		//imshow("input",frame);
		//output = frame;
	}
};
