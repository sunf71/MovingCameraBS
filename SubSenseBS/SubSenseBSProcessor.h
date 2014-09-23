#pragma once
#include <vector>
#include "BackgroundSubtractorSuBSENSE.h"
#include "BGSSubsenseM.h"
#include "videoprocessor.h"

class SubSenseBSProcessor : public FrameProcessor
{
private:
	//BackgroundSubtractorSuBSENSE _bgs;
	BGSSubsenseM _bgs;
	std::vector<cv::KeyPoint> _voKeyPoints;
	bool _initFlag;
public:
	SubSenseBSProcessor():_initFlag(false)
	{}
	void  process(cv:: Mat &frame, cv:: Mat &output)
	{
		/*cv::Mat gray;
		cv::cvtColor(frame, gray, CV_BGR2GRAY); */
		static int frameNo = 1;
		if (!_initFlag)
		{
			_bgs.initialize(frame,_voKeyPoints);
			_initFlag = true;
		}
		_bgs(frame,output);
		char fileName[50];
		sprintf(fileName,"..\\result\\subsensem\\ptz\\input0\\bin%06d.png",frameNo++);
		imwrite(fileName,output);
		imshow("input",frame);
		//output = frame;
	}
};
