#pragma once
#include <direct.h>
#include <io.h>
#include <vector>
#include "GpuBackgroundSubtractor.h"
#include "WarpBackgroundSubtractor.h"
#include "videoprocessor.h"
#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
int CreateDir(char *pszDir);
class SubSenseBSProcessor : public FrameProcessor
{
private:
	//GpuBackgroundSubtractor _bgs;
	WarpBackgroundSubtractor _bgs;
	//GpuWarpBackgroundSubtractor _bgs;
	//BGSSubsenseM _bgs;
	std::vector<cv::KeyPoint> _voKeyPoints;
	bool _initFlag;
	char fileName[150];
	char pathName[150];
	int _offset;
public:
	SubSenseBSProcessor(const char* str,int offset=0):_initFlag(false),_offset(offset)
	{
		
		sprintf(pathName,str);
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

		sprintf(fileName,"%sbin%06d.png",pathName,_offset+frameNo++);
		imwrite(fileName,output);
		//imshow("input",frame);
		//output = frame;
	}
};
