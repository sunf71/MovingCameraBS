#pragma once
#include <direct.h>
#include <io.h>
#include <vector>
#include "GpuBackgroundSubtractor.h"
#include "WarpBackgroundSubtractor.h"
#include "videoprocessor.h"
#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Common.h"
class BSProcesor : public FrameProcessor
{
private:
	GpuBackgroundSubtractor _gbgs;
	
	std::vector<cv::KeyPoint> _voKeyPoints;
	bool _initFlag;
	char fileName[150];
	char pathName[150];
	int _offset;
public:
	BSProcesor(const char* str,int offset=0):_initFlag(false),_offset(offset)
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
			_gbgs.initialize(frame,_voKeyPoints);
			
		}
		_gbgs(frame,output);

		sprintf(fileName,"%s\\bin%06d.png",pathName,_offset+frameNo++);
		imwrite(fileName,output);
		//imshow("input",frame);
		//output = frame;
	}
};

class WarpBSProcessor : public FrameProcessor
{
private:
	
	
	WarpBackgroundSubtractor* _bgs;
	//BGSSubsenseM _bgs;
	std::vector<cv::KeyPoint> _voKeyPoints;
	bool _initFlag;
	char fileName[150];
	char pathName[150];
	int _offset;
public:
	WarpBSProcessor(int procId,const char* str,int offset=0, int warpId = 1,  float rggThreshold= 1.0, float rggSeedThreshold = 0.4, float mdlConfidence = 0.75, float tcConfidence = 0.15, float scConfidence = 0.35):_initFlag(false),_offset(offset)
	{
		if (procId==0)
		{
			_bgs = new GpuWarpBackgroundSubtractor(warpId, rggThreshold, rggSeedThreshold, mdlConfidence, tcConfidence);
			
		}
		else if (procId == 1)
		{
			_bgs = new WarpBackgroundSubtractor(rggThreshold,rggSeedThreshold,mdlConfidence,tcConfidence);
			
		}
		
		sprintf(pathName,str);
		CreateDir(pathName);
	}
	~WarpBSProcessor()
	{
		safe_delete(_bgs);
		
	}
	virtual void initialize(cv::Mat& frame0)
	{
		std::vector<cv::KeyPoint> emptyKepPoints;
		_bgs->initialize(frame0, emptyKepPoints);
	}
	void  process(cv:: Mat &frame, cv:: Mat &output)
	{
		_bgs->operator()(frame,output);
	}
};
