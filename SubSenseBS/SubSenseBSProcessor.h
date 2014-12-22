#pragma once
#include <direct.h>
#include <io.h>
#include <vector>
#include "BGSSubsenseM.h"
#include "videoprocessor.h"
#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
int CreateDir(char *pszDir)	
{
	int i = 0;
	int iRet;
	int iLen = strlen(pszDir);
	//在末尾加/
	if (pszDir[iLen - 1] != '\\' && pszDir[iLen - 1] != '/')
	{
		pszDir[iLen] = '/';
		pszDir[iLen + 1] = '\0';
	}

	// 创建目录
	for (i = 0;i < iLen;i ++)
	{
		if (pszDir[i] == '\\' || pszDir[i] == '/')
		{ 
			pszDir[i] = '\0';

			//如果不存在,创建
			iRet = _access(pszDir,0);
			if (iRet != 0)
			{
				iRet = _mkdir(pszDir);
				if (iRet != 0)
				{
					return -1;
				} 
			}
			//支持linux,将所有\换成/
			pszDir[i] = '/';
		} 
	}

	return 0;
}
class SubSenseBSProcessor : public FrameProcessor
{
private:
	//BackgroundSubtractorSuBSENSE _bgs;
	BGSSubsenseM _bgs;
	std::vector<cv::KeyPoint> _voKeyPoints;
	bool _initFlag;
	char fileName[150];
	char pathName[150];
public:
	SubSenseBSProcessor():_initFlag(false)
	{

		sprintf(pathName,"..\\result\\subsensem\\moseg\\people1\\warp\\");
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
