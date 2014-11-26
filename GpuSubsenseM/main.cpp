#include "GpuBackgroundSubtractor.h"
#include "SubSenseBSProcessor.h"
#include "videoprocessor.h"
#include "timer.h"
//void TestGpuSubsense()
//{
//	VideoProcessor processor;
//	
//	// Create feature tracker instance
//	SubSenseBSProcessor tracker;
//	std::vector<std::string> fileNames;
//	int start = 1;
//	int end = 1130;
//	for(int i=start; i<=end;i++)
//	{
//		char name[50];
//		sprintf(name,"..\\ptz\\input0\\in%06d.jpg",i);
//		//sprintf(name,"..\\PTZ\\input4\\drive1_%03d.png",i);
//		fileNames.push_back(name);
//	}
//	// Open video file
//	processor.setInput(fileNames);
//	//processor.setInput("..\\ptz\\woman.avi");
//	// set frame processor
//	processor.setFrameProcessor(&tracker);
//
//	processor.dontDisplay();
//	// Declare a window to display the video
//	//processor.displayOutput("Tracked Features");
//
//	// Play the video at the original frame rate
//	//processor.setDelay(1000./processor.getFrameRate());
//	processor.setDelay(0);
//
//	nih::Timer timer;
//	timer.start();
//	// Start the process
//	processor.run();
//	timer.stop();
//
//	std::cout<<(end-start+1)/timer.seconds()<<" fps"<<std::endl;
//	
//
//	cv::waitKey();
//
//
//	
//
//}
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
int cpu_main()
{
	//TestGpuSubsense();
	return 0;
}