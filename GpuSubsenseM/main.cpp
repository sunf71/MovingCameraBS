#include "GpuBackgroundSubtractor.h"
#include "SubSenseBSProcessor.h"
#include "videoprocessor.h"
#include "timer.h"
void TestGpuSubsense()
{
	VideoProcessor processor;
	
	// Create feature tracker instance
	SubSenseBSProcessor tracker;
	std::vector<std::string> fileNames;
	int start = 1;
	int end = 1130;
	for(int i=start; i<=end;i++)
	{
		char name[50];
		sprintf(name,"..\\baseline\\input0\\in%06d.jpg",i);
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
	

	cv::waitKey();


	

}

int main()
{
	TestGpuSubsense();
	return 0;
}