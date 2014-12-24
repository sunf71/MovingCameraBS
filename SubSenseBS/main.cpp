#include "SubSenseBSProcessor.h"
#include "BGSMovieMaker.h"
#include "timer.h"
#include "ASAPWarping.h"
using namespace cv;
void TestASAPWarping()
{
	cv::Mat img = cv::imread("..\\ptz\\input0\\in000001.jpg");
	Mesh mesh(img.rows,img.cols,img.cols/8,img.rows/8);
	cv::Mat nimg;
	mesh.drawMesh(img,0,nimg);
	cv::imshow("mesh",nimg);
	cv::waitKey();
}
int main()
{
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