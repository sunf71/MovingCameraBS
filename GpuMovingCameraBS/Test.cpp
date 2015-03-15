#include "Test.h"
#include "flowIO.h"
#include "FlowComputer.h"
#include "ASAPWarping.h"
#include "FeaturePointRefine.h"
#include "SuperpixelComputer.h"
#include "DistanceUtils.h"
#include "Common.h"
#include "findHomography.h"
#include "BlockWarping.h"
#include "GpuTimer.h"
#include "LBP.h"
#include <fstream>
#include <algorithm>
#include <queue>
void testCudaGpu()
{
	try

	{

		cv::Mat src_host = cv::imread("in000001.jpg");

		cv::gpu::GpuMat dst, src;

		src.upload(src_host);

		cv::gpu::cvtColor(src,src,CV_BGR2GRAY);

		cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

		//cv::Mat result_host = dst;

		cv::Mat result_host;

		dst.download(result_host);

		cv::imshow("Result", result_host);

		cv::waitKey();

	}

	catch(const cv::Exception& ex)

	{

		std::cout << "Error: " << ex.what() << std::endl;

	}
}

void CpuSuperpixel(unsigned int* data, int width, int height, int step, float alpha = 0.9)
{
	int size = width*height;
	int* labels = new int[size];
	unsigned int* idata = new unsigned int[size];
	memcpy(idata,data,sizeof(unsigned int)*size);
	int numlabels(0);
	ComSuperpixel CS;
	//CS.Superpixel(idata,width,height,7000,0.9,labels);
#ifdef REPORT
	nih::Timer timer;
	timer.start();
#endif
	//CS.SuperpixelLattice(idata,width,height,step,alpha,numlabels,labels);
	CS.Superpixel(idata,width,height,step,alpha,numlabels,labels);
#ifdef REPORT
	timer.stop();
	std::cout<<"SLIC SuperPixel "<<timer.seconds()<<std::endl;
#endif
	SLIC aslic;
	aslic.DrawContoursAroundSegments(idata, labels, width, height,0x00ff00);
	PictureHandler handler;
	char name[50];
	sprintf(name,"%f_cpusuper.jpg",alpha);
	handler.SavePicture(idata,width,height,std::string(name),std::string(".\\"));
	aslic.SaveSuperpixelLabels(labels,width,height,std::string("cpuSp.txt"),std::string(".\\"));
	delete[] labels;
	delete[] idata;
}

void TestSuperpixel()
{
	using namespace cv;
	Mat img = imread("..//moseg//people1//in000001.jpg");
	cv::cvtColor(img,img,CV_BGR2BGRA);
	//cv::resize(img,img,cv::Size(16,16));
	uchar4* imgData = new uchar4[img.rows*img.cols];
	unsigned int* idata = new unsigned int[img.rows*img.cols];
	unsigned char tmp[4];
	for(int i=0; i< img.cols; i++)
	{

		for(int j=0; j<img.rows; j++)
		{
			int idx = img.step[0]*j + img.step[1]*i;
			for(int k=0; k<4; k++)
				tmp[k] = img.data[idx + img.elemSize1()*k];
			imgData[i + j*img.cols].x = tmp[0];
			imgData[i + j*img.cols].y = tmp[1];
			imgData[i + j*img.cols].z = tmp[2];
			imgData[i + j*img.cols].w = tmp[3];


			idata[i + j*img.cols] = tmp[3]<<24 | tmp[2]<<16| tmp[1]<<8 | tmp[0];
		}
	}
	for(int i =0; i<10; i++)
		CpuSuperpixel(idata,img.cols,img.rows,40,i*0.1);
	GpuSuperpixel gs(img.cols,img.rows,40);
	int num(0);
	int* labels = new int[img.rows*img.cols];

	GpuTimer timer;
	SLIC aslic;	
	PictureHandler handler;

	timer.Start();
	gs.Superpixel(imgData,num,labels);
	timer.Stop();
	std::cout<<timer.Elapsed()<<"ms"<<std::endl;

	aslic.DrawContoursAroundSegments(idata, labels, img.cols,img.rows,0x00ff00);
	handler.SavePicture(idata,img.cols,img.rows,std::string("GpuSp.jpg"),std::string(".\\"));
	aslic.SaveSuperpixelLabels(labels,img.cols,img.rows,std::string("GpuSp.txt"),std::string(".\\"));

	memset(labels,0,sizeof(int)*img.rows*img.cols);
	timer.Start();
	gs.SuperpixelLattice(imgData,num,labels);
	timer.Stop();
	std::cout<<timer.Elapsed()<<"ms"<<std::endl;
	aslic.DrawContoursAroundSegments(idata, labels, img.cols,img.rows,0x00ff00);
	handler.SavePicture(idata,img.cols,img.rows,std::string("GpuSpLatttice.jpg"),std::string(".\\"));
	aslic.SaveSuperpixelLabels(labels,img.cols,img.rows,std::string("GpuSpLatttice.txt"),std::string(".\\"));
	delete[] labels;
	delete[] idata;
	delete[] imgData;
}

void MRFOptimization()
{
	using namespace std;
	char imgFileName[150];
	char maskFileName[150];
	char featureMaskFileName[150];
	char resultFileName[150];
	int cols = 320;
	int rows = 240;
	GpuSuperpixel gs(cols,rows,5);
	MRFOptimize optimizer(cols,rows,5);
	nih::Timer timer;
	timer.start();
	int start = 1;
	int end = 1130;
	for(int i=start; i<=end;i++)
	{
		sprintf(imgFileName,"..\\ptz\\input0\\in%06d.jpg",i);
		sprintf(maskFileName,"..\\result\\subsensex\\ptz\\input0\\o\\bin%06d.png",i);
		sprintf(featureMaskFileName,"..\\result\\subsensex\\ptz\\input0\\features\\features%06d.jpg",i);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\ptz\\input0\\bin%06d.png",i);

		/*sprintf(imgFileName,"..\\baseline\\input0\\in%06d.jpg",i);
		sprintf(maskFileName,"..\\result\\sobs\\baseline\\input0\\bin%06d.png",i);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\baseline\\input0\\bin%06d.png",i);*/
		//optimizer.Optimize(&gs,string(imgFileName),string(maskFileName),string(resultFileName));
		optimizer.Optimize(&gs,string(imgFileName),string(maskFileName),string(featureMaskFileName),string(resultFileName));
	}
	timer.stop();
	std::cout<<(end-start+1)/timer.seconds()<<" fps\n";
}
void warmUpDevice()
{
	cv::Mat cpuMat = cv::Mat::ones(100,100,CV_8UC3);
	cv::gpu::GpuMat gmat;
	gmat.upload(cpuMat);
	gmat.download(cpuMat);
}
void TestRandom()
{
	int width(8);
	int height(6);
	int* d_rand,*h_rand;
	cudaMalloc(&d_rand,width*height*2*sizeof(int));
	h_rand = new int[width*height*2];
	TestRandNeighbour(width,height,d_rand);	
	cudaMemcpy(h_rand,d_rand,sizeof(int)*width*height*2,cudaMemcpyDeviceToHost);
	for(int i=0; i<width*height; i++)
	{
		std::cout<<h_rand[2*i]<<","<<h_rand[2*i+1]<<" ";
		if ((2*i+1)%width == 0)
			std::cout<<std::endl;
	}
	std::cout<<"---------\n";
	TestRandNeighbour(width,height,d_rand);	
	cudaMemcpy(h_rand,d_rand,sizeof(int)*width*height*2,cudaMemcpyDeviceToHost);
	for(int i=0; i<width*height; i++)
	{
		std::cout<<h_rand[2*i]<<","<<h_rand[2*i+1]<<" ";
		if ((2*i+1)%width == 0)
			std::cout<<std::endl;
	}
	std::cout<<"---------\n";
	int x_rand,y_rand;
	cv::Size size(width,height);
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			getRandNeighborPosition_3x3(x_rand,y_rand,i,j,2,size);
			std::cout<<x_rand<<","<<y_rand<<" ";

		}
		std::cout<<"\n";
	}
	std::cout<<"---------\n";
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			getRandNeighborPosition_3x3(x_rand,y_rand,i,j,2,size);
			std::cout<<x_rand<<","<<y_rand<<" ";

		}
		std::cout<<"\n";
	}
	cudaFree(d_rand);
	delete[] h_rand;
}
void TestGpuSubsense(int procId, int start, int end, const char* input, const char* output, float rggThre, float rggSeedThres, float mdlConfidence, float tcConfidence, float scConfidence)
{

	warmUpDevice();
	VideoProcessor processor;	
	FrameProcessor* tracker;
	// Create feature tracker instance"..\\result\\subsensex\\moseg\\people1\\"
	if (procId >=0)
	{
		tracker = new WarpBSProcessor(procId,output,start-1,rggThre,rggSeedThres,mdlConfidence,tcConfidence);
	}
	else
	{
		tracker = new BSProcesor(output,start-1);
	}
	std::vector<std::string> fileNames;

	for(int i=start; i<=end;i++)
	{
		char name[50];
		//"..\\moseg\\people1\\in%06d.jpg"
		sprintf(name,"%s\\in%06d.jpg",input,i);
		//sprintf(name,"..\\PTZ\\input4\\drive1_%03d.png",i);
		fileNames.push_back(name);
	}
	// Open video file
	processor.setInput(fileNames);
	//processor.setInput("..\\ptz\\woman.avi");
	// set frame processor
	processor.setFrameProcessor(tracker);

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

	safe_delete(tracker);


}
void TestMotionEstimate()
{
	char fileName[100];
	int start = 2;
	int end = 40;
	cv::Mat curImg,prevImg,transM;
	MotionEstimate me(640,480,5);
	cv::Mat mask;
	for(int i=start; i<=end; i++)
	{
		sprintf(fileName,"..//moseg//people1//in%06d.jpg",i);
		curImg = cv::imread(fileName);
		sprintf(fileName,"..//moseg//people1//in%06d.jpg",i-1);
		prevImg = cv::imread(fileName);
		//me.EstimateMotionMeanShift(curImg,prevImg,transM,mask);
		me.EstimateMotion(curImg,prevImg,transM,mask);
		//me.EstimateMotionHistogram(curImg,prevImg,transM,mask);
		sprintf(fileName,".//features//people1//features%06d.jpg",i);
		cv::imwrite(fileName,mask);
		/*cv::imshow("curImg",curImg);
		cv::imshow("prevImg",prevImg);*/
		cv::waitKey();
	}
}
void TestRegionGrowing()
{
	std::vector<cv::Point2f> seeds;
	seeds.push_back(cv::Point2f(30,66));
	cv::Mat img = cv::imread("..//ptz//input0in000225.jpg");
	cv::Mat result(img.size(),CV_8U);
	result = cv::Scalar(0);
	RegionGrowing(seeds,img,result);
	cv::imshow("region grow",result);
	cv::waitKey();
}



void GetHomography(const cv::Mat& gray,const cv::Mat& pre_gray, cv::Mat& homography)
{
	int max_count = 50000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 2;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	std::vector<cv::Point2f> features1,features2;
	// detect the features
	cv::goodFeaturesToTrack(gray, // the image 
		features1,   // the output detected features
		max_count,  // the maximum number of features 
		qlevel,     // quality level
		minDist);   // min distance between two features

	// 2. track features
	cv::calcOpticalFlowPyrLK(gray, pre_gray, // 2 consecutive images
		features1, // input point position in first image
		features2, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

	int k=0;

	for( int i= 0; i < features1.size(); i++ ) 
	{

		// do we keep this point?
		if (status[i] == 1) 
		{

			//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
			// keep this point in vector
			features1[k] = features1[i];
			features2[k++] = features2[i];
		}
	}
	features1.resize(k);
	features2.resize(k);

	std::vector<uchar> inliers(features1.size());
	homography= cv::findHomography(
		cv::Mat(features1), // corresponding
		cv::Mat(features2), // points
		inliers, // outputted inliers matches
		CV_RANSAC, // RANSAC method
		0.1); // max distance to reprojection point
}
void TestFlow()
{
	DenseOpticalFlowProvier* DOFP = new GpuDenseOptialFlow();
	cv::Mat preImg = cv::imread("..//ptz//input3//in000289.jpg");
	cv::Mat curImg = cv::imread("..//ptz//input3//in000290.jpg");
	cv::cvtColor(preImg,preImg,CV_BGR2GRAY);
	cv::cvtColor(curImg,curImg,CV_BGR2GRAY);
	cv::Mat preMsk = cv::imread("..//result//subsensem//ptz//input3//wap//bin000289.png");
	cv::Mat curMsk = cv::imread("..//result//subsensem//ptz//input3//warp//bin000290.png");
	cv::cvtColor(preMsk,preMsk,CV_BGR2GRAY);
	cv::cvtColor(curMsk,curMsk,CV_BGR2GRAY);

	cv::Mat homography;
	GetHomography(curImg,preImg,homography);
	std::vector<cv::Point2f> curPts,prevPts;
	int width = curImg.cols;
	int height = curImg.rows;
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			int idx = i+j*width;
			if (curMsk.data[idx] == 0xff)
				curPts.push_back(cv::Point2f(i,j));
		}
	}
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	// 2. track features
	cv::calcOpticalFlowPyrLK(curImg, preImg, // 2 consecutive images
		curPts, // input point position in first image
		prevPts, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error
	cv::Mat mask(height,width,CV_32F);
	mask = cv::Scalar(0);
	double* data = (double*)homography.data;
	double maxD(0),minD(1e10);

	for(int i=0; i<status.size(); i++)
	{
		if (status[i] == 1)
		{
			int x = (int)(curPts[i].x);
			int y = (int)(curPts[i].y);
			float wx = data[0]*x + data[1]*y + data[2];
			float wy = data[3]*x + data[4]*y + data[5];
			float w = data[6]*x + data[7]*y + data[8];
			wx /= w;
			wy /= w;
			int px = (int)(prevPts[i].x);
			int py = (int)(prevPts[i].y);
			int idx = py*width + px;
			float d = abs(px-wx) + abs(py-wy);

			if (d>maxD)
				maxD = d;
			if (d<minD)
				minD = d;
			float * ptr = (float*)(mask.data + idx*4);
			*ptr= d;
		}

	}

	cv::Mat B;
	mask.convertTo(B,CV_8U,255.0/(maxD-minD),0);
	cv::imwrite("tracked.jpg",B);
}


void SuperpixelGrowingFlow(const cv::Mat& sgray, const cv::Mat& tgray,int step, int spSize, const SLICClusterCenter* centers, const int* labels, cv::Mat& flow)
{
	std::vector<cv::Point2f> features0,features1;
	std::vector<uchar> status;
	std::vector<float> err;
	int spWidth = (sgray.cols+step-1)/step;
	int spHeight = (sgray.rows+step-1)/step;
	int width = sgray.cols;
	int height = sgray.rows;
	flow.create(spHeight,spWidth,CV_32FC2);
	flow = cv::Scalar(0);
	cv::goodFeaturesToTrack(sgray,features0,50000,0.05,2);
	cv::calcOpticalFlowPyrLK(sgray,tgray,features0,features1,status,err);

	int k=0; 
	for(int i=0; i<features0.size(); i++)
	{
		if (status[i] == 1)
		{
			features0[k] = features0[i];
			features1[k++] = features1[i];
		}
	}
	for(int i=0; i<k; i++)
	{
		int ix = (int)(features0[i].x+0.5);
		int iy = (int)(features0[i].y+0.5);
		int label = labels[ix + iy*width];
		float2* ptr = (float2*)(flow.data + label*8);
		*ptr = make_float2(features1[i].x - features0[i].x,features1[i].y - features0[i].y);
	}

	std::cout<<"tracking succeeded "<<k<<" total "<<spSize<<std::endl;
}
void SuperpixelFlowToPixelFlow(const int* labels, const SLICClusterCenter* centers, const cv::Mat& sflow, int spSize, int step, int width, int height, cv::Mat& flow)
{

	flow.create(height,width,CV_32FC2);
	flow = cv::Scalar(0);
	for(int i=0; i<spSize; i++)
	{		
		int k = centers[i].xy.x;
		int j = centers[i].xy.y;
		float2 flowValue = *((float2*)(sflow.data + i*8));
		if (centers[i].nPoints >0)			
		{

			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = step;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>width-1 || y<0 || y> height-1)
						continue;
					int idx = x+y*width;

					if (labels[idx] == i )
					{		
						float2* fptr = (float2*)(flow.data + idx*8);
						*fptr = flowValue;
					}				
				}
			}

		}
	}
}
float L1Dist(const float2& p1, const float2& p2)
{
	return abs(p1.x-p2.x) + abs(p1.y - p2.y);
}

//利用warping的mapX和mapY进行superpixel matching
//比较matching后超像素的颜色差，理论上前景应该较大，背景较小，但是结果显示在图像边缘部分会有较大的误差
void SuperpixelMatching(const int* labels0, const SLICClusterCenter* centers0, const int* labels1, const SLICClusterCenter* centers1, int spSize, int spStep, int width, int height,
	const cv::Mat& mapX, const cv::Mat& mapY,std::vector<int> matchedId, cv::Mat& diff)
{
	int spWidth = (width+spStep-1)/spStep;
	diff.create(height,width,CV_8UC3);
	std::vector<int> ids;
	for(int i=0; i<spSize; i++)
	{		
		int k = centers0[i].xy.x;
		int j = centers0[i].xy.y;
		ids.clear();
		if (centers0[i].nPoints >0)			
		{
			float avgX(0),avgY(0);
			int n(0);
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = spStep;
			for (int x = k- radius; x<= k+radius; x++)
			{
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (x<0 || x>width-1 || y<0 || y> height-1)
						continue;
					int idx = x+y*width;

					if (labels0[idx] == i )
					{		
						ids.push_back(idx);
						float nx = *((float*)(mapX.data + idx*4));
						avgX += nx;
						float ny = *((float*)(mapY.data + idx*4));
						avgY += ny;
						n++;
					}				
				}
			}
			avgX /= n;
			avgY /= n;
			float2 point = make_float2(avgX,avgY);
			int iavgX = (int)(avgX +0.5);
			int iavgY = (int)(avgY + 0.5);
			int label = labels1[iavgX + iavgY*width];
			//在8邻域内查找中心点与avgX和avgY最接近的超像素
			float disMin = L1Dist(centers1[label].xy,point);
			int minLabel = label;
			//left
			if (label-1 >=0)
			{
				float dist =L1Dist(centers1[label-1].xy,point);
				if (dist<disMin)
				{
					minLabel = label-1;
					disMin = dist;
				}
				if (label-1-spWidth >=0)
				{
					dist =L1Dist(centers1[label-1-spWidth-1].xy,point);
					if (dist<disMin)
					{
						minLabel = label-1-spWidth;
						disMin = dist;
					}
				}
				if(label-1+spWidth < spSize)
				{
					dist =L1Dist(centers1[label-1+spWidth].xy,point);
					if (dist<disMin)
					{
						minLabel = label-1+spWidth;
						disMin = dist;
					}
				}
			}
			if (label+1 <spSize)
			{
				float dist =L1Dist(centers1[label+1].xy,point);
				if (dist<disMin)
				{
					minLabel = label+1;
					disMin = dist;
				}
				if (label+1-spWidth >=0)
				{
					dist =L1Dist(centers1[label+1-spWidth-1].xy,point);
					if (dist<disMin)
					{
						minLabel = label+1-spWidth;
						disMin = dist;
					}
				}
				if(label+1+spWidth < spSize)
				{
					dist =L1Dist(centers1[label+1+spWidth].xy,point);
					if (dist<disMin)
					{
						minLabel = label+1+spWidth;
						disMin = dist;
					}
				}
			}
			if (label+spWidth <spSize)
			{
				float dist =L1Dist(centers1[label+spWidth].xy,point);
				if (dist<disMin)
				{
					minLabel = label+spWidth;
					disMin = dist;
				}
			}
			if (label-spWidth >=0)
			{
				float dist =L1Dist(centers1[label-spWidth].xy,point);
				if (dist<disMin)
				{
					minLabel = label-spWidth;
					disMin = dist;
				}
			}
			matchedId.push_back(minLabel);
			uchar cdx = abs(centers0[i].rgb.x - centers1[minLabel].rgb.x);
			uchar cdy = abs(centers0[i].rgb.y - centers1[minLabel].rgb.y);
			uchar cdz = abs(centers0[i].rgb.z - centers1[minLabel].rgb.z);
			uchar3 val = make_uchar3(cdx,cdy,cdz);
			for(int k=0; k<ids.size(); k++)
			{
				uchar3* ptr = (uchar3*)(diff.data + 3*ids[k]);
				*ptr = val;
			}

		}
	}
}
//利用稠密光流进行superpixel matching,同时计算superpixel光流（超像素内像素光流平均值）
void SuperpixelMatching(const int* labels0, const SLICClusterCenter* centers0, const int* labels1, const SLICClusterCenter* centers1, int spSize, int spStep, int width, int height,
	const cv::Mat& flow,std::vector<int> matchedId, cv::Mat& spFlow, cv::Mat& diff)
{

	int spWidth = (width+spStep-1)/spStep;
	int spHeight = (height+spStep-1)/spStep;
	diff.create(height,width,CV_8UC3);
	spFlow.create(spHeight,spWidth,CV_32FC2);
	std::vector<int> ids;
	matchedId.clear();
	matchedId.resize(spSize);
	memset(&matchedId[0],-1,sizeof(int)*spSize);
	for(int i=0; i<spSize; i++)
	{		

		int k = (int)(centers0[i].xy.x+0.5);
		int j = (int)(centers0[i].xy.y+0.5);
		ids.clear();
		if (centers0[i].nPoints >0)			
		{
			float avgX(0),avgY(0);
			int n(0);
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = spStep;
			for (int x = k- radius; x<= k+radius; x++)
			{
				if  (x<0 || x>width-1)
					continue;
				for(int y = j - radius; y<= j+radius; y++)
				{
					if  (y<0 || y> height-1)
						continue;

					int idx = x+y*width;

					if (labels0[idx] == i )
					{		
						/*if (x == 144 && y==96)
						std::cout<<i<<std::endl;*/
						ids.push_back(idx);
						float2 dxy = *((float2*)(flow.data + idx*4*2));
						avgX += dxy.x;						
						avgY += dxy.y;
						n++;
					}				
				}
			}
			avgX /= n;
			avgY /= n;
			* (float2*)(spFlow.data+i*8) = make_float2(avgX,avgY);
			float2 point = make_float2(avgX+k,avgY+j);
			int iavgX = (int)(avgX +0.5+k);
			int iavgY = (int)(avgY + 0.5+j);
			if (iavgX <0 || iavgX > width-1 || iavgY <0 || iavgY > height-1)
				continue;
			int label = labels1[iavgX + iavgY*width];
			//在8邻域内查找中心点与avgX和avgY最接近的超像素
			float disMin = L1Dist(centers1[label].xy,point);
			int minLabel = label;
			//left
			if (label-1 >=0)
			{
				float dist =L1Dist(centers1[label-1].xy,point);
				if (dist<disMin)
				{
					minLabel = label-1;
					disMin = dist;
				}
				if (label-1-spWidth >=0)
				{
					dist =L1Dist(centers1[label-1-spWidth-1].xy,point);
					if (dist<disMin)
					{
						minLabel = label-1-spWidth;
						disMin = dist;
					}
				}
				if(label-1+spWidth < spSize)
				{
					dist =L1Dist(centers1[label-1+spWidth].xy,point);
					if (dist<disMin)
					{
						minLabel = label-1+spWidth;
						disMin = dist;
					}
				}
			}
			if (label+1 <spSize)
			{
				float dist =L1Dist(centers1[label+1].xy,point);
				if (dist<disMin)
				{
					minLabel = label+1;
					disMin = dist;
				}
				if (label+1-spWidth >=0)
				{
					dist =L1Dist(centers1[label+1-spWidth-1].xy,point);
					if (dist<disMin)
					{
						minLabel = label+1-spWidth;
						disMin = dist;
					}
				}
				if(label+1+spWidth < spSize)
				{
					dist =L1Dist(centers1[label+1+spWidth].xy,point);
					if (dist<disMin)
					{
						minLabel = label+1+spWidth;
						disMin = dist;
					}
				}
			}
			if (label+spWidth <spSize)
			{
				float dist =L1Dist(centers1[label+spWidth].xy,point);
				if (dist<disMin)
				{
					minLabel = label+spWidth;
					disMin = dist;
				}
			}
			if (label-spWidth >=0)
			{
				float dist =L1Dist(centers1[label-spWidth].xy,point);
				if (dist<disMin)
				{
					minLabel = label-spWidth;
					disMin = dist;
				}
			}

			matchedId[i] = (minLabel);
			uchar cdx = abs(centers0[i].rgb.x - centers1[minLabel].rgb.x);
			uchar cdy = abs(centers0[i].rgb.y - centers1[minLabel].rgb.y);
			uchar cdz = abs(centers0[i].rgb.z - centers1[minLabel].rgb.z);
			/*float t = 30;
			cdx = cdx < t ? 0 : cdx;
			cdy = cdy < t ? 0 : cdy;
			cdz = cdz < t ? 0 : cdz;*/
			uchar3 val = make_uchar3(cdx,cdy,cdz);
			for(int k=0; k<ids.size(); k++)
			{
				uchar3* ptr = (uchar3*)(diff.data + 3*ids[k]);
				*ptr = val;
			}

		}
	}

}


void TestSuperpixelMatching()
{
	SLIC aslic;	
	PictureHandler handler;

	int cols = 320;
	int rows = 240;
	int step = 5;
	ASAPWarping asap(cols,rows,8,1.0);

	int size = rows*cols;
	int num(0);
	GpuSuperpixel gs(cols,rows,step);
	cv::Mat simg,timg,wimg,sgray,tgray;
	cv::Mat img0,img1;
	img0 = cv::imread("..//ptz//input3//in000222.jpg");
	img1 = cv::imread("..//ptz//input3//in000221.jpg");

	std::vector<cv::Point2f> features0,features1;
	std::vector<uchar> status;
	std::vector<float> err;

	cv::cvtColor(img0,sgray,CV_BGR2GRAY);
	cv::cvtColor(img1,tgray,CV_BGR2GRAY);
	cv::cvtColor(img0,simg,CV_BGR2BGRA);
	cv::cvtColor(img1,timg,CV_BGR2BGRA);


	SLICClusterCenter* centers0(NULL),*centers1(NULL);
	int * labels0(NULL), * labels1(NULL);
	labels0 = new int[size];
	labels1 = new int[size];
	int spSize = ((rows+step-1)/step) * ((cols+step-1)/step);
	centers0 = new SLICClusterCenter[spSize];
	centers1 = new SLICClusterCenter[spSize];

	gs.Superpixel(simg,num,labels0,centers0);
	gs.Superpixel(timg,num,labels1,centers1);
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	cv::Mat flow,spFlow;
	DOFP->DenseOpticalFlow(sgray,tgray,flow);
	WriteFlowFile(flow,"flow.flo");
	std::vector<int> matchedId;
	cv::Mat diffMat;
	SuperpixelMatching(labels0,centers0,labels1,centers1,spSize,step,cols,rows,flow,matchedId,spFlow,diffMat);
	WriteFlowFile(spFlow,"spFlow.flo");
	KLTFeaturesMatching(sgray,tgray,features0,features1);
	cv::Mat homography;
	FeaturePointsRefineRANSAC(features0,features1,homography);
	asap.SetControlPts(features0,features1);
	asap.Solve();
	asap.Warp(simg,wimg);
	cv::Mat warpFlow;
	asap.getFlow(warpFlow);
	cv::Mat flowDiff;
	cv::absdiff(flow,warpFlow,flowDiff);
	cv::Mat mask(rows,cols,CV_8U);
	mask = cv::Scalar(0);
	for(int i=0; i<flowDiff.rows; i++)
	{
		float2* ptrw = warpFlow.ptr<float2>(i);
		float2* ptrf = flow.ptr<float2>(i);
		uchar * mptr = mask.ptr<uchar>(i);
		for(int j=0; j<flowDiff.cols; j++)
		{
			float diff = abs(ptrw[j].x - ptrf[j].x) + abs(ptrw[j].y - ptrf[j].y);
			mptr[j] = diff;
		}
	}
	cv::imshow("flow diff", mask);
	/*std::vector<int> matchedId;
	cv::Mat diffMat;
	SuperpixelMatching(labels0,centers0,labels1,centers1,spSize,step,cols,rows,asap.getMapX(),asap.getMapY(),matchedId,diffMat);*/
	cv::imshow("matched err", diffMat);	
	cv::imshow("source image",img0);
	cv::imshow("dest image", img1);
	//cv::imshow("warped image", wimg);
	cv::waitKey();
	delete[] labels0;
	delete[] labels1;
	delete[] centers0;
	delete[] centers1;
	delete DOFP;
}
void TestSuperpixelFlow()
{
	SLIC aslic;	
	PictureHandler handler;

	int cols = 570;
	int rows = 340;
	int step = 5;
	ASAPWarping asap(cols,rows,8,1.0);

	int size = rows*cols;
	int num(0);
	GpuSuperpixel gs(cols,rows,step);
	cv::Mat simg,timg,wimg,sgray,tgray;
	cv::Mat img0,img1;
	img0 = cv::imread("..//ptz//input2//in000002.jpg");
	img1 = cv::imread("..//ptz//input2//in000001.jpg");

	std::vector<cv::Point2f> features0,features1;
	std::vector<uchar> status;
	std::vector<float> err;

	cv::cvtColor(img0,sgray,CV_BGR2GRAY);
	cv::cvtColor(img1,tgray,CV_BGR2GRAY);
	cv::cvtColor(img0,simg,CV_BGR2BGRA);
	cv::cvtColor(img1,timg,CV_BGR2BGRA);


	SLICClusterCenter* centers0(NULL),*centers1(NULL);
	int * labels0(NULL), * labels1(NULL);
	labels0 = new int[size];
	labels1 = new int[size];
	int spHeight = (rows+step-1)/step;
	int spWidth = (cols+step-1)/step;
	int spSize = spHeight*spWidth;
	centers0 = new SLICClusterCenter[spSize];
	centers1 = new SLICClusterCenter[spSize];

	gs.Superpixel(simg,num,labels0,centers0);
	gs.Superpixel(timg,num,labels1,centers1);

	/*KLTFeaturesMatching(sgray,tgray,features0,features1);
	cv::Mat homography;
	FeaturePointsRefineRANSAC(features0,features1,homography);
	asap.SetControlPts(features0,features1);
	asap.Solve();
	asap.Warp(simg,wimg);*/

	cv::Mat spFlow,flow,flowField,pflowField;
	std::vector<cv::Mat> flows(2);
	std::vector<cv::Point2f> f0,f1;
	//SuperpixelFlow(sgray,tgray,step,spSize,centers0,f0,f1,spFlow);
	GpuSuperpixelFlow(sgray,tgray,step,spSize,centers0,f0,f1,spFlow);
	//SuperpixelGrowingFlow(sgray,tgray,step,spSize,centers0,labels0,spFlow);
	std::vector<float> flowHist,avgX,avgY;
	std::vector<std::vector<int>> ids;
	cv::Mat idMat;
	OpticalFlowHistogram(spFlow,flowHist,avgX,avgY,ids,idMat,20,36);
	double minVal,maxVal;
	cv::minMaxLoc(flowHist,&minVal,&maxVal);
	std::ofstream file("out.txt");
	cv::Mat histImg(spHeight,spWidth,CV_32F,cv::Scalar(0));
	for(int i=0; i<spHeight; i++)
	{
		unsigned short* idPtr = idMat.ptr<unsigned short>(i);
		float* histImgPtr = histImg.ptr<float>(i);
		for(int j=0; j<spWidth; j++)
		{
			histImgPtr[j] = flowHist[idPtr[j]]/maxVal;
			file<<histImgPtr[j]<<"\t"; 
		}
		file<<std::endl;
	}
	file.close();

	//cv::normalize(histImg,histImg,0, 100, NORM_MINMAX, -1, Mat());
	cv::imshow("histImg",histImg);
	cv::split(spFlow,flows);
	getFlowField(flows[0],flows[1],flowField);
	cv::imshow("superpixel flow",flowField);
	SuperpixelFlowToPixelFlow(labels0,centers0,spFlow,spSize,step,cols,rows,flow);
	WriteFlowFile(flow,"spflow.flo");
	cv::split(flow,flows);
	getFlowField(flows[0],flows[1],pflowField);
	cv::imshow("pixel flow",pflowField);
	cv::imshow("simg",simg);
	cv::imshow("timg",timg);
	cv::imshow("timg",timg);
	cv::waitKey();

	delete[] labels0;
	delete[] labels1;
	delete[] centers0;
	delete[] centers1;
	//for(int i=0; i<spSize; i++)
	//{
	//	features0.push_back(cv::Point2f(centers0[i].xy.x,centers0[i].xy.y));
	//}
	//cv::calcOpticalFlowPyrLK(sgray,tgray,features0,features1,status,err);
	////跟踪成功的label
	//std::vector<int> mlabels0;
	//int k=0; 
	//for(int i=0; i<spSize; i++)
	//{
	//	if (status[i] ==1)
	//	{
	//		mlabels0.push_back(i);
	//		features0[k] = features0[i];
	//		features1[k++] = features1[i];
	//		
	//	}
	//}
	//asap.SetControlPts(features0,features1);
	//asap.Solve();
	//asap.Warp(img0,wimg);
	//cv::Mat mapX = asap.getMapX();
	//cv::Mat mapY = asap.getMapY();
	//int radius = 2*step;
	//cv::RNG rgn =  cv::theRNG();
	//for( int i=0; i<k; i++)
	//{
	//	
	//	int label0 = mlabels0[i];
	//	int ix1 = (int)(features1[i].x+0.5);
	//	int iy1 = (int)(features1[i].y+0.5);
	//	int idx = ix1+ iy1*cols;
	//	int ix0 = (int)(features0[i].x+0.5);
	//	int iy0 = (int)(features0[i].y+0.5);
	//	int idx0 = ix0+ iy0*cols;

	//	float dx = features1[i].x - mapX.at<float>(idx0);
	//	float dy = features1[i].y - mapY.at<float>(idx0);
	//	float dis = abs(dx) + abs(dy);

	//	int label1 = labels1[idx];
	//	float4 avg0 = centers0[label0].rgb;
	//	float4 avg1 = centers1[label1].rgb;
	//	uchar diff = (abs(avg0.x-avg0.y) + abs(avg0.y - avg1.y) + abs(avg0.z-avg1.z) + abs(avg0.w-avg1.w))/4;
	//	cv::Scalar color = cv::Scalar(diff,diff,diff,255);
	//	if (dis < 2)
	//		color = cv::Scalar(255,0,0,255);

	//	
	//	for(int m = -radius; m<= radius; m++)
	//	{
	//		for(int n = -radius; n<= radius; n++)
	//		{
	//			int ix = features0[i].x+m;
	//			int iy = features0[i].y+n;
	//			
	//		
	//			if (ix >=0 && ix < cols && iy >=0 && iy<rows)
	//			{	
	//				int idx0 = ix + iy*cols;
	//				if (labels0[idx0] == label0)
	//				{						
	//					simg.at<Vec4b>(iy,ix) = color;
	//				}
	//			}
	//			
	//			ix = centers1[label1].xy.x+m;
	//			iy = centers1[label1].xy.y+n;
	//			if (ix >=0 && ix < cols && iy >=0 && iy<rows)
	//			{
	//				int idx1 = ix + iy*cols;
	//				if (labels1[idx1] == label1)
	//				{
	//					timg.at<Vec4b>(iy,ix) = color;
	//				}
	//			}
	//		}
	//	}
	//	
	//		
	//}

	//unsigned int* idata = new unsigned[size];
	//memcpy(idata,simg.data,size*4);
	//aslic.DrawContoursAroundSegments(idata, labels0, simg.cols,simg.rows,0x00ff00);
	//aslic.SaveSuperpixelLabels(labels0,cols,rows,std::string("labels0.txt"),std::string(".\\"));
	//handler.SavePicture(idata,simg.cols,simg.rows,std::string("GpuSp0.jpg"),std::string(".\\"));
	//memcpy(idata,timg.data,size*4);
	//aslic.DrawContoursAroundSegments(idata, labels1, simg.cols,simg.rows,0x00ff00);
	//aslic.SaveSuperpixelLabels(labels1,cols,rows,std::string("labels1.txt"),std::string(".\\"));
	//handler.SavePicture(idata,simg.cols,simg.rows,std::string("GpuSp1.jpg"),std::string(".\\"));

	//cv::imshow("s",simg);
	//cv::imshow("t", timg);
	//cv::waitKey(0);
	//if(labels0 != NULL)
	//{
	//	delete[] labels0;
	//	delete[] labels1;
	//	delete[] centers0;
	//	delete[] centers1;
	//	delete[] idata;
	//	idata = NULL;
	//	centers0 = NULL;
	//	centers1 = NULL;
	//	labels0 = NULL;
	//	labels1 = NULL;
	//}
}
void TCMRFOptimization()
{
	/*TestFlow();
	return;*/
	using namespace std;
	char imgFileName[150];
	char maskFileName[150];
	char resultFileName[150];
	int cols = 320;
	int rows = 240;
	GpuSuperpixel gs(cols,rows,5);
	MRFOptimize optimizer(cols,rows,5);
	nih::Timer timer;
	timer.start();
	int start = 300;
	int end = 330;
	std::vector<cv::Mat> imgs;
	std::vector<cv::Mat> masks;
	cv::Mat curImg,prevImg,mask,prevMask,resultImg,gray,preGray;

	cv::Mat flow;
	//DenseOpticalFlowProvier* DOFP = new GpuDenseOptialFlow();
	//DenseOpticalFlowProvier* DOFP = new SFDenseOptialFlow();
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	for(int i=start; i<=end;i++)
	{
		sprintf(imgFileName,"..\\ptz\\input3\\in%06d.jpg",i);		
		curImg = cv::imread(imgFileName);
		imgs.push_back(curImg.clone());
		sprintf(maskFileName,"..\\result\\subsensem\\ptz\\input3\\bin%06d.png",i);
		curImg = cv::imread(maskFileName);
		cv::cvtColor(curImg,curImg,CV_BGR2GRAY);
		masks.push_back(curImg.clone());		
	}
	for (int i= 1; i<end-start+1; i++)
	{
		curImg = imgs[i];
		prevImg = imgs[i-1];
		prevMask = masks[i-1];
		cv::cvtColor(curImg,gray,CV_BGR2GRAY);
		cv::cvtColor(prevImg,preGray,CV_BGR2GRAY);
		mask = masks[i];
		cv::Mat homography;
		GetHomography(gray,preGray,homography);
		DOFP->DenseOpticalFlow(gray,preGray,flow);
		//WriteFlowFile(flow,"flow.flo");
		//ReadFlowFile(flow,"ptz0_87.flo");
		optimizer.Optimize(&gs,curImg,mask,prevMask,flow,homography,resultImg);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\ptz\\input3\\bin%06d.png",i);
		cv::imwrite(resultFileName,resultImg);
	}
	timer.stop();
	std::cout<<(end-start+1)/timer.seconds()<<" fps\n";
	delete DOFP;
}

void TestFlowHistogram()
{
	using namespace std;
	using namespace cv;
	char imgFileName[150];
	char resultFileName[150];
	int cols = 640;
	int rows = 480;
	int step = 5;
	int spWidth = (cols+step-1)/step;
	int spHeight = (rows + step -1)/step;
	int spSize = spWidth * spHeight;
	int* labels = new int[rows*cols];
	GpuSuperpixel gs(cols,rows,step);
	SLICClusterCenter* centers = new SLICClusterCenter[spSize];
	int start=1;
	int end = 18;
	cv::Mat img0,img1,gray0,gray1;
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	cv::Mat flow,spFlow;
	std::vector<float> flowHist,avgDx,avgDy;
	std::vector<std::vector<int>> ids;
	cv::Mat idMat;
	float minVal,maxVal;
	int maxIdx(0),minIdx(0);
	int num(0);
	for(int i=start; i<=end;i++)
	{		
		sprintf(imgFileName,"..//moseg//cars2//in%06d.jpg",i);
		img0=cv::imread(imgFileName);
		sprintf(imgFileName,"..//moseg//cars2//in%06d.jpg",i+1);
		img1=cv::imread(imgFileName);
		cv::cvtColor(img0,gray0,CV_BGR2GRAY);
		cv::cvtColor(img1,gray1,CV_BGR2GRAY);
		cv::cvtColor(img0,img0,CV_BGR2BGRA);
		DOFP->DenseOpticalFlow(gray1,gray0,flow);
		/*gs.Superpixel(img0,num,labels,centers);
		SuperpixelFlow(gray1,gray0,5,spSize,centers,spFlow);
		SuperpixelFlowToPixelFlow(labels,centers,spFlow,spSize,step,cols,rows,flow);*/
		OpticalFlowHistogram(flow,flowHist,avgDx,avgDy,ids,idMat,10,36);

		for(int i=0; i<avgDx.size(); i++)
		{
			avgDx[i] /= ids[i].size();
			avgDy[i] /= ids[i].size();
		}
		minMaxLoc(flowHist,maxVal,minVal,maxIdx,minIdx);
		float maxAvgDx = avgDx[maxIdx];
		float maxAvgDy = avgDy[maxIdx];
		cv::Mat histImg(rows,cols,CV_32F,cv::Scalar(0));
		float threshold = 3;
		float avg = 0;
		float variance(0);
		for(int i=0; i<rows; i++)
		{
			unsigned short* idPtr = idMat.ptr<unsigned short>(i);
			uchar* histImgPtr = histImg.ptr<uchar>(i);
			for(int j=0; j<cols; j++)
			{
				float dist = abs(avgDx[idPtr[j]] - maxAvgDx)+ abs(avgDy[idPtr[j]] - maxAvgDy);
				avg+=dist;
				//histImgPtr[j] = dist >threshold ? 255 : 0;
				//histImgPtr[j] = dist;
				/*if (i == 70 && j==50)
				{
				std::cout<<dist<<" , "<<flowHist[idPtr[j]]<<" "<<histImgPtr[j]<<"\n";
				}*/
			}
		}
		avg/=(rows*cols);
		std::cout<<"avg dist "<<avg<<std::endl;
		threshold = avg*2.0;
		for(int i=0; i<rows; i++)
		{
			unsigned short* idPtr = idMat.ptr<unsigned short>(i);
			float* histImgPtr = histImg.ptr<float>(i);
			for(int j=0; j<cols; j++)
			{
				float dist = abs(avgDx[idPtr[j]] - maxAvgDx)+ abs(avgDy[idPtr[j]] - maxAvgDy);
				variance += (dist - avg)*(dist-avg);
				histImgPtr[j] = exp(dist/threshold);
				//histImgPtr[j] = dist;
				/*if (i == 70 && j==50)
				{
				std::cout<<dist<<" , "<<flowHist[idPtr[j]]<<" "<<histImgPtr[j]<<"\n";
				}*/
			}
		}
		variance /= rows*cols;
		std::cout<<"variance : "<<variance<<std::endl;
		sprintf(resultFileName,".\\histogram\\input3\\bin%06d.jpg",i);
		cv::imwrite(resultFileName,histImg);
		/*	cv::imshow("histImg",histImg);
		cv::waitKey();*/
	}
	delete DOFP;
	delete[] labels;
	delete[] centers;
}
void TestColorHistogram()
{
	using namespace std;
	using namespace cv;
	char imgFileName[150];
	char maskFileName[150];
	char resultFileName[150];
	int cols = 320;
	int rows = 240;
	int step = 5;
	int size = rows*cols;
	GpuSuperpixel gs(cols,rows,step);
	int spNum = gs.GetSuperPixelNum();
	int * labels = new int[rows*cols];
	SLICClusterCenter* centers = new SLICClusterCenter[spNum];

	cv::Mat img,mask,rst;
	rst = Mat::zeros(rows,cols,CV_8U);
	vector<Mat> bgr_planes;
	vector<int> spIdx;
	/// Establish the number of bins
	int histSize = 16;
	int binSize = 256/histSize;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	Mat b_hist, g_hist, r_hist;
	int start = 500;
	int end = 500;
	for(int i=start; i<=end;i++)
	{
		rst = cv::Scalar(0);
		sprintf(imgFileName,"..\\ptz\\input3\\in%06d.jpg",i);
		sprintf(maskFileName,"H:\\changeDetection2014\\PTZ\\PTZ\\zoomInZoomOut\\groundtruth\\gt%06d.png",i);		
		sprintf(resultFileName,".\\hist%06d.png",i);
		img = imread(imgFileName);

		mask = imread(maskFileName);
		cvtColor(mask,mask,CV_BGR2GRAY);
		split( img, bgr_planes );
		namedWindow("mask", CV_WINDOW_AUTOSIZE );
		imshow("mask", mask );
		/// Compute the histograms:
		calcHist( &bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate );
		calcHist( &bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate );
		calcHist( &bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate );
		std::cout<<"bhist\n";
		for(int i=0; i<b_hist.rows;i++)
		{
			std::cout<<b_hist.at<float>(i)<<" ";
		}
		std::cout<<"\n";
		std::cout<<"ghist\n";
		for(int i=0; i<b_hist.rows;i++)
		{
			std::cout<<g_hist.at<float>(i)<<" ";
		}
		std::cout<<"\n";
		std::cout<<"rhist\n";
		for(int i=0; i<b_hist.rows;i++)
		{
			std::cout<<r_hist.at<float>(i)<<" ";
		}
		std::cout<<"\n";
		/// Normalize the result to [ 0, histImage.rows ]
		/*normalize(b_hist, b_hist, 0, 255, NORM_MINMAX, -1, Mat() );
		normalize(g_hist, g_hist, 0, 255, NORM_MINMAX, -1, Mat() );
		normalize(r_hist, r_hist, 0, 255, NORM_MINMAX, -1, Mat() );*/
		// Draw the histograms for B, G and R
		int hist_w = 512; int hist_h = 256;
		int bin_w = cvRound( (double) hist_w/histSize );

		Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );



		/// Draw for each channel
		for( int i = 1; i < histSize; i++ )
		{
			line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
				cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
				Scalar( 255, 0, 0), 2, 8, 0  );
			line( histImage,  cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
				cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
				Scalar( 0, 255, 0), 2, 8, 0  );
			line( histImage,  cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
				cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
				Scalar( 0, 0, 255), 2, 8, 0  );
		}

		for(int i=0; i<rows; i++)
		{
			for(int j=0; j<cols; j++)
			{

				int idx = i*cols +j;
				uchar b = bgr_planes[0].data[idx];
				uchar g =  bgr_planes[1].data[idx];
				uchar r =  bgr_planes[2].data[idx];

				float bhist = b_hist.at<float>(bgr_planes[0].data[idx]/binSize);
				float ghist = g_hist.at<float>(bgr_planes[1].data[idx]/binSize);
				float rhist = r_hist.at<float>(bgr_planes[2].data[idx]/binSize);
				uchar val = (bhist+ghist+rhist)/3;
				if (j==74 && i==26)
				{	std::cout<<"row "<<i<<" , col "<<j<<" (r,g,b) = "<<(int)r<<","<<(int)g<<","<<(int)b<<std::endl;
				std::cout<<"bhist "<<bhist<<" ghist "<<ghist<<" rhist  "<<rhist<<std::endl;
				}
				if (j==92 && i==117)
				{	std::cout<<"row "<<i<<" , col "<<j<<" (r,g,b) = "<<(int)r<<","<<(int)g<<","<<(int)b<<std::endl;
				std::cout<<"bhist "<<bhist<<" ghist "<<ghist<<" rhist  "<<rhist<<std::endl;
				}
				if (j==93 && i==131)
				{	std::cout<<"row "<<i<<" , col "<<j<<" (r,g,b) = "<<(int)r<<","<<(int)g<<","<<(int)b<<std::endl;
				std::cout<<"bhist "<<bhist<<" ghist "<<ghist<<" rhist  "<<rhist<<std::endl;
				}
				rst.data[idx] = val;
			}
		}
		/// Display
		namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
		imshow("calcHist Demo", histImage );

		imshow("rst", rst );
		waitKey(0);
		cvtColor(img,img,CV_BGR2BGRA);
		int num(0);
		gs.Superpixel(img,num,labels,centers);
		for(int i=0; i<spNum; i++)
		{
			int k = centers[i].xy.x;
			int j = centers[i].xy.y;
			spIdx.clear();

			int c(0);
			if (centers[i].nPoints !=0)			
			{

				float bHists(0),gHists(0),rHists(0);
				//以原来的中心点为中心，step +2　为半径进行更新
				int radius = step;
				for (int x = k- radius; x<= k+radius; x++)
				{
					if (x<0 || x>cols-1 )
						continue;

					for(int y = j - radius; y<= j+radius; y++)
					{
						if  (y<0 || y> rows-1)
							continue;
						int idx = x+y*cols;					
						//std::cout<<idx<<std::endl;
						if (labels[idx] == i )
						{		
							uchar b = bgr_planes[0].data[idx];
							uchar g =  bgr_planes[1].data[idx];
							uchar r =  bgr_planes[2].data[idx];
							spIdx.push_back(idx);
							float bhist = b_hist.at<float>(bgr_planes[0].data[idx]/bin_w);
							float ghist = g_hist.at<float>(bgr_planes[1].data[idx]/bin_w);
							float rhist = r_hist.at<float>(bgr_planes[2].data[idx]/bin_w);
							bHists+=bhist;
							gHists+=ghist;
							rHists += rhist;
							c++;							
						}					
					}
				}

				/*bHists/=c;
				gHists/=c;
				rHists/=c;
				uchar val = (bHists+gHists+rHists)/3;
				for(int s=0; s<spIdx.size();s++)
				{
				int idx = spIdx[s];
				rst.data[idx] = val;
				}*/


			}


		}
		imwrite(resultFileName,rst);



	}


	delete[] labels;
	delete[] centers;
}
void TestSuperpielxComputer()
{
	using namespace std;
	using namespace cv;
	char imgFileName[150];
	char maskFileName[150];
	char resultFileName[150];
	int cols = 640;
	int rows = 480;
	int step = 5;
	int size = rows*cols;
	int spWidth = (cols+step-1)/step;
	int spHeight = (rows+step-1)/step;
	int spSize = spWidth*spHeight;
	cv::Mat img,preImg,gray,preGray,mask,preMask;
	SuperpixelComputer spComputer(cols,rows,step);
	int * labels(NULL), *preLabels(NULL);
	SLICClusterCenter* centers(NULL), *preCenters(NULL);
	int num(0);
	cv::Mat spFlow,flow;
	int start = 2;
	int end = 5;
	std::vector<int> matchedId;
	std::vector<int> revMatchedId;
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	std::vector<cv::Point2f> f0,f1;

	for(int i=start; i<=end;i++)
	{

		sprintf(imgFileName,"..\\moseg\\cars3\\in%06d.jpg",i);		
		sprintf(maskFileName, "..\\moseg\\cars3\\groundtruth\\gt%06d.png",i-1);
		sprintf(resultFileName,".\\test\\spFlow%06d.flo",i);
		matchedId.clear();
		img = imread(imgFileName);
		cv::cvtColor(img,gray,CV_BGR2GRAY);
		mask = imread(maskFileName);
		if (mask.channels()==3)
			cv::cvtColor(mask,mask,CV_BGR2GRAY);

		if (preGray.empty())
		{
			preImg = img.clone();
			preGray = gray.clone();
		}
		spComputer.ComputeSuperpixel(img,num,labels,centers);
		spComputer.GetPreSuperpixelResult(num,preLabels,preCenters);
		SuperpixelFlow(gray,preGray,step,num,centers,f0,f1,spFlow);
		SuperpixelMatching(labels,centers,img,preLabels,preCenters,preImg,num,step,cols,rows,spFlow,matchedId);
		//
		preMask = cv::Mat::zeros(rows,cols,CV_8U);
		for (int si=0; si<matchedId.size(); si++)
		{
			int cx = (int)(preCenters[matchedId[si]].xy.x+0.5);
			int cy = (int)(preCenters[matchedId[si]].xy.y+0.5);
			int idx = cx + cy*cols;
			if (mask.data[idx] == 0xff)
			{
				int x = (int)(centers[si].xy.x+0.5);
				int y = (int)(centers[si].xy.y + 0.5);
				for(int m = x - step; m<=x+step; m++)
				{
					if (m<0 || m > cols-1)
						continue;
					for(int n = y - step; n<=y+step; n++)
					{
						if (n>=0 && n<rows && labels[n*cols+m] == si)
						{
							preMask.data[n*cols+m] =  0xff;
						}
					}
				}
			}
		}
		SuperpixelFlow(preGray,gray,step,num,centers,f0,f1,spFlow);
		SuperpixelMatching(preLabels,preCenters,preImg,labels,centers,img,num,step,cols,rows,spFlow,revMatchedId);
		std::set<int> checkedId;
		for(int i=0; i<matchedId.size(); i++)
		{
			int matched = matchedId[i];
			if (matched >=0 && matched < spSize && revMatchedId[matched] == i)
				checkedId.insert(checkedId.begin(),i);
		}
		cv::Mat chkMat(rows,cols,CV_8U);
		cv::Mat diffMat(rows,cols,CV_8UC3,cv::Scalar(0));
		for(int i=0; i<num; i++)
		{
			if (matchedId[i] >0)
			{
				float4 color0 = centers[i].rgb;
				float4 color1 = preCenters[matchedId[i]].rgb;
				int r = abs(color0.x - color1.x);
				int g = abs(color0.y - color1.y);
				int b = abs(color0.z - color1.z);
				int x = (int)(centers[i].xy.x + 0.5);
				int y = (int)(centers[i].xy.y + 0.5);
				for(int k= x-step; k<= x+step; k++)
				{
					if (k<0 || k>cols-1)
						continue;
					for(int j=y-step; j<= y+step; j++)
					{
						if (j<0 || j>rows-1)
							continue;
						int idx = j*cols +k;
						if (labels[idx] == i)
						{
							uchar* rgb = (uchar*)(diffMat.data+idx*3);
							rgb[0] = r;
							rgb[1] = g;
							rgb[2] = b;
						}
						if (checkedId.find(i) != checkedId.end())
						{
							chkMat.data[idx] = gray.data[idx];
						}
					}
				}
			}
		}

		//WriteFlowFile(spFlow,resultFileName);
		sprintf(resultFileName,".\\test\\sp_matchedDiff%06d.jpg",i);
		cv::imwrite(resultFileName,diffMat);
		sprintf(resultFileName,".\\test\\sp_dchked%06d.jpg",i);
		cv::imwrite(resultFileName,chkMat);
		sprintf(resultFileName,".\\test\\preMask%06d.jpg",i);
		cv::imwrite(resultFileName,preMask);


		/*DOFP->DenseOpticalFlow(gray,preGray,flow);
		cv::Mat diff;
		SuperpixelMatching(labels,centers,preLabels,preCenters,spSize,step,cols,rows,flow,matchedId,spFlow,diff);
		sprintf(resultFileName,".\\test\\matchedDiff%06d.jpg",i);
		cv::imwrite(resultFileName,diff);*/
		cv::swap(gray,preGray);
		cv::swap(img,preImg);
	}
	delete DOFP;
}

void TestDescDiff()
{
	char filename[20];
	cv::Mat cmat,gmat,hmat;
	hmat = cv::imread("hdesc.png",CV_LOAD_IMAGE_UNCHANGED);

	int width = hmat.cols;
	int height = hmat.rows;
	cv::Mat cr = cv::Mat::zeros(hmat.size(),CV_8U);

	std::vector<cv::Mat> cmats,gmats;
	cv::Mat tmp;
	for(int i=0; i<50; i++)
	{
		sprintf(filename,"cpu%ddescmodel.png",i);		
		cmat = cv::imread(filename,CV_LOAD_IMAGE_UNCHANGED );

		sprintf(filename,"gpu%ddescmodel.png",i);		
		gmat = cv::imread(filename,CV_LOAD_IMAGE_UNCHANGED );
		cmats.push_back(cmat.clone());

		gmats.push_back(gmat.clone());
	}
	for(int r=2; r< height-2; r++)
	{

		cv::Vec4w* hptr = hmat.ptr<cv::Vec4w>(r);
		uchar* crPtr = cr.ptr<uchar>(r);

		for(int c=2; c<width-2; c++)
		{
			int idx = (r*width+c)*3*2;
			int cidx = (r*width+c)*3*2;
			int b(0);
			int i=0;
			while(i<50 && b<2)
			{

				//cv::Vec4w gptr = gmats[i].at<cv::Vec4w>(r,c);
				ushort* cptr = (ushort*)(cmats[i].data+cidx);
				ushort* gptr = (ushort*)(gmats[i].data+idx);
				//std::cout<<i<<" "<<r<<" , "<<c<<std::endl;

				for(int k=0; k<3; k++)
				{					
					size_t d = hdist_ushort_8bitLUT(hptr[c][k],gptr[k]);		
					//size_t d = hdist_ushort_8bitLUT(hptr[c][k],gptr[k]);
					if (d/2*15 >36)
						goto failed;

				}
				b++;

failed:
				i++;
			}
			if (b<2)
			{
				crPtr[c] = 0xff;
				/*std::cout<<hptr[c][0]<<" "<<hptr[c][1]<<" "<<hptr[c][2]<<" "<<hptr[c][3]<<"\n";
				for(int t=0; t<50; t++)
				{
				ushort* cptr = (ushort*)(cmats[t].data+cidx);
				ushort* gptr = (ushort*)(gmats[t].data+idx);
				std::cout<<t<<"--------------\n cpu:\n";
				std::cout<<cptr[0]<<" "<<cptr[1]<<" "<<cptr[2]<<"\n gpu:";
				std::cout<<gptr[0]<<" "<<gptr[1]<<" "<<gptr[2]<<" "<<gptr[3]<<"\n";

				}*/

			}


		}
	}
	cv::imshow("result",cr);

	cv::waitKey();
}


void TestSuperpixelDownSample()
{
	char imgFileName[150];
	char resultFileName[150];
	int cols = 640;
	int rows = 480;
	int step = 5;
	int start =1;
	int end = 40;
	cv::Mat curImg,dsImg;
	SuperpixelComputer spComputer(cols,rows,step);
	int num(0);
	int * labels;
	SLICClusterCenter* centers;
	for(int i=start; i<=end;i++)
	{
		sprintf(imgFileName,"..\\moseg\\people1\\in%06d.jpg",i);		
		curImg = cv::imread(imgFileName);
		spComputer.ComputeSuperpixel(curImg,num,labels,centers);
		spComputer.GetSuperpixelDownSampleImg(dsImg);
		sprintf(resultFileName,"..\\moseg\\people1\\downsample\\in%06d.jpg",i);
		cv::imwrite(resultFileName,dsImg);
	}
}

void GpuSubsenseMain(int argc, char* argv[])
{
	printf("gpu 0 cpu 1 %s\n",argv[1]);
	printf("from %s\n",argv[2]);
	printf("to %s\n",argv[3]);
	printf("input %s\n",argv[4]);
	printf("output %s\n",argv[5]);
	
	if (argc == 6)
		TestGpuSubsense(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),argv[4],argv[5]);
	else if (argc == 10)
	{
		printf("region growing threshold %s\n",argv[6]);
		printf("region growing seed threshold %s\n",argv[7]);
		printf("model confidence %s\n",argv[8]);
		printf("tc confidence %s\n",argv[9]);
		TestGpuSubsense(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),argv[4],argv[5],atof(argv[6]),atof(argv[7]),atof(argv[8]),atof(argv[9]));
	}
	else if (argc == 11)
	{
		printf("region growing threshold %s\n",argv[6]);
		printf("region growing seed threshold %s\n",argv[7]);
		printf("model confidence %s\n",argv[8]);
		printf("tc confidence %s\n",argv[9]);
		printf("sc confidence %s\n",argv[10]);
		TestGpuSubsense(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),argv[4],argv[5],atof(argv[6]),atof(argv[7]),atof(argv[8]),atof(argv[9]),atof(argv[10]));
	}
}

void TestBlockHomography()
{
	/*std::vector<cv::Point2f> f1,f2;
	f1.push_back(cv::Point2f(139,23));
	f1.push_back(cv::Point2f(434,326));
	f1.push_back(cv::Point2f(599,185));
	f1.push_back(cv::Point2f(332,400));
	f1.push_back(cv::Point2f(22,25));

	f2.push_back(cv::Point2f(282,240));
	f2.push_back(cv::Point2f(357,223));
	f2.push_back(cv::Point2f(607,458));
	f2.push_back(cv::Point2f(272,146));
	f2.push_back(cv::Point2f(33,23));
	cv::Mat homography;
	findHomographyNormalizedDLT(f1,f2,homography);
	std::cout<<homography<<std::endl;
	findHomographyEqa(f1,f2,homography);
	std::cout<<homography<<std::endl;*/
	int start = 1;
	int end = 200;
	int width = 720;
	int height = 480;
	int quadWidth = 4;
	cv::Size size2(width*2,height);
	std::vector<cv::Mat> homographies;
	std::vector<float> blkWeights(quadWidth*quadWidth,0);
	const char path[] = "..//particle//vcar";
	char dstPath[200];
	sprintf(dstPath,"..//warpRst//mywarp");
	CreateDir(dstPath);
	char fileName[200];
	cv::Mat img1,img0,gray1,gray0,wimg,gtImg;
	std::vector<cv::Point2f> features1,features0;
	std::vector<cv::Point2f> sf1,sf0;

	ASAPWarping asap(width,height,quadWidth,1.0);
	for (int i=start; i<=end; i++)
	{
		printf("%d------------\n",i);
		sprintf(fileName,"%s//in%06d.jpg",path,i);
		img1 = imread(fileName);
		sprintf(fileName,"%s//groundtruth//gt%06d.png",path,i);
		gtImg = imread(fileName);
		cv::cvtColor(gtImg,gtImg,CV_BGR2GRAY);
		cv::cvtColor(img1,gray1,CV_BGR2GRAY);
		if (gray0.empty())
		{
			gray0 = gray1.clone();
			img0 = img1.clone();
		}
		/*nih::Timer timer;
		timer.start();*/
		KLTFeaturesMatching(gray1,gray0,features1,features0,5000,0.05,5);
		/*timer.stop();
		std::cout<<"klt tracking "<<timer.seconds()*1000<<"ms\n";
		timer.start();*/
		//BC2FFeaturePointsRefineHistogram(width,height,features1,features0,blkWeights,4,radSize1,thetaSize1,radSize2,thetaSize2);
		//C2FFeaturePointsRefineHistogram(width,height,features1,features0,5,36,2,2);
		FeaturePointsRefineHistogram(width,height,features1,features0);
		/*cv::Mat homo;
		FeaturePointsRefineRANSAC(features1,features0,homo,0.1);*/
		if (i>1)
		{
			bool save = false;
			int ec(0);
			cv::Mat rstImg(size2,CV_8UC3);
			img0.copyTo(rstImg(cv::Rect(0,0,width,height)));
			img1.copyTo(rstImg(cv::Rect(width,0,width,height)));
			for(int j=0; j< features1.size(); j++)
			{
				int x = (int)(features1[j].x+0.5);
				int y = (int)(features1[j].y+0.5);
				if (gtImg.data[x+y*width] == 0xff)
				{
					cv::line(rstImg,cv::Point(features0[j].x,features0[j].y),cv::Point(features1[j].x+width,features1[j].y),cv::Scalar(255,0,0));
					ec++;
					save = true;
				}
			}
			if (save)
			{
				sprintf(fileName,"..//warpRst//mywarp//error//%d_error_%d.jpg",i,ec);
				cv::imwrite(fileName,rstImg);
			}
		}
		std::cout<<"features remians "<<features1.size()<<std::endl;
		int num = BlockDltHomography(width,height,quadWidth,features1,features0,homographies,blkWeights,sf1,sf0);
		
		

		asap.Reset();
		/*asap.CreateSmoothCons(1.0);
		asap.SetControlPts(features1,features0);
		asap.Solve();*/
		
		cv::Mat b;
		asap.CreateSmoothCons(blkWeights);		
		asap.CreateMyDataCons(num,homographies,b);
		asap.MySolve(b);
		asap.Warp(img1,wimg);
		/*cv::imshow("warped image",wimg);
		cv::waitKey();*/
		sprintf(fileName,"%swin%06d.jpg",dstPath,i);
		cv::imwrite(fileName,wimg);
		cv::Mat diff;
		cv::absdiff(wimg,img0,diff);
		sprintf(fileName,"%sdiff%06d.jpg",dstPath,i);
		
		cv::imwrite(fileName,diff);
		
		homographies.clear();
		cv::swap(img0,img1);
		cv::swap(gray0,gray1);
	}

}
//测试直方图投票的方式选取背景特征点
void TestFeaturesRefineHistogram(int argc, char* argv[])
{
	int start = atoi(argv[1]);
	int end = atoi(argv[2]);
	int width = atoi(argv[3]);
	int height = atoi(argv[4]);
	int radSize1 = atoi(argv[5]);
	int thetaSize1 = atoi(argv[6]);
	int radSize2 = atoi(argv[7]);
	int thetaSize2 = atoi(argv[8]);
	char* path = argv[9];
	cv::Size size2(width*2,height);
	char fileName[200];
	cv::Mat img0,gray0,img1,gray1,gtImg;
	std::vector<cv::Point2f> features0,features1;
	std::vector<uchar>status;
	std::vector<float>err;
	std::vector<float> blkWeights;
	for (int i=start; i<=end; i++)
	{
		sprintf(fileName,"%s//in%06d.jpg",path,i);
		img1 = imread(fileName);
		sprintf(fileName,"%s//groundtruth//gt%06d.png",path,i);
		gtImg = imread(fileName);
		cv::cvtColor(gtImg,gtImg,CV_BGR2GRAY);
		cv::cvtColor(img1,gray1,CV_BGR2GRAY);
		if (gray0.empty())
		{
			gray0 = gray1.clone();
			img0 = img1.clone();
		}
		/*nih::Timer timer;
		timer.start();*/
		KLTFeaturesMatching(gray1,gray0,features1,features0,5000,0.05,5);
		/*timer.stop();
		std::cout<<"klt tracking "<<timer.seconds()*1000<<"ms\n";
		timer.start();*/
		//BC2FFeaturePointsRefineHistogram(width,height,features1,features0,blkWeights,4,radSize1,thetaSize1,radSize2,thetaSize2);
		C2FFeaturePointsRefineHistogram(width,height,features1,features0,radSize1,thetaSize1,radSize2,thetaSize2);
		/*cv::Mat homo;
		FeaturePointsRefineRANSAC(features1,features0,homo);*/
		std::cout<<"features remians "<<features1.size()<<std::endl;
	/*	timer.stop();
		std::cout<<"refine "<<timer.seconds()*1000<<"ms\n";*/
		//cv::Mat homography;
		//FeaturePointsRefineRANSAC(features1,features0,homography);
		if (i>1)
		{
			bool save = false;
			int ec(0);
			cv::Mat rstImg(size2,CV_8UC3);
			img0.copyTo(rstImg(cv::Rect(0,0,width,height)));
			img1.copyTo(rstImg(cv::Rect(width,0,width,height)));
			for(int j=0; j< features1.size(); j++)
			{
				int x = (int)(features1[j].x+0.5);
				int y = (int)(features1[j].y+0.5);
				if (gtImg.data[x+y*width] == 0xff)
				{
					cv::line(rstImg,cv::Point(features0[j].x,features0[j].y),cv::Point(features1[j].x+width,features1[j].y),cv::Scalar(255,0,0));
					ec++;
					save = true;
				}
			}
			if (save)
			{
				sprintf(fileName,".//error//%d_error_%d.jpg",i,ec);
				cv::imwrite(fileName,rstImg);
			}
		}
		cv::swap(img0,img1);
		cv::swap(gray0,gray1);

	}
	


}

void TestBlockWarping()
{
	int start = 1;
	int end = 30;
	int width = 640;
	int height = 480;
	float warpErr = 0;
	int quadWidth = 8;
	cv::Size size2(width*2,height);
	std::vector<cv::Mat> homographies;
	std::vector<float> blkWeights(quadWidth*quadWidth,0);
	const char path[] = "..//moseg//cars2";
	char dstPath[200];
	sprintf(dstPath,"..//warpRst//mywarp//");
	CreateDir(dstPath);
	char fileName[200];
	cv::Mat img1,img0,gray1,gray0,wimg,gtImg;
	std::vector<cv::Point2f> features1,features0;
	std::vector<cv::Point2f> sf1,sf0;

	BlockWarping blkWarping(width,height,quadWidth);
	ASAPWarping asapWarping(width,height,quadWidth,1.0);
	for (int i=start; i<=end; i++)
	{
		printf("%d------------\n",i);
		sprintf(fileName,"%s//in%06d.jpg",path,i);
		img1 = imread(fileName);
		sprintf(fileName,"%s//groundtruth//gt%06d.png",path,i);
		gtImg = imread(fileName);
		cv::cvtColor(gtImg,gtImg,CV_BGR2GRAY);
		cv::cvtColor(img1,gray1,CV_BGR2GRAY);
		if (gray0.empty())
		{
			gray0 = gray1.clone();
			img0 = img1.clone();
		}
		/*nih::Timer timer;
		timer.start();*/
		KLTFeaturesMatching(gray1,gray0,features1,features0,5000,0.05,5);
		//SURFFeaturesMatching(gray1,gray0,features1,features0);
		/*timer.stop();
		std::cout<<"klt tracking "<<timer.seconds()*1000<<"ms\n";*/
		nih::Timer timer;
		timer.start();
		//BC2FFeaturePointsRefineHistogram(width,height,features1,features0,blkWeights,4,radSize1,thetaSize1,radSize2,thetaSize2);
		//C2FFeaturePointsRefineHistogram(width,height,features1,features0,3,1,1,3);
		
		//FeaturePointsRefineHistogram(width,height,features1,features0,10,36);
		cv::Mat homo;
		FeaturePointsRefineRANSAC(features1,features0,homo,0.1);
		if (i>1)
		{
			bool save = false;
			int ec(0);
			cv::Mat rstImg(size2,CV_8UC3);
			img0.copyTo(rstImg(cv::Rect(0,0,width,height)));
			img1.copyTo(rstImg(cv::Rect(width,0,width,height)));
			for(int j=0; j< features1.size(); j++)
			{
				int x = (int)(features1[j].x+0.5);
				int y = (int)(features1[j].y+0.5);
				if (gtImg.data[x+y*width] == 0xff)
				{
					cv::line(rstImg,cv::Point(features0[j].x,features0[j].y),cv::Point(features1[j].x+width,features1[j].y),cv::Scalar(255,0,0));
					ec++;
					save = true;
				}
			}
			if (save)
			{
				sprintf(fileName,"..//warpRst//mywarp//error//%d_error_%d.jpg",i,ec);
				cv::imwrite(fileName,rstImg);
			}
		}
		timer.stop();
		std::cout<<"featues refine "<<timer.seconds()*1000<<std::endl;
		std::cout<<"features remians "<<features1.size()<<std::endl;

		//nih::Timer timer;
		//timer.start();
		//asapWarping.CreateSmoothCons(1.0);	
		///*timer.stop();
		//std::cout<<"create smooth cons "<<timer.seconds()*1000<<std::endl;
		//timer.start();*/
		//asapWarping.SetControlPts(features1,features0);
		///*timer.stop();
		//std::cout<<"SetControlPts "<<timer.seconds()*1000<<std::endl;
		//timer.start();*/
		//asapWarping.Solve();
		///*timer.stop();
		//std::cout<<"solve "<<timer.seconds()*1000<<std::endl;
		//timer.start();*/
		//asapWarping.Warp(img1,wimg);
		///*timer.stop();
		//std::cout<<"warp "<<timer.seconds()*1000<<std::endl;*/
		//asapWarping.Reset();
		//timer.stop();
		//std::cout<<"asap warping "<<timer.seconds()*1000<<std::endl;

		
		
		timer.start();
		blkWarping.SetFeaturePoints(features1,features0);

		timer.stop();
		std::cout<<"set featurepoints "<<timer.seconds()*1000<<std::endl;
		timer.start();
		blkWarping.CalcBlkHomography();
		timer.stop();
		std::cout<<"calc homo "<<timer.seconds()*1000<<std::endl;
		//timer.start();
		//blkWarping.Warp(img1,wimg);
		GpuTimer gtimer;
		gtimer.Start();
		cv::gpu::GpuMat dimg,dwimg;
		dimg.upload(img1);
		blkWarping.GpuWarp(dimg,dwimg);
		dwimg.download(wimg);
		gtimer.Stop();

		//timer.stop();
		std::cout<<"blk warp "<<gtimer.Elapsed()<<std::endl;
		cv::Mat flow;
		blkWarping.getFlow(flow);
		timer.start();
		blkWarping.Reset();
		timer.stop();
		std::cout<<"reset "<<timer.seconds()*1000<<std::endl;
		
		timer.start();
		/*cv::Mat homo;
		findHomographyEqa(features1,features0,homo);*/
		/*cv::warpPerspective(img1,wimg,homo,img1.size());
		timer.stop();
		std::cout<<"warpPerspective "<<timer.seconds()*1000<<std::endl;*/
		/*cv::imshow("warped image",wimg);
		cv::waitKey();*/
		sprintf(fileName,"%swin%06d.jpg",dstPath,i);
		cv::imwrite(fileName,wimg);
		cv::Mat diff;
		cv::absdiff(wimg,img0,diff);
		sprintf(fileName,"%sdiff%06d.jpg",dstPath,i);
		cv::Scalar sum= cv::sum(diff);
		std::cout<<sum[0]<<std::endl;
		warpErr += sum[0];
		cv::imwrite(fileName,diff);
		
		cv::swap(img0,img1);
		cv::swap(gray0,gray1);
	}
	std::cout<<"avg err "<<warpErr/(end-start+1);
}
float L2Dist(const float4& a, const float4& b)
{
	double dx = a.x - b.x;
	double dy = a.y - b.y;
	double dz = a.z - b.z;
	double dw = a.w - b.w;
	return sqrt(dx*dx + dy*dy + dz*dz + dw*dw);
}
void LBPHistogram(cv::Mat& LbpImg, std::vector<uint2>& poses,std::vector<float>& histogram)
{
	int width = LbpImg.cols;
	int height = LbpImg.rows;
	int binSize = 59;
	histogram.resize(binSize);
	memset(&histogram[0],0,sizeof(float)*histogram.size());
	for(int i=0; i< poses.size(); i++)
	{
		int idx = poses[i].x + poses[i].y*width;
		uchar ptr = *(uchar*)(LbpImg.data + idx);
		histogram[ptr]++;
	}
	
}
void RGBHistogram(cv::Mat& fImg, std::vector<uint2>& poses, int bins, float min, float max, std::vector<float>& histogram)
{
	int width = fImg.cols;
	int height = fImg.rows;
	int typeLen = 4;//4 for float 
	float step = (max-min)/bins;
	histogram.resize(bins*bins*bins);
	memset(&histogram[0],0,sizeof(float)*histogram.size());
	for(int i=0; i< poses.size(); i++)
	{
		int idx = poses[i].x + poses[i].y*width;
		float*ptr = (float*)(fImg.data + idx*12);
		int id = 0;
		int s = 1;
		for(int c=0; c<3; c++)
		{
			id += s*min(ceil(ptr[c] /step),bins-1);
			s*=bins;
		}
		histogram[id]++;

		
	}
	//cv::normalize(histogram,histogram,1.0,0.0,NORM_MINMAX);
}
void HOG(cv::Mat&mag, cv::Mat& ang, std::vector<uint2>& poses, int bins, std::vector<float>& histogram)
{
	int width = mag.cols;
	int height = ang.rows;
	float step = 360/bins;
	histogram.resize(bins);
	memset(&histogram[0],0,sizeof(float)*histogram.size());
	for(int i=0; i< poses.size(); i++)
	{
		int idx = poses[i].x + poses[i].y*width;
		float m = *(float*)(mag.data + idx*4);
		float a = *(float*)(ang.data + idx*4);
		int id = min(floor(a/step),bins-1);
		histogram[id] += m;
	}
}
void SuperPixelRegionMerging(int width, int height, int step,const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented, 
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors,float confidence = 0.6)
{
	//std::ofstream file("mergeOut.txt");
	const int dx4[] = {-1,0,1,0};
	const int dy4[] = {0,-1,0,1};
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int spWidth = (width+step-1)/step;
	int spHeight = (height+step-1)/step;
	float pixDist(0);
	float regMaxDist = threshold;
	regSizes.clear();
	int regSize(0);
	//当前新标签
	int curLabel(0);
	int imgSize = spWidth*spHeight;
	char* visited = new char[imgSize];
	memset(visited ,0,imgSize);
	memset(segmented,0,sizeof(int)*width*height);
	std::vector<cv::Point2i> neighbors;
	float4 regMean;
	std::vector<int> singleLabels;
	//region growing 后的新label
	std::vector<int> newLabels;
	
	newLabels.resize(imgSize);
	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	boundarySet.insert(rand()%imgSize);
	//boundarySet.insert(3);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;
	
	while(!boundarySet.empty())
	{
		//std::cout<<boundarySet.size()<<std::endl;
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		//file<<"seed: "<<label<<"\n";
		visited[label] = true;

		labelGroup.push_back(label);
		
		//newLabels[label] = curLabel;
		boundarySet.erase(itr);
		SLICClusterCenter cc = centers[label];
		int k = cc.xy.x;
		int j = cc.xy.y;		
		float4 regColor = cc.rgb;
		int ix = label%spWidth;
		int iy = label/spWidth;
		pixDist = 0;
		regSize = 1;
		//segmented[ix+iy*spWidth] = curLabel;
		/*for(int j=0; j<neighbors.size(); j++)
		{
			size_t idx = neighbors[j].x+neighbors[j].y*spWidth;
			visited[idx] = false;
		}*/
		neighbors.clear();
		regMean = cc.rgb;
		
		
		while(pixDist < regMaxDist && regSize<imgSize)
		{
			//file<<"iy:"<<iy<<"ix:"<<ix<<"\n";
			
			for(int d=0; d<4; d++)
			{
				int x = ix+dx4[d];
				int y = iy + dy4[d];
				if (x>=0 && x<spWidth && y>=0 && y<spHeight && !visited[x+y*spWidth])
				{
					neighbors.push_back(cv::Point2i(x,y));
					visited[x+y*spWidth] = true;
					
				}
			}
			//file<<"	neighbors: ";
			for (int i=0; i<neighbors.size(); i++)
			{
				int x = neighbors[i].x;
				int y = neighbors[i].y;
				//file<<x+y*spWidth<<"("<<y<<","<<x<<"),";
			}
			//file<<"\n";
			int idxMin = 0;
			pixDist = 255;
			if (neighbors.size() == 0)
				break;

			for(int j=0; j<neighbors.size(); j++)
			{
				size_t idx = neighbors[j].x+neighbors[j].y*spWidth;
				float rd = cv::compareHist(histograms[idx],histograms[label],CV_COMP_BHATTACHARYYA);
				float hd = cv::compareHist(lhistograms[idx],lhistograms[label],CV_COMP_BHATTACHARYYA);
				float dist = confidence*rd + 	hd*(1-confidence);
				/*float4 acolor = centers[idx].rgb;
				float cd = L2Dist(acolor,regColor)/255;
				float hd = cv::compareHist(lhistograms[idx],lhistograms[label],CV_COMP_BHATTACHARYYA);		
				float dist = confidence*cd + (1-confidence)*hd;*/
				//float dist = (abs(dx) + abs(dy)+ abs(dz))/3;
				
				if (dist < pixDist)
				{
					pixDist = dist;
					idxMin = j;
				}				
			}
			if (pixDist < regMaxDist)
			{
				ix = neighbors[idxMin].x;
				iy = neighbors[idxMin].y;
				int minIdx =ix + iy*spWidth;			
				//file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"\n";
				/*regColor.x = (regColor.x*regSize + centers[minIdx].rgb.x)/(regSize+1);
				regColor.y = (regColor.y*regSize + centers[minIdx].rgb.y)/(regSize+1);
				regColor.z = (regColor.z*regSize + centers[minIdx].rgb.z)/(regSize+1);*/
				regColor.x += centers[minIdx].rgb.x;
				regColor.y += centers[minIdx].rgb.y;
				regColor.z += centers[minIdx].rgb.z;
				regSize++;
				labelGroup.push_back(minIdx);
				for(int i=0; i<histograms[label].size(); i++)
				{
					histograms[label][i] += histograms[minIdx][i];

				}
				//cv::normalize(histogram,histogram,1,0,NORM_L1 );
				for(int i=0; i<lhistograms[label].size(); i++)
				{
					lhistograms[label][i] += lhistograms[minIdx][i];
				}
				//cv::normalize(lhistogram,lhistogram,1,0,NORM_L1 );
				visited[minIdx] = true;
				/*segmented[minIdx] = k;*/
				//result.data[minIdx] = 0xff;
				//smask.data[minIdx] = 0xff;
				neighbors[idxMin] = neighbors[neighbors.size()-1];
				neighbors.pop_back();
				std::set<int>::iterator itr =boundarySet.find(minIdx);
				if ( itr!= boundarySet.end())
				{
					boundarySet.erase(itr);
				}
			}
			else
			{
				ix = neighbors[idxMin].x;
				iy = neighbors[idxMin].y;
				int minIdx =ix + iy*spWidth;			
				//file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"overpass threshold "<<regMaxDist<<"\n";
			}
		}
		newHistograms.push_back(histograms[label]);		
		for(int i=0; i<labelGroup.size(); i++)
		{
			newLabels[labelGroup[i]] = curLabel;
		}
		regColor.x/=regSize;
		regColor.y/=regSize;
		regColor.z/=regSize;
		regAvgColors.push_back(regColor);
		regSizes.push_back(regSize);
		curLabel++;		
		for(int i=0; i<neighbors.size(); i++)
		{
			int label = neighbors[i].x + neighbors[i].y*spWidth;
			visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);
			
		}
		if (regSize <2)
			singleLabels.push_back(label);
	}
	
	for(int i=0; i<newLabels.size(); i++)
	{
		int x = centers[i].xy.x;
		int y = centers[i].xy.y;
		for(int dx= -step; dx<=step; dx++)
		{
			int sx = x+dx;
			if (sx<0 || sx>=width)
				continue;
			for(int dy = -step; dy<=step; dy++)
			{
				
				int sy = y + dy;
				if(  sy>=0 && sy<height)
				{
					int idx = sx+sy*width;
					if (labels[idx] == i)
						segmented[idx] = newLabels[i];
				}
			}
		}

	}
	//for (int i=0; i<singleLabels.size(); i++)
	//{
	//	int label = singleLabels[i];
	//	int ix = label%spWidth;
	//	int iy = label/spWidth;
	//	std::vector<int> ulabel;
	//	//对单个超像素，检查其周围是还有单个超像素
	//	for(int d=0; d<4; d++)
	//	{
	//		int x = ix+dx4[d];
	//		int y = iy + dy4[d];
	//		if (x>=0 && x<spWidth && y>=0 && y<spHeight)
	//		{
	//			int nlabel = x+y*spWidth;		
	//			if (std::find(ulabel.begin(),ulabel.end(),newLabels[nlabel]) == ulabel.end())
	//				ulabel.push_back(newLabels[nlabel]);
	//		}
	//		
	//	}
	//	if (ulabel.size()<=2)
	//			newLabels[label] = ulabel[0];
	//}
	delete[] visited;
	//delete[] segmented;
	//file.close();
}
struct RegInfo
{
	RegInfo(){}
	RegInfo(int l, int _x, int _y, float d):dist(d),label(l),x(_x),y(_y){} 
	float dist;
	int label;
	int x,y;
};
//结构体的比较方法 改写operator()  
struct RegInfoCmp  
{  
    bool operator()(const RegInfo &na, const RegInfo &nb)  
    {  
		return na.dist > nb.dist;
    }  
};
typedef std::priority_queue<RegInfo,std::vector<RegInfo>,RegInfoCmp> RegInfos;
void SuperPixelRegionMergingFast(int width, int height, int step,const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented, 
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors,float confidence = 0.6)
{
	//std::ofstream file("mergeOut.txt");
	const int dx4[] = {-1,0,1,0};
	const int dy4[] = {0,-1,0,1};
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int spWidth = (width+step-1)/step;
	int spHeight = (height+step-1)/step;
	float pixDist(0);
	float regMaxDist = threshold;
	regSizes.clear();
	int regSize(0);
	//当前新标签
	int curLabel(0);
	int imgSize = spWidth*spHeight;
	char* visited = new char[imgSize];
	memset(visited ,0,imgSize);
	memset(segmented,0,sizeof(int)*width*height);
	//std::vector<cv::Point2i> neighbors;
	
	float4 regMean;
	std::vector<int> singleLabels;
	//region growing 后的新label
	std::vector<int> newLabels;
	
	newLabels.resize(imgSize);
	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	boundarySet.insert(rand()%imgSize);
	//boundarySet.insert(3);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;
	
	while(!boundarySet.empty())
	{
		//std::cout<<boundarySet.size()<<std::endl;
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		//file<<"seed: "<<label<<"\n";
		visited[label] = true;

		labelGroup.push_back(label);
		
		//newLabels[label] = curLabel;
		boundarySet.erase(itr);
		SLICClusterCenter cc = centers[label];
		int k = cc.xy.x;
		int j = cc.xy.y;		
		float4 regColor = cc.rgb;
		int ix = label%spWidth;
		int iy = label/spWidth;
		pixDist = 0;
		regSize = 1;
		//segmented[ix+iy*spWidth] = curLabel;
		/*for(int j=0; j<neighbors.size(); j++)
		{
			size_t idx = neighbors[j].x+neighbors[j].y*spWidth;
			visited[idx] = false;
		}*/
		RegInfos neighbors, tneighbors;
		regMean = cc.rgb;
		
		
		while(pixDist < regMaxDist && regSize<imgSize)
		{
			//file<<"iy:"<<iy<<"ix:"<<ix<<"\n";
			
			for(int d=0; d<4; d++)
			{
				int x = ix+dx4[d];
				int y = iy + dy4[d];
				size_t idx = x+y*spWidth;
				if (x>=0 && x<spWidth && y>=0 && y<spHeight && !visited[idx])
				{
					
					visited[idx] = true;
					float rd = cv::compareHist(histograms[idx],histograms[label],CV_COMP_BHATTACHARYYA);
					float hd = cv::compareHist(lhistograms[idx],lhistograms[label],CV_COMP_BHATTACHARYYA);
					float dist = confidence*rd + 	hd*(1-confidence);
					neighbors.push(RegInfo(idx,x,y,dist));
				}
			}
			//file<<"neighbors: ";
			/*vector<RegInfo> *vtor = (vector<RegInfo> *)&neighbors;
			for(int i=0; i<vtor->size(); i++)
			{
				int label = ((RegInfo)vtor->operator [](i)).label;
				int x = ((RegInfo)vtor->operator [](i)).x;
				int y=  ((RegInfo)vtor->operator [](i)).y;
				file<<label<<"("<<y<<","<<x<<"),";
			}*/
			//file<<"\n";
			if (neighbors.empty())
				break;
			RegInfo sp = neighbors.top();
			pixDist = sp.dist;
			
			int minIdx = sp.label;
			ix = sp.x;
			iy = sp.y;
			if (pixDist < regMaxDist)
			{
				neighbors.pop();
				//file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"\n";
				float tmpx = regColor.x;
				float tmpy = regColor.y;
				float tmpz = regColor.z;
				regColor.x = (regColor.x*regSize + centers[minIdx].rgb.x)/(regSize+1);
				regColor.y = (regColor.y*regSize + centers[minIdx].rgb.y)/(regSize+1);
				regColor.z = (regColor.z*regSize + centers[minIdx].rgb.z)/(regSize+1);
				float t = 2.0;
				float dx = abs(tmpx - regColor.x);
				float dy = abs(tmpy - regColor.y);
				float dz = abs(tmpz - regColor.z);
			
				/*regColor.x += centers[minIdx].rgb.x;
				regColor.y += centers[minIdx].rgb.y;
				regColor.z += centers[minIdx].rgb.z;*/
				regSize++;
				labelGroup.push_back(minIdx);
				
				for(int i=0; i<histograms[label].size(); i++)
				{
					histograms[label][i] += histograms[minIdx][i];

				}
				//cv::normalize(histogram,histogram,1,0,NORM_L1 );
				for(int i=0; i<lhistograms[label].size(); i++)
				{
					lhistograms[label][i] += lhistograms[minIdx][i];
				}
				//cv::normalize(lhistogram,lhistogram,1,0,NORM_L1 );
				visited[minIdx] = true;
				if (sqrt(dx*dx +dy*dy +dz*dz) > t)
				{
					while(!tneighbors.empty())
						tneighbors.pop();
					while(!neighbors.empty())
					{
						RegInfo sp = neighbors.top();
						neighbors.pop();
						float rd = cv::compareHist(histograms[sp.label],histograms[label],CV_COMP_BHATTACHARYYA);
						float hd = cv::compareHist(lhistograms[sp.label],lhistograms[label],CV_COMP_BHATTACHARYYA);
						sp.dist =  confidence*rd + 	hd*(1-confidence);
						tneighbors.push(sp);
					}
					std::swap(neighbors,tneighbors);
				}
				/*segmented[minIdx] = k;*/
				//result.data[minIdx] = 0xff;
				//smask.data[minIdx] = 0xff;
				
				std::set<int>::iterator itr =boundarySet.find(minIdx);
				if ( itr!= boundarySet.end())
				{
					boundarySet.erase(itr);
				}
			}
			else
			{			
				//file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"overpass threshold "<<regMaxDist<<"\n";
			}
		}
		newHistograms.push_back(histograms[label]);		
		for(int i=0; i<labelGroup.size(); i++)
		{
			newLabels[labelGroup[i]] = curLabel;
		}
	/*	regColor.x/=regSize;
		regColor.y/=regSize;
		regColor.z/=regSize;*/
		regAvgColors.push_back(regColor);
		regSizes.push_back(regSize);
		curLabel++;		
		vector<RegInfo> *vtor = (vector<RegInfo> *)&neighbors;
		for(int i=0; i<vtor->size(); i++)
		{
			int label = ((RegInfo)vtor->operator [](i)).label;
			visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);
			
		}
		if (regSize <2)
			singleLabels.push_back(label);
	}
	
	for(int i=0; i<newLabels.size(); i++)
	{
		int x = centers[i].xy.x;
		int y = centers[i].xy.y;
		for(int dx= -step; dx<=step; dx++)
		{
			int sx = x+dx;
			if (sx<0 || sx>=width)
				continue;
			for(int dy = -step; dy<=step; dy++)
			{
				
				int sy = y + dy;
				if(  sy>=0 && sy<height)
				{
					int idx = sx+sy*width;
					if (labels[idx] == i)
						segmented[idx] = newLabels[i];
				}
			}
		}

	}
	//对单个超像素，检查其是否在大区域之中（周边三个以上label一样）
	for (int i=0; i<singleLabels.size(); i++)
	{
		int label = singleLabels[i];
		int ix = label%spWidth;
		int iy = label/spWidth;
		std::vector<int> ulabel;
		
		for(int d=0; d<4; d++)
		{
			int x = ix+dx4[d];
			int y = iy + dy4[d];
			if (x>=0 && x<spWidth && y>=0 && y<spHeight)
			{
				int nlabel = x+y*spWidth;		
				if (std::find(ulabel.begin(),ulabel.end(),newLabels[nlabel]) == ulabel.end())
					ulabel.push_back(newLabels[nlabel]);
			}
			
		}
		if (ulabel.size()<=2)
				newLabels[label] = ulabel[0];
	}
	delete[] visited;
	//delete[] segmented;
	//file.close();
}
void histogram()
{
	cv::Mat fimg1,fimg2,fimg3;
	cv::Mat img1 = cv::imread("p1.png");
	img1.convertTo(fimg1,CV_32FC3,1/255.0);
	cv::cvtColor(img1,img1,CV_BGR2GRAY);
	cv::Mat img2 = cv::imread("p2.png");
	img2.convertTo(fimg2,CV_32FC3,1/255.0);
	cv::cvtColor(img2,img2,CV_BGR2GRAY);
	cv::Mat img3 = cv::imread("p3.png");
	img3.convertTo(fimg3,CV_32FC3,1/255.0);
	cv::cvtColor(img3,img3,CV_BGR2GRAY);
	cv::Mat lbp1,lbp2,lbp3;
	LBPGRAY(img1,lbp1);
	GaussianBlur( img1, img1, cv::Size(3,3), 0, 0, BORDER_DEFAULT );
	GaussianBlur( img2, img2, cv::Size(3,3), 0, 0, BORDER_DEFAULT );
	GaussianBlur( img2, img3, cv::Size(3,3), 0, 0, BORDER_DEFAULT );
	cv::Mat dx1,dy1,dx2,dy2,dx3,dy3;
	cv::Mat mag1,ang1,mag2,ang2,mag3,ang3;
	cv::Scharr(img1,dx1, CV_32F ,1,0);
	cv::Scharr(img1,dy1, CV_32F ,0,1);
	cv::cartToPolar(dx1,dy1,mag1,ang1,true);
	cv::Scharr(img2,dx2, CV_32F ,1,0);
	cv::Scharr(img2,dy2, CV_32F ,0,1);
	cv::cartToPolar(dx2,dy2,mag2,ang2,true);
	cv::Scharr(img3,dx3, CV_32F ,1,0);
	cv::Scharr(img3,dy3, CV_32F ,0,1);
	cv::cartToPolar(dx3,dy3,mag3,ang3,true);
	std::vector<float> hog1,hog2,hog3;

	cv::imwrite("lbp1.png",lbp1);
	LBPGRAY(img2,lbp2);
	LBPGRAY(img3,lbp3);
	std::vector<uint2> poses;
	for(int i=1; i< img1.rows-1; i++)
	{
		for(int j=1; j<img1.cols-1; j++)
			poses.push_back(make_uint2(j,i));
	}
	std::vector<float> hist1,hist2,hist3;
	RGBHistogram(fimg1,poses,12,0,1,hist1);
	RGBHistogram(fimg2,poses,12,0,1,hist2);
	RGBHistogram(fimg3,poses,12,0,1,hist3);
	HOG(mag1,ang1,poses,36,hog1);
	HOG(mag2,ang2,poses,36,hog2);
	HOG(mag3,ang3,poses,36,hog3);
	/*cv::normalize(hist1,hist1,1,0,NORM_L1 );
	cv::normalize(hist2,hist2,1,0,NORM_L1 );
	cv::normalize(hist3,hist3,1,0,NORM_L1 );*/
	
	float cd1 = cv::compareHist(hist1,hist2,CV_COMP_BHATTACHARYYA);
	float cd2 = cv::compareHist(hist1,hist3,CV_COMP_BHATTACHARYYA);
	float cd3 = cv::compareHist(hist2,hist3,CV_COMP_BHATTACHARYYA);

	float hd1 = cv::compareHist(hog1,hog2,CV_COMP_BHATTACHARYYA);
	float hd2 = cv::compareHist(hog1,hog3,CV_COMP_BHATTACHARYYA);
	float hd3 = cv::compareHist(hog2,hog3,CV_COMP_BHATTACHARYYA);
	std::cout<<cd1<<" "<<cd2<<" "<<cd3<<std::endl;
	std::cout<<hd1<<" "<<hd2<<" "<<hd3<<std::endl;
	DrawHistogram(hist1,hist1.size(),"hist1");
	DrawHistogram(hist2,hist2.size(),"hist2");
	DrawHistogram(hist3,hist3.size(),"hist3");
	int sum1(0),sum2(0),sum3(0);
	for(int i=0; i<hist2.size(); i++)
	{
		sum1 += hist1[i];
		sum2 += hist2[i];
		sum3 += hist3[i];
	}
	DrawHistogram(hog1,hog1.size(),"ghist1");
	DrawHistogram(hog2,hog2.size(),"ghist2");
	DrawHistogram(hog3,hog3.size(),"ghist3");
	cv::waitKey();
}

void SaliencyTest(const char* path,int pid, int width, int height, int step)
{
	/*histogram();*/
	
	
	char imgName[200];
	sprintf(imgName,"%s\\in%06d.jpg",path,pid);
	cv::Mat img = cv::imread(imgName);
	cv::Mat fimg,gray,lbpImg;
	img.convertTo(fimg,CV_32FC3,1.0/255);
	cv::cvtColor(img,gray,CV_BGR2GRAY);
	cv::GaussianBlur(gray,gray,cv::Size(3,3),0);
	cv::Mat dx,dy,ang,mag;
	cv::Scharr(gray,dx,CV_32F,1,0);
	cv::Scharr(gray,dy,CV_32F,0,1);
	cv::cartToPolar(dx,dy,mag,ang,true);

 	/*LBPGRAY(gray,lbpImg);*/
	cv::Mat simg;
	cv::Mat diff(height,width,CV_8U);
	diff = cv::Scalar(0);
	SuperpixelComputer computer(width,height,step);
	computer.ComputeSuperpixel(img);

	computer.GetVisualResult(img,simg);
	sprintf(imgName,"%s//superpixel_%d.jpg",path,pid);
	cv::imwrite(imgName,simg);

	//计算每个超像素与周围超像素的差别
	int spHeight = computer.GetSPHeight();
	int spWidth = computer.GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int num(0);
	computer.GetSuperpixelResult(num,labels,centers);
	//每个超像素中包含的像素以及位置
	std::vector<std::vector<uchar4>> pixels(num);
	std::vector<std::vector<uint2>> pos(num);
	std::vector<std::vector<float>> histogram(num);
	std::vector<std::vector<float>> lhistogram(num);
	/*std::vector<uchar3> avgColor(num);*/
	float rgbHConfidence = 0.6;
	int k=0;
	cv::Mat avgImg(height,width,CV_8UC3);
	for(int i=0; i< spHeight; i++)
	{
		for(int j=0; j<spWidth; j++)
		{
			int idx = i*spWidth+j;
			int x = int(centers[idx].xy.x+0.5);
			int y = int(centers[idx].xy.y+0.5);
			for( int m=-step+y; m<=step+y; m++)
			{
				if (m<0 || m>= height)
					continue;
				cv::Vec3b* ptr = img.ptr<cv::Vec3b>(m);
				for(int n=-step+x; n<=step+x; n++)
				{
					if (n<0 || n>=width)
						continue;
					int id = m*width+n;
					if (labels[id] == idx)
					{
						pixels[idx].push_back(make_uchar4(ptr[n][0],ptr[n][1],ptr[n][2],0));
						pos[idx].push_back(make_uint2(n,m));
						uchar* ptr = (avgImg.data+id*3);
						ptr[0] = centers[idx].rgb.x;
						ptr[1] = centers[idx].rgb.y;
						ptr[2] = centers[idx].rgb.z;
					}
				}
			}
		}
	}
	cv::imwrite("avgImg.jpg",avgImg);
	//计算每个超像素的直方图
	for(int i=0; i<pos.size(); i++)
	{

		RGBHistogram(fimg,pos[i],12,0,1,histogram[i]);
		//cv::normalize(histogram[i],histogram[i],1,0,NORM_L1 );

		HOG(mag,ang,pos[i],36,lhistogram[i]);
		//cv::normalize(lhistogram[i],lhistogram[i],1,0,NORM_L1 );
		//LBPHistogram(lbpImg,pos[i],lhistogram[i]);
		//DrawHistogram(histogram,histogram.size());
		//cv::waitKey();
	}
	/*int minX = width;
	int maxX = 0;
	int minY = height;
	int maxY = 0;
	for(int i=0; i<pos[135].size(); i++)
	{
		if (pos[135][i].x < minX)
			minX = pos[135][i].x;
		else if (pos[135][i].x > maxX)
			maxX = pos[135][i].x;
		if (pos[135][i].y < minY)
			minY = pos[135][i].y;
		else if (pos[135][i].y > maxY)
			maxY = pos[135][i].y;
	}
	cv::Mat p1 = img(cv::Rect(minX,minY,maxX-minX+1,maxY-minY+1));
	cv::imwrite("p1.png",p1);
	int minX2 = width;
	int maxX2 = 0;
	int minY2 = height;
	int maxY2 = 0;
	for(int i=0; i<pos[134].size(); i++)
	{
		if (pos[134][i].x < minX2)
			minX2 = pos[134][i].x;
		else if (pos[134][i].x > maxX2)
			maxX2 = pos[134][i].x;
		if (pos[134][i].y < minY2)
			minY2 = pos[134][i].y;
		else if (pos[134][i].y > maxY2)
			maxY2 = pos[134][i].y;
	}
	cv::Mat p2 = img(cv::Rect(minX2,minY2,maxX2-minX2+1,maxY2-minY2+1));
	cv::imwrite("p2.png",p2);
	int minX3 = width;
	int maxX3 = 0;
	int minY3 = height;
	int maxY3 = 0;
	for(int i=0; i<pos[133].size(); i++)
	{
		if (pos[133][i].x < minX3)
			minX3 = pos[133][i].x;
		else if (pos[133][i].x > maxX3)
			maxX3 = pos[133][i].x;
		if (pos[134][i].y < minY3)
			minY3 = pos[133][i].y;
		else if (pos[133][i].y > maxY3)
			maxY3 = pos[133][i].y;
	}
	cv::Mat p3 = img(cv::Rect(minX3,minY3,maxX3-minX3+1,maxY3-minY3+1));
	cv::imwrite("p3.png",p3);
	std::vector<float> h = lhistogram[133];
	float d1 = cv::compareHist(lhistogram[135],lhistogram[134],CV_COMP_BHATTACHARYYA);
	float d2 = cv::compareHist(lhistogram[133],lhistogram[134],CV_COMP_BHATTACHARYYA);
	DrawHistogram(lhistogram[135],lhistogram[135].size(),"lhistogram_135");
	DrawHistogram(lhistogram[134],lhistogram[134].size(),"lhistogram_134");
	DrawHistogram(histogram[135],histogram[135].size(),"histogram_135");
	DrawHistogram(histogram[134],histogram[134].size(),"histogram_134");
	cv::waitKey();*/
	
	//计算平均相邻超像素距离之间的距离
	const int dx4[] = {-1,0,1,0};
	const int dy4[] = {0,-1,0,1};
	float avgCDist(0);
	float avgGDist(0);
	float avgDist(0);
	int nc(0);
	for (int i=0; i<spHeight; i++)
	{
		for(int j=0; j<spWidth; j++)
		{
			int idx = i*spWidth + j;			
			for(int n=0; n<4; n++)
			{
				int dy = i+dy4[n];
				int dx = j+dx4[n];
				if (dy>=0 && dy<spHeight && dx >=0 && dx < spWidth)
				{
					int nIdx = dy*spWidth + dx;
					nc++;
					avgCDist += cv::compareHist(histogram[idx],histogram[nIdx],CV_COMP_BHATTACHARYYA);
					avgGDist += cv::compareHist(lhistogram[idx],lhistogram[nIdx],CV_COMP_BHATTACHARYYA);
					avgDist += L2Dist(centers[idx].rgb,centers[nIdx].rgb);
				}
			}
			
		}
	}
	avgCDist/=nc;
	avgGDist/=nc;
	avgDist/=(nc*255);
	//std::vector<int>minIdx(num);
	//std::vector<float>minDist(num);
	////计算每个超像素的最接近的邻居
	//for(int i=0; i<num; i++)
	//{
	//	float minD = 255;
	//	int minId(i);
	//	for(int j=0; j<num; j++)
	//	{
	//		if (j!=i)
	//		{
	//			float dist = cv::compareHist(histogram[i],histogram[j]);
	//			if (dist<minD)
	//			{
	//				minId = j;
	//				minD = dist;
	//			}
	//		}
	//	}
	//	minIdx[i] = minId;
	//	minDist[i] = minD;

	//}
	//

	//cv::Mat salImg(img.size(),CV_8U);
	//for(int i=0; i<num; i++)
	//{
	//	for(int j=0; j<pos[i].size(); j++)
	//	{
	//		int idx = pos[i][j].x+pos[i][j].y*width;
	//		salImg.data[idx] = floor(minDist[i]*255);
	//	}
	//}
	//for(int i=0; i< minIdx.size();i++)
	//	std::cout<<i<<":"<<minIdx[i]<<", ";
	std::vector<std::vector<float>> newHistograms;
	std::vector<std::vector<uint2>> newpos(num);
	std::vector<int>regSizes;
	int* segmented = new int[width*height];
	rgbHConfidence = (avgGDist)/(avgCDist+avgGDist);
	//rgbHConfidence = (avgGDist)/(avgDist+avgGDist);
	float threshold = ((avgCDist*rgbHConfidence+(1-rgbHConfidence)*avgGDist))*1.2;	
	std::cout<<"threshold: "<<threshold<<std::endl;
	//float threshold = (avgDist*rgbHConfidence+(1-rgbHConfidence)*avgGDist);
	std::vector<float4> avgColors;
	nih::Timer timer;
	timer.start();
	//SuperPixelRegionMerging(width,height,step,labels,centers,pos,histogram,lhistogram,newpos,newHistograms,threshold,segmented,regSizes,avgColors,rgbHConfidence);
	SuperPixelRegionMergingFast(width,height,step,labels,centers,pos,histogram,lhistogram,newpos,newHistograms,threshold,segmented,regSizes,avgColors,rgbHConfidence);
	
	timer.stop();
	std::cout<<"merging time: "<<timer.seconds()<<"s\n";
	std::cout<<regSizes.size()<<std::endl;
	//求大区域的颜色
	//假定背景占大部分，大区域门限为n个超像素
	int backRegThreshold = 12;
	std::vector<int> backReg;
	cv::vector<float4> backColors;
	for (int i=0; i< regSizes.size(); i++)
	{
		if (regSizes[i] > backRegThreshold)
		{
			backReg.push_back(i);
		}
	}
	
	float sumRegSize(0);
	for(int i=0; i<backReg.size(); i++)
	{
		sumRegSize+=backReg[i];
		float maxV(0);
		int maxId(0);
		for (int j=1; j<newHistograms[backReg[i]].size(); j++)
		{
			if (newHistograms[backReg[i]][j] > maxV)
			{
				maxV = newHistograms[backReg[i]][j];
				maxId = j;
			}
		}
		float step = 1.f/12;
		int b = maxId%(12);
		int cb = (int)b*step*255;
		maxId = (maxId-b)/12;
		int g = (maxId)%(12);
		int cg = (int)g*step*255;
		maxId = (maxId - g)/12;
		int r = maxId;
		int cr = (int)r*step*255;
		std::cout<<"back color "<<cr<<" "<<cg<<" "<<cb<<"\n";

		cv::Mat pallet(512,512,CV_8UC3);
		cv::Mat pallet2 = pallet.clone();
		pallet = cv::Scalar(avgColors[backReg[i]].x,avgColors[backReg[i]].y,avgColors[backReg[i]].z);
		//cv::imshow("pallet",pallet);
		pallet2 = cv::Scalar(cb,cg,cr);
		/*cv::imshow("pallet2",pallet2);
		cv::waitKey();*/
		backColors.push_back(make_float4(avgColors[backReg[i]].x,avgColors[backReg[i]].y,avgColors[backReg[i]].z,0));
	}
	cv::Mat mask,SalMask;
	mask.create(height,width,CV_8UC3);
	SalMask.create(height,width,CV_8U);
	int spSize = spHeight*spWidth;
	std::vector<int> color(spSize);
	CvRNG rng= cvRNG(cvGetTickCount());
	color[0] = 0;
	for(int i=1;i<spSize;i++)
		color[i] = cvRandInt(&rng);
	// Draw random color
	for(int i=0;i<height;i++)
	{
		cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
		uchar* sptr = SalMask.ptr<uchar>(i);
		for(int j=0;j<width;j++)
		{ 
			int cl = segmented[i*width+j];
			((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = (color[cl])&255;
			((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = (color[cl]>>8)&255;
			((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = (color[cl]>>16)&255;
			//((uchar *)(SalMask.data + i*SalMask.step.p[0]))[j*SalMask.step.p[1]] = 255* (1-1.0*regSizes[cl]/spSize);
			if (std::find(backReg.begin(),backReg.end(),cl) == backReg.end())
			{
				//float sal(0);
				//float sumReg(0);
				///*for(int b=0; b<backReg.size(); b++)
				//{
				//	float dr = backColors[b].x - ptr[j][0];
				//	float dg = backColors[b].y - ptr[j][1];
				//	float db = backColors[b].z - ptr[j][2];
				//	sal += sqrt(dr*dr+dg*dg+db*db)*backReg[b]/sumRegSize;
				//}*/
				//for(int b=0; b<backReg.size(); b++)
				//{
				//	sal += cv::compareHist(newHistograms[cl],newHistograms[backReg[b]],CV_COMP_BHATTACHARYYA)*backReg[b]/sumRegSize;
				//
				//}
				//sptr[j] = min(sal,1.0)*255;
				sptr[j] = (ptr[j][0]+ptr[j][1]+ptr[j][2])/3.0;
			}
			else
				sptr[j] = 0;
		}
	}
		
	//cv::imshow("region merging",mask);
		sprintf(imgName,"%s\\Region%06d.jpg",path,pid);
		cv::imwrite(imgName,mask);
		sprintf(imgName,"%s\\Sal%06d.jpg",path,pid);
		cv::imwrite(imgName,SalMask);
	//cv::imshow("superpixels",simg);
	/*cv::imshow("lbpImg",lbpImg);
	cv::imwrite("lbpImg.png",lbpImg);*/
	/*cv::imshow("salImg",salImg);*/
	//cv::waitKey();
	delete[] segmented;
}
void TestSaliency(int argC, char** argv)
{
	int start = atoi(argv[1]);
	int end = atoi(argv[2]);
	char* path = argv[3];
	int width = atoi(argv[4]);
	int height = atoi(argv[5]);
	int step = atoi(argv[6]);
	
	for (int i=start; i<=end; i++)
	{
		
		SaliencyTest((const char*)path,i,width,height,step);
	}
}


void TestLBP()
{
	cv::Mat img = cv::imread("..//moseg//cars1//in000001.jpg");
	cv::Mat lbpImg;
	//LBPRGB(img,1,8,lbpImg);
	LBPRGB(img,lbpImg);
	cv::Vec3b  val = lbpImg.at<cv::Vec3b>(3,3);
	std::cout<< (int)val[0]<<","<<(int)val[1]<<","<<(int)val[2]<<std::endl;
	val = lbpImg.at<cv::Vec3b>(30,30);
	std::cout<< (int)val[0]<<","<<(int)val[1]<<","<<(int)val[2]<<std::endl;
	val = lbpImg.at<cv::Vec3b>(220,330);
	std::cout<< (int)val[0]<<","<<(int)val[1]<<","<<(int)val[2]<<std::endl;
	cv::imwrite("lbpimg.png",lbpImg);
	cv::imshow("lbpimg",lbpImg);
	cv::waitKey();
}

template<class T, int D> inline T vecSqrDist(const Vec<T, D> &v1, const Vec<T, D> &v2) {T s = 0; for (int i=0; i<D; i++) s += sqrt((v1[i] - v2[i])*1.0f); return s;} // out of range risk for T = byte, ...
int Quantize(cv::Mat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio, const int clrNums[3])
{
	float clrTmp[3] = {clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f};
	int w[3] = {clrNums[1] * clrNums[2], clrNums[2], 1};

	CV_Assert(img3f.data != NULL);
	idx1i = Mat::zeros(img3f.size(), CV_32S);
	int rows = img3f.rows, cols = img3f.cols;
	if (img3f.isContinuous() && idx1i.isContinuous()){
		cols *= rows;
		rows = 1;
	}

	// Build color pallet
	map<int, int> pallet;
	for (int y = 0; y < rows; y++)
	{
		const float* imgData = img3f.ptr<float>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++, imgData += 3)
		{
			idx[x] = (int)(imgData[0]*clrTmp[0])*w[0] + (int)(imgData[1]*clrTmp[1])*w[1] + (int)(imgData[2]*clrTmp[2]);
			pallet[idx[x]] ++;
		}
	}

	// Find significant colors
	int maxNum = 0;
	{
		int count = 0;
		vector<pair<int, int>> num; // (num, color) pairs in num
		num.reserve(pallet.size());
		for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
			num.push_back(pair<int, int>(it->second, it->first)); // (color, num) pairs in pallet
		sort(num.begin(), num.end(), std::greater<pair<int, int>>());

		maxNum = (int)num.size();
		int maxDropNum = cvRound(rows * cols * (1-ratio));
		for (int crnt = num[maxNum-1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
			crnt += num[maxNum - 2].first;
		maxNum = min(maxNum, 256); // To avoid very rarely case
		if (maxNum <= 10)
			maxNum = min(10, (int)num.size());

		pallet.clear();
		for (int i = 0; i < maxNum; i++)
			pallet[num[i].second] = i; 

		vector<Vec3i> color3i(num.size());
		for (unsigned int i = 0; i < num.size(); i++)
		{
			color3i[i][0] = num[i].second / w[0];
			color3i[i][1] = num[i].second % w[0] / w[1];
			color3i[i][2] = num[i].second % w[1];
		}

		for (unsigned int i = maxNum; i < num.size(); i++)
		{
			int simIdx = 0, simVal = INT_MAX;
			for (int j = 0; j < maxNum; j++)
			{
				int d_ij = vecSqrDist<int, 3>(color3i[i], color3i[j]);
				if (d_ij < simVal)
					simVal = d_ij, simIdx = j;
			}
			pallet[num[i].second] = pallet[num[simIdx].second];
		}
	}

	_color3f = Mat::zeros(1, maxNum, CV_32FC3);
	_colorNum = Mat::zeros(_color3f.size(), CV_32S);

	Vec3f* color = (Vec3f*)(_color3f.data);
	int* colorNum = (int*)(_colorNum.data);
	for (int y = 0; y < rows; y++) 
	{
		const Vec3f* imgData = img3f.ptr<Vec3f>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++)
		{
			idx[x] = pallet[idx[x]];
			color[idx[x]] += imgData[x];
			colorNum[idx[x]] ++;
		}
	}
	for (int i = 0; i < _color3f.cols; i++)
		color[i] /= (float)colorNum[i];

	return _color3f.cols;
}

void TestQuantize()
{
	cv::Mat img = cv::imread("..//moseg//people1//in000001.jpg");
	img.convertTo(img, CV_32FC3, 1.0 / 255);
	cv::Mat idx1i,_color3f, _colorNum;
	double ratio = 0.95;
	const int clrNums[3] = {12,12,12};

	int num = Quantize(img,idx1i,_color3f,_colorNum,ratio,clrNums);
	std::cout<<_colorNum;
	cv::Mat qImg(img.size(),CV_8UC3);
	for(int i=0; i<qImg.rows; i++)
	{
		int * ptr = idx1i.ptr<int>(i);
		cv::Vec3b* vptr = qImg.ptr<cv::Vec3b>(i);
		for(int j=0; j<qImg.cols; j++)
		{
			int idx = ptr[j];
			float* color = (float*)(_color3f.data + idx*12);
			for(int c=0; c<3; c++)
			{
				vptr[j][c] = floor(255*color[c]);
			}
		}

	}
	
	cv::imshow("qimg",qImg);
	cv::waitKey();
	printf("%d colors",num);
}