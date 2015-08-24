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
#include "RTBackgroundSubtractor.h"
#include "RegionMerging.h"
#include "GpuTimer.h"
#include "LBP.h"
#include <fstream>
#include <algorithm>
#include <queue>
#include "HistComparer.h"
#include "FGMaskPostProcess.h"
#include "GCoptimization.h"
#include "LSE.h"
#include <numeric>
#include "Dijkstra.h"
#include "FileNameHelper.h"
void testCudaGpu()
{
	try

	{

		cv::Mat src_host = cv::imread("in000001.jpg");

		cv::gpu::GpuMat dst, src;

		src.upload(src_host);

		cv::gpu::cvtColor(src, src, CV_BGR2GRAY);

		cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

		//cv::Mat result_host = dst;

		cv::Mat result_host;

		dst.download(result_host);

		cv::imshow("Result", result_host);

		cv::waitKey();

	}

	catch (const cv::Exception& ex)

	{

		std::cout << "Error: " << ex.what() << std::endl;

	}
}

void CpuSuperpixel(unsigned int* data, int width, int height, int step, float alpha = 0.9)
{
	int size = width*height;
	int* labels = new int[size];
	unsigned int* idata = new unsigned int[size];
	memcpy(idata, data, sizeof(unsigned int)*size);
	int numlabels(0);
	ComSuperpixel CS;
	//CS.Superpixel(idata,width,height,7000,0.9,labels);
#ifndef REPORT
	nih::Timer timer;
	timer.start();
#endif
	//CS.SuperpixelLattice(idata,width,height,step,alpha,numlabels,labels);
	CS.Superpixel(idata, width, height, step, alpha, numlabels, labels);
#ifndef REPORT
	timer.stop();
	std::cout << "CPU SuperPixel " << timer.seconds() << std::endl;
#endif
	SLIC aslic;
	aslic.DrawContoursAroundSegments(idata, labels, width, height, 0x00ff00);
	PictureHandler handler;
	char name[50];
	sprintf(name, "%f_cpusuper.jpg", alpha);
	handler.SavePicture(idata, width, height, std::string(name), std::string(".\\"));
	aslic.SaveSuperpixelLabels(labels, width, height, std::string("cpuSp.txt"), std::string(".\\"));
	delete[] labels;
	delete[] idata;
}

void TestSuperpixel(int argc, char* argv[])
{
	printf("arg1:1 for gpu,0 for cpu, 3 for slic, arg2:alpha(0-1), arg3:Step, arg4: img fileName\n");
	int flag = atoi(argv[1]);
	float alpha = atof(argv[2]);
	int step = atoi(argv[3]);
	char* fileName = argv[4];
	using namespace cv;

	Mat img = imread(fileName);
	if (flag == 1)
	{
		int num(0);

		cv::Mat rst;

		SuperpixelComputer computer(img.cols, img.rows, step, alpha);
		cv::gpu::GpuMat dImg;
		//cv::gpu::cvtColor(dImg, dImg, CV_BGR2BGRA);		
		cv::Mat imgBGRA;
		cv::cvtColor(img, imgBGRA, CV_BGR2BGRA);
		dImg.upload(imgBGRA);
		computer.ComputeBigSuperpixel(dImg.ptr<uchar4>());
		computer.GetVisualResult(img, rst);
		char name[50];
		sprintf(name, "%f_gpusuper.jpg", alpha);
		cv::imwrite(name, rst);
	}
	else if (flag == 0)
	{
		cv::cvtColor(img, img, CV_BGR2BGRA);
		CpuSuperpixel((unsigned int*)img.data, img.cols, img.rows, step, alpha);
	}
	else
	{
		SuperpixelComputer computer(img.cols, img.rows, step, alpha);
		computer.ComputeSLICSuperpixel(img);
		int* labels; SLICClusterCenter* centers;
		int num(0);
		computer.GetSuperpixelResult(num, labels, centers);
		cv::Mat rst;
		computer.GetVisualResult(img, rst);

		char name[50];
		sprintf(name, "%f_SLICSuper.jpg", alpha);
		cv::imwrite(name, rst);
	}




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
	GpuSuperpixel gs(cols, rows, 5);
	MRFOptimize optimizer(cols, rows, 5);
	nih::Timer timer;
	timer.start();
	int start = 1;
	int end = 1130;
	for (int i = start; i <= end; i++)
	{
		sprintf(imgFileName, "..\\ptz\\input0\\in%06d.jpg", i);
		sprintf(maskFileName, "..\\result\\subsensex\\ptz\\input0\\o\\bin%06d.png", i);
		sprintf(featureMaskFileName, "..\\result\\subsensex\\ptz\\input0\\features\\features%06d.jpg", i);
		sprintf(resultFileName, "..\\result\\SubsenseMMRF\\ptz\\input0\\bin%06d.png", i);

		/*sprintf(imgFileName,"..\\baseline\\input0\\in%06d.jpg",i);
		sprintf(maskFileName,"..\\result\\sobs\\baseline\\input0\\bin%06d.png",i);
		sprintf(resultFileName,"..\\result\\SubsenseMMRF\\baseline\\input0\\bin%06d.png",i);*/
		//optimizer.Optimize(&gs,string(imgFileName),string(maskFileName),string(resultFileName));
		optimizer.Optimize(&gs, string(imgFileName), string(maskFileName), string(featureMaskFileName), string(resultFileName));
	}
	timer.stop();
	std::cout << (end - start + 1) / timer.seconds() << " fps\n";
}
void warmUpDevice()
{
	cv::Mat cpuMat = cv::Mat::ones(100, 100, CV_8UC3);
	cv::gpu::GpuMat gmat;
	gmat.upload(cpuMat);
	gmat.download(cpuMat);
}
void warmCuda()
{
	int* tmp(NULL);
	cudaMalloc(&tmp, 800);
	cudaFree(tmp);
}
void TestRandom()
{
	int width(8);
	int height(6);
	int* d_rand, *h_rand;
	cudaMalloc(&d_rand, width*height * 2 * sizeof(int));
	h_rand = new int[width*height * 2];
	TestRandNeighbour(width, height, d_rand);
	cudaMemcpy(h_rand, d_rand, sizeof(int)*width*height * 2, cudaMemcpyDeviceToHost);
	for (int i = 0; i < width*height; i++)
	{
		std::cout << h_rand[2 * i] << "," << h_rand[2 * i + 1] << " ";
		if ((2 * i + 1) % width == 0)
			std::cout << std::endl;
	}
	std::cout << "---------\n";
	TestRandNeighbour(width, height, d_rand);
	cudaMemcpy(h_rand, d_rand, sizeof(int)*width*height * 2, cudaMemcpyDeviceToHost);
	for (int i = 0; i < width*height; i++)
	{
		std::cout << h_rand[2 * i] << "," << h_rand[2 * i + 1] << " ";
		if ((2 * i + 1) % width == 0)
			std::cout << std::endl;
	}
	std::cout << "---------\n";
	int x_rand, y_rand;
	cv::Size size(width, height);
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			getRandNeighborPosition_3x3(x_rand, y_rand, i, j, 2, size);
			std::cout << x_rand << "," << y_rand << " ";

		}
		std::cout << "\n";
	}
	std::cout << "---------\n";
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			getRandNeighborPosition_3x3(x_rand, y_rand, i, j, 2, size);
			std::cout << x_rand << "," << y_rand << " ";

		}
		std::cout << "\n";
	}
	cudaFree(d_rand);
	delete[] h_rand;
}
void TestGpuSubsense(int procId, int start, int end, const char* input, const char* output, int warpId, float rggThre, float rggSeedThres, float mdlConfidence, float tcConfidence, float scConfidence)
{
	FrameProcessor* BSProcessor(NULL);
	// Create feature tracker instance"..\\result\\subsensex\\moseg\\people1\\"
	if (procId >= 0)
	{
		BSProcessor = new WarpBSProcessor(procId, output, start - 1, warpId, rggThre, rggSeedThres, mdlConfidence, tcConfidence);
	}
	else
	{
		BSProcessor = new BSProcesor(output, start - 1);
	}
	char name[150];
	std::vector<std::string> fileNames;
	std::vector<cv::Mat> inputImgs;
	std::vector<std::string> outFileNames;
	cv::Mat img;
	sprintf(name, "%s\\in%06d.jpg", input, start);
	img = cv::imread(name);
	BSProcessor->initialize(img);
	for (int i = start; i <= end; i++)
	{
		sprintf(name, "%s\\in%06d.jpg", input, i);
		fileNames.push_back(name);
		sprintf(name, "%sbin%06d.png", output, i);
		outFileNames.push_back(name);
	}


	cv::Mat outImg;
	nih::Timer timer;
	timer.start();

	for (int i = 0; i < fileNames.size(); i++)
	{
		img = cv::imread(fileNames[i]);
		BSProcessor->process(img, outImg);
		cv::imwrite(outFileNames[i], outImg);
	}

	timer.stop();
	std::cout << (end - start + 1) / timer.seconds() << " fps" << std::endl;




	safe_delete(BSProcessor);


}
void TestMotionEstimate()
{
	char fileName[100];
	int start = 2;
	int end = 40;
	cv::Mat curImg, prevImg, transM;
	MotionEstimate me(640, 480, 5);
	cv::Mat mask;
	for (int i = start; i <= end; i++)
	{
		sprintf(fileName, "..//moseg//people1//in%06d.jpg", i);
		curImg = cv::imread(fileName);
		sprintf(fileName, "..//moseg//people1//in%06d.jpg", i - 1);
		prevImg = cv::imread(fileName);
		//me.EstimateMotionMeanShift(curImg,prevImg,transM,mask);
		me.EstimateMotion(curImg, prevImg, transM, mask);
		//me.EstimateMotionHistogram(curImg,prevImg,transM,mask);
		sprintf(fileName, ".//features//people1//features%06d.jpg", i);
		cv::imwrite(fileName, mask);
		/*cv::imshow("curImg",curImg);
		cv::imshow("prevImg",prevImg);*/
		cv::waitKey();
	}
}
void TestRegionGrowing()
{
	std::vector<cv::Point2f> seeds;
	seeds.push_back(cv::Point2f(30, 66));
	cv::Mat img = cv::imread("..//ptz//input0in000225.jpg");
	cv::Mat result(img.size(), CV_8U);
	result = cv::Scalar(0);
	RegionGrowing(seeds, img, result);
	cv::imshow("region grow", result);
	cv::waitKey();
}



void GetHomography(const cv::Mat& gray, const cv::Mat& pre_gray, cv::Mat& homography)
{
	int max_count = 50000;	  // maximum number of features to detect
	double qlevel = 0.05;    // quality level for feature detection
	double minDist = 2;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
	std::vector<float> err;    // error in tracking
	std::vector<cv::Point2f> features1, features2;
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

	int k = 0;

	for (int i = 0; i < features1.size(); i++)
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
	homography = cv::findHomography(
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
	cv::cvtColor(preImg, preImg, CV_BGR2GRAY);
	cv::cvtColor(curImg, curImg, CV_BGR2GRAY);
	cv::Mat preMsk = cv::imread("..//result//subsensem//ptz//input3//wap//bin000289.png");
	cv::Mat curMsk = cv::imread("..//result//subsensem//ptz//input3//warp//bin000290.png");
	cv::cvtColor(preMsk, preMsk, CV_BGR2GRAY);
	cv::cvtColor(curMsk, curMsk, CV_BGR2GRAY);

	cv::Mat homography;
	GetHomography(curImg, preImg, homography);
	std::vector<cv::Point2f> curPts, prevPts;
	int width = curImg.cols;
	int height = curImg.rows;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			int idx = i + j*width;
			if (curMsk.data[idx] == 0xff)
				curPts.push_back(cv::Point2f(i, j));
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
	cv::Mat mask(height, width, CV_32F);
	mask = cv::Scalar(0);
	double* data = (double*)homography.data;
	double maxD(0), minD(1e10);

	for (int i = 0; i < status.size(); i++)
	{
		if (status[i] == 1)
		{
			int x = (int)(curPts[i].x);
			int y = (int)(curPts[i].y);
			float wx = data[0] * x + data[1] * y + data[2];
			float wy = data[3] * x + data[4] * y + data[5];
			float w = data[6] * x + data[7] * y + data[8];
			wx /= w;
			wy /= w;
			int px = (int)(prevPts[i].x);
			int py = (int)(prevPts[i].y);
			int idx = py*width + px;
			float d = abs(px - wx) + abs(py - wy);

			if (d > maxD)
				maxD = d;
			if (d < minD)
				minD = d;
			float * ptr = (float*)(mask.data + idx * 4);
			*ptr = d;
		}

	}

	cv::Mat B;
	mask.convertTo(B, CV_8U, 255.0 / (maxD - minD), 0);
	cv::imwrite("tracked.jpg", B);
}


void SuperpixelGrowingFlow(const cv::Mat& sgray, const cv::Mat& tgray, int step, int spSize, const SLICClusterCenter* centers, const int* labels, cv::Mat& flow)
{
	std::vector<cv::Point2f> features0, features1;
	std::vector<uchar> status;
	std::vector<float> err;
	int spWidth = (sgray.cols + step - 1) / step;
	int spHeight = (sgray.rows + step - 1) / step;
	int width = sgray.cols;
	int height = sgray.rows;
	flow.create(spHeight, spWidth, CV_32FC2);
	flow = cv::Scalar(0);
	cv::goodFeaturesToTrack(sgray, features0, 50000, 0.05, 2);
	cv::calcOpticalFlowPyrLK(sgray, tgray, features0, features1, status, err);

	int k = 0;
	for (int i = 0; i < features0.size(); i++)
	{
		if (status[i] == 1)
		{
			features0[k] = features0[i];
			features1[k++] = features1[i];
		}
	}
	for (int i = 0; i < k; i++)
	{
		int ix = (int)(features0[i].x + 0.5);
		int iy = (int)(features0[i].y + 0.5);
		int label = labels[ix + iy*width];
		float2* ptr = (float2*)(flow.data + label * 8);
		*ptr = make_float2(features1[i].x - features0[i].x, features1[i].y - features0[i].y);
	}

	std::cout << "tracking succeeded " << k << " total " << spSize << std::endl;
}
void SuperpixelFlowToPixelFlow(const int* labels, const SLICClusterCenter* centers, const cv::Mat& sflow, int spSize, int step, int width, int height, cv::Mat& flow)
{

	flow.create(height, width, CV_32FC2);
	flow = cv::Scalar(0);
	for (int i = 0; i < spSize; i++)
	{
		int k = centers[i].xy.x;
		int j = centers[i].xy.y;
		float2 flowValue = *((float2*)(sflow.data + i * 8));
		if (centers[i].nPoints > 0)
		{

			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = step;
			for (int x = k - radius; x <= k + radius; x++)
			{
				for (int y = j - radius; y <= j + radius; y++)
				{
					if (x<0 || x>width - 1 || y<0 || y> height - 1)
						continue;
					int idx = x + y*width;

					if (labels[idx] == i)
					{
						float2* fptr = (float2*)(flow.data + idx * 8);
						*fptr = flowValue;
					}
				}
			}

		}
	}
}
float L1Dist(const float2& p1, const float2& p2)
{
	return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

//利用warping的mapX和mapY进行superpixel matching
//比较matching后超像素的颜色差，理论上前景应该较大，背景较小，但是结果显示在图像边缘部分会有较大的误差
void SuperpixelMatching(const int* labels0, const SLICClusterCenter* centers0, const int* labels1, const SLICClusterCenter* centers1, int spSize, int spStep, int width, int height,
	const cv::Mat& mapX, const cv::Mat& mapY, std::vector<int> matchedId, cv::Mat& diff)
{
	int spWidth = (width + spStep - 1) / spStep;
	diff.create(height, width, CV_8UC3);
	std::vector<int> ids;
	for (int i = 0; i < spSize; i++)
	{
		int k = centers0[i].xy.x;
		int j = centers0[i].xy.y;
		ids.clear();
		if (centers0[i].nPoints > 0)
		{
			float avgX(0), avgY(0);
			int n(0);
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = spStep;
			for (int x = k - radius; x <= k + radius; x++)
			{
				for (int y = j - radius; y <= j + radius; y++)
				{
					if (x<0 || x>width - 1 || y<0 || y> height - 1)
						continue;
					int idx = x + y*width;

					if (labels0[idx] == i)
					{
						ids.push_back(idx);
						float nx = *((float*)(mapX.data + idx * 4));
						avgX += nx;
						float ny = *((float*)(mapY.data + idx * 4));
						avgY += ny;
						n++;
					}
				}
			}
			avgX /= n;
			avgY /= n;
			float2 point = make_float2(avgX, avgY);
			int iavgX = (int)(avgX + 0.5);
			int iavgY = (int)(avgY + 0.5);
			int label = labels1[iavgX + iavgY*width];
			//在8邻域内查找中心点与avgX和avgY最接近的超像素
			float disMin = L1Dist(centers1[label].xy, point);
			int minLabel = label;
			//left
			if (label - 1 >= 0)
			{
				float dist = L1Dist(centers1[label - 1].xy, point);
				if (dist < disMin)
				{
					minLabel = label - 1;
					disMin = dist;
				}
				if (label - 1 - spWidth >= 0)
				{
					dist = L1Dist(centers1[label - 1 - spWidth - 1].xy, point);
					if (dist < disMin)
					{
						minLabel = label - 1 - spWidth;
						disMin = dist;
					}
				}
				if (label - 1 + spWidth < spSize)
				{
					dist = L1Dist(centers1[label - 1 + spWidth].xy, point);
					if (dist < disMin)
					{
						minLabel = label - 1 + spWidth;
						disMin = dist;
					}
				}
			}
			if (label + 1 < spSize)
			{
				float dist = L1Dist(centers1[label + 1].xy, point);
				if (dist < disMin)
				{
					minLabel = label + 1;
					disMin = dist;
				}
				if (label + 1 - spWidth >= 0)
				{
					dist = L1Dist(centers1[label + 1 - spWidth - 1].xy, point);
					if (dist < disMin)
					{
						minLabel = label + 1 - spWidth;
						disMin = dist;
					}
				}
				if (label + 1 + spWidth < spSize)
				{
					dist = L1Dist(centers1[label + 1 + spWidth].xy, point);
					if (dist < disMin)
					{
						minLabel = label + 1 + spWidth;
						disMin = dist;
					}
				}
			}
			if (label + spWidth < spSize)
			{
				float dist = L1Dist(centers1[label + spWidth].xy, point);
				if (dist < disMin)
				{
					minLabel = label + spWidth;
					disMin = dist;
				}
			}
			if (label - spWidth >= 0)
			{
				float dist = L1Dist(centers1[label - spWidth].xy, point);
				if (dist < disMin)
				{
					minLabel = label - spWidth;
					disMin = dist;
				}
			}
			matchedId.push_back(minLabel);
			uchar cdx = abs(centers0[i].rgb.x - centers1[minLabel].rgb.x);
			uchar cdy = abs(centers0[i].rgb.y - centers1[minLabel].rgb.y);
			uchar cdz = abs(centers0[i].rgb.z - centers1[minLabel].rgb.z);
			uchar3 val = make_uchar3(cdx, cdy, cdz);
			for (int k = 0; k < ids.size(); k++)
			{
				uchar3* ptr = (uchar3*)(diff.data + 3 * ids[k]);
				*ptr = val;
			}

		}
	}
}
//利用稠密光流进行superpixel matching,同时计算superpixel光流（超像素内像素光流平均值）
void SuperpixelMatching(const int* labels0, const SLICClusterCenter* centers0, const int* labels1, const SLICClusterCenter* centers1, int spSize, int spStep, int width, int height,
	const cv::Mat& flow, std::vector<int> matchedId, cv::Mat& spFlow, cv::Mat& diff)
{

	int spWidth = (width + spStep - 1) / spStep;
	int spHeight = (height + spStep - 1) / spStep;
	diff.create(height, width, CV_8UC3);
	spFlow.create(spHeight, spWidth, CV_32FC2);
	std::vector<int> ids;
	matchedId.clear();
	matchedId.resize(spSize);
	memset(&matchedId[0], -1, sizeof(int)*spSize);
	for (int i = 0; i < spSize; i++)
	{

		int k = (int)(centers0[i].xy.x + 0.5);
		int j = (int)(centers0[i].xy.y + 0.5);
		ids.clear();
		if (centers0[i].nPoints > 0)
		{
			float avgX(0), avgY(0);
			int n(0);
			//以原来的中心点为中心，step +2　为半径进行更新
			int radius = spStep;
			for (int x = k - radius; x <= k + radius; x++)
			{
				if (x<0 || x>width - 1)
					continue;
				for (int y = j - radius; y <= j + radius; y++)
				{
					if (y<0 || y> height - 1)
						continue;

					int idx = x + y*width;

					if (labels0[idx] == i)
					{
						/*if (x == 144 && y==96)
						std::cout<<i<<std::endl;*/
						ids.push_back(idx);
						float2 dxy = *((float2*)(flow.data + idx * 4 * 2));
						avgX += dxy.x;
						avgY += dxy.y;
						n++;
					}
				}
			}
			avgX /= n;
			avgY /= n;
			*(float2*)(spFlow.data + i * 8) = make_float2(avgX, avgY);
			float2 point = make_float2(avgX + k, avgY + j);
			int iavgX = (int)(avgX + 0.5 + k);
			int iavgY = (int)(avgY + 0.5 + j);
			if (iavgX <0 || iavgX > width - 1 || iavgY <0 || iavgY > height - 1)
				continue;
			int label = labels1[iavgX + iavgY*width];
			//在8邻域内查找中心点与avgX和avgY最接近的超像素
			float disMin = L1Dist(centers1[label].xy, point);
			int minLabel = label;
			//left
			if (label - 1 >= 0)
			{
				float dist = L1Dist(centers1[label - 1].xy, point);
				if (dist < disMin)
				{
					minLabel = label - 1;
					disMin = dist;
				}
				if (label - 1 - spWidth >= 0)
				{
					dist = L1Dist(centers1[label - 1 - spWidth - 1].xy, point);
					if (dist < disMin)
					{
						minLabel = label - 1 - spWidth;
						disMin = dist;
					}
				}
				if (label - 1 + spWidth < spSize)
				{
					dist = L1Dist(centers1[label - 1 + spWidth].xy, point);
					if (dist < disMin)
					{
						minLabel = label - 1 + spWidth;
						disMin = dist;
					}
				}
			}
			if (label + 1 < spSize)
			{
				float dist = L1Dist(centers1[label + 1].xy, point);
				if (dist < disMin)
				{
					minLabel = label + 1;
					disMin = dist;
				}
				if (label + 1 - spWidth >= 0)
				{
					dist = L1Dist(centers1[label + 1 - spWidth - 1].xy, point);
					if (dist < disMin)
					{
						minLabel = label + 1 - spWidth;
						disMin = dist;
					}
				}
				if (label + 1 + spWidth < spSize)
				{
					dist = L1Dist(centers1[label + 1 + spWidth].xy, point);
					if (dist < disMin)
					{
						minLabel = label + 1 + spWidth;
						disMin = dist;
					}
				}
			}
			if (label + spWidth < spSize)
			{
				float dist = L1Dist(centers1[label + spWidth].xy, point);
				if (dist < disMin)
				{
					minLabel = label + spWidth;
					disMin = dist;
				}
			}
			if (label - spWidth >= 0)
			{
				float dist = L1Dist(centers1[label - spWidth].xy, point);
				if (dist < disMin)
				{
					minLabel = label - spWidth;
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
			uchar3 val = make_uchar3(cdx, cdy, cdz);
			for (int k = 0; k < ids.size(); k++)
			{
				uchar3* ptr = (uchar3*)(diff.data + 3 * ids[k]);
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
	ASAPWarping asap(cols, rows, 8, 1.0);

	int size = rows*cols;
	int num(0);
	GpuSuperpixel gs(cols, rows, step);
	cv::Mat simg, timg, wimg, sgray, tgray;
	cv::Mat img0, img1;
	img0 = cv::imread("..//ptz//input3//in000222.jpg");
	img1 = cv::imread("..//ptz//input3//in000221.jpg");

	std::vector<cv::Point2f> features0, features1;
	std::vector<uchar> status;
	std::vector<float> err;

	cv::cvtColor(img0, sgray, CV_BGR2GRAY);
	cv::cvtColor(img1, tgray, CV_BGR2GRAY);
	cv::cvtColor(img0, simg, CV_BGR2BGRA);
	cv::cvtColor(img1, timg, CV_BGR2BGRA);


	SLICClusterCenter* centers0(NULL), *centers1(NULL);
	int * labels0(NULL), *labels1(NULL);
	labels0 = new int[size];
	labels1 = new int[size];
	int spSize = ((rows + step - 1) / step) * ((cols + step - 1) / step);
	centers0 = new SLICClusterCenter[spSize];
	centers1 = new SLICClusterCenter[spSize];

	gs.Superpixel(simg, num, labels0, centers0);
	gs.Superpixel(timg, num, labels1, centers1);
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	cv::Mat flow, spFlow;
	DOFP->DenseOpticalFlow(sgray, tgray, flow);
	WriteFlowFile(flow, "flow.flo");
	std::vector<int> matchedId;
	cv::Mat diffMat;
	SuperpixelMatching(labels0, centers0, labels1, centers1, spSize, step, cols, rows, flow, matchedId, spFlow, diffMat);
	WriteFlowFile(spFlow, "spFlow.flo");
	KLTFeaturesMatching(sgray, tgray, features0, features1);
	cv::Mat homography;
	FeaturePointsRefineRANSAC(features0, features1, homography);
	asap.SetFeaturePoints(features0, features1);
	asap.Solve();
	asap.Warp(simg, wimg);
	cv::Mat warpFlow;
	asap.getFlow(warpFlow);
	cv::Mat flowDiff;
	cv::absdiff(flow, warpFlow, flowDiff);
	cv::Mat mask(rows, cols, CV_8U);
	mask = cv::Scalar(0);
	for (int i = 0; i < flowDiff.rows; i++)
	{
		float2* ptrw = warpFlow.ptr<float2>(i);
		float2* ptrf = flow.ptr<float2>(i);
		uchar * mptr = mask.ptr<uchar>(i);
		for (int j = 0; j < flowDiff.cols; j++)
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
	cv::imshow("source image", img0);
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
	ASAPWarping asap(cols, rows, 8, 1.0);

	int size = rows*cols;
	int num(0);
	GpuSuperpixel gs(cols, rows, step);
	cv::Mat simg, timg, wimg, sgray, tgray;
	cv::Mat img0, img1;
	img0 = cv::imread("..//ptz//input2//in000002.jpg");
	img1 = cv::imread("..//ptz//input2//in000001.jpg");

	std::vector<cv::Point2f> features0, features1;
	std::vector<uchar> status;
	std::vector<float> err;

	cv::cvtColor(img0, sgray, CV_BGR2GRAY);
	cv::cvtColor(img1, tgray, CV_BGR2GRAY);
	cv::cvtColor(img0, simg, CV_BGR2BGRA);
	cv::cvtColor(img1, timg, CV_BGR2BGRA);


	SLICClusterCenter* centers0(NULL), *centers1(NULL);
	int * labels0(NULL), *labels1(NULL);
	labels0 = new int[size];
	labels1 = new int[size];
	int spHeight = (rows + step - 1) / step;
	int spWidth = (cols + step - 1) / step;
	int spSize = spHeight*spWidth;
	centers0 = new SLICClusterCenter[spSize];
	centers1 = new SLICClusterCenter[spSize];

	gs.Superpixel(simg, num, labels0, centers0);
	gs.Superpixel(timg, num, labels1, centers1);

	/*KLTFeaturesMatching(sgray,tgray,features0,features1);
	cv::Mat homography;
	FeaturePointsRefineRANSAC(features0,features1,homography);
	asap.SetControlPts(features0,features1);
	asap.Solve();
	asap.Warp(simg,wimg);*/

	cv::Mat spFlow, flow, flowField, pflowField;
	std::vector<cv::Mat> flows(2);
	std::vector<cv::Point2f> f0, f1;
	//SuperpixelFlow(sgray,tgray,step,spSize,centers0,f0,f1,spFlow);
	GpuSuperpixelFlow(sgray, tgray, step, spSize, centers0, f0, f1, spFlow);
	//SuperpixelGrowingFlow(sgray,tgray,step,spSize,centers0,labels0,spFlow);
	std::vector<float> flowHist, avgX, avgY;
	std::vector<std::vector<int>> ids;
	cv::Mat idMat;
	OpticalFlowHistogram(spFlow, flowHist, avgX, avgY, ids, idMat, 20, 36);
	double minVal, maxVal;
	cv::minMaxLoc(flowHist, &minVal, &maxVal);
	std::ofstream file("out.txt");
	cv::Mat histImg(spHeight, spWidth, CV_32F, cv::Scalar(0));
	for (int i = 0; i < spHeight; i++)
	{
		unsigned short* idPtr = idMat.ptr<unsigned short>(i);
		float* histImgPtr = histImg.ptr<float>(i);
		for (int j = 0; j < spWidth; j++)
		{
			histImgPtr[j] = flowHist[idPtr[j]] / maxVal;
			file << histImgPtr[j] << "\t";
		}
		file << std::endl;
	}
	file.close();

	//cv::normalize(histImg,histImg,0, 100, NORM_MINMAX, -1, Mat());
	cv::imshow("histImg", histImg);
	cv::split(spFlow, flows);
	getFlowField(flows[0], flows[1], flowField);
	cv::imshow("superpixel flow", flowField);
	SuperpixelFlowToPixelFlow(labels0, centers0, spFlow, spSize, step, cols, rows, flow);
	WriteFlowFile(flow, "spflow.flo");
	cv::split(flow, flows);
	getFlowField(flows[0], flows[1], pflowField);
	cv::imshow("pixel flow", pflowField);
	cv::imshow("simg", simg);
	cv::imshow("timg", timg);
	cv::imshow("timg", timg);
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
	GpuSuperpixel gs(cols, rows, 5);
	MRFOptimize optimizer(cols, rows, 5);
	nih::Timer timer;
	timer.start();
	int start = 300;
	int end = 330;
	std::vector<cv::Mat> imgs;
	std::vector<cv::Mat> masks;
	cv::Mat curImg, prevImg, mask, prevMask, resultImg, gray, preGray;

	cv::Mat flow;
	//DenseOpticalFlowProvier* DOFP = new GpuDenseOptialFlow();
	//DenseOpticalFlowProvier* DOFP = new SFDenseOptialFlow();
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	for (int i = start; i <= end; i++)
	{
		sprintf(imgFileName, "..\\ptz\\input3\\in%06d.jpg", i);
		curImg = cv::imread(imgFileName);
		imgs.push_back(curImg.clone());
		sprintf(maskFileName, "..\\result\\subsensem\\ptz\\input3\\bin%06d.png", i);
		curImg = cv::imread(maskFileName);
		cv::cvtColor(curImg, curImg, CV_BGR2GRAY);
		masks.push_back(curImg.clone());
	}
	for (int i = 1; i < end - start + 1; i++)
	{
		curImg = imgs[i];
		prevImg = imgs[i - 1];
		prevMask = masks[i - 1];
		cv::cvtColor(curImg, gray, CV_BGR2GRAY);
		cv::cvtColor(prevImg, preGray, CV_BGR2GRAY);
		mask = masks[i];
		cv::Mat homography;
		GetHomography(gray, preGray, homography);
		DOFP->DenseOpticalFlow(gray, preGray, flow);
		//WriteFlowFile(flow,"flow.flo");
		//ReadFlowFile(flow,"ptz0_87.flo");
		optimizer.Optimize(&gs, curImg, mask, prevMask, flow, homography, resultImg);
		sprintf(resultFileName, "..\\result\\SubsenseMMRF\\ptz\\input3\\bin%06d.png", i);
		cv::imwrite(resultFileName, resultImg);
	}
	timer.stop();
	std::cout << (end - start + 1) / timer.seconds() << " fps\n";
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
	int spWidth = (cols + step - 1) / step;
	int spHeight = (rows + step - 1) / step;
	int spSize = spWidth * spHeight;
	int* labels = new int[rows*cols];
	GpuSuperpixel gs(cols, rows, step);
	SLICClusterCenter* centers = new SLICClusterCenter[spSize];
	int start = 1;
	int end = 18;
	cv::Mat img0, img1, gray0, gray1;
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	cv::Mat flow, spFlow;
	std::vector<float> flowHist, avgDx, avgDy;
	std::vector<std::vector<int>> ids;
	cv::Mat idMat;
	float minVal, maxVal;
	int maxIdx(0), minIdx(0);
	int num(0);
	for (int i = start; i <= end; i++)
	{
		sprintf(imgFileName, "..//moseg//cars2//in%06d.jpg", i);
		img0 = cv::imread(imgFileName);
		sprintf(imgFileName, "..//moseg//cars2//in%06d.jpg", i + 1);
		img1 = cv::imread(imgFileName);
		cv::cvtColor(img0, gray0, CV_BGR2GRAY);
		cv::cvtColor(img1, gray1, CV_BGR2GRAY);
		cv::cvtColor(img0, img0, CV_BGR2BGRA);
		DOFP->DenseOpticalFlow(gray1, gray0, flow);
		/*gs.Superpixel(img0,num,labels,centers);
		SuperpixelFlow(gray1,gray0,5,spSize,centers,spFlow);
		SuperpixelFlowToPixelFlow(labels,centers,spFlow,spSize,step,cols,rows,flow);*/
		OpticalFlowHistogram(flow, flowHist, avgDx, avgDy, ids, idMat, 10, 36);

		for (int i = 0; i < avgDx.size(); i++)
		{
			avgDx[i] /= ids[i].size();
			avgDy[i] /= ids[i].size();
		}
		minMaxLoc(flowHist, maxVal, minVal, maxIdx, minIdx);
		float maxAvgDx = avgDx[maxIdx];
		float maxAvgDy = avgDy[maxIdx];
		cv::Mat histImg(rows, cols, CV_32F, cv::Scalar(0));
		float threshold = 3;
		float avg = 0;
		float variance(0);
		for (int i = 0; i < rows; i++)
		{
			unsigned short* idPtr = idMat.ptr<unsigned short>(i);
			uchar* histImgPtr = histImg.ptr<uchar>(i);
			for (int j = 0; j < cols; j++)
			{
				float dist = abs(avgDx[idPtr[j]] - maxAvgDx) + abs(avgDy[idPtr[j]] - maxAvgDy);
				avg += dist;
				//histImgPtr[j] = dist >threshold ? 255 : 0;
				//histImgPtr[j] = dist;
				/*if (i == 70 && j==50)
				{
				std::cout<<dist<<" , "<<flowHist[idPtr[j]]<<" "<<histImgPtr[j]<<"\n";
				}*/
			}
		}
		avg /= (rows*cols);
		std::cout << "avg dist " << avg << std::endl;
		threshold = avg*2.0;
		for (int i = 0; i < rows; i++)
		{
			unsigned short* idPtr = idMat.ptr<unsigned short>(i);
			float* histImgPtr = histImg.ptr<float>(i);
			for (int j = 0; j < cols; j++)
			{
				float dist = abs(avgDx[idPtr[j]] - maxAvgDx) + abs(avgDy[idPtr[j]] - maxAvgDy);
				variance += (dist - avg)*(dist - avg);
				histImgPtr[j] = exp(dist / threshold);
				//histImgPtr[j] = dist;
				/*if (i == 70 && j==50)
				{
				std::cout<<dist<<" , "<<flowHist[idPtr[j]]<<" "<<histImgPtr[j]<<"\n";
				}*/
			}
		}
		variance /= rows*cols;
		std::cout << "variance : " << variance << std::endl;
		sprintf(resultFileName, ".\\histogram\\input3\\bin%06d.jpg", i);
		cv::imwrite(resultFileName, histImg);
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
	GpuSuperpixel gs(cols, rows, step);
	int spNum = gs.GetSuperPixelNum();
	int * labels = new int[rows*cols];
	SLICClusterCenter* centers = new SLICClusterCenter[spNum];

	cv::Mat img, mask, rst;
	rst = Mat::zeros(rows, cols, CV_8U);
	vector<Mat> bgr_planes;
	vector<int> spIdx;
	/// Establish the number of bins
	int histSize = 16;
	int binSize = 256 / histSize;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	Mat b_hist, g_hist, r_hist;
	int start = 500;
	int end = 500;
	for (int i = start; i <= end; i++)
	{
		rst = cv::Scalar(0);
		sprintf(imgFileName, "..\\ptz\\input3\\in%06d.jpg", i);
		sprintf(maskFileName, "H:\\changeDetection2014\\PTZ\\PTZ\\zoomInZoomOut\\groundtruth\\gt%06d.png", i);
		sprintf(resultFileName, ".\\hist%06d.png", i);
		img = imread(imgFileName);

		mask = imread(maskFileName);
		cvtColor(mask, mask, CV_BGR2GRAY);
		split(img, bgr_planes);
		namedWindow("mask", CV_WINDOW_AUTOSIZE);
		imshow("mask", mask);
		/// Compute the histograms:
		calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate);
		std::cout << "bhist\n";
		for (int i = 0; i < b_hist.rows; i++)
		{
			std::cout << b_hist.at<float>(i) << " ";
		}
		std::cout << "\n";
		std::cout << "ghist\n";
		for (int i = 0; i < b_hist.rows; i++)
		{
			std::cout << g_hist.at<float>(i) << " ";
		}
		std::cout << "\n";
		std::cout << "rhist\n";
		for (int i = 0; i < b_hist.rows; i++)
		{
			std::cout << r_hist.at<float>(i) << " ";
		}
		std::cout << "\n";
		/// Normalize the result to [ 0, histImage.rows ]
		/*normalize(b_hist, b_hist, 0, 255, NORM_MINMAX, -1, Mat() );
		normalize(g_hist, g_hist, 0, 255, NORM_MINMAX, -1, Mat() );
		normalize(r_hist, r_hist, 0, 255, NORM_MINMAX, -1, Mat() );*/
		// Draw the histograms for B, G and R
		int hist_w = 512; int hist_h = 256;
		int bin_w = cvRound((double)hist_w / histSize);

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));



		/// Draw for each channel
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
				cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
			line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
				cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
			line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
				cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
		}

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{

				int idx = i*cols + j;
				uchar b = bgr_planes[0].data[idx];
				uchar g = bgr_planes[1].data[idx];
				uchar r = bgr_planes[2].data[idx];

				float bhist = b_hist.at<float>(bgr_planes[0].data[idx] / binSize);
				float ghist = g_hist.at<float>(bgr_planes[1].data[idx] / binSize);
				float rhist = r_hist.at<float>(bgr_planes[2].data[idx] / binSize);
				uchar val = (bhist + ghist + rhist) / 3;
				if (j == 74 && i == 26)
				{
					std::cout << "row " << i << " , col " << j << " (r,g,b) = " << (int)r << "," << (int)g << "," << (int)b << std::endl;
					std::cout << "bhist " << bhist << " ghist " << ghist << " rhist  " << rhist << std::endl;
				}
				if (j == 92 && i == 117)
				{
					std::cout << "row " << i << " , col " << j << " (r,g,b) = " << (int)r << "," << (int)g << "," << (int)b << std::endl;
					std::cout << "bhist " << bhist << " ghist " << ghist << " rhist  " << rhist << std::endl;
				}
				if (j == 93 && i == 131)
				{
					std::cout << "row " << i << " , col " << j << " (r,g,b) = " << (int)r << "," << (int)g << "," << (int)b << std::endl;
					std::cout << "bhist " << bhist << " ghist " << ghist << " rhist  " << rhist << std::endl;
				}
				rst.data[idx] = val;
			}
		}
		/// Display
		namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
		imshow("calcHist Demo", histImage);

		imshow("rst", rst);
		waitKey(0);
		cvtColor(img, img, CV_BGR2BGRA);
		int num(0);
		gs.Superpixel(img, num, labels, centers);
		for (int i = 0; i < spNum; i++)
		{
			int k = centers[i].xy.x;
			int j = centers[i].xy.y;
			spIdx.clear();

			int c(0);
			if (centers[i].nPoints != 0)
			{

				float bHists(0), gHists(0), rHists(0);
				//以原来的中心点为中心，step +2　为半径进行更新
				int radius = step;
				for (int x = k - radius; x <= k + radius; x++)
				{
					if (x<0 || x>cols - 1)
						continue;

					for (int y = j - radius; y <= j + radius; y++)
					{
						if (y<0 || y> rows - 1)
							continue;
						int idx = x + y*cols;
						//std::cout<<idx<<std::endl;
						if (labels[idx] == i)
						{
							uchar b = bgr_planes[0].data[idx];
							uchar g = bgr_planes[1].data[idx];
							uchar r = bgr_planes[2].data[idx];
							spIdx.push_back(idx);
							float bhist = b_hist.at<float>(bgr_planes[0].data[idx] / bin_w);
							float ghist = g_hist.at<float>(bgr_planes[1].data[idx] / bin_w);
							float rhist = r_hist.at<float>(bgr_planes[2].data[idx] / bin_w);
							bHists += bhist;
							gHists += ghist;
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
		imwrite(resultFileName, rst);



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
	int spWidth = (cols + step - 1) / step;
	int spHeight = (rows + step - 1) / step;
	int spSize = spWidth*spHeight;
	cv::Mat img, preImg, gray, preGray, mask, preMask;
	SuperpixelComputer spComputer(cols, rows, step);
	int * labels(NULL), *preLabels(NULL);
	SLICClusterCenter* centers(NULL), *preCenters(NULL);
	int num(0);
	cv::Mat spFlow, flow;
	int start = 2;
	int end = 5;
	std::vector<int> matchedId;
	std::vector<int> revMatchedId;
	DenseOpticalFlowProvier* DOFP = new EPPMDenseOptialFlow();
	std::vector<cv::Point2f> f0, f1;

	for (int i = start; i <= end; i++)
	{

		sprintf(imgFileName, "..\\moseg\\cars3\\in%06d.jpg", i);
		sprintf(maskFileName, "..\\moseg\\cars3\\groundtruth\\gt%06d.png", i - 1);
		sprintf(resultFileName, ".\\test\\spFlow%06d.flo", i);
		matchedId.clear();
		img = imread(imgFileName);
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		mask = imread(maskFileName);
		if (mask.channels() == 3)
			cv::cvtColor(mask, mask, CV_BGR2GRAY);

		if (preGray.empty())
		{
			preImg = img.clone();
			preGray = gray.clone();
		}
		spComputer.ComputeSuperpixel(img, num, labels, centers);
		spComputer.GetPreSuperpixelResult(num, preLabels, preCenters);
		SuperpixelFlow(gray, preGray, step, num, centers, f0, f1, spFlow);
		SuperpixelMatching(labels, centers, img, preLabels, preCenters, preImg, num, step, cols, rows, spFlow, matchedId);
		//
		preMask = cv::Mat::zeros(rows, cols, CV_8U);
		for (int si = 0; si < matchedId.size(); si++)
		{
			int cx = (int)(preCenters[matchedId[si]].xy.x + 0.5);
			int cy = (int)(preCenters[matchedId[si]].xy.y + 0.5);
			int idx = cx + cy*cols;
			if (mask.data[idx] == 0xff)
			{
				int x = (int)(centers[si].xy.x + 0.5);
				int y = (int)(centers[si].xy.y + 0.5);
				for (int m = x - step; m <= x + step; m++)
				{
					if (m<0 || m > cols - 1)
						continue;
					for (int n = y - step; n <= y + step; n++)
					{
						if (n >= 0 && n < rows && labels[n*cols + m] == si)
						{
							preMask.data[n*cols + m] = 0xff;
						}
					}
				}
			}
		}
		SuperpixelFlow(preGray, gray, step, num, centers, f0, f1, spFlow);
		SuperpixelMatching(preLabels, preCenters, preImg, labels, centers, img, num, step, cols, rows, spFlow, revMatchedId);
		std::set<int> checkedId;
		for (int i = 0; i < matchedId.size(); i++)
		{
			int matched = matchedId[i];
			if (matched >= 0 && matched < spSize && revMatchedId[matched] == i)
				checkedId.insert(checkedId.begin(), i);
		}
		cv::Mat chkMat(rows, cols, CV_8U);
		cv::Mat diffMat(rows, cols, CV_8UC3, cv::Scalar(0));
		for (int i = 0; i < num; i++)
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
				for (int k = x - step; k <= x + step; k++)
				{
					if (k<0 || k>cols - 1)
						continue;
					for (int j = y - step; j <= y + step; j++)
					{
						if (j<0 || j>rows - 1)
							continue;
						int idx = j*cols + k;
						if (labels[idx] == i)
						{
							uchar* rgb = (uchar*)(diffMat.data + idx * 3);
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
		sprintf(resultFileName, ".\\test\\sp_matchedDiff%06d.jpg", i);
		cv::imwrite(resultFileName, diffMat);
		sprintf(resultFileName, ".\\test\\sp_dchked%06d.jpg", i);
		cv::imwrite(resultFileName, chkMat);
		sprintf(resultFileName, ".\\test\\preMask%06d.jpg", i);
		cv::imwrite(resultFileName, preMask);


		/*DOFP->DenseOpticalFlow(gray,preGray,flow);
		cv::Mat diff;
		SuperpixelMatching(labels,centers,preLabels,preCenters,spSize,step,cols,rows,flow,matchedId,spFlow,diff);
		sprintf(resultFileName,".\\test\\matchedDiff%06d.jpg",i);
		cv::imwrite(resultFileName,diff);*/
		cv::swap(gray, preGray);
		cv::swap(img, preImg);
	}
	delete DOFP;
}

void TestDescDiff()
{
	char filename[20];
	cv::Mat cmat, gmat, hmat;
	hmat = cv::imread("hdesc.png", CV_LOAD_IMAGE_UNCHANGED);

	int width = hmat.cols;
	int height = hmat.rows;
	cv::Mat cr = cv::Mat::zeros(hmat.size(), CV_8U);

	std::vector<cv::Mat> cmats, gmats;
	cv::Mat tmp;
	for (int i = 0; i < 50; i++)
	{
		sprintf(filename, "cpu%ddescmodel.png", i);
		cmat = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);

		sprintf(filename, "gpu%ddescmodel.png", i);
		gmat = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
		cmats.push_back(cmat.clone());

		gmats.push_back(gmat.clone());
	}
	for (int r = 2; r < height - 2; r++)
	{

		cv::Vec4w* hptr = hmat.ptr<cv::Vec4w>(r);
		uchar* crPtr = cr.ptr<uchar>(r);

		for (int c = 2; c < width - 2; c++)
		{
			int idx = (r*width + c) * 3 * 2;
			int cidx = (r*width + c) * 3 * 2;
			int b(0);
			int i = 0;
			while (i < 50 && b < 2)
			{

				//cv::Vec4w gptr = gmats[i].at<cv::Vec4w>(r,c);
				ushort* cptr = (ushort*)(cmats[i].data + cidx);
				ushort* gptr = (ushort*)(gmats[i].data + idx);
				//std::cout<<i<<" "<<r<<" , "<<c<<std::endl;

				for (int k = 0; k < 3; k++)
				{
					size_t d = hdist_ushort_8bitLUT(hptr[c][k], gptr[k]);
					//size_t d = hdist_ushort_8bitLUT(hptr[c][k],gptr[k]);
					if (d / 2 * 15 >36)
						goto failed;

				}
				b++;

			failed:
				i++;
			}
			if (b < 2)
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
	cv::imshow("result", cr);

	cv::waitKey();
}


void TestSuperpixelDownSample()
{
	char imgFileName[150];
	char resultFileName[150];
	int cols = 640;
	int rows = 480;
	int step = 5;
	int start = 1;
	int end = 40;
	cv::Mat curImg, dsImg;
	SuperpixelComputer spComputer(cols, rows, step);
	int num(0);
	int * labels;
	SLICClusterCenter* centers;
	for (int i = start; i <= end; i++)
	{
		sprintf(imgFileName, "..\\moseg\\people1\\in%06d.jpg", i);
		curImg = cv::imread(imgFileName);
		spComputer.ComputeSuperpixel(curImg, num, labels, centers);
		spComputer.GetSuperpixelDownSampleImg(dsImg);
		sprintf(resultFileName, "..\\moseg\\people1\\downsample\\in%06d.jpg", i);
		cv::imwrite(resultFileName, dsImg);
	}
}

void GpuSubsenseMain(int argc, char* argv[])
{
	printf("gpu 0 cpu 1: %s\n", argv[1]);
	printf("from: %s\n", argv[2]);
	printf("to: %s\n", argv[3]);
	printf("input: %s\n", argv[4]);
	printf("output: %s\n", argv[5]);
	if (argc == 6)
		TestGpuSubsense(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4], argv[5]);
	else if (argc == 7)
	{
		printf("warpId 1 my asap,2 block 3 global: %s\n", argv[6]);

		TestGpuSubsense(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4], argv[5], atoi(argv[6]));
	}
	else if (argc == 11)
	{
		printf("warpId: 1\n");
		printf("region growing threshold: %s\n", argv[6]);
		printf("region growing seed threshold: %s\n", argv[7]);
		printf("model confidence: %s\n", argv[8]);
		printf("tc confidence: %s\n", argv[9]);
		printf("sc confidence: %s\n", argv[10]);
		TestGpuSubsense(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4], argv[5], 1, atoi(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]));
	}
	else if (argc == 12)
	{
		printf("warpId 1:my asap,2:block 3: global %s\n", argv[6]);
		printf("region growing threshold: %s\n", argv[7]);
		printf("region growing seed threshold: %s\n", argv[8]);
		printf("model confidence: %s\n", argv[9]);
		printf("tc confidence: %s\n", argv[10]);
		printf("sc confidence: %s\n", argv[11]);
		TestGpuSubsense(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4], argv[5], atoi(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]), atof(argv[11]));
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
	cv::Size size2(width * 2, height);
	std::vector<cv::Mat> homographies;
	std::vector<float> blkWeights(quadWidth*quadWidth, 0);
	const char path[] = "..//particle//vcar";
	char dstPath[200];
	sprintf(dstPath, "..//warpRst//mywarp");
	CreateDir(dstPath);
	char fileName[200];
	cv::Mat img1, img0, gray1, gray0, wimg, gtImg;
	std::vector<cv::Point2f> features1, features0;
	std::vector<cv::Point2f> sf1, sf0;

	ASAPWarping asap(width, height, quadWidth, 1.0);
	for (int i = start; i <= end; i++)
	{
		printf("%d------------\n", i);
		sprintf(fileName, "%s//in%06d.jpg", path, i);
		img1 = imread(fileName);
		sprintf(fileName, "%s//groundtruth//gt%06d.png", path, i);
		gtImg = imread(fileName);
		cv::cvtColor(gtImg, gtImg, CV_BGR2GRAY);
		cv::cvtColor(img1, gray1, CV_BGR2GRAY);
		if (gray0.empty())
		{
			gray0 = gray1.clone();
			img0 = img1.clone();
		}
		/*nih::Timer timer;
		timer.start();*/
		KLTFeaturesMatching(gray1, gray0, features1, features0, 5000, 0.05, 5);
		/*timer.stop();
		std::cout<<"klt tracking "<<timer.seconds()*1000<<"ms\n";
		timer.start();*/
		//BC2FFeaturePointsRefineHistogram(width,height,features1,features0,blkWeights,4,radSize1,thetaSize1,radSize2,thetaSize2);
		//C2FFeaturePointsRefineHistogram(width,height,features1,features0,5,36,2,2);
		FeaturePointsRefineHistogram(width, height, features1, features0);
		/*cv::Mat homo;
		FeaturePointsRefineRANSAC(features1,features0,homo,0.1);*/
		if (i > 1)
		{
			bool save = false;
			int ec(0);
			cv::Mat rstImg(size2, CV_8UC3);
			img0.copyTo(rstImg(cv::Rect(0, 0, width, height)));
			img1.copyTo(rstImg(cv::Rect(width, 0, width, height)));
			for (int j = 0; j < features1.size(); j++)
			{
				int x = (int)(features1[j].x + 0.5);
				int y = (int)(features1[j].y + 0.5);
				if (gtImg.data[x + y*width] == 0xff)
				{
					cv::line(rstImg, cv::Point(features0[j].x, features0[j].y), cv::Point(features1[j].x + width, features1[j].y), cv::Scalar(255, 0, 0));
					ec++;
					save = true;
				}
			}
			if (save)
			{
				sprintf(fileName, "..//warpRst//mywarp//error//%d_error_%d.jpg", i, ec);
				cv::imwrite(fileName, rstImg);
			}
		}
		std::cout << "features remians " << features1.size() << std::endl;
		int num = BlockDltHomography(width, height, quadWidth, features1, features0, homographies, blkWeights, sf1, sf0);



		asap.Reset();
		asap.SetFeaturePoints(features1, features0);
		asap.Solve();


		/*cv::imshow("warped image",wimg);
		cv::waitKey();*/
		sprintf(fileName, "%swin%06d.jpg", dstPath, i);
		cv::imwrite(fileName, wimg);
		cv::Mat diff;
		cv::absdiff(wimg, img0, diff);
		sprintf(fileName, "%sdiff%06d.jpg", dstPath, i);

		cv::imwrite(fileName, diff);

		homographies.clear();
		cv::swap(img0, img1);
		cv::swap(gray0, gray1);
	}

}


void RansacVsVoting(cv::Mat& img1, std::vector<cv::Point2f>& features1,
	cv::Mat& img0, std::vector<cv::Point2f>&features0)
{

	std::vector<uchar> inliers;
	std::vector<float> histogram;
	std::vector<std::vector<int>> ids;

	OpticalFlowHistogram(features1, features0, histogram, ids, 10, 90);

	//最大bin
	int max = ids[0].size();
	int idx(0);
	for (int i = 1; i < ids.size(); i++)
	{
		if (ids[i].size() > max)
		{
			max = ids[i].size();
			idx = i;
		}
	}
	int width = img1.cols;
	int height = img1.rows;
	cv::Size size2(width * 2, height);
	cv::Mat rstImg(size2, CV_8UC3);
	img1.copyTo(rstImg(cv::Rect(0, 0, width, height)));
	img0.copyTo(rstImg(cv::Rect(width, 0, width, height)));
	cv::Mat rstImg2 = rstImg.clone();
	for (int i = 0; i < ids[idx].size(); i++)
	{
		int j = ids[idx][i];
		cv::line(rstImg, cv::Point(features0[j].x, features0[j].y), cv::Point(features1[j].x + width, features1[j].y), cv::Scalar(255, 0, 0));
	}
	cv::imwrite("votingOut.jpg", rstImg);
	std::vector<int> rids;
	cv::Mat homography = cv::findHomography(features1, features0, inliers, CV_RANSAC, 0.1);
	for (int i = 0; i < inliers.size(); i++)
	{
		if (inliers[i] == 1)
			rids.push_back(i);
	}
	for (int i = 0; i < rids.size(); i++)
	{
		int j = rids[i];
		cv::line(rstImg, cv::Point(features0[j].x, features0[j].y), cv::Point(features1[j].x + width, features1[j].y), cv::Scalar(255, 0, 0));
	}
	cv::imwrite("RansacOut.jpg", rstImg);
}
struct Item
{
	int id;
	double dist;

};
struct ItemComparer
{
	bool operator()(const Item& item1, const Item & item2)
	{
		return item1.dist > item2.dist;
	}
};
typedef std::priority_queue<Item, std::vector<Item>, ItemComparer> Items;
void RelFlowRefine(int width, int height, std::vector<cv::Point2f>& features1, std::vector<cv::Point2f>&features0, std::vector<uchar>& inliers, int& ankorId)
{
	//select a anchor
	inliers.resize(features0.size());
	double minDistance(1e10);
	int anchor = 0;
	std::vector<float> dx0, dy0, dx1, dy1;
	dx0.resize(features0.size());
	dy0.resize(dx0.size());
	dx1.resize(dx0.size());
	dy1.resize(dy0.size());
	for (int a = 0; a < features0.size(); a++)
	{
		inliers[a] = 0;
		cv::Point2f anchorPt1 = features1[a];
		cv::Point2f anchorPt0 = features0[a];

		for (int i = 0; i < features1.size(); i++)
		{
			dx1[i] = features1[i].x - anchorPt1.x;
			dy1[i] = features1[i].y - anchorPt1.y;
		}

		double sum(0);
		double maxD(0);

		for (int i = 0; i < features0.size(); i++)
		{
			double dx0 = features0[i].x - anchorPt0.x;
			double dy0 = features0[i].y - anchorPt0.y;
			double diffX = dx0 - dx1[i];
			double diffY = dy0 - dy1[i];
			double diff = sqrt(diffX*diffX + diffY*diffY);
			sum += diff;
		}
		if (sum < minDistance)
		{
			minDistance = sum;
			anchor = a;
		}
	}
	ankorId = anchor;
	cv::Point2f anchorPt1 = features1[anchor];
	cv::Point2f anchorPt0 = features0[anchor];

	for (int i = 0; i < features1.size(); i++)
	{

		dx1[i] = features1[i].x - anchorPt1.x;
		dy1[i] = features1[i].y - anchorPt1.y;
	}
	double threshold = minDistance / (features0.size() - 1);
	threshold = 1.2;
	std::cout << "threshold = " << threshold << "\n";


	inliers[anchor] = 1;
	cv::Mat diffMat(height, width, CV_32F);
	Items items;
	for (int i = 0; i < features0.size(); i++)
	{

		double dx0 = features0[i].x - anchorPt0.x;
		double dy0 = features0[i].y - anchorPt0.y;
		double diffX = dx0 - dx1[i];
		double diffY = dy0 - dy1[i];
		double diff = sqrt(diffX*diffX + diffY*diffY);
		items.push(Item{ i, diff });
		int idx = (int)(features1[i].x + 0.5) + (int)(features1[i].y + 0.5)*width;
		*(float*)(diffMat.data + idx * 4) = diff;
		if (diff > threshold)
			inliers[i] = 0;
		else
			inliers[i] = 1;


	}
	//for (size_t i = 0; i < features0.size()*0.6; i++)
	//{
	//	Item item = items.top();
	//	inliers[item.id] = 1;
	//	items.pop();

	//}
	/*cv::normalize(diffMat, diffMat, 1.0, 0.0, NORM_MINMAX);
	diffMat.convertTo(diffMat, CV_8U, 255);
	cv::imshow("diffMat", diffMat);
	cv::waitKey();*/
}
//test the optical orientation 
void HomographyFlowOrientation()
{
	cv::Mat img1, img0;
	img1 = cv::imread("..//moseg//cars7//in000024.jpg");
	img0 = cv::imread("..//moseg//cars7//in000023.jpg");
	std::vector<cv::Point2f> f1, f0;
	std::vector<uchar> inliers;
	KLTFeaturesMatching(img1, img0, f1, f0);
	cv::Mat homo = cv::findHomography(f1, f0, inliers, CV_RANSAC);
	std::cout << "homography: \n" << homo << "\n";
	cv::Point2f src = cv::Point2f(0, 0);
	cv::Point2f dst;
	std::cout << "src " << src << "\n";
	MatrixTimesPoint(homo, src, dst);
	std::cout << "dst " << dst << "\n";
	std::cout << "orientaion " << atan2(dst.y - src.y, dst.x - src.x) / M_PI * 180 + 180 << " degree\n";

	src = cv::Point2f(img1.cols, 0);
	std::cout << "src " << src << "\n";
	MatrixTimesPoint(homo, src, dst);
	std::cout << "dst " << dst << "\n";
	std::cout << "orientaion " << atan2(dst.y - src.y, dst.x - src.x) / M_PI * 180 + 180 << " degree\n";

	src = cv::Point2f(0, img1.rows);
	std::cout << "src " << src << "\n";
	MatrixTimesPoint(homo, src, dst);
	std::cout << "dst " << dst << "\n";
	std::cout << "orientaion " << atan2(dst.y - src.y, dst.x - src.x) / M_PI * 180 + 180 << " degree\n";

	src = cv::Point2f(img1.cols, img1.rows);
	std::cout << "src " << src << "\n";
	MatrixTimesPoint(homo, src, dst);
	std::cout << "dst " << dst << "\n";
	std::cout << "orientaion " << atan2(dst.y - src.y, dst.x - src.x) / M_PI * 180 + 180 << " degree\n";

	src = cv::Point2f(img1.cols / 2, img1.rows / 2);
	std::cout << "src " << src << "\n";
	MatrixTimesPoint(homo, src, dst);
	std::cout << "dst " << dst << "\n";
	std::cout << "orientaion " << atan2(dst.y - src.y, dst.x - src.x) / M_PI * 180 + 180 << " degree\n";
}
void TestFeaturesRefine(int argc, char* argv[])
{

	int start = atoi(argv[1]);
	int end = atoi(argv[2]);
	int width = atoi(argv[3]);
	int height = atoi(argv[4]);
	int method = atoi(argv[5]);
	char* path = argv[6];
	char* outPath = argv[7];
	CreateDir(outPath);
	int dBinNum(10), aBinNum(90);
	float threshold(1.0);
	if (method == 3 && argc == 10)
	{
		dBinNum = atoi(argv[8]);
		aBinNum = atoi(argv[9]);
	}
	else if (method == 2 && argc == 9)
	{
		threshold = atof(argv[8]);
	}
	else if (argc == 9)
	{
		aBinNum = atoi(argv[8]);
	}
	cv::Size size2(width * 2, height);
	char fileName[200];
	cv::Mat img0, gray0, img1, gray1, gtImg;
	std::vector<cv::Point2f> features0, features1;
	std::vector<uchar> inliers;
	double warpErr(0);
	char methodName[20];

	for (int i = start; i <= end; i++)
	{
		sprintf(fileName, "%s//in%06d.jpg", path, i);
		img1 = imread(fileName);
		/*sprintf(fileName,"%s//groundtruth//gt%06d.png",path,i);
		gtImg = imread(fileName);
		cv::cvtColor(gtImg,gtImg,CV_BGR2GRAY);*/
		cv::cvtColor(img1, gray1, CV_BGR2GRAY);
		if (gray0.empty())
		{
			gray0 = gray1.clone();
			img0 = img1.clone();
		}
		nih::Timer timer;
		timer.start();
		KLTFeaturesMatching(gray1, gray0, features1, features0, 500, 0.05, 10);
		timer.stop();
		std::cout << i << "-----\nKLT " << timer.seconds() * 1000 << "ms\n";

		BlockRelFlowRefine BRFR(width, height, 2);

		cv::Mat homo;
		int anchorId(0);

		timer.start();
		switch (method)
		{
		case 1:
			sprintf(methodName, "Ransac");
			cv::findHomography(features1, features0, inliers, CV_RANSAC, 0.1);
			break;
		case 2:
			sprintf(methodName, "RelFlow");
			RelFlowRefine(features1, features0, inliers, anchorId);
			//RelFlowRefine(width, height, features1, features0, inliers, anchorId);
			break;
		case 3:
			sprintf(methodName, "Histogram");
			FeaturePointsRefineHistogram(features1, features0, inliers, dBinNum, aBinNum);
			break;
		case 4:
			sprintf(methodName, "BlockRefFlow");
			//BRFR.Refine(2, features1, features0, inliers, anchorId);
			BRFR.Refine(features1, features0, inliers);
			break;
		case 5:
			sprintf(methodName, "HistogramO");
			FeaturePointsRefineHistogramO(features1, features0, inliers, aBinNum);
			break;
		case 6:
			sprintf(methodName, "HistogramZ");
			FeaturePointsRefineZoom(width, height, features1, features0, inliers, aBinNum);
			break;
		default:
			sprintf(methodName, "Ransac");
			cv::findHomography(features1, features0, inliers, CV_RANSAC, 0.1);
			break;
		}

		timer.stop();
		std::cout << "Refine " << timer.seconds() * 1000 << "ms\n";
		sprintf(fileName, "%s\\%s%d.jpg", outPath, methodName, i);
		if (method == 2 || method == 4)
		{
			ShowFeatureRefine(img1, features1, img0, features0, inliers, std::string(fileName), anchorId);
		}
		else
		{
			/*ShowFeatureRefine(img1, features1, img0, features0, inliers, std::string(fileName));
			sprintf(fileName, "%s\\%s%dL.jpg", outPath, methodName, i);
			ShowFeatureRefine(img1, features1, img0, features0, inliers, std::string(fileName), true);*/

			ShowFeatureRefineSingle(img1, features1, img0, features0, inliers, fileName);
		}


		int k(0);
		for (size_t i = 0; i < inliers.size(); i++)
		{
			if (inliers[i] == 1)
			{
				features1[k] = features1[i];
				features0[k] = features0[i];
				k++;
			}

		}
		features0.resize(k);
		features1.resize(k);
		//findHomographyDLT(features1, features0, homo);
		findHomographyEqa(features1, features0, homo);
		//homo = cv::findHomography(features1, features0, CV_LMEDS);
		/*std::cout << "homoCV: \n" << homoCV << "\n";
		std::cout << "homo: \n" << homo << "\n";*/


		cv::Mat wimg0;
		cv::warpPerspective(img1, wimg0, homo, img1.size());
		cv::Mat diff;
		cv::absdiff(wimg0, img0, diff);
		sprintf(fileName, "%s\\%sDiff%d.jpg", outPath, methodName, i);
		double wr = cv::sum(cv::sum(diff))[0];
		std::cout << "total diff " << wr << " \n";
		warpErr += wr;
		cv::imwrite(fileName, diff);
		cv::swap(img0, img1);
		cv::swap(gray0, gray1);
	}

	std::cout << "warp err " << warpErr / (end - start + 1);
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
	cv::Size size2(width * 2, height);
	char fileName[200];
	cv::Mat img0, gray0, img1, gray1, gtImg;
	std::vector<cv::Point2f> features0, features1;
	std::vector<uchar>status;
	std::vector<float>err;
	std::vector<float> blkWeights;
	int errorImgNum(0);

	for (int i = start; i <= end; i++)
	{
		sprintf(fileName, "%s//in%06d.jpg", path, i);
		img1 = imread(fileName);

		cv::cvtColor(img1, gray1, CV_BGR2GRAY);
		if (gray0.empty())
		{
			gray0 = gray1.clone();
			img0 = img1.clone();
		}
		nih::Timer timer;
		timer.start();
		KLTFeaturesMatching(gray1, gray0, features1, features0, 500, 0.05, 10);
		timer.stop();
		std::cout << i << "-----\nKLT " << timer.seconds() * 1000 << "ms\n";

		RansacVsVoting(img1, features1, img0, features0);
		/*timer.stop();
		std::cout<<"klt tracking "<<timer.seconds()*1000<<"ms\n";
		timer.start();*/
		//BC2FFeaturePointsRefineHistogram(width,height,features1,features0,blkWeights,4,radSize1,thetaSize1,radSize2,thetaSize2);
		//C2FFeaturePointsRefineHistogram(width,height,features1,features0,radSize1,thetaSize1,radSize2,thetaSize2);
		//FeaturePointsRefineHistogram(width, height, features1, features0, radSize1, thetaSize1);
		std::vector<uchar> inliers;
		FeaturePointsRefineHistogram(features1, features0, inliers, radSize1, thetaSize1);

		sprintf(fileName, "histoRefine%06d.jpg", i);
		ShowFeatureRefine(img1, features1, img0, features0, inliers, std::string(fileName));
		/*cv::Mat homo;
		FeaturePointsRefineRANSAC(features1,features0,homo);*/
		std::cout << "features remians " << features1.size() << std::endl;
		/*	timer.stop();
			std::cout<<"refine "<<timer.seconds()*1000<<"ms\n";*/
		//cv::Mat homography;
		//FeaturePointsRefineRANSAC(features1,features0,homography);
		sprintf(fileName, "%s//groundtruth//gt%06d.png", path, i);
		gtImg = imread(fileName);
		cv::cvtColor(gtImg, gtImg, CV_BGR2GRAY);
		if (i > 1)
		{
			bool save = false;
			int ec(0);
			cv::Mat rstImg(size2, CV_8UC3);
			img0.copyTo(rstImg(cv::Rect(0, 0, width, height)));
			img1.copyTo(rstImg(cv::Rect(width, 0, width, height)));
			for (int j = 0; j < features1.size(); j++)
			{
				int x = (int)(features1[j].x + 0.5);
				int y = (int)(features1[j].y + 0.5);
				if (gtImg.data[x + y*width] == 0xff)
				{
					cv::line(rstImg, cv::Point(features0[j].x, features0[j].y), cv::Point(features1[j].x + width, features1[j].y), cv::Scalar(255, 0, 0));
					ec++;
					save = true;
				}
			}
			if (save)
			{
				errorImgNum++;
				sprintf(fileName, ".//error//%d_error_%d.jpg", i, ec);
				cv::imwrite(fileName, rstImg);
			}
		}
		cv::swap(img0, img1);
		cv::swap(gray0, gray1);

	}

	std::cout << "包含错误的图像数是" << errorImgNum << "\n";

}

void TestBlockWarping()
{
	int start = 230;
	int end = 260;
	int width = 1280;
	int height = 720;
	float warpErr = 0;
	int quadWidth = 8;
	cv::Size size2(width * 2, height);
	std::vector<cv::Mat> homographies;
	std::vector<float> blkWeights(quadWidth*quadWidth, 0);
	const char path[] = "..//iphone//mbh";
	char dstPath[200];
	sprintf(dstPath, "..//warpRst//mywarp//");
	CreateDir(dstPath);
	char fileName[200];
	cv::Mat img1, img0, gray1, gray0, wimg, gtImg;
	std::vector<cv::Point2f> features1, features0;
	std::vector<cv::Point2f> sf1, sf0;
	NBlockWarping nblkWarping(width, height, quadWidth);
	BlockWarping blkWarping(width, height, quadWidth);
	ASAPWarping asapWarping(width, height, quadWidth, 1.0);
	for (int i = start; i <= end; i++)
	{
		printf("%d------------\n", i);
		sprintf(fileName, "%s//in%06d.jpg", path, i);
		img1 = imread(fileName);
		/*sprintf(fileName,"%s//groundtruth//gt%06d.png",path,i);
		gtImg = imread(fileName);
		cv::cvtColor(gtImg,gtImg,CV_BGR2GRAY);*/
		cv::cvtColor(img1, gray1, CV_BGR2GRAY);
		if (gray0.empty())
		{
			gray0 = gray1.clone();
			img0 = img1.clone();
		}
		/*nih::Timer timer;
		timer.start();*/
		KLTFeaturesMatching(gray1, gray0, features1, features0, 5000, 0.05, 5);
		//SURFFeaturesMatching(gray1,gray0,features1,features0);
		/*timer.stop();
		std::cout<<"klt tracking "<<timer.seconds()*1000<<"ms\n";*/
		nih::Timer timer;
		timer.start();
		//BC2FFeaturePointsRefineHistogram(width,height,features1,features0,blkWeights,4,radSize1,thetaSize1,radSize2,thetaSize2);
		//C2FFeaturePointsRefineHistogram(width,height,features1,features0,3,1,1,3);

		//FeaturePointsRefineHistogram(width,height,features1,features0,10,36);
		cv::Mat homo;
		FeaturePointsRefineRANSAC(features1, features0, homo, 0.1);
		//if (i>1)
		//{
		//	bool save = false;
		//	int ec(0);
		//	cv::Mat rstImg(size2,CV_8UC3);
		//	img0.copyTo(rstImg(cv::Rect(0,0,width,height)));
		//	img1.copyTo(rstImg(cv::Rect(width,0,width,height)));
		//	for(int j=0; j< features1.size(); j++)
		//	{
		//		int x = (int)(features1[j].x+0.5);
		//		int y = (int)(features1[j].y+0.5);
		//		if (gtImg.data[x+y*width] == 0xff)
		//		{
		//			cv::line(rstImg,cv::Point(features0[j].x,features0[j].y),cv::Point(features1[j].x+width,features1[j].y),cv::Scalar(255,0,0));
		//			ec++;
		//			save = true;
		//		}
		//	}
		//	if (save)
		//	{
		//		sprintf(fileName,"..//warpRst//mywarp//error//%d_error_%d.jpg",i,ec);
		//		cv::imwrite(fileName,rstImg);
		//	}
		//}
		timer.stop();
		std::cout << "featues refine " << timer.seconds() * 1000 << std::endl;
		std::cout << "features remians " << features1.size() << std::endl;

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
		//blkWarping.SetFeaturePoints(features1,features0);
		nblkWarping.SetFeaturePoints(features1, features0);
		timer.stop();
		std::cout << "set featurepoints " << timer.seconds() * 1000 << std::endl;
		timer.start();
		//blkWarping.CalcBlkHomography();
		nblkWarping.Solve();
		timer.stop();
		std::cout << "calc homo " << timer.seconds() * 1000 << std::endl;
		//timer.start();
		//blkWarping.Warp(img1,wimg);
		GpuTimer gtimer;
		gtimer.Start();
		cv::gpu::GpuMat dimg, dwimg;
		dimg.upload(img1);
		//blkWarping.GpuWarp(dimg,dwimg);
		nblkWarping.GpuWarp(dimg, dwimg);
		dwimg.download(wimg);
		gtimer.Stop();

		//timer.stop();
		std::cout << "blk warp " << gtimer.Elapsed() << std::endl;
		cv::Mat flow;
		//blkWarping.getFlow(flow);
		nblkWarping.getFlow(flow);
		timer.start();
		//blkWarping.Reset();
		nblkWarping.Reset();
		//nblkWarping.getFlow(flow);
		timer.stop();
		std::cout << "reset " << timer.seconds() * 1000 << std::endl;

		timer.start();
		/*cv::Mat homo;
		findHomographyEqa(features1,features0,homo);*/
		/*cv::warpPerspective(img1,wimg,homo,img1.size());
		timer.stop();
		std::cout<<"warpPerspective "<<timer.seconds()*1000<<std::endl;*/
		/*cv::imshow("warped image",wimg);
		cv::waitKey();*/
		sprintf(fileName, "%swin%06d.jpg", dstPath, i);
		cv::imwrite(fileName, wimg);
		cv::Mat diff;
		cv::absdiff(wimg, img0, diff);
		sprintf(fileName, "%sdiff%06d.jpg", dstPath, i);
		cv::Scalar sum = cv::sum(diff);
		std::cout << sum[0] << std::endl;
		warpErr += sum[0];
		cv::imwrite(fileName, diff);

		cv::swap(img0, img1);
		cv::swap(gray0, gray1);
	}
	std::cout << "avg err " << warpErr / (end - start + 1);
}
float L2Dist(const float4& a, const float4& b)
{
	double dx = a.x - b.x;
	double dy = a.y - b.y;
	double dz = a.z - b.z;
	double dw = a.w - b.w;
	return sqrt(dx*dx + dy*dy + dz*dz + dw*dw);
}
//量化后的稀疏直方图，index是量化后的颜色id，一共有colorNum种颜色
void SparseHistogram(cv::Mat& indexImg, int colorNum, std::vector<uint2>& poses, std::vector<float>& histogram)
{
	int width = indexImg.cols;
	int height = indexImg.rows;
	int typeLen = 4;//4 for int 

	histogram.resize(colorNum);
	memset(&histogram[0], 0, sizeof(float)*histogram.size());
	for (int i = 0; i < poses.size(); i++)
	{
		int idx = poses[i].x + poses[i].y*width;
		int*ptr = (int*)(indexImg.data + idx * 4);
		int id = histogram[*ptr]++;
	}
}

void RGBHistogram(cv::Mat& fImg, std::vector<uint2>& poses, int bins, float min, float max, std::vector<float>& histogram)
{
	int width = fImg.cols;
	int height = fImg.rows;
	int typeLen = 4;//4 for float 
	float step = (max - min) / bins;
	histogram.resize(bins*bins*bins);
	memset(&histogram[0], 0, sizeof(float)*histogram.size());
	//std::ofstream file("tmp.txt");
	for (int i = 0; i < poses.size(); i++)
	{
		//file << "(" << poses[i].y << "," << poses[i].x << ")\n";
		int idx = poses[i].x + poses[i].y*width;
		float*ptr = (float*)(fImg.data + idx * 12);
		int id = 0;
		int s = 1;
		for (int c = 0; c < 3; c++)
		{
			/*file << "chanell " << c << ":" << ptr[c] * 255;
			file << " id " << (int)(ptr[c] / step) << "\n";*/
			id += s*std::min((int)(ptr[c] / step), bins - 1);
			s *= bins;
		}
		/*file << " hist id " << id << "\n";*/
		histogram[id]++;


	}
	//file.close();

	/*int minId, maxId;
	float minV, maxV;
	minMaxLoc(histogram, maxV, minV, maxId, minId);
	DrawHistogram(histogram, histogram.size(), "histogram");
	std::cout << maxId << "," << maxV << "\n";
	cv::waitKey();*/
	//cv::normalize(histogram,histogram,1.0,0.0,NORM_MINMAX);
}
void LABHistogram(cv::Mat& fImg, std::vector<uint2>& poses, int bins, std::vector<float>& histogram)
{
	int width = fImg.cols;
	int height = fImg.rows;
	int typeLen = 4;//4 for float 
	float step[3] = { 100.0 / bins, 255.0 / bins, 255.0 / bins };
	float minV[3] = { 0, -127, -127 };
	histogram.resize(bins*bins*bins);
	memset(&histogram[0], 0, sizeof(float)*histogram.size());
	/*cv::Mat lab[3];
	cv::split(fImg,lab);
	double minV(0),maxV(0);
	cv::minMaxLoc(lab[0],&minV,&maxV);
	cv::minMaxLoc(lab[1],&minV,&maxV);
	cv::minMaxLoc(lab[2],&minV,&maxV);*/
	for (int i = 0; i < poses.size(); i++)
	{
		int idx = poses[i].x + poses[i].y*width;
		float*ptr = (float*)(fImg.data + idx * 12);
		int id = 0;
		int s = 1;
		for (int c = 0; c < 3; c++)
		{
			id += s*std::min((int)ceil((ptr[c] - minV[c]) / step[c]), bins - 1);
			s *= bins;
		}
		histogram[id]++;


	}
	//cv::normalize(histogram,histogram,1.0,0.0,NORM_MINMAX);
}
void HOG(cv::Mat&mag, cv::Mat& ang, std::vector<uint2>& poses, int bins, std::vector<float>& histogram)
{
	int width = mag.cols;
	int height = ang.rows;
	float step = 360 / bins;
	histogram.resize(bins);
	/*memset(&histogram[0],0,sizeof(float)*histogram.size());*/
	for (size_t i = 0; i < histogram.size(); i++)
		histogram[i] = 1e-3;
	for (int i = 0; i < poses.size(); i++)
	{
		int idx = poses[i].x + poses[i].y*width;
		float m = *(float*)(mag.data + idx * 4);
		float a = *(float*)(ang.data + idx * 4);
		int id = std::min((int)floor(a / step), bins - 1);
		histogram[id] += m;
	}
}

void histogram()
{
	cv::Mat fimg1, fimg2, fimg3;
	cv::Mat labImg1, labImg2, labImg3;
	cv::Mat img1 = cv::imread("p1.png");
	img1.convertTo(fimg1, CV_32FC3, 1 / 255.0);
	cv::cvtColor(fimg1, labImg1, CV_BGR2Lab);
	cv::cvtColor(img1, img1, CV_BGR2GRAY);
	cv::Mat img2 = cv::imread("p2.png");
	img2.convertTo(fimg2, CV_32FC3, 1 / 255.0);
	cv::cvtColor(fimg2, labImg2, CV_BGR2Lab);
	cv::cvtColor(img2, img2, CV_BGR2GRAY);
	cv::Mat img3 = cv::imread("p3.png");
	img3.convertTo(fimg3, CV_32FC3, 1 / 255.0);
	cv::cvtColor(fimg3, labImg3, CV_BGR2Lab);
	cv::cvtColor(img3, img3, CV_BGR2GRAY);
	cv::Mat lbp1, lbp2, lbp3;
	LBPGRAY(img1, lbp1);
	GaussianBlur(img1, img1, cv::Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(img2, img2, cv::Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(img2, img3, cv::Size(3, 3), 0, 0, BORDER_DEFAULT);
	cv::Mat dx1, dy1, dx2, dy2, dx3, dy3;
	cv::Mat mag1, ang1, mag2, ang2, mag3, ang3;
	cv::Scharr(img1, dx1, CV_32F, 1, 0);
	cv::Scharr(img1, dy1, CV_32F, 0, 1);
	cv::cartToPolar(dx1, dy1, mag1, ang1, true);
	cv::Scharr(img2, dx2, CV_32F, 1, 0);
	cv::Scharr(img2, dy2, CV_32F, 0, 1);
	cv::cartToPolar(dx2, dy2, mag2, ang2, true);
	cv::Scharr(img3, dx3, CV_32F, 1, 0);
	cv::Scharr(img3, dy3, CV_32F, 0, 1);
	cv::cartToPolar(dx3, dy3, mag3, ang3, true);
	std::vector<float> hog1, hog2, hog3;

	cv::imwrite("lbp1.png", lbp1);
	LBPGRAY(img2, lbp2);
	LBPGRAY(img3, lbp3);
	std::vector<uint2> poses;
	for (int i = 1; i < img1.rows - 1; i++)
	{
		for (int j = 1; j < img1.cols - 1; j++)
			poses.push_back(make_uint2(j, i));
	}
	std::vector<float> hist1, hist2, hist3;
	RGBHistogram(fimg1, poses, 12, 0, 1, hist1);
	RGBHistogram(fimg2, poses, 12, 0, 1, hist2);
	RGBHistogram(fimg3, poses, 12, 0, 1, hist3);
	HOG(mag1, ang1, poses, 36, hog1);
	HOG(mag2, ang2, poses, 36, hog2);
	HOG(mag3, ang3, poses, 36, hog3);
	std::vector<float> labHist1, labHist2, labHist3;
	LABHistogram(labImg1, poses, 12, labHist1);
	LABHistogram(labImg2, poses, 12, labHist2);
	LABHistogram(labImg3, poses, 12, labHist3);
	std::vector<float> lbpHist1, lbpHist2, lbpHist3;
	LBPHistogram(lbp1, poses, lbpHist1);
	LBPHistogram(lbp2, poses, lbpHist2);
	LBPHistogram(lbp3, poses, lbpHist3);
	/*cv::normalize(hist1,hist1,1,0,NORM_L1 );
	cv::normalize(hist2,hist2,1,0,NORM_L1 );
	cv::normalize(hist3,hist3,1,0,NORM_L1 );*/

	float cd1 = cv::compareHist(hist1, hist2, CV_COMP_BHATTACHARYYA);
	float cd2 = cv::compareHist(hist1, hist3, CV_COMP_BHATTACHARYYA);
	float cd3 = cv::compareHist(hist2, hist3, CV_COMP_BHATTACHARYYA);

	float hd1 = cv::compareHist(hog1, hog2, CV_COMP_BHATTACHARYYA);
	float hd2 = cv::compareHist(hog1, hog3, CV_COMP_BHATTACHARYYA);
	float hd3 = cv::compareHist(hog2, hog3, CV_COMP_BHATTACHARYYA);

	float ld1 = cv::compareHist(labHist1, labHist2, CV_COMP_BHATTACHARYYA);
	float ld2 = cv::compareHist(labHist1, labHist3, CV_COMP_BHATTACHARYYA);
	float ld3 = cv::compareHist(labHist2, labHist3, CV_COMP_BHATTACHARYYA);

	float pd1 = cv::compareHist(lbpHist1, lbpHist2, CV_COMP_BHATTACHARYYA);
	float pd2 = cv::compareHist(lbpHist1, lbpHist3, CV_COMP_BHATTACHARYYA);
	float pd3 = cv::compareHist(lbpHist2, lbpHist3, CV_COMP_BHATTACHARYYA);

	std::cout << cd1 << " " << cd2 << " " << cd3 << std::endl;
	std::cout << hd1 << " " << hd2 << " " << hd3 << std::endl;
	std::cout << ld1 << " " << ld2 << " " << ld3 << std::endl;
	std::cout << pd1 << " " << pd2 << " " << pd3 << std::endl;
	DrawHistogram(hist1, hist1.size(), "hist1");
	DrawHistogram(hist2, hist2.size(), "hist2");
	DrawHistogram(hist3, hist3.size(), "hist3");
	int sum1(0), sum2(0), sum3(0);
	for (int i = 0; i < hist2.size(); i++)
	{
		sum1 += hist1[i];
		sum2 += hist2[i];
		sum3 += hist3[i];
	}
	DrawHistogram(hog1, hog1.size(), "ghist1");
	DrawHistogram(hog2, hog2.size(), "ghist2");
	DrawHistogram(hog3, hog3.size(), "ghist3");

	DrawHistogram(labHist1, labHist1.size(), "lab hist1");
	DrawHistogram(labHist2, labHist2.size(), "lab hist2");
	DrawHistogram(labHist3, labHist3.size(), "lab hist3");
	cv::waitKey();
}
//calculate color moments
void ColorMoments(std::vector<cv::Vec3b>& colors, double* moments)
{
	double coeff[] = { 0.4, 0.3, 0.3 };
	double avg[] = { 0, 0, 0 };
	double std[] = { 0, 0, 0 };
	double skw[] = { 0, 0, 0 };
	for (size_t i = 0; i < colors.size(); i++)
	{
		for (size_t c = 0; c < 3; c++)
		{
			avg[c] += colors[i][c];
		}

	}
	for (size_t c = 0; c < 3; c++)
	{
		avg[c] /= colors.size();
	}
	for (size_t i = 0; i < colors.size(); i++)
	{
		for (size_t c = 0; c < 3; c++)
		{
			double d = colors[i][c] - avg[c];
			std[c] += d*d;
			skw[c] += std[c] * d;

		}

	}
	for (size_t c = 0; c < 3; c++)
	{
		std[c] = sqrt(std[c] / colors.size());
		if (skw[c] < 0)
		{
			skw[c] = -pow(-1 * skw[c] / colors.size(), 1.0 / 3);
		}
		else
		{
			skw[c] = pow(skw[c] / colors.size(), 1.0 / 3);

		}

	}
	for (size_t i = 0; i < 3; i++)
	{
		moments[i] = coeff[0] * avg[i] + coeff[1] * std[i] + coeff[2] * skw[i];
	}
}


struct MortonCode
{
	int id;
	unsigned int code;
};
struct CodeComparer
{
	bool operator()(const MortonCode& a, const MortonCode& b)
	{
		return a.code > b.code;
	}
};
void TestMontonCode(cv::Mat& img)
{
	

	cv::Mat fimg, labImg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	cv::cvtColor(fimg, labImg, CV_BGR2Lab);
	int width = img.cols;
	int height = img.rows;
	SuperpixelComputer computer(width, height, 16, 0.65);
	int spWidth = computer.GetSPWidth();
	int spHeight = computer.GetSPHeight();
	int spSize(0), *labels(NULL);
	SLICClusterCenter* centers(NULL);
	computer.ComputeSuperpixel(img);
	cv::Mat spMap;
	computer.GetVisualResult(img, spMap);
	cv::imshow("sp", spMap);
	cv::waitKey(0);
	computer.GetSuperpixelResult(spSize, labels, centers);
	std::vector<float> x, y, r, g, b, gray;
	for (size_t i = 0; i < spHeight; i++)
	{
		for (size_t j = 0; j < spWidth; j++)
		{
			x.push_back(j*1.0 / spWidth);
			y.push_back(i*1.0 / spHeight);
			float4 rgb = centers[i*spWidth + j].rgb;
			r.push_back(rgb.x / 255);
			g.push_back(rgb.y / 255);
			b.push_back(rgb.z / 255);
			gray.push_back((rgb.x + rgb.y + rgb.z) / 3 / 255);
		}
	}
	std::vector<MortonCode> mortonCodes;
	for (size_t i = 0; i < x.size(); i++)
	{
		MortonCode code;
		code.id = i;
		code.code = morton3D(r[i],g[i],b[i]);
		mortonCodes.push_back(code);
	}
	
	std::sort(mortonCodes.begin(), mortonCodes.end(), CodeComparer());
	
	int k = spSize / 2;
	cv::Mat mask(height, width, CV_8U);
	std::vector<std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);
	for (size_t i = 0; i < k; i++)
	{
		std::vector<uint2> sps = spPoses[mortonCodes[i].id];
		for (size_t j = 0; j < sps.size(); j++)
		{
			mask.data[sps[j].x + sps[j].y*width] = 0xff;
		}
		
	}
	cv::imshow("mtest", mask);
	cv::waitKey();
}
void RegionMerging(const char* inputPath, const char* fileName, const char* outputPath, int step, bool debug = false)
{
	nih::Timer timer;
	char imgName[200];
	sprintf(imgName, "%s\\%s.jpg", inputPath, fileName);
	char outPath[200];
	sprintf(outPath, "%s%s\\", outputPath, fileName);
	
	cv::Mat img = cv::imread(imgName);
	cv::Mat fimg, gray, labImg, lbpImg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	cv::cvtColor(fimg, labImg, CV_BGR2Lab);
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
	cv::Mat dx, dy, _angImg, _magImg;
	cv::Scharr(gray, dx, CV_32F, 1, 0);
	cv::Scharr(gray, dy, CV_32F, 0, 1);
	cv::cartToPolar(dx, dy, _magImg, _angImg, true);
	sprintf(imgName, "%s\\Edges\\%s_edge.png", inputPath, fileName);
	cv::Mat edgeMap = cv::imread(imgName, -1);
	if (edgeMap.channels() == 3)
		cv::cvtColor(edgeMap, edgeMap, CV_BGR2GRAY);
	edgeMap.convertTo(edgeMap, CV_32F, 1.0 / 255);

	int _width = img.cols;
	int _height = img.rows;
	cv::Mat simg;

	SuperpixelComputer computer(_width, _height, step, 0.55);
	
	
	timer.start();
	computer.ComputeSuperpixel(img);
	timer.stop();
	

	if (debug)
	{
		std::cout << "superpixel " << timer.seconds() * 1000 << "ms\n";
		computer.GetVisualResult(img, simg);
		sprintf(imgName, "%ssuperpixel_%s.jpg", outputPath, fileName);
		cv::imwrite(imgName, simg);
	}
	

	//计算每个超像素与周围超像素的差别
	int spHeight = computer.GetSPHeight();
	int spWidth = computer.GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int _spSize(0);
	computer.GetSuperpixelResult(_spSize, labels, centers);
	//每个超像素中包含的像素以及位置	
	std::vector<std::vector<uint2>> _spPoses;
	computer.GetSuperpixelPoses(_spPoses);

	std::vector<int> nLabels;
	std::vector<SPRegion> regions;
	std::vector < std::vector<int>> neighbors;
	timer.start();
	IterativeRegionGrowing(img, edgeMap, outPath, computer, nLabels, regions, neighbors, 0.2, 20);
	timer.stop();
	if (debug)
		std::cout << "IterativeRegionGrowing " << timer.seconds() * 1000 << "ms\n";

	//求每个区域的中心距
	cv::Mat momentMask;
	momentMask.create(_height, _width, CV_32F);
	std::vector<float> moments;
	for (size_t i = 0; i < regions.size(); i++)
	{
		float dx(0),dy(0);
		float d(0);
		for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		{
			int y = regions[i].spIndices[j] / spWidth;
			int x = regions[i].spIndices[j] % spWidth;
			dx += abs(x*1.0 / spWidth - 0.5);
			dy += abs(y*1.0 / spHeight - 0.5);
			d += dx + dy;
		}
		regions[i].moment = d/regions[i].size;
		moments.push_back(regions[i].moment);
		regions[i].ad2c = make_float2(dx / regions[i].size, dy / regions[i].size);
		//for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		//{
		//	for (size_t s = 0; s < _spPoses[regions[i].spIndices[j]].size(); s++)
		//	{
		//		uint2 xy = _spPoses[regions[i].spIndices[j]][s];

		//		int idx = xy.x + xy.y*_width;
		//		//mask.at<cv::Vec3b>(xy.y, xy.x) = color;
		//		momentMask.at<float>(xy.y, xy.x) = regions[i].moment;
		//		//*(float*)(mask.data + idx * 4) = minDist;
		//		//mask.at<float>(xy.y, xy.x) = (regions[minId].color.x + regions[minId].color.y + regions[minId].color.z) / 3 / 255;
		//	}
		//}
	}
	normalize(moments, moments, 1.0, 0.0, cv::NORM_MINMAX);
	for (size_t i = 0; i < regions.size(); i++)
	{
		for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (size_t s = 0; s < _spPoses[regions[i].spIndices[j]].size(); s++)
			{
				uint2 xy = _spPoses[regions[i].spIndices[j]][s];

				int idx = xy.x + xy.y*_width;
				//mask.at<cv::Vec3b>(xy.y, xy.x) = color;
				momentMask.at<float>(xy.y, xy.x) = moments[i]*255;
				//*(float*)(mask.data + idx * 4) = minDist;
				//mask.at<float>(xy.y, xy.x) = (regions[minId].color.x + regions[minId].color.y + regions[minId].color.z) / 3 / 255;
			}
		}
	}
	
	//normalize(momentMask, momentMask, 1.0, 0, cv::NORM_MINMAX, CV_32F);
	/*double min, max;
	cv::minMaxLoc(mask, &min, &max);*/
	//cv::threshold(mask, mask, 1.5, 255, CV_THRESH_BINARY);
	momentMask.convertTo(momentMask, CV_8U, 255);
	sprintf(imgName, "%smoment_%s.jpg", outputPath, fileName);
	cv::imwrite(imgName, momentMask);


	cv::Mat salMap;	
	GetContrastMap(_width, _height, &computer, nLabels, _spPoses, regions, neighbors, salMap);
	
	sprintf(imgName, "%ssaliency_%s.jpg", outputPath, fileName);
	cv::imwrite(imgName, salMap);
	
	cv::Mat sMask;
	sMask.create(_height, _width, CV_32S);
	std::vector<int> sortedLabels;	
	for (size_t i = 0; i < nLabels.size(); i++)
	{
		if (std::find(sortedLabels.begin(), sortedLabels.end(), nLabels[i]) == sortedLabels.end())
		{
			sortedLabels.push_back(nLabels[i]);
		}
	}
	

	
	int *pixSeg = new int[_width*_height];
	for (int i = 0; i < _height; i++)
	{
		for (int j = 0; j < _width; j++)
		{
			int idx = i*_width + j;
			int id = std::find(sortedLabels.begin(), sortedLabels.end(), nLabels[labels[idx]]) - sortedLabels.begin();
			pixSeg[idx] = id;
		}
	}
	memcpy(sMask.data, pixSeg, sizeof(int)*_width*_height);
	delete[] pixSeg;
	sprintf(imgName, "%ssegment_%s.bmp", outputPath, fileName);
	cv::imwrite(imgName, sMask);

	cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, nLabels, regions, rmask);
	sprintf(imgName, "%sregion_%s_%d.jpg", outputPath, fileName, regions.size());
	cv::imwrite(imgName, rmask);
}
void SaliencyTest(const char* inputPath, const char* fileName, const char* outputPath, int step)
{
	nih::Timer timer;
	char imgName[200];
	sprintf(imgName, "%s\\%s.jpg", inputPath, fileName);
	cv::Mat img = cv::imread(imgName);
	cv::Mat fimg, gray, labImg, lbpImg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	cv::cvtColor(fimg, labImg, CV_BGR2Lab);
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
	cv::Mat dx, dy, _angImg, _magImg;
	cv::Scharr(gray, dx, CV_32F, 1, 0);
	cv::Scharr(gray, dy, CV_32F, 0, 1);
	cv::cartToPolar(dx, dy, _magImg, _angImg, true);

	int _width = img.cols;
	int _height = img.rows;
	cv::Mat simg;

	SuperpixelComputer computer(_width, _height, step, 0.55);
	timer.start();
	computer.ComputeSuperpixel(img);
	timer.stop();
	std::cout << "superpixel " << timer.seconds() * 1000 << "ms\n";

	computer.GetVisualResult(img, simg);
	sprintf(imgName, "%ssuperpixel_%s.jpg", outputPath, fileName);
	cv::imwrite(imgName, simg);

	//计算每个超像素与周围超像素的差别
	int spHeight = computer.GetSPHeight();
	int spWidth = computer.GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int _spSize(0);
	computer.GetSuperpixelResult(_spSize, labels, centers);
	//每个超像素中包含的像素以及位置	
	std::vector<std::vector<uint2>> _spPoses;
	computer.GetSuperpixelPoses(_spPoses);

	std::vector<int> nLabels;
	std::vector<SPRegion> regions;
	std::vector < std::vector<int>> neighbors;
	timer.start();
	IterativeRegionGrowing(img, computer, nLabels, regions, neighbors, 0.2, 2);
	timer.stop();
	std::cout << "IterativeRegionGrowing " << timer.seconds() * 1000 << "ms\n";
	cv::Mat regMask;
	
	
	GetRegionMap(_width, _height, &computer, nLabels, regions, regMask);
	
	sprintf(imgName, "%sregion_%s.jpg", outputPath, fileName);
	cv::imwrite(imgName, regMask);
	return;


	

	HISTOGRAMS _colorHists;
	HISTOGRAMS _HOGs;
	BuildHistogram(img, &computer, _colorHists, _HOGs);
	timer.stop();
	std::cout << "build histogram " << timer.seconds() * 1000 << "ms\n";

	

	//计算平均相邻超像素距离之间的距离
	const int dx4[] = { -1, 0, 1, 0 };
	const int dy4[] = { 0, -1, 0, 1 };

	float avgCDist(0);
	float avgGDist(0);
	float avgDist(0);
	int nc(0);
	for (int i = 0; i < spHeight; i++)
	{
		for (int j = 0; j < spWidth; j++)
		{
			int idx = i*spWidth + j;
			for (int n = 0; n < 4; n++)
			{
				int dy = i + dy4[n];
				int dx = j + dx4[n];
				if (dy >= 0 && dy < spHeight && dx >= 0 && dx < spWidth)
				{
					int nIdx = dy*spWidth + dx;
					nc++;
					avgCDist += cv::compareHist(_colorHists[idx], _colorHists[nIdx], CV_COMP_BHATTACHARYYA);
					avgGDist += cv::compareHist(_HOGs[idx], _HOGs[nIdx], CV_COMP_BHATTACHARYYA);
					avgDist += L2Dist(centers[idx].rgb, centers[nIdx].rgb);
				}
			}

		}
	}
	avgCDist /= nc;
	avgGDist /= nc;

	avgDist /= (nc * 255);

	std::vector<std::vector<float>> nColorHist, nHOGs;
	std::vector<std::vector<int>> _regIdices;
	std::vector<int>regSizes;
	std::vector<float4> regColors;
	int* segmented = new int[_width*_height];
	memset(&segmented[0], -1, sizeof(int)*_width*_height);
	float rgbHConfidence = (avgGDist) / (avgCDist + avgGDist);
	rgbHConfidence =1;
	float threshold = ((avgCDist*rgbHConfidence + (1 - rgbHConfidence)*avgGDist));
	std::cout << "rgbHConfidence " << rgbHConfidence << "\n";
	std::cout << "threshold: " << threshold << std::endl;
	//threshold = 0.4;
	//float threshold = (avgDist*rgbHConfidence+(1-rgbHConfidence)*avgGDist);

	//float qrgbHConfidence = (qavgGDist) / (qavgCDist + qavgGDist);
	////float qrgbHConfidence = 1.0;
	//float qthreshold = ((qavgCDist*qrgbHConfidence + (1 - qrgbHConfidence)*qavgGDist));
	//std::cout << "qrgbHConfidence " << qrgbHConfidence << "\n";
	//std::cout << "qthreshold: " << qthreshold << std::endl;


	std::vector<int> newLabels;
	timer.start();
	//SuperPixelRegionMerging(width, height, step, labels, centers, pos, histogram, lhistogram, qrgbComp, qgradComp, newpos, newHistograms, qthreshold, segmented, regSizes, avgColors, qrgbHConfidence);
	//SuperPixelRegionMergingFast(_width, _height, &computer, _spPoses, _colorHists, _HOGs, _regIdices, newLabels, nColorHist, nHOGs, threshold, segmented, regSizes, regColors, rgbHConfidence);
	/*cv::Mat regionLabel = cv::Mat(_height, _width, CV_32S);
	memcpy(regionLabel.data, segmented, sizeof(int)*_width*_height);
	sprintf(imgName, "%s\\Region%06d.png", path, pid);
	cv::imwrite(imgName, regionLabel);*/
	std::vector<SPRegion> Regions;
	SuperPixelRegionMergingFast(_width, _height, &computer, _spPoses, _colorHists, _HOGs, newLabels, Regions, segmented, threshold, rgbHConfidence);
	timer.stop();
	std::cout << "merging time: " << timer.seconds() * 1000 << "ms\n";
	std::vector<std::vector<int>> regNeighrbors;
	timer.start();

	//RegionAnalysis(_width, _height, &computer, segmented, newLabels, _spPoses, regSizes, _regIdices, regColors, regions);
	RegionAnalysis(_width, _height, &computer, segmented, newLabels, _spPoses, Regions, regNeighrbors);
	timer.stop();
	std::cout << "RegionAnalysis time: " << timer.seconds() * 1000 << "ms\n";

	std::cout << Regions.size() << std::endl;
	cv::Mat mask, SalMask;

	HISTOGRAMS hogs;
	HISTOGRAMS colorHists;
	int _colorBins(16);
	int _hogBins(36);
	int _totalColorBins = _colorBins*_colorBins*_colorBins;
	int _hogStep = 360.0 / _hogBins;
	float _colorSteps[3], _colorMins[3];
	//rgb color space
	_colorSteps[0] = _colorSteps[1] = _colorSteps[2] = 255.0 / _colorBins;
	_colorMins[0] = _colorMins[1] = _colorMins[2] = 0;
	colorHists.resize(Regions.size());
	hogs.resize(Regions.size());
	for (size_t i = 0; i <Regions.size(); i++)
	{
		colorHists[i].resize(_totalColorBins);
		memset(&colorHists[i][0], 0, sizeof(float)*_totalColorBins);
		hogs[i].resize(_hogBins);
		for (size_t j = 0; j < _hogBins; j++)
		{
			hogs[i][j] = 1e-2;
		}
	}
	for (size_t i = 0; i < Regions.size(); i++)
	{
		for (size_t j = 0; j < Regions[i].spIndices.size(); j++)
		{
			std::vector<uint2> poses = _spPoses[Regions[i].spIndices[j]];
			for (size_t k = 0; k < poses.size(); k++)
			{
				int idx = poses[k].x + poses[k].y*_width;
				unsigned char* rgbPtr = img.data + idx * 3;
				float* magPtr = (float*)(_magImg.data + idx * 4);
				float* angPtr = (float*)(_angImg.data + idx * 4);
				int bin = std::min<float>(floor(angPtr[0] / _hogStep), _hogBins - 1);
				hogs[i][bin] += magPtr[0];
				bin = 0;
				int s = 1;
				for (int c = 0; c < 3; c++)
				{
					//bin += s*std::min<float>(floor((labPtr[n][c]-_colorMins[c]) /_colorSteps[c]),_colorBins-1);
					bin += s*std::min<float>(floor((rgbPtr[c] - _colorMins[c]) / _colorSteps[c]), _colorBins - 1);
					s *= _colorBins;
				}
				colorHists[i][bin] ++;
			}
		}
		cv::normalize(colorHists[i], colorHists[i], 1, 0, NORM_L1);
	}
	for (int i = 0; i < Regions.size(); i++)
	{
		//float dist = cv::compareHist(colorHists[i], Regions[i].colorHist,CV_COMP_BHATTACHARYYA);
		//std::cout << "compare color hist " << dist << "\n";
		Regions[i].colorHist = colorHists[i];
	}
	GetRegionMap(_width, _height, &computer, segmented, Regions, mask);

	//cv::imshow("region merging",mask);
	sprintf(imgName, "%sRegion%s.jpg", outputPath, fileName);
	cv::imwrite(imgName, mask);
	cv::Mat idxImg(_height, _width, CV_32S);
	for (int i = 0; i < _height; i++)
	{
		int* ptr = idxImg.ptr<int>(i);
		for (int j = 0; j < _width; j++)
		{
			int idx = i*_width + j;
			ptr[j] = segmented[idx];
		}
	}
	sprintf(imgName, "%sSegmented%s.bmp", outputPath, fileName);
	cv::imwrite(imgName, idxImg);

	timer.start();
	GetSaliencyMap(_width, _height, &computer, newLabels, _spPoses, Regions, SalMask);
	sprintf(imgName, "%s%s_SalPIF.png", outputPath, fileName);
	timer.stop();
	std::cout << "GetSaliencyMap time: " << timer.seconds() * 1000 << "ms\n";
	cv::imwrite(imgName, SalMask);

	timer.start();
	cv::Mat ConMask;
	GetContrastMap(_width, _height, &computer, newLabels, _spPoses, Regions, regNeighrbors, ConMask);
	sprintf(imgName, "%s%s_SF.png", outputPath, fileName);
	timer.stop();
	std::cout << "GetContrastMap time: " << timer.seconds() * 1000 << "ms\n";
	cv::imwrite(imgName, ConMask);
	
	delete[] segmented;
	//safe_delete(qrgbComp);
	//safe_delete(qgradComp);
}
void SaliencyTest(const char* path, int pid, int step)
{
	typedef std::vector<float> HISTOGRAM;
	typedef std::vector<HISTOGRAM> HISTOGRAMS;
	nih::Timer timer;
	char imgName[200];
	sprintf(imgName, "in%06d.jpg", pid);
	char outPath[200];
	sprintf(outPath, "%s\\saliency", path);
	SaliencyTest(path, imgName, outPath, step);
}

void TestRegionDist()
{
	cv::Mat img = cv::imread("124.jpg");
	SuperpixelComputer computer(img.cols, img.rows, 40, 0.55);
	int spSize, *labels;
	SLICClusterCenter* centers;
	computer.ComputeSuperpixel(img);
	cv::Mat spImg;
	computer.GetVisualResult(img, spImg);
	cv::imwrite("spImg.jpg", spImg);

	computer.GetSuperpixelResult(spSize, labels, centers);
	cv::Mat fimg;
	img.convertTo(fimg, CV_32F, 1.0 / 255);
	HISTOGRAMS colorHist, gradHist, qColorHist;
	BuildHistogram(img, &computer, colorHist, gradHist);

	cv::Mat idxImg, colorf, colorNum;
	const int colorSize[] = { 12, 12, 12 };
	int colors = Quantize(fimg, idxImg, colorf, colorNum, 0.95, colorSize);
	BuildQHistorgram(idxImg, colors, &computer, qColorHist);
	cv::cvtColor(colorf, colorf, CV_BGR2Lab);
	cv::Mat_<float> cDistCache1f = cv::Mat::zeros(colorf.cols, colorf.cols, CV_32F); {
		cv::Vec3f* pColor = (cv::Vec3f*)colorf.data;
		for (int i = 0; i < cDistCache1f.rows; i++)
			for (int j = i + 1; j < cDistCache1f.cols; j++)
			{
				float dist = vecDist<float, 3>(pColor[i], pColor[j]);
				cDistCache1f[i][j] = cDistCache1f[j][i] = vecDist<float, 3>(pColor[i], pColor[j]);

			}

	}
	std::vector<int> newLabels(spSize);
	std::vector<SPRegion> regions;
	//init regions 
	for (int i = 0; i < spSize; i++)
	{
		newLabels[i] = i;
		SPRegion region;
		region.cX = centers[i].xy.x;
		region.cY = centers[i].xy.y;
		region.color = centers[i].rgb;
		region.colorHist = qColorHist[i];
		region.hog = gradHist[i];
		region.size = 1;
		region.dist = 0;
		region.id = i;
		region.neighbors = computer.GetNeighbors4(i);
		region.spIndices.push_back(i);
		regions.push_back(region);
		//regNeighbors.push_back(computer.GetNeighbors(i));
	}
	/*int regi = 10*4+2;
	int regj = 10*5+1;
	double qcolorDist = RegionDist(regions[regi], regions[regj], cDistCache1f);
	double clolorHistDist = cv::compareHist(colorHist[regi], colorHist[regj], CV_COMP_BHATTACHARYYA);
	double gradHistDist = cv::compareHist(gradHist[regi], gradHist[regj], CV_COMP_BHATTACHARYYA);
	std::cout << "Q colorDist = " << qcolorDist << "\n";
	std::cout << "clolorHistDist = " << clolorHistDist << "\n";
	std::cout << "gradHistDist = " << gradHistDist << "\n";*/
	std::ofstream ofile("dist.txt");
	std::vector<RegDist> QRegDists,HCRegDists,HGRegDists;
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (i < n)
			{
				double dist = RegionDist(regions[i], regions[n], cDistCache1f);

				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				rd.colorDist = dist;
				QRegDists.push_back(rd);
				ofile << "Region " << i << " ,Region " << n << " QC dist " << rd.colorDist;
				rd.colorDist = cv::compareHist(colorHist[i], colorHist[n], CV_COMP_BHATTACHARYYA);
				ofile << " HC dist " << rd.colorDist;
				HCRegDists.push_back(rd);
				rd.colorDist = cv::compareHist(gradHist[i], gradHist[n], CV_COMP_BHATTACHARYYA);
				ofile << " HG dist " << rd.colorDist << "\n";
				HGRegDists.push_back(rd);
			}
		}
	}
	ofile.close();
	std::sort(QRegDists.begin(), QRegDists.end(), RegDistDescComparer());
	std::sort(HCRegDists.begin(), HCRegDists.end(), RegDistDescComparer());
	std::sort(HGRegDists.begin(), HGRegDists.end(), RegDistDescComparer());
	
}
void TestRegionMerging(int argc, char* argv[])
{
	char* path = argv[1];
	int step = atoi(argv[2]);
	std::vector<std::string> fileNames;
	FileNameHelper::GetAllFormatFiles(path, fileNames, "*.jpg");
	char outPath[200];
	sprintf(outPath, "%s\\Regions\\", path);
	CreateDir(outPath);
	for (size_t i = 0; i < fileNames.size(); i++)
	{
		RegionMerging(path, fileNames[i].c_str(), outPath, step);
	}
}
	

void TestSaliency(int argC, char** argv)
{
	/*TestRegionDist();
	return;*/
	//histogram();
	//return;
	if (argC == 5)
	{
		int start = atoi(argv[1]);
		int end = atoi(argv[2]);
		char* path = argv[3];
		int step = atoi(argv[4]);
		for (int i = start; i <= end; i++)
		{
			SaliencyTest((const char*)path, i, step);
		}
	}
	else
	{
		char* path = argv[1];
		int step = atoi(argv[2]);
		std::vector<std::string> fileNames;
		FileNameHelper::GetAllFormatFiles(path, fileNames, "*.jpg");
		char outPath[200];
		sprintf(outPath, "%s\\saliency\\", path);
		CreateDir(outPath);
		//SaliencyTest(path, "75", outPath, step);
		
		for (int i = 6770; i < fileNames.size(); i++)
		{
			std::cout <<i<< " processing "<<fileNames[i].c_str() << "\n";
			SaliencyTest(path, fileNames[i].c_str(), outPath, step);
		}
	}
	
}


void TestLBP()
{
	cv::Mat img = cv::imread("..//moseg//cars1//in000001.jpg");
	cv::Mat lbpImg;
	//LBPRGB(img,1,8,lbpImg);
	LBPRGB(img, lbpImg);
	cv::Vec3b  val = lbpImg.at<cv::Vec3b>(3, 3);
	std::cout << (int)val[0] << "," << (int)val[1] << "," << (int)val[2] << std::endl;
	val = lbpImg.at<cv::Vec3b>(30, 30);
	std::cout << (int)val[0] << "," << (int)val[1] << "," << (int)val[2] << std::endl;
	val = lbpImg.at<cv::Vec3b>(220, 330);
	std::cout << (int)val[0] << "," << (int)val[1] << "," << (int)val[2] << std::endl;
	cv::imwrite("lbpimg.png", lbpImg);
	cv::imshow("lbpimg", lbpImg);
	cv::waitKey();
}



void TestQuantize()
{
	cv::Mat img = cv::imread("..//moseg//people1//in000001.jpg");
	img.convertTo(img, CV_32FC3, 1.0 / 255);
	cv::Mat idx1i, _color3f, _colorNum;
	double ratio = 0.95;
	const int clrNums[3] = { 12, 12, 12 };

	int num = Quantize(img, idx1i, _color3f, _colorNum, ratio, clrNums);
	std::cout << _colorNum;
	int sum(0);
	for (int i = 0; i < _colorNum.rows; i++)
	{
		int* ptr = _colorNum.ptr<int>(i);
		for (int j = 0; j < _colorNum.cols; j++)
		{
			sum += ptr[j];
		}
	}
	std::cout << "\n total pixels " << sum << "\n";
	cv::Mat qImg(img.size(), CV_8UC3);
	for (int i = 0; i < qImg.rows; i++)
	{
		int * ptr = idx1i.ptr<int>(i);
		cv::Vec3b* vptr = qImg.ptr<cv::Vec3b>(i);
		for (int j = 0; j < qImg.cols; j++)
		{
			int idx = ptr[j];
			float* color = (float*)(_color3f.data + idx * 12);
			for (int c = 0; c < 3; c++)
			{
				vptr[j][c] = floor(255 * color[c]);
			}
		}

	}
	printf("%d colors", num);
	cv::imshow("qimg", qImg);
	cv::waitKey();



}

void TestRTBS(int argc, char** argv)
{
	//warmUpDevice();
	//warmCuda();
	int start = atoi(argv[1]);
	int end = atoi(argv[2]);
	char* input = argv[3];
	char* output = argv[4];

	//VideoProcessor processor;	
	RTBackgroundSubtractor* bs;
	if (argc == 6)
	{
		float mthreshold = atof(argv[5]);
		bs = new RTBackgroundSubtractor(mthreshold);
	}
	else if (argc == 7)
	{
		float mthreshold = atof(argv[5]);
		float alpha = atof(argv[6]);
		bs = new RTBackgroundSubtractor(mthreshold, alpha);
	}
	else if (argc == 8)
	{
		float mthreshold = atof(argv[5]);
		float alpha = atof(argv[6]);
		int warpId = atoi(argv[7]);
		bs = new RTBackgroundSubtractor(mthreshold, alpha, warpId);
	}
	else
	{
		bs = new RTBackgroundSubtractor();
	}
	CreateDir(output);
	std::vector<std::string> fileNames;
	std::vector<std::string> oFileNames;
	std::vector<std::string> tfileNames;
	std::vector<std::string> salencyFileNames;
	std::vector<std::string> sfileNames;
	std::vector<cv::Mat> frames;
	for (int i = start; i <= end; i++)
	{
		char name[150];
		sprintf(name, "%s\\in%06d.jpg", input, i);
		//sprintf(name,"..\\PTZ\\input4\\drive1_%03d.png",i);
		fileNames.push_back(name);
		sprintf(name, "%sbin%06d.png", output, i);
		oFileNames.push_back(name);
		sprintf(name, "%sregion%06d.jpg", output, i);
		tfileNames.push_back(name);
		sprintf(name, "%ssal%06d.jpg", output, i);
		salencyFileNames.push_back(name);
		sprintf(name, "%ssuperpixel%06d.jpg", output, i);
		sfileNames.push_back(name);
	}
	cv::Mat frame = cv::imread(fileNames[0]);
	cv::Mat gray;
	cv::cvtColor(frame, gray, CV_BGR2GRAY);
	bs->Initialize(frame);
	cv::Mat result;
	for (int i = 0; i < fileNames.size(); i++)
	{
		frame = cv::imread(fileNames[i]);
		frames.push_back(frame);
		bs->operator()(frame, result);
		cv::imwrite(oFileNames[i], result);
		bs->GetSaliencyMap(result);
		cv::imwrite(salencyFileNames[i], result);
		bs->GetRegionMap(result);
		cv::imwrite(tfileNames[i], result);
		bs->GetSuperpixelMap(result);
		cv::imwrite(sfileNames[i], result);
		/*cv::imshow("regions",result);
		cv::waitKey();*/
	}
	bs->SetPreGray(gray);
	nih::Timer timer;
	timer.start();

	for (int i = 0; i < frames.size(); i++)
	{
		bs->operator()(frames[i], result);
		//cv::imwrite(oFileNames[i],result);
	}

	//
	timer.stop();
	std::cout << (end - start + 1) / timer.seconds() << " fps" << std::endl;


	cv::waitKey();

	safe_delete(bs);

	/*char fileName[200];
	cv::Mat fgmask,mask;
	cv::Mat img = cv::imread("..\\moseg\\cars2\\in000002.jpg");
	RTBackgroundSubtractor rtbs;
	rtbs.Initialize(img);
	rtbs.operator()(img,fgmask);
	rtbs.GetRegionMap(mask);

	cv::imshow("regions",mask);
	cv::waitKey(0);*/



}

struct Proposal
{
	Proposal(cv::Rect _window, cv::Rect _kernel, float _score) : window(_window),
		kernel(_kernel),
		score(_score){}
	cv::Rect kernel;
	cv::Rect window;
	float score;
};

void TestGpuKLT()
{
	cv::Mat img1, img0, gray1, gray0;
	std::vector<cv::Point2f> f1, f0;
	img1 = cv::imread("..//moseg//cars1//in000002.jpg");
	img0 = cv::imread("..//moseg//cars1//in000001.jpg");
	cv::cvtColor(img1, gray1, CV_BGR2GRAY);
	cv::cvtColor(img0, gray0, CV_BGR2GRAY);
	KLTFeaturesMatching(img1, img0, f1, f0);

	cv::gpu::GpuMat dImg1, dImg0, dF1, dF0, dStatus;
	cv::Mat cf1, cf0, cStatus;
	std::vector<cv::Point2f> vf1, vf0;
	std::vector<uchar> vStatus;
	dImg1.upload(gray1);
	dImg0.upload(gray0);
	cf1.create(1, f1.size(), CV_32FC2);
	for (size_t i = 0; i < f1.size(); i++)
	{
		float* ptr = (float*)(cf1.data + i * 8);
		ptr[0] = f1[i].x;
		ptr[1] = f1[i].y;
	}

	dF1.upload(cf1);
	/*cv::gpu::GoodFeaturesToTrackDetector_GPU dGFTTD(100,0.05,10);
	dGFTTD.operator()(dImg1, dF1);*/
	cv::gpu::PyrLKOpticalFlow gKLT;
	gKLT.sparse(dImg1, dImg0, dF1, dF0, dStatus);
	//dF1.download(cf1);
	dF0.download(cf0);
	dStatus.download(cStatus);
	for (size_t i = 0; i < cf1.cols; i++)
	{
		if (cStatus.data[i] == 1)
		{
			float x = *(float*)(cf1.data + i * 8);
			float y = *(float*)(cf1.data + i * 8 + 4);
			vf1.push_back(cv::Point2f(x, y));
			x = *(float*)(cf0.data + i * 8);
			y = *(float*)(cf0.data + i * 8 + 4);
			vf0.push_back(cv::Point2f(x, y));
		}
	}
	std::cout << "GPU " << vf1.size() << "\n";
	std::cout << "CPU " << f1.size() << "\n";
	int err(0);
	for (size_t i = 0; i < f1.size(); i++)
	{
		if (
			abs(vf0[i].x - f0[i].x) > 1e-1 ||
			abs(vf0[i].y - f0[i].y) > 1e-1)
		{
			std::cout << "vf0: " << vf0[i].x << " , " << vf0[i].y;
			std::cout << " f0: " << f0[i].x << " , " << f0[i].y << "\n";

			err++;

		}

	}
	std::cout << "ERROR " << err << "\n";
}

float GetWindowValue(const cv::Mat& img, size_t llx, size_t lly, size_t rrx, size_t rry)
{
	float v00 = img.ptr<int>(lly + 1)[llx + 1];
	float v01 = img.ptr<int>(lly + 1)[rrx + 1];
	float v10 = img.ptr<int>(rry + 1)[llx + 1];
	float v11 = img.ptr<int>(rry + 1)[rrx + 1];
	return v11 - v01 - v10 + v00;
}

float GetProposalScore(const cv::Mat& img, size_t i, size_t j, size_t rows, size_t cols, size_t border)
{
	size_t llx = j;
	size_t lly = i;
	size_t rrx = j + cols;
	size_t rry = i + rows;
	size_t kllx = llx + border;
	size_t klly = lly + border;
	size_t krrx = rrx - border;
	size_t krry = rry - border;
	size_t kernelSize = (rows - border)*(cols - border);
	size_t borderSize = rows*cols - kernelSize;
	float kernelV = GetWindowValue(img, kllx, klly, krrx, krry);
	float borderV = GetWindowValue(img, llx, lly, rrx, rry) - kernelV;
	return borderV / (rows + cols);
}

//img is the integral image
void WindowIteration(const cv::Mat& img, size_t rows, size_t cols, size_t step, std::vector<Proposal>& proposals)
{
	const float borderf = 0.1;
	size_t border = (max(rows, cols) - 1)*borderf;
	size_t kernelSize = (rows - border)*(cols - border);
	size_t borderSize = rows*cols - kernelSize;
	if (img.rows < rows + 1 || img.cols < cols + 1)
		return;
	for (size_t i = 0; i < img.rows - 1 - rows; i += step)
	{
		size_t lly = i;
		size_t rry = i + rows;
		if (lly<0 || rry >img.rows - 1)
			continue;
		for (size_t j = 0; j < img.cols - 1 - cols; j += step)
		{
			size_t llx = j;
			size_t rrx = j + cols;
			size_t kllx = llx + border;
			size_t klly = lly + border;
			size_t krrx = rrx - border;
			size_t krry = rry - border;
			if (llx < 0 || rrx >= img.cols - 1)
				continue;
			//std::cout << llx << "," << lly << "," << rrx << "," << rry << "\n";
			cv::Rect kernel = cv::Rect(kllx, klly, cols - border * 2, rows - border * 2);
			cv::Rect window = cv::Rect(llx, lly, cols, rows);
			float kernelV = GetWindowValue(img, kllx, klly, krrx, krry);
			float borderV = GetWindowValue(img, llx, lly, rrx, rry) - kernelV;
			float score = borderV / (rows + cols);

			Proposal p(window, kernel, score);
			proposals.push_back(p);
		}
	}
}
struct ProposalComparer
{
	bool operator()(const Proposal& p1, const Proposal& p2)
	{
		return p1.score > p2.score;
	}
};
void ProposalLocate(const cv::Mat& img, std::vector<Proposal>& proposals)
{
	size_t maxRC = max(img.rows, img.cols);
	float wRows[] = { 0.25, 0.3, 0.5, 0.7 }; // Window row size relative to image size max(imRow, imCol))
	float wCols[] = { 0.1, 0.3, 0.5, 0.4 }; // Window column size relative to image size max(imRow, imCol))
	float sampleStep[] = { 0.01, 0.015, 0.03, 0.04 }; // Window sampling step relative to image size max(imRow, imCol)) (one for each window size)
	int configs = sizeof(wRows) / sizeof(float);
	cv::Mat intImg;
	cv::integral(img, intImg);
	for (size_t i = 0; i < configs; i++)
	{
		size_t rows = wRows[i] * maxRC;
		size_t cols = wCols[i] * maxRC;
		size_t step = sampleStep[i] * maxRC;
		WindowIteration(intImg, rows, cols, step, proposals);
	}

	std::sort(proposals.begin(), proposals.end(), ProposalComparer());
}

typedef Graph<float, float, float> GraphType;
void addEdge(SLICClusterCenter* centers, GraphType* g, int i, int j, float avgDistance, float lmd1 = 0.30, float lmd2 = 0.3)
{
	float cap = lmd1 + lmd2*exp(-L1Distance(centers[i].rgb, centers[j].rgb) / 2 / avgDistance);
	g->add_edge(i, j, cap, cap);
}
void SuperpixelOptimize(SuperpixelComputer& computer, cv::Mat& img, cv::Mat& seed, cv::Mat& result, cv::Mat& contoursRst, int step = 40)
{

	size_t width = img.cols;
	size_t height = img.rows;
	contoursRst = cv::Mat::zeros(height, width, CV_8UC3);
	int* labels;
	int num(0);
	SLICClusterCenter* centers;
	computer.ComputeSuperpixel(img);
	cv::Mat spImg;
	/*computer.GetVisualResult(img, spImg);
	cv::imshow("superpixel", spImg);
	cv::waitKey();*/
	computer.GetSuperpixelResult(num, labels, centers);
	float avgDist = computer.ComputAvgColorDistance();

	GraphType *g;
	g = new GraphType(/*estimated # of nodes*/ num, /*estimated # of edges*/ num * 4);
	for (size_t i = 0; i < num; i++)
	{
		g->add_node();
		int cx = (int)(centers[i].xy.x + 0.5);
		int cy = (int)(centers[i].xy.y + 0.5);
		double num(0);
		for (int m = cx - step; m <= cx + step; m++)
		{
			if (m < 0 || m >= width)
				continue;
			for (int n = cy - step; n < cy + step; n++)
			{
				if (n < 0 || n >= height)
					continue;
				int idx = n*width + m;
				if (labels[idx] == i && seed.data[idx] == 0xff)
				{
					num++;
				}

			}
		}
		float sp = num / centers[i].nPoints;
		float d = min(1.0f, sp * 2);
		d = max(1e-20f, d);
		float d1 = -log(d);
		float d2 = max(1e-20f, 1 - d);
		d2 = -log(d2);
		g->add_tweights(i, d1, d2);
	}
	int spWidth = computer.GetSPWidth();
	int spHeight = computer.GetSPHeight();
	for (int i = 0; i < num; i++)
	{

		if (i - 1 >= 0)
		{
			addEdge(centers, g, i, i - 1, avgDist);
			if (i + spWidth - 1 < num)
				addEdge(centers, g, i, i + spWidth - 1, avgDist);
			if (i - spWidth - 1 >= 0)
				addEdge(centers, g, i, i - spWidth - 1, avgDist);
		}
		if (i + 1 < num)
		{
			addEdge(centers, g, i, i + 1, avgDist);
			if (i + spWidth + 1 < num)
				addEdge(centers, g, i, i + spWidth + 1, avgDist);
			if (i - spWidth + 1 >= 0)
				addEdge(centers, g, i, i - spWidth + 1, avgDist);
		}

		if (i - spWidth >= 0)
			addEdge(centers, g, i, i - spWidth, avgDist);

		if (i + spWidth < num)
			addEdge(centers, g, i, i + spWidth, avgDist);

	}
	float flow = g->maxflow();
	result.create(img.size(), CV_8U);
	int* labelResult = new int[num];
	std::vector<cv::Point> seeds;
	for (int i = 0; i < num; i++)
	{

		if (g->what_segment(i) == GraphType::SINK)
		{
			labelResult[i] = 1;
			seeds.push_back(cv::Point((int)(centers[i].xy.x + 0.5), (int)(centers[i].xy.y + 0.5)));
		}
		else
		{
			labelResult[i] = 0;
		}
	}

	/*computer.RegionGrowing(seeds, avgDist, labelResult);*/
	result = cv::Scalar(0);
	unsigned char* imgPtr = result.data;
	for (int i = 0; i < num; i++)
	{
		if (labelResult[i] == 1)
		{
			/*for(int j=0; j<m_spPtr[i].pixels.size(); j++)
			{
			int idx = m_spPtr[i].pixels[j].first + m_spPtr[i].pixels[j].second*m_width;
			imgPtr[idx] = 0xff;
			}*/
			int k = centers[i].xy.x;
			int j = centers[i].xy.y;
			int radius = step + 5;
			for (int x = k - radius; x <= k + radius; x++)
			{
				for (int y = j - radius; y <= j + radius; y++)
				{
					if (x<0 || x>width - 1 || y<0 || y> height - 1)
						continue;
					int idx = x + y*width;
					if (labels[idx] == i)
					{
						imgPtr[idx] = 0xff;
					}
				}
			}
		}
	}
	if (seeds.size() > 0)
	{
		cv::cvtColor(result, result, CV_GRAY2BGR);
		std::vector<cv::Point> hull;
		cv::convexHull(seeds, hull, false);
		vector<vector<cv::Point>> convexContour;  // Convex hull contour points   
		convexContour.push_back(hull);
		//approxPolyDP(hull, convexContour, 0.001, true);
		for (int i = 0; i < convexContour.size(); i++)
		{
			//drawContours(result, convexContour, i, cv::Scalar(0, 0, 255), 1, 8);
			drawContours(contoursRst, convexContour, i, cv::Scalar(255, 255, 255), CV_FILLED);
		}
	}
	delete g;
	safe_delete_array(labelResult);
}
float calcPIF(const cv::Mat& img, int i, int j, int k)
{

	int minX = max(j - k, 0);
	int minY = max(i - k, 0);
	int maxX = min(j + k, img.cols - 1);
	int maxY = min(i + k, img.rows - 1);
	int width = maxX - minX + 1;
	int height = maxY - minY + 1;
	cv::Rect roi(minX, minY, width, height);
	cv::Mat window = img(roi);
	int rows = window.rows;
	int cols = window.cols;
	uchar center = img.at<uchar>(i, j);
	float result(0);

	for (size_t i = 0; i < rows; i++)
	{
		uchar* ptr = window.ptr<uchar>(i);
		for (size_t j = 0; j < cols; j++)
		{
			result += abs(ptr[j] - center);
		}
	}
	return result / rows / cols;
}
float calcPIF(const cv::Mat& img, int i, int j, int k, float threshold)
{

	int minX = max(j - k, 0);
	int minY = max(i - k, 0);
	int maxX = min(j + k, img.cols - 1);
	int maxY = min(i + k, img.rows - 1);
	int width = maxX - minX + 1;
	int height = maxY - minY + 1;
	cv::Rect roi(minX, minY, width, height);
	cv::Mat window = img(roi);
	int rows = window.rows;
	int cols = window.cols;
	uchar center = img.at<uchar>(i, j);
	float result(0);

	for (size_t i = 0; i < rows; i++)
	{
		uchar* ptr = window.ptr<uchar>(i);
		for (size_t j = 0; j < cols; j++)
		{
			if (abs(ptr[j] - center) > threshold)
				result++;
		}
	}
	return result / rows / cols;
}
void PIF(const cv::Mat& img, cv::Mat& PIF, int k = 5)
{
	PIF.create(img.size(), CV_32F);
	cv::Mat gray;
	if (img.channels() == 3)
		cv::cvtColor(img, gray, CV_BGR2GRAY);
	else
	{
		gray = img.clone();
	}
	cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);
	size_t rows(img.rows), cols(img.cols);
	float avgK(0);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			avgK += calcPIF(gray, i, j, k);
		}

	}
	avgK /= rows*cols;
	for (size_t i = 0; i < rows; i++)
	{
		float* ptr = PIF.ptr<float>(i);

		for (size_t j = 0; j < cols; j++)
		{
			ptr[j] = calcPIF(gray, i, j, k, avgK);
		}

	}
}
void TestPIF()
{
	SuperpixelComputer computer(640, 480, 10);
	cv::Mat img = cv::imread("..//moseg//cars2//in000001.jpg");
	computer.ComputeSuperpixel(img);
	cv::Mat simg;
	computer.GetSuperpixelDownSampleImg(simg);
	cv::imshow("simg", simg);
	/*cv::Mat pimg;
	PIF(img, pimg);
	pimg.convertTo(pimg,CV_8U,255);
	cv::imwrite("pif.jpg", pimg);
	cv::imshow("pif",pimg);*/
	cv::waitKey();
}
void TestLSE()
{
	cv::Mat img = cv::imread("..//release//warperror//warp1//cars1//warpErr9.jpg");
	cv::Mat mask, u;
	/*mask = cv::Mat::zeros(img.size(), CV_8U);
	mask(cv::Rect(9, 9, mask.cols - 10 * 2 + 1, mask.rows - 10 * 2 + 1)) = cv::Scalar(255);*/
	mask = cv::imread("..//release//warperror//warp1//cars1//bin000009.png");
	cv::cvtColor(mask, mask, CV_BGR2GRAY);
	int N = 150;

	LSEWR(img, 1.5, 1.5, 0.04, 5, 3, 4, N, mask, u);
	cv::bitwise_not(u, mask);
	cv::imwrite("mask9.jpg", mask);
	img = cv::imread("..//moseg//cars1//in000009.jpg");
	N = 100;
	LSEWR(img, 1.5, 1.5, 0.04, 5, -3, 4, N, mask, u);
	/*cv::Mat f = (cv::Mat_<float>(4, 5) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
	std::cout << f << "\n";
	NeumannBoundCond<float>(f);
	std::cout << f;
	cv::Mat dx, dy;
	MGradient<float>(f, dx, dy);
	std::cout <<"\n"<< dx << "\n" << dy << "\n";
	cv::Mat lap;
	cv::Laplacian(f, lap, CV_32F);
	std::cout << lap << "\n";*/
}
void MatchContour(const vector<vector<cv::Point>> & contours, const cv::Mat& contourImg, const cv::Mat& edgeImg, vector<int>& matchedContours, cv::Mat& matchedEdgeImg)
{
	int width = edgeImg.cols;
	matchedEdgeImg = contourImg.clone();
	cv::cvtColor(matchedEdgeImg, matchedEdgeImg, CV_GRAY2BGR);
	float threshold = 0.2;//匹配条件
	float threshold2 = 20;//有效边缘的最小长度
	std::vector<cv::Point> mPonits;
	for (int i = 0; i < contours.size(); i++)
	{
		size_t m(0);
		std::vector<cv::Point> ponits;
		for (size_t j = 0; j < contours[i].size(); j++)
		{
			int idx = contours[i][j].x + contours[i][j].y*width;
			if (edgeImg.data[idx] == 0xff)
			{
				m++;
				ponits.push_back(contours[i][j]);
				mPonits.push_back(contours[i][j]);
			}

		}
		if (m > threshold*contours[i].size() && contours[i].size() > threshold2)
		{
			matchedContours.push_back(i);
			for (size_t j = 0; j < ponits.size(); j++)
			{

				cv::circle(matchedEdgeImg, ponits[j], 2, cv::Scalar(255, 0, 0));
			}

		}

	}
	if (mPonits.size() > 0)
	{
		std::vector<cv::Point> hull;
		cv::convexHull(mPonits, hull, false);
		vector<vector<cv::Point>> convexContour;  // Convex hull contour points   
		convexContour.push_back(hull);
		//approxPolyDP(hull, convexContour, 0.001, true);
		for (int i = 0; i < convexContour.size(); i++)
		{
			drawContours(matchedEdgeImg, convexContour, i, cv::Scalar(0, 0, 255), 1, 8);
		}
	}

}
void TestWarpError(int argc, char**argv)
{
	TestPIF();
	return;
	char fileName[200];
	int start = atoi(argv[1]);
	int end = atoi(argv[2]);
	int method = atoi(argv[3]);
	char* path = argv[4];
	char* outPath = argv[5];
	CreateDir(outPath);
	sprintf(fileName, "%s\\in%06d.jpg", path, start);
	cv::Mat prevImg = cv::imread(fileName);
	int width = prevImg.cols;
	int height = prevImg.rows;
	ImageWarping* warper;
	int spStep = 5;
	SuperpixelComputer spComputer(width, height, spStep);
	switch (method)
	{
	case 0:
		warper = new ASAPWarping(width, height, 8, 1.0);
		break;
	case 1:
		warper = new MyASAPWarping(width, height, 8, 1.0);
		break;
	case 2:
		warper = new BlockWarping(width, height, 8);
		break;
	case 3:
		warper = new NBlockWarping(width, height, 8);
		break;
	case 4:
		warper = new GlobalWarping(width, height);
		break;
	default:
		warper = new ASAPWarping(width, height, 8, 1.0);
		break;
	}
	cv::Mat curImg, warpImg, warpError;
	cv::Mat gray0, gray1;
	std::vector<cv::Point2f> features0, features1;
	cv::Mat homography;
	cv::cvtColor(prevImg, gray0, CV_BGR2GRAY);
	for (size_t i = start; i <= end; i++)
	{
		std::cout << i << "\n";
		sprintf(fileName, "%s\\in%06d.jpg", path, i);
		curImg = cv::imread(fileName);
		cv::cvtColor(curImg, gray1, CV_BGR2GRAY);
		KLTFeaturesMatching(gray1, gray0, features1, features0, 500);
		FeaturePointsRefineRANSAC(features1, features0, homography);
		warper->SetFeaturePoints(features1, features0);
		warper->Solve();
		warper->Reset();
		warper->Warp(gray1, warpImg);
		cv::absdiff(warpImg, gray0, warpError);
		cv::Mat outMask;
		warper->GetOutMask(outMask);
		/*cv::imshow("outmask", outMask);
		cv::waitKey(0);*/
		cv::bitwise_not(outMask, outMask);
		cv::bitwise_and(outMask, warpError, warpError);
		cv::GaussianBlur(warpError, warpError, cv::Size(3, 3), 1.0);
		cv::threshold(warpError, warpError, 50, 255, CV_THRESH_BINARY);
		sprintf(fileName, "%swarpErr%d.jpg", outPath, i);
		cv::imwrite(fileName, warpError);

		cv::Mat result, contourRst;
		SuperpixelOptimize(spComputer, curImg, warpError, result, contourRst, spStep);

		sprintf(fileName, "%soptimized%d.jpg", outPath, i);
		cv::imwrite(fileName, result);
		sprintf(fileName, "%sbin%06d.jpg", outPath, i);
		cv::imwrite(fileName, contourRst);
		cv::cvtColor(contourRst, contourRst, CV_BGR2GRAY);
		int N = 200;
		cv::Mat u, mask;
		LSEWR(warpError, 1.5, 1.5, 0.04, 5, 3, 4, N, contourRst, u);
		cv::bitwise_not(u, mask);
		sprintf(fileName, "%shbin%06d.png", outPath, i);
		cv::imwrite(fileName, mask);
		N = 100;
		sprintf(fileName, "%s\\region%06d.jpg", path, i);
		curImg = cv::imread(fileName);
		LSEWR(curImg, 1.5, 1.5, 0.04, 5, -3, 4, N, mask, u);
		cv::bitwise_not(u, u);
		sprintf(fileName, "%sbin%06d.png", outPath, i);
		cv::bitwise_not(u, mask);
		cv::imwrite(fileName, u);

		/*cv::Mat wpImg;
		curImg.copyTo(wpImg, warpError);
		sprintf(fileName, "%swarpErrImg%d.jpg", outPath, i);
		cv::imwrite(fileName, wpImg);*/

		/*cv::Mat warpErrors;
		cv::resize(warpError, warpErrors, cv::Size(width / 4, height / 4));
		sprintf(fileName, "%swarpErrS%d.jpg", outPath, i);
		cv::imwrite(fileName, warpErrors);*/

		/*vector<vector<cv::Point>> contours;
		cv::Mat imgC;
		drawImageContour(curImg, imgC, contours);
		sprintf(fileName, "%simgContour%d.jpg", outPath, i);
		cv::imwrite(fileName, imgC);*/

		/*cv::Mat mask;
		cv::Canny(warpError, mask, 100, 300);
		sprintf(fileName, "%swarpErrEdge%d.jpg", outPath, i);
		cv::imwrite(fileName, mask);*/

		/*std::vector<Proposal> proposals;
		ProposalLocate(mask, proposals);
		cv::Mat boxImg = mask.clone();
		cv::cvtColor(boxImg, boxImg, CV_GRAY2BGR);
		for (int p = 0; p < 5; p++)
		{
		cv::rectangle(boxImg, proposals[p].window, cv::Scalar(0, 0, 0xff));
		cv::rectangle(boxImg, proposals[p].kernel, cv::Scalar(0xff, 0, 0));
		}

		sprintf(fileName, "%sbox%d.jpg", outPath, i);
		cv::imwrite(fileName, boxImg);*/
		/*std::vector<int> matchedContours;
		cv::Mat contourImg = cv::Mat(curImg.size(), curImg.type(),cv::Scalar(0,0,0));
		cv::Mat matchedEdge;
		MatchContour(contours, imgC, mask, matchedContours, matchedEdge);
		for (int i = 0; i< matchedContours.size(); i++)
		{
		drawContours(contourImg, contours, matchedContours[i], cv::Scalar(255, 255, 255), 1, 8);
		}
		cv::cvtColor(contourImg, contourImg, CV_BGR2GRAY);
		sprintf(fileName, "%sWarpErrMatchedImgC%d.jpg", outPath, i);
		cv::imwrite(fileName, contourImg);
		sprintf(fileName, "%sMatchedEdges%d.jpg", outPath, i);
		cv::imwrite(fileName, matchedEdge);*/




		cv::swap(prevImg, curImg);
		cv::swap(gray1, gray0);
	}
}
