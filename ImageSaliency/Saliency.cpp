#include <stdio.h>
#include <vector>
#include <algorithm>
#include "../GpuMovingCameraBS/FileNameHelper.h"
#include "Saliency.h"
#include "../GpuMovingCameraBS/common.h"
#include "../GpuMovingCameraBS/RegionMerging.h"
#include "../GpuMovingCameraBS/timer.h"
#include "ImageFocusness.h"
#include "RegionObjectness.h"

void RegionMerging(const char* workingPath, const char* imgPath, const char* fileName, const char* outputPath, int step, bool debug = false)
{
	nih::Timer timer;
	char imgName[200];
	sprintf(imgName, "%s\\%s\\%s.jpg", workingPath, imgPath, fileName);
	
	if (debug)
	{
		char debugPath[200];
		sprintf(debugPath, "%s%s\\", outputPath, fileName);
		CreateDir(debugPath);
	}
	

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
	sprintf(imgName, "%s\\Edges\\%s_edge.png", workingPath, fileName);
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

	if (debug)
		std::cout << "IterativeRegionGrowing begin\n";

	std::vector<int> nLabels;
	std::vector<SPRegion> regions;
	std::vector < std::vector<int>> neighbors;
	cv::Mat salMap;
	timer.start();
	IterativeRegionGrowing(img, edgeMap, fileName, outputPath, computer, nLabels, regions, neighbors, 0.2, salMap, 20, debug);
	timer.stop();
	if (debug)
		std::cout << "IterativeRegionGrowing " << timer.seconds() * 1000 << "ms\n";


	
	//GetContrastMap(_width, _height, &computer, nLabels, _spPoses, regions, neighbors, salMap);
	//PickSaliencyRegion(_width, _height, &computer, nLabels, regions, salMap, 0.4);
	sprintf(imgName, "%s%s_RM.png", outputPath, fileName);
	cv::imwrite(imgName, salMap);
	sprintf(imgName, "%s%s.jpg", outputPath, fileName);
	cv::imwrite(imgName, img);
	sprintf(imgName, "%s\\gt\\%s.png", workingPath, fileName);
	cv::Mat gt = cv::imread(imgName);
	sprintf(imgName, "%s\\%s\\%s.png", workingPath, imgPath, fileName);
	cv::imwrite(imgName, gt);
	/*cv::Mat sMask;
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
	cv::imwrite(imgName, sMask);*/

	cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, nLabels, regions, rmask);
	sprintf(imgName, "%s%s_region_%d.jpg", outputPath, fileName, regions.size());
	cv::imwrite(imgName, rmask);
}
void TestImageRegionObjectness()
{
	cv::Mat img = cv::imread("0161.jpg");
	cv::Mat edgeMap = cv::imread("0161_edge.png", -1);
	cv::Mat gtSal = cv::imread("0161.png", -1);
	if (gtSal.channels() == 3)
		cv::cvtColor(gtSal, gtSal, CV_BGR2GRAY);
	cv::Mat scaleMap;

	cv::Mat fimg, gray, labImg, lbpImg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	cv::cvtColor(fimg, labImg, CV_BGR2Lab);
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
	cv::Mat dx, dy, _angImg, _magImg;
	cv::Scharr(gray, dx, CV_32F, 1, 0);
	cv::Scharr(gray, dy, CV_32F, 0, 1);
	cv::cartToPolar(dx, dy, _magImg, _angImg, true);

	if (edgeMap.channels() == 3)
		cv::cvtColor(edgeMap, edgeMap, CV_BGR2GRAY);
	edgeMap.convertTo(edgeMap, CV_32F, 1.0 / 255);

	int _width = img.cols;
	int _height = img.rows;
	cv::Mat simg;

	SuperpixelComputer computer(_width, _height, 16, 0.55);
	computer.ComputeSuperpixel(img);



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


	//build historgram
	HISTOGRAMS colorHist, gradHist, lbpHist;
	
	std::vector<SPRegion> regions(2);
	BuildHistogram(img, &computer, colorHist, gradHist, lbpHist);
	float threshold = 0.5;
	//根据gtmask 分成2个区域
	std::vector<int> nLabels(_spSize);
	int* segment = new int[_width*_height];
	float4 c0, c1;
	c0 = make_float4(0, 0, 0,0);
	c1 = c0;
	
	std::vector<float> bgHist(colorHist[0].size(), 0);
	std::vector<float> fgHist(colorHist[0].size(), 0);

	for (size_t i = 0; i < _spSize; i++)
	{
		float salP(0);
		float4 avgColor = make_float4(0, 0, 0,0);
		for (size_t j = 0; j < _spPoses[i].size(); j++)
		{
			int pos = _spPoses[i][j].x + _spPoses[i][j].y*_width;
			avgColor.x += *(char*)(img.data + pos * 3);
			avgColor.y += *(char*)(img.data + pos * 3 +1);
			avgColor.z += *(char*)(img.data + pos * 3+2 );
			if (gtSal.data[pos] == 255)
				salP++;
		}
		avgColor = avgColor * (1.0 / _spPoses[i].size());
		if (salP > threshold*_spPoses[i].size())
		{
			regions[0].spIndices.push_back(i);
			nLabels[i] = 0;
			c0 = c0 + avgColor;
			for (size_t j = 0; j < colorHist[i].size(); j++)
			{
				fgHist[j] += colorHist[i][j];
			}
		}
		else
		{
			regions[1].spIndices.push_back(i);
			nLabels[i] = 1;
			c1 = c1 + avgColor;
			for (size_t j = 0; j < colorHist[i].size(); j++)
			{
				bgHist[j] += colorHist[i][j];
			}
		}
	}
	regions[0].size = regions[0].spIndices.size();
	regions[0].color = c0 * (1.0/regions[0].size);
	regions[0].neighbors.push_back(1);
	regions[1].neighbors.push_back(0);
	regions[1].size = regions[1].spIndices.size();
	regions[1].color = c1 * (1.0 / regions[1].size);
	cv::normalize(fgHist, fgHist, 1, 0, cv::NORM_L1);
	cv::normalize(bgHist, bgHist, 1, 0, cv::NORM_L1);
	regions[0].colorHist = fgHist;
	regions[1].colorHist = bgHist;
	cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, nLabels, regions, rmask, 0, false);
	cv::imshow("region", rmask);
	
	UpdateRegionInfo(img.cols, img.rows, &computer, nLabels, regions, segment);
	delete[] segment;
	int trid = 0;
	std::vector<int> borderSPs;
	borderSPs.push_back(trid);
	RegionOutBorder(trid, regions);
	for (size_t i = 0; i < regions[trid].outBorderSPs.size(); i++)
	{
		borderSPs.push_back(regions[trid].outBorderSPs[i]);
	}
	std::vector<float> borderHist(colorHist[0].size(), 0);
	for (size_t i = 0; i < regions[trid].outBorderSPs.size(); i++)
	{
		int id = regions[trid].outBorderSPs[i];
		for (size_t j = 0; j < colorHist[id].size(); j++)
		{
			borderHist[j] += colorHist[id][j];
		}
	}
	cv::normalize(borderHist, borderHist, 1, 0, cv::NORM_L1);
	double dist = cv::compareHist(borderHist, regions[trid].colorHist, CV_COMP_BHATTACHARYYA);
	std::cout << dist << "\n";
	GetRegionMap(img.cols, img.rows, &computer, nLabels, regions, borderSPs, rmask);
	cv::imshow("region border",rmask);
	cv::waitKey();

}

void TestImageFocusness()
{
	
	cv::Mat img = cv::imread("0161.jpg");	
	cv::Mat edgeMap = cv::imread("0161_edge.png", -1);
	cv::Mat scaleMap;
	
	cv::Mat fimg, gray, labImg, lbpImg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	cv::cvtColor(fimg, labImg, CV_BGR2Lab);
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
	cv::Mat dx, dy, _angImg, _magImg;
	cv::Scharr(gray, dx, CV_32F, 1, 0);
	cv::Scharr(gray, dy, CV_32F, 0, 1);
	cv::cartToPolar(dx, dy, _magImg, _angImg, true);
	
	if (edgeMap.channels() == 3)
		cv::cvtColor(edgeMap, edgeMap, CV_BGR2GRAY);
	edgeMap.convertTo(edgeMap, CV_32F, 1.0 / 255);

	int _width = img.cols;
	int _height = img.rows;
	cv::Mat simg;

	SuperpixelComputer computer(_width, _height, 16, 0.55);	
	computer.ComputeSuperpixel(img);
	
	

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

	
	//build historgram
	HISTOGRAMS colorHist, gradHist, lbpHist;
	std::vector<int> nLabels;
	std::vector<SPRegion> regions;
	BuildHistogram(img, &computer, colorHist, gradHist, lbpHist);
	
	nLabels.resize(_spSize);
	//init regions 
	for (int i = 0; i < _spSize; i++)
	{
		nLabels[i] = i;
		SPRegion region;
		region.cX = i%spWidth;
		region.cY = i / spWidth;
		region.color = centers[i].rgb;
		region.colorHist = colorHist[i];
		region.hog = gradHist[i];
		region.lbpHist = lbpHist[i];
		region.size = 1;
		region.dist = 0;
		region.id = i;
		region.neighbors = computer.GetNeighbors4(i);
		region.spIndices.push_back(i);
		regions.push_back(region);

	}
	
	int * segment = new int[img.cols*img.rows];
	UpdateRegionInfo(img.cols, img.rows, &computer, nLabels, regions, segment);
	delete[] segment;
	GetRegionEdgeness(edgeMap, regions);
	RegionGrowing(img, computer, nLabels, regions, 0.4);
	cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, nLabels, regions, rmask, 0, false);
	cv::Mat focus;
	CalScale(gray, scaleMap);
	CalRegionFocusness(gray, scaleMap, edgeMap, _spPoses, regions, focus);
	cv::imshow("regions", rmask);
	cv::imshow("focusness", focus);
	cv::waitKey();
	

}

void EvaluateSaliency(cv::Mat& salMap)
{
	using namespace cv;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::cvtColor(salMap, salMap, CV_BGR2GRAY);
	cv::Mat outputImg;
	cv::threshold(salMap, outputImg, 128, 255, CV_THRESH_BINARY);
	/// Find contours
	findContours(outputImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	/*/// Find the convex hull object for each contour
	std::vector<std::vector<cv::Point> >hull(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
	convexHull(cv::Mat(contours[i]), hull[i], false);
	}*/

	/// Draw contours + hull results
	cv::Mat drawing = cv::Mat::zeros(salMap.size(), CV_8UC3);
	cv::RNG rng(12345);

	for (int i = 0; i< 1; i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point());
		//cv::drawContours(drawing, hull, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
	}
	cv::imshow("drawing", drawing);
	cv::waitKey(0);

}
void GetImgSaliency(int argc, char* argv[])
{
	TestImageRegionObjectness();
	/*TestImageFocusness();*/
	return;
	char* workingPath = argv[1];
	char* imgFolder = argv[2];
	char* outFolder = argv[3];
	int step = atoi(argv[4]);
	int debug(1);
	
	char path[200];
	sprintf(path, "%s\\%s\\", workingPath, imgFolder);
	std::vector<std::string> fileNames;
	FileNameHelper::GetAllFormatFiles(path, fileNames, "*.jpg");
	std::sort(fileNames.begin(), fileNames.end());
	char outPath[200];
	sprintf(outPath, "%s\\%s\\", workingPath, outFolder);
	CreateDir(outPath);
	int start = 0;
	if (argc == 7)
		start = atoi(argv[6]);
	if (argc >= 6)
	{
		debug = atoi(argv[5]);
	}
	for (size_t i = start; i < fileNames.size(); i++)
	{
		std::cout << i << ":" << fileNames[i] << "\n";
		RegionMerging(workingPath, imgFolder, fileNames[i].c_str(), outPath, step, debug);
	}
}