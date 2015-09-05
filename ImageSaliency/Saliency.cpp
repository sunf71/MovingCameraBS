#include <stdio.h>
#include <vector>
#include <algorithm>
#include "../GpuMovingCameraBS/FileNameHelper.h"
#include "Saliency.h"
#include "../GpuMovingCameraBS/common.h"
#include "../GpuMovingCameraBS/RegionMerging.h"
#include "../GpuMovingCameraBS/timer.h"


void RegionMerging(const char* workingPath, const char* imgPath, const char* fileName, const char* outputPath, int step, bool debug = false)
{
	nih::Timer timer;
	char imgName[200];
	sprintf(imgName, "%s\\%s\\%s.jpg", workingPath, imgPath, fileName);
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
	timer.start();
	IterativeRegionGrowing(img, edgeMap, outPath, computer, nLabels, regions, neighbors, 0.2, 20, true);
	timer.stop();
	if (debug)
		std::cout << "IterativeRegionGrowing " << timer.seconds() * 1000 << "ms\n";

	//求每个区域的中心距
	//cv::Mat momentMask;
	//momentMask.create(_height, _width, CV_32F);
	//std::vector<float> moments;
	//for (size_t i = 0; i < regions.size(); i++)
	//{
	//	float dx(0),dy(0);
	//	float d(0);
	//	for (size_t j = 0; j < regions[i].spIndices.size(); j++)
	//	{
	//		int y = regions[i].spIndices[j] / spWidth;
	//		int x = regions[i].spIndices[j] % spWidth;
	//		dx += abs(x*1.0 / spWidth - 0.5);
	//		dy += abs(y*1.0 / spHeight - 0.5);
	//		d += dx + dy;
	//	}
	//	regions[i].moment = d/regions[i].size;
	//	moments.push_back(regions[i].moment);
	//	regions[i].ad2c = make_float2(dx / regions[i].size, dy / regions[i].size);
	//	
	//}
	//normalize(moments, moments, 1.0, 0.0, cv::NORM_MINMAX);
	//for (size_t i = 0; i < regions.size(); i++)
	//{
	//	for (size_t j = 0; j < regions[i].spIndices.size(); j++)
	//	{
	//		for (size_t s = 0; s < _spPoses[regions[i].spIndices[j]].size(); s++)
	//		{
	//			uint2 xy = _spPoses[regions[i].spIndices[j]][s];

	//			int idx = xy.x + xy.y*_width;
	//			//mask.at<cv::Vec3b>(xy.y, xy.x) = color;
	//			momentMask.at<float>(xy.y, xy.x) = moments[i]*255;
	//			//*(float*)(mask.data + idx * 4) = minDist;
	//			//mask.at<float>(xy.y, xy.x) = (regions[minId].color.x + regions[minId].color.y + regions[minId].color.z) / 3 / 255;
	//		}
	//	}
	//}
	//momentMask.convertTo(momentMask, CV_8U, 255);
	//sprintf(imgName, "%smoment_%s.jpg", outputPath, fileName);
	//cv::imwrite(imgName, momentMask);


	cv::Mat salMap;
	//GetContrastMap(_width, _height, &computer, nLabels, _spPoses, regions, neighbors, salMap);
	PickSaliencyRegion(_width, _height, &computer, nLabels, regions, salMap, 0.4);
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
	char* workingPath = argv[1];
	char* imgFolder = argv[2];
	char* outFolder = argv[3];
	int step = atoi(argv[4]);
	char path[200];
	sprintf(path, "%s\\%s\\", workingPath, imgFolder);
	std::vector<std::string> fileNames;
	FileNameHelper::GetAllFormatFiles(path, fileNames, "*.jpg");
	std::sort(fileNames.begin(), fileNames.end());
	char outPath[200];
	sprintf(outPath, "%s\\%s\\", workingPath, outFolder);
	CreateDir(outPath);
	int start = 0;
	if (argc == 6)
		start = atoi(argv[5]);
	for (size_t i = start; i < fileNames.size(); i++)
	{
		std::cout << i << ":" << fileNames[i] << "\n";
		RegionMerging(workingPath, imgFolder, fileNames[i].c_str(), outPath, step, true);
	}
}