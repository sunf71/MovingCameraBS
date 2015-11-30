#include "RegionMerging.h"
#include <fstream>
#include <time.h>       /* time */
#include <numeric>
#include <queue>
#include "../Gpumovingcamerabs/DistanceUtils.h"
#include "../Gpumovingcamerabs/Dijkstra.h"
#include "../Gpumovingcamerabs/LBP.h"
#include "../Gpumovingcamerabs/Common.h"
#include "../Gpumovingcamerabs/timer.h"
#include "ImageFocusness.h"
#include "RegionObjectness.h"
#include "../Gpumovingcamerabs/GCoptimization.h"
#include "../QC/sparse_matlab_like_matrix.hpp"
#include "../QC/QC_full_sparse.hpp"
#include "../QC/ind_sim_pair.hpp"
#include "../QC/sparse_similarity_matrix_utils.hpp"
#include <iostream>
#include "../Gpumovingcamerabs/HistComparer.h"
extern HistComparer* histComparer;
extern std::vector< std::vector<ind_sim_pair> > A;
const float compactnessTheta = 0.4;
const float compactnessMean = 0.7;

void PrepareQCSimMatrix(int N)
{
	A.resize(N);
	int threshold = 3;
	for (int i = 0; i<N; ++i) A[i].push_back(ind_sim_pair(i, 1.0));
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = i + 1; j < i + threshold && j < N; j++)
		{
			float sim = 1 - (j - i) / threshold;
			sparse_similarity_matrix_utils::insert_into_A_symmetric_sim(A, i, j, sim); // A(0,1)= 0.2 and A(1,0)= 0.2
		}
	}


}

double QCHistogramDist(const HISTOGRAM& h1, const HISTOGRAM& h2)
{

	// The normalization factor
	double m = 0.5;
	QC_full_sparse qc_full_sparse;
	return qc_full_sparse(&h1[0], &h2[0], A, m, h1.size());

}

double RegionColorDist(const HISTOGRAM& h1, const HISTOGRAM& h2, float4 avgc1, float4 avgc2)
{
	//double QCdist = QCHistogramDist(h1, h2);
	//return QCdist;
	return cv::compareHist(h1, h2, CV_COMP_BHATTACHARYYA);
	double histDist = cv::compareHist(h1, h2, CV_COMP_BHATTACHARYYA);

	double reg0V = HistogramVariance(h1);
	double reg1V = HistogramVariance(h2);
	//处理当颜色分布集中，而且平均颜色一致时，直方图距离仍然很大的问题
	double avgDist = L2Distance(avgc1, avgc2) / 255;
	if (avgDist < 0.2 && std::max(reg0V, reg1V)> 0.1 && histDist >0.7)
	{

		return avgDist;
	}
	else
		return histDist;
	double dist = RegionDist(h1, h2, gColorDist);

	//return dist;
}
double RegionColorDist(const SPRegion& reg0, const SPRegion& reg1)
{
	return cv::compareHist(reg0.colorHist, reg1.colorHist, CV_COMP_BHATTACHARYYA);
	double QCdist = QCHistogramDist(reg0.colorHist, reg1.colorHist);
	return QCdist;
	static int idx = 0;
	double histDist = cv::compareHist(reg0.colorHist, reg1.colorHist, CV_COMP_BHATTACHARYYA);
	double avgDist = L2Distance(reg0.color, reg1.color) / 255;
	double reg0V = HistogramVariance(reg0.colorHist);
	double reg1V = HistogramVariance(reg1.colorHist);
	//处理当颜色分布集中，而且平均颜色一致时，直方图距离仍然很大的问题
	if (avgDist < 0.2 && std::max(reg0V, reg1V)> 0.1 && histDist >0.7)
	{


		if (avgDist < 0.2)
		{
			int width = 400;
			int height = 200;
			cv::Mat map(height, width, CV_8UC3);
			for (int i = 0; i < height; i++)
			{
				uchar3* ptr = map.ptr<uchar3>(i);
				for (int j = 0; j < width / 2; j++)
				{
					ptr[j].x = (uchar)reg0.color.x;
					ptr[j].y = (uchar)reg0.color.y;
					ptr[j].z = (uchar)reg0.color.z;
				}
				for (int j = width / 2; j < width; j++)
				{
					ptr[j].x = (uchar)reg1.color.x;
					ptr[j].y = (uchar)reg1.color.y;
					ptr[j].z = (uchar)reg1.color.z;
				}
			}
			char name[100];
			sprintf(name, "%dimg_%d_%d_%d_%d_%d.jpg", idx, (int)(avgDist * 100), (int)(histDist * 100), (int)(QCdist * 100), (int)(reg0V * 100), (int)(reg1V * 100));
			cv::imwrite(name, map);
			idx++;

		}

		return avgDist;

	}
	else
		return histDist;

	//double dist = RegionDist(reg0, reg1, gColorDist);

	//return dist;

}


void SuperPixelRegionMerging(int width, int height, int step, const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	HistComparer* histComp1,
	HistComparer* histComp2,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented,
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors, float confidence)
{
	std::ofstream file("mergeOut.txt");
	const int dx4[] = { -1, 0, 1, 0 };
	const int dy4[] = { 0, -1, 0, 1 };
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int spWidth = (width + step - 1) / step;
	int spHeight = (height + step - 1) / step;
	float pixDist(0);
	float regMaxDist = threshold;
	regSizes.clear();
	int regSize(0);
	//当前新标签
	int curLabel(0);
	int imgSize = spWidth*spHeight;
	char* visited = new char[imgSize];
	memset(visited, 0, imgSize);
	memset(segmented, 0, sizeof(int)*width*height);
	std::vector<cv::Point2i> neighbors;
	float4 regMean;
	std::vector<int> singleLabels;
	//region growing 后的新label
	std::vector<int> newLabels;

	newLabels.resize(imgSize);
	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	/* initialize random seed: */
	srand(time(NULL));
	int seed = rand() % imgSize;
	std::cout << "seed " << seed << "\n";
	boundarySet.insert(seed);
	//boundarySet.insert(5);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;

	while (!boundarySet.empty())
	{
		//std::cout<<boundarySet.size()<<std::endl;
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		file << "seed: " << label << "\n";
		visited[label] = true;

		labelGroup.push_back(label);

		//newLabels[label] = curLabel;
		boundarySet.erase(itr);
		SLICClusterCenter cc = centers[label];
		int k = cc.xy.x;
		int j = cc.xy.y;
		float4 regColor = cc.rgb;
		int ix = label%spWidth;
		int iy = label / spWidth;
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


		while (pixDist < regMaxDist && regSize < imgSize)
		{
			file << "iy:" << iy << " ix:" << ix << "\n";

			for (int d = 0; d < 4; d++)
			{
				int x = ix + dx4[d];
				int y = iy + dy4[d];
				if (x >= 0 && x < spWidth && y >= 0 && y < spHeight && !visited[x + y*spWidth])
				{
					neighbors.push_back(cv::Point2i(x, y));
					visited[x + y*spWidth] = true;

				}
			}
			file << "	neighbors: ";
			for (int i = 0; i < neighbors.size(); i++)
			{
				int x = neighbors[i].x;
				int y = neighbors[i].y;
				file << x + y*spWidth << "(" << y << "," << x << "),";
			}
			file << "\n";
			int idxMin = 0;
			pixDist = 255;
			if (neighbors.size() == 0)
				break;

			for (int j = 0; j < neighbors.size(); j++)
			{
				size_t idx = neighbors[j].x + neighbors[j].y*spWidth;
				float rd = histComp1->Distance(histograms[idx], histograms[label]);
				float hd = histComp2->Distance(lhistograms[idx], lhistograms[label]);
				float dist = confidence*rd + hd*(1 - confidence);
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
				int minIdx = ix + iy*spWidth;
				file << "nearst neighbor " << minIdx << "(" << iy << "," << ix << ") with distance:" << pixDist << "\n";
				/*regColor.x = (regColor.x*regSize + centers[minIdx].rgb.x)/(regSize+1);
				regColor.y = (regColor.y*regSize + centers[minIdx].rgb.y)/(regSize+1);
				regColor.z = (regColor.z*regSize + centers[minIdx].rgb.z)/(regSize+1);*/
				regColor.x += centers[minIdx].rgb.x;
				regColor.y += centers[minIdx].rgb.y;
				regColor.z += centers[minIdx].rgb.z;
				regSize++;
				labelGroup.push_back(minIdx);
				for (int i = 0; i < histograms[label].size(); i++)
				{
					histograms[label][i] += histograms[minIdx][i];

				}
				cv::normalize(histograms[label], histograms[label], 1, 0, cv::NORM_L1);
				for (int i = 0; i < lhistograms[label].size(); i++)
				{
					lhistograms[label][i] += lhistograms[minIdx][i];
				}
				cv::normalize(lhistograms[label], lhistograms[label], 1, 0, cv::NORM_L1);
				visited[minIdx] = true;
				/*segmented[minIdx] = k;*/
				//result.data[minIdx] = 0xff;
				//smask.data[minIdx] = 0xff;
				neighbors[idxMin] = neighbors[neighbors.size() - 1];
				neighbors.pop_back();
				std::set<int>::iterator itr = boundarySet.find(minIdx);
				if (itr != boundarySet.end())
				{
					boundarySet.erase(itr);
				}
			}
			else
			{
				ix = neighbors[idxMin].x;
				iy = neighbors[idxMin].y;
				int minIdx = ix + iy*spWidth;
				file << "nearst neighbor " << minIdx << "(" << iy << "," << ix << ") with distance:" << pixDist << "overpass threshold " << regMaxDist << "\n";
			}
		}
		newHistograms.push_back(histograms[label]);
		for (int i = 0; i < labelGroup.size(); i++)
		{
			newLabels[labelGroup[i]] = curLabel;
		}
		regColor.x /= regSize;
		regColor.y /= regSize;
		regColor.z /= regSize;
		regAvgColors.push_back(regColor);
		regSizes.push_back(regSize);
		curLabel++;
		for (int i = 0; i < neighbors.size(); i++)
		{
			int label = neighbors[i].x + neighbors[i].y*spWidth;
			visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);

		}
		if (regSize < 2)
			singleLabels.push_back(label);
	}

	for (int i = 0; i < newLabels.size(); i++)
	{
		int x = centers[i].xy.x;
		int y = centers[i].xy.y;
		for (int dx = -step; dx <= step; dx++)
		{
			int sx = x + dx;
			if (sx < 0 || sx >= width)
				continue;
			for (int dy = -step; dy <= step; dy++)
			{

				int sy = y + dy;
				if (sy >= 0 && sy < height)
				{
					int idx = sx + sy*width;
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

void SuperPixelRegionMerging(int width, int height, int step, const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented,
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors, float confidence)
{
	std::ofstream file("mergeOut.txt");
	const int dx4[] = { -1, 0, 1, 0 };
	const int dy4[] = { 0, -1, 0, 1 };
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int spWidth = (width + step - 1) / step;
	int spHeight = (height + step - 1) / step;
	float pixDist(0);
	float regMaxDist = threshold;
	regSizes.clear();
	int regSize(0);
	//当前新标签
	int curLabel(0);
	int imgSize = spWidth*spHeight;
	char* visited = new char[imgSize];
	memset(visited, 0, imgSize);
	memset(segmented, 0, sizeof(int)*width*height);
	std::vector<cv::Point2i> neighbors;
	float4 regMean;
	std::vector<int> singleLabels;
	//region growing 后的新label
	std::vector<int> newLabels;

	newLabels.resize(imgSize);
	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	boundarySet.insert(rand() % imgSize);
	//boundarySet.insert(95);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;

	while (!boundarySet.empty())
	{
		//std::cout<<boundarySet.size()<<std::endl;
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		file << "seed: " << label << "\n";
		visited[label] = true;

		labelGroup.push_back(label);

		//newLabels[label] = curLabel;
		boundarySet.erase(itr);
		SLICClusterCenter cc = centers[label];
		int k = cc.xy.x;
		int j = cc.xy.y;
		float4 regColor = cc.rgb;
		int ix = label%spWidth;
		int iy = label / spWidth;
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


		while (pixDist < regMaxDist && regSize < imgSize)
		{
			file << "iy:" << iy << " ix:" << ix << "\n";

			for (int d = 0; d < 4; d++)
			{
				int x = ix + dx4[d];
				int y = iy + dy4[d];
				if (x >= 0 && x < spWidth && y >= 0 && y < spHeight && !visited[x + y*spWidth])
				{
					neighbors.push_back(cv::Point2i(x, y));
					visited[x + y*spWidth] = true;

				}
			}
			file << "	neighbors: ";
			for (int i = 0; i < neighbors.size(); i++)
			{
				int x = neighbors[i].x;
				int y = neighbors[i].y;
				file << x + y*spWidth << "(" << y << "," << x << "),";
			}
			file << "\n";
			int idxMin = 0;
			pixDist = 255;
			if (neighbors.size() == 0)
				break;

			for (int j = 0; j < neighbors.size(); j++)
			{
				size_t idx = neighbors[j].x + neighbors[j].y*spWidth;
				float rd = cv::compareHist(histograms[idx], histograms[label], CV_COMP_BHATTACHARYYA);
				float hd = cv::compareHist(lhistograms[idx], lhistograms[label], CV_COMP_BHATTACHARYYA);
				float dist = confidence*rd + hd*(1 - confidence);
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
				int minIdx = ix + iy*spWidth;
				file << "nearst neighbor " << minIdx << "(" << iy << "," << ix << ") with distance:" << pixDist << "\n";
				/*regColor.x = (regColor.x*regSize + centers[minIdx].rgb.x)/(regSize+1);
				regColor.y = (regColor.y*regSize + centers[minIdx].rgb.y)/(regSize+1);
				regColor.z = (regColor.z*regSize + centers[minIdx].rgb.z)/(regSize+1);*/
				regColor.x += centers[minIdx].rgb.x;
				regColor.y += centers[minIdx].rgb.y;
				regColor.z += centers[minIdx].rgb.z;
				regSize++;
				labelGroup.push_back(minIdx);
				for (int i = 0; i < histograms[label].size(); i++)
				{
					histograms[label][i] += histograms[minIdx][i];

				}
				//cv::normalize(histograms[label], histograms[label], 1, 0, cv::NORM_L1);
				for (int i = 0; i < lhistograms[label].size(); i++)
				{
					lhistograms[label][i] += lhistograms[minIdx][i];
				}
				//cv::normalize(lhistograms[label], lhistograms[label], 1, 0, cv::NORM_L1);
				visited[minIdx] = true;
				/*segmented[minIdx] = k;*/
				//result.data[minIdx] = 0xff;
				//smask.data[minIdx] = 0xff;
				neighbors[idxMin] = neighbors[neighbors.size() - 1];
				neighbors.pop_back();
				std::set<int>::iterator itr = boundarySet.find(minIdx);
				if (itr != boundarySet.end())
				{
					boundarySet.erase(itr);
				}
			}
			else
			{
				ix = neighbors[idxMin].x;
				iy = neighbors[idxMin].y;
				int minIdx = ix + iy*spWidth;
				file << "nearst neighbor " << minIdx << "(" << iy << "," << ix << ") with distance:" << pixDist << "overpass threshold " << regMaxDist << "\n";
			}
		}
		newHistograms.push_back(histograms[label]);
		for (int i = 0; i < labelGroup.size(); i++)
		{
			newLabels[labelGroup[i]] = curLabel;
		}
		regColor.x /= regSize;
		regColor.y /= regSize;
		regColor.z /= regSize;
		regAvgColors.push_back(regColor);
		regSizes.push_back(regSize);
		curLabel++;
		for (int i = 0; i < neighbors.size(); i++)
		{
			int label = neighbors[i].x + neighbors[i].y*spWidth;
			visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);

		}
		if (regSize < 2)
			singleLabels.push_back(label);
	}

	for (int i = 0; i < newLabels.size(); i++)
	{
		int x = centers[i].xy.x;
		int y = centers[i].xy.y;
		for (int dx = -step; dx <= step; dx++)
		{
			int sx = x + dx;
			if (sx < 0 || sx >= width)
				continue;
			for (int dy = -step; dy <= step; dy++)
			{

				int sy = y + dy;
				if (sy >= 0 && sy < height)
				{
					int idx = sx + sy*width;
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

void SuperPixelRegionMergingFast(int width, int height, SuperpixelComputer* computer,
	std::vector<std::vector<uint2>>& _spPoses,
	std::vector<std::vector<float>>& _colorHists,
	std::vector<std::vector<float>>& _HOGs,
	std::vector<std::vector<int>>& _regIdices,
	std::vector<int>& newLabels,
	std::vector<std::vector<float>>& _nColorHists,
	std::vector<std::vector<float>>& _nHOGs,
	float threshold, int*& segmented,
	std::vector<int>& _regSizes, std::vector<float4>& _regColors, float confidence)
{
	//计算平均相邻超像素距离之间的距离
	static const int dx4[] = { -1, 0, 1, 0 };
	static const int dy4[] = { 0, -1, 0, 1 };
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	float avgCDist(0);
	float avgGDist(0);
	int _spSize;
	int _spWidth = computer->GetSPWidth();
	int _spHeight = computer->GetSPHeight();
	int* labels;
	SLICClusterCenter* centers;
	computer->GetSuperpixelResult(_spSize, labels, centers);
	newLabels.resize(_spSize);
	int nc(0);
#pragma omp parallel for
	for (int idx = 0; idx < _spSize; idx++)
	{
		int i = idx / _spWidth;
		int j = idx%_spWidth;
		for (int n = 0; n < 4; n++)
		{
			int dy = i + dy4[n];
			int dx = j + dx4[n];
			if (dy >= 0 && dy < _spHeight && dx >= 0 && dx < _spWidth)
			{
				int nIdx = dy*_spWidth + dx;
				nc++;
				avgCDist += cv::compareHist(_colorHists[idx], _colorHists[nIdx], CV_COMP_BHATTACHARYYA);
				avgGDist += cv::compareHist(_HOGs[idx], _HOGs[nIdx], CV_COMP_BHATTACHARYYA);

			}
		}
	}
	avgCDist /= nc;
	avgGDist /= nc;
	confidence = (avgGDist) / (avgCDist + avgGDist);
	//rgbHConfidence = (avgGDist)/(avgDist+avgGDist);
	threshold = ((avgCDist*confidence + (1 - confidence)*avgGDist));
	int _totalColorBins = _colorHists[0].size();
	int _hogBins = _HOGs[0].size();
	//std::cout<<"threshold: "<<threshold<<std::endl;
	//std::ofstream file("mergeOut.txt");

	bool* _visited = new bool[_spSize];
	float pixDist(0);
	_regSizes.clear();
	_regColors.clear();
	int regSize(0);
	//当前新标签
	int curLabel(0);
	memset(_visited, 0, _spSize);
	memset(segmented, 0, sizeof(int)*_spSize);
	std::vector<int> singleLabels;
	//region growing 后的新label



	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	boundarySet.insert(rand() % _spSize);
	//boundarySet.insert(38);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;
	float4 regColor;
	while (!boundarySet.empty())
	{
		//std::cout<<boundarySet.size()<<std::endl;
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		//file<<"seed: "<<label<<"\n";
		_visited[label] = true;
		labelGroup.push_back(label);
		//newLabels[label] = curLabel;
		boundarySet.erase(itr);
		SLICClusterCenter cc = centers[label];
		int k = cc.xy.x;
		int j = cc.xy.y;
		regColor = cc.rgb;
		int ix = label%_spWidth;
		int iy = label / _spWidth;
		pixDist = 0;
		regSize = 1;

		RegInfos neighbors, tneighbors;
		while (pixDist < threshold && regSize < _spSize)
		{
			//file<<"iy:"<<iy<<"ix:"<<ix<<"\n";

			for (int d = 0; d < 4; d++)
			{
				int x = ix + dx4[d];
				int y = iy + dy4[d];
				size_t idx = x + y*_spWidth;
				if (x >= 0 && x < _spWidth && y >= 0 && y < _spHeight && !_visited[idx])
				{
					_visited[idx] = true;
					float rd = cv::compareHist(_colorHists[idx], _colorHists[label], CV_COMP_BHATTACHARYYA);
					float hd = cv::compareHist(_HOGs[idx], _HOGs[label], CV_COMP_BHATTACHARYYA);
					float dist = confidence*rd + hd*(1 - confidence);
					neighbors.push(RegInfo(idx, x, y, dist));
				}
			}

			if (neighbors.empty())
				break;
			RegInfo sp = neighbors.top();
			pixDist = sp.dist;

			int minIdx = sp.label;
			ix = sp.x;
			iy = sp.y;
			if (pixDist < threshold)
			{
				neighbors.pop();
				//file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"\n";
				float tmpx = regColor.x;
				float tmpy = regColor.y;
				float tmpz = regColor.z;
				regColor.x = (regColor.x*regSize + centers[minIdx].rgb.x) / (regSize + 1);
				regColor.y = (regColor.y*regSize + centers[minIdx].rgb.y) / (regSize + 1);
				regColor.z = (regColor.z*regSize + centers[minIdx].rgb.z) / (regSize + 1);
				float t = 5.0;
				float dx = abs(tmpx - regColor.x);
				float dy = abs(tmpy - regColor.y);
				float dz = abs(tmpz - regColor.z);

				/*regColor.x += centers[minIdx].rgb.x;
				regColor.y += centers[minIdx].rgb.y;
				regColor.z += centers[minIdx].rgb.z;*/
				regSize++;
				labelGroup.push_back(minIdx);

#pragma omp parallel for
				for (int i = 0; i < _totalColorBins; i++)
				{
					_colorHists[label][i] += _colorHists[minIdx][i];
				}
				cv::normalize(_colorHists[label], _colorHists[label], 1, 0, cv::NORM_L1);
#pragma omp parallel for
				for (int i = 0; i < _hogBins; i++)
				{
					_HOGs[label][i] += _HOGs[minIdx][i];
				}
				cv::normalize(_HOGs[label], _HOGs[label], 1, 0, cv::NORM_L1);
				_visited[minIdx] = true;
				if (sqrt(dx*dx + dy*dy + dz*dz) > t)
				{
					while (!tneighbors.empty())
						tneighbors.pop();
					while (!neighbors.empty())
					{
						RegInfo sp = neighbors.top();
						neighbors.pop();
						float rd = cv::compareHist(_colorHists[sp.label], _colorHists[label], CV_COMP_BHATTACHARYYA);
						float hd = cv::compareHist(_HOGs[sp.label], _HOGs[label], CV_COMP_BHATTACHARYYA);
						sp.dist = confidence*rd + hd*(1 - confidence);
						tneighbors.push(sp);
					}
					std::swap(neighbors, tneighbors);
				}
				/*segmented[minIdx] = k;*/
				//result.data[minIdx] = 0xff;
				//smask.data[minIdx] = 0xff;

				std::set<int>::iterator itr = boundarySet.find(minIdx);
				if (itr != boundarySet.end())
				{
					boundarySet.erase(itr);
				}
			}
			/*else
			{
			file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"overpass threshold "<<regMaxDist<<"\n";
			}*/
		}
		//检查是否可以合并到其它区域里面去
		//		float minDist = threshold;
		//		int minId = -1;
		//		for (size_t r = 0; r < _nColorHists.size(); r++)
		//		{
		//			float rd = cv::compareHist(_nColorHists[r], _colorHists[label], CV_COMP_BHATTACHARYYA);
		//			float hd = cv::compareHist(_nHOGs[r], _HOGs[label], CV_COMP_BHATTACHARYYA);
		//			float dist = confidence*rd + hd*(1 - confidence);
		//			if (dist < minDist)
		//			{
		//				minDist = dist;
		//				minId = r;
		//			}
		//		}
		//		if (minId >= 0)
		//		{
		//			//合并到已有区域
		//#pragma omp parallel for
		//			for (int i = 0; i < _totalColorBins; i++)
		//			{
		//				_nColorHists[minId][i] += _colorHists[label][i];
		//			}
		//			cv::normalize(_nColorHists[minId], _nColorHists[minId], 1, 0, cv::NORM_L1);
		//#pragma omp parallel for
		//			for (int i = 0; i < _hogBins; i++)
		//			{
		//				_nHOGs[minId][i] += _HOGs[label][i];
		//			}
		//			cv::normalize(_nHOGs[minId], _nHOGs[minId], 1, 0, cv::NORM_L1);
		//			_regColors[minId] = (_regColors[minId] * _regSizes[minId] + regColor*regSize) *(1.0 /(regSize + _regSizes[minId]));
		//				
		//			for (int i = 0; i < labelGroup.size(); i++)
		//			{
		//				_regIdices[minId].push_back(labelGroup[i]);
		//				newLabels[labelGroup[i]] = minId;
		//			}
		//
		//			_regSizes[minId] += regSize;
		//		}
		//		else
		//		{
		//			//新区域
		//			_nColorHists.push_back(_colorHists[label]);
		//			_nHOGs.push_back(_HOGs[label]);
		//			_regIdices.push_back(labelGroup);
		//			_regColors.push_back(regColor);
		//			for (int i = 0; i < labelGroup.size(); i++)
		//			{
		//				newLabels[labelGroup[i]] = curLabel;
		//			}
		//
		//			_regSizes.push_back(regSize);
		//			curLabel++;
		//		}
		//新区域
		_nColorHists.push_back(_colorHists[label]);
		_nHOGs.push_back(_HOGs[label]);
		_regIdices.push_back(labelGroup);
		_regColors.push_back(regColor);
		for (int i = 0; i < labelGroup.size(); i++)
		{
			newLabels[labelGroup[i]] = curLabel;
		}

		_regSizes.push_back(regSize);
		curLabel++;
		std::vector<RegInfo> *vtor = (std::vector<RegInfo> *)&neighbors;
		for (int i = 0; i < vtor->size(); i++)
		{
			int label = ((RegInfo)vtor->operator [](i)).label;
			_visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);
		}

	}

#pragma omp parallel for
	for (int i = 0; i < newLabels.size(); i++)
	{

		for (int j = 0; j < _spPoses[i].size(); j++)
			segmented[_spPoses[i][j].x + _spPoses[i][j].y*width] = newLabels[i];
	}
}

void SuperPixelRegionMergingFast(int width, int height, SuperpixelComputer* computer,
	std::vector<std::vector<uint2>>& _spPoses,
	std::vector<std::vector<float>>& _colorHists,
	std::vector<std::vector<float>>& _HOGs,
	std::vector<int>& newLabels,
	std::vector<SPRegion>& regions,
	int*& segmented,
	float threshold, float confidence
	)
{
	//计算平均相邻超像素距离之间的距离
	static const int dx4[] = { -1, 0, 1, 0 };
	static const int dy4[] = { 0, -1, 0, 1 };
	int _spSize;
	int _spWidth = computer->GetSPWidth();
	int _spHeight = computer->GetSPHeight();
	int* labels;
	SLICClusterCenter* centers;
	computer->GetSuperpixelResult(_spSize, labels, centers);
	newLabels.resize(_spSize);

	int _totalColorBins = _colorHists[0].size();
	int _hogBins = _HOGs[0].size();

	bool* _visited = new bool[_spSize];
	float pixDist(0);

	int regSize(0);
	//当前新标签
	int curLabel(0);
	memset(_visited, 0, _spSize);
	memset(segmented, 0, sizeof(int)*_spSize);

	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	boundarySet.insert(rand() % _spSize);
	//boundarySet.insert(38);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;
	float4 regColor;
	while (!boundarySet.empty())
	{
		//std::cout<<boundarySet.size()<<std::endl;
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		//file<<"seed: "<<label<<"\n";
		_visited[label] = true;
		labelGroup.push_back(label);
		//newLabels[label] = curLabel;
		boundarySet.erase(itr);
		SLICClusterCenter cc = centers[label];
		if (_spPoses[label].size() == 0)
		{
			regSize = 0;
			continue;

		}

		int k = cc.xy.x;
		int j = cc.xy.y;
		regColor = cc.rgb;
		int ix = label%_spWidth;
		int iy = label / _spWidth;
		pixDist = 0;
		regSize = 1;

		SPRegionPQ neighbors, tneighbors;
		while (pixDist < threshold && regSize < _spSize)
		{
			//file<<"iy:"<<iy<<"ix:"<<ix<<"\n";

			for (int d = 0; d < 4; d++)
			{
				int x = ix + dx4[d];
				int y = iy + dy4[d];
				size_t idx = x + y*_spWidth;
				if (x >= 0 && x < _spWidth && y >= 0 && y < _spHeight && !_visited[idx])
				{
					if (_spPoses[idx].size() == 0)
						continue;
					_visited[idx] = true;
					float rd = cv::compareHist(_colorHists[idx], _colorHists[label], CV_COMP_BHATTACHARYYA);
					float hd = cv::compareHist(_HOGs[idx], _HOGs[label], CV_COMP_BHATTACHARYYA);
					float dist = confidence*rd + hd*(1 - confidence);
					neighbors.push(SPRegion(idx, x, y, dist));
				}
			}

			if (neighbors.empty())
				break;
			SPRegion sp = neighbors.top();
			pixDist = sp.dist;

			int minIdx = sp.id;
			ix = sp.cX;
			iy = sp.cY;
			if (pixDist < threshold)
			{
				neighbors.pop();
				//file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"\n";
				float tmpx = regColor.x;
				float tmpy = regColor.y;
				float tmpz = regColor.z;
				regColor.x = (regColor.x*regSize + centers[minIdx].rgb.x) / (regSize + 1);
				regColor.y = (regColor.y*regSize + centers[minIdx].rgb.y) / (regSize + 1);
				regColor.z = (regColor.z*regSize + centers[minIdx].rgb.z) / (regSize + 1);
				float t = 5.0;
				float dx = abs(tmpx - regColor.x);
				float dy = abs(tmpy - regColor.y);
				float dz = abs(tmpz - regColor.z);

				/*regColor.x += centers[minIdx].rgb.x;
				regColor.y += centers[minIdx].rgb.y;
				regColor.z += centers[minIdx].rgb.z;*/
				regSize++;
				labelGroup.push_back(minIdx);

#pragma omp parallel for
				for (int i = 0; i < _totalColorBins; i++)
				{
					_colorHists[label][i] += _colorHists[minIdx][i];
				}
				cv::normalize(_colorHists[label], _colorHists[label], 1, 0, cv::NORM_L1);
#pragma omp parallel for
				for (int i = 0; i < _hogBins; i++)
				{
					_HOGs[label][i] += _HOGs[minIdx][i];
				}
				cv::normalize(_HOGs[label], _HOGs[label], 1, 0, cv::NORM_L1);
				_visited[minIdx] = true;
				if (sqrt(dx*dx + dy*dy + dz*dz) > t)
				{
					while (!tneighbors.empty())
						tneighbors.pop();
					while (!neighbors.empty())
					{
						SPRegion sp = neighbors.top();
						neighbors.pop();
						float rd = cv::compareHist(_colorHists[sp.id], _colorHists[label], CV_COMP_BHATTACHARYYA);
						float hd = cv::compareHist(_HOGs[sp.id], _HOGs[label], CV_COMP_BHATTACHARYYA);
						sp.dist = confidence*rd + hd*(1 - confidence);
						tneighbors.push(sp);
					}
					std::swap(neighbors, tneighbors);
				}
				/*segmented[minIdx] = k;*/
				//result.data[minIdx] = 0xff;
				//smask.data[minIdx] = 0xff;

				std::set<int>::iterator itr = boundarySet.find(minIdx);
				if (itr != boundarySet.end())
				{
					boundarySet.erase(itr);
				}
			}
			/*else
			{
			file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"overpass threshold "<<regMaxDist<<"\n";
			}*/
		}
		if (regSize > 0)
		{
			SPRegion region;
			region.color = regColor;
			region.id = curLabel;
			region.colorHist = _colorHists[curLabel];
			region.hog = _HOGs[curLabel];
			region.spIndices = labelGroup;
			region.size = regSize;
			regions.push_back(region);
			for (int i = 0; i < labelGroup.size(); i++)
			{
				newLabels[labelGroup[i]] = curLabel;
			}
			curLabel++;
		}

		std::vector<SPRegion> *vtor = (std::vector<SPRegion> *)&neighbors;
		for (int i = 0; i < vtor->size(); i++)
		{
			int label = ((SPRegion)vtor->operator [](i)).id;
			_visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);
		}

	}

#pragma omp parallel for
	for (int i = 0; i < newLabels.size(); i++)
	{
		for (int j = 0; j < _spPoses[i].size(); j++)
			segmented[_spPoses[i][j].x + _spPoses[i][j].y*width] = newLabels[i];
	}
}
void GetRegionMap(int width, int height, SuperpixelComputer* computer, int* segmented, std::vector<int>& regions, std::vector<float4>& regColors, cv::Mat& mask)
{
	mask.create(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int cl = segmented[i*width + j];
			if (std::find(regions.begin(), regions.end(), cl) != regions.end())
			{
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = 0;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = 255;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = 0;
			}
			else
			{
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = (uchar)regColors[cl].x;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = (uchar)regColors[cl].y;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = (uchar)regColors[cl].z;
			}


		}
	}
}

//superpixel-level border
void GetRegionBorder(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int* segment)
{
	int * labels;
	SLICClusterCenter* centers;
	int num;
	computer->GetSuperpixelResult(num, labels, centers);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size == 0)
			continue;
		int labelI = nLabels[regions[i].spIndices[0]];
		regions[i].borders.resize(regions[i].neighbors.size());
		memset(&regions[i].borders[0], 0, sizeof(int)*regions[i].borders.size());
		regions[i].borderSpIndices.resize(regions[i].neighbors.size());
		for (int k = 0; k < regions[i].neighbors.size(); k++)
			regions[i].borderSpIndices[k].clear();
		for (int p = 0; p < regions[i].spIndices.size(); p++)
		{
			//std::cout << "p="<<p << "\n"; 
			int spIdx = regions[i].spIndices[p];
			std::vector<int> spNeighbors = computer->GetNeighbors4(spIdx);
			for (size_t n = 0; n < spNeighbors.size(); n++)
			{
				//std::cout << "n=" << n << "\n";
				for (size_t j = 0; j < regions[i].neighbors.size(); j++)
				{
					//std::cout << "j=" << j << "\n";
					int regId = regions[i].neighbors[j];
					int labelJ = nLabels[regions[regId].spIndices[0]];
					if (nLabels[spNeighbors[n]] == labelJ)
					{
						regions[i].borders[j]++;
						if (std::find(regions[i].borderSpIndices[j].begin(),
							regions[i].borderSpIndices[j].end(),
							spIdx) == regions[i].borderSpIndices[j].end())
							regions[i].borderSpIndices[j].push_back(spIdx);
					}
				}
			}
		}
	}
}

void GetRegionEdgeness(const cv::Mat& edgeMap, std::vector<SPRegion>& regions)
{
	int width(edgeMap.cols);
	int height(edgeMap.rows);
	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size == 0)
			continue;
		regions[i].edgeness.resize(regions[i].borderPixels.size());
		memset(&regions[i].edgeness[0], 0, sizeof(float)*regions[i].edgeness.size());
		for (size_t j = 0; j < regions[i].borderPixels.size(); j++)
		{
			std::vector<cv::Point>& pixels = regions[i].borderPixels[j];
			for (size_t p = 0; p < pixels.size(); p++)
			{
				int idx = (pixels[p].y*width + pixels[p].x) * 4;
				float* edgeness = (float*)(edgeMap.data + idx);
				/*if (edgeMap.data[idx] == 0xff)
				regions[i].edgeness[j]++;*/
				/*	if (*edgeness > 0.99 && *edgeness<1.01)
				regions[i].edgeness[j]++;*/
				regions[i].edgeness[j] += *edgeness;
			}

		}
	}
}



//pixel-level border
void GetRegionPixelBorder(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions)
{
	static int dx4[] = { 1, 0, -1, 0 };
	static int dy4[] = { 0, 1, 0, -1 };
	int * labels;
	SLICClusterCenter* centers;
	int num;
	computer->GetSuperpixelResult(num, labels, centers);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);
	int spWidth = computer->GetSPWidth();
	int spHeight = computer->GetSPHeight();

	for (size_t i = 0; i < regions.size(); i++)
	{
		float minX(spWidth), maxX(0), minY(spHeight), maxY(0);
		if (regions[i].size == 0)
			continue;
		int labelI = nLabels[regions[i].spIndices[0]];
		regions[i].borders.resize(regions[i].neighbors.size());
		regions[i].borderPixelNum.resize(regions[i].neighbors.size());
		memset(&regions[i].borders[0], 0, sizeof(int)*regions[i].borders.size());
		memset(&regions[i].borderPixelNum[0], 0, sizeof(int)*regions[i].borderPixelNum.size());
		regions[i].borderSpIndices.resize(regions[i].neighbors.size());
		regions[i].borderPixels.resize(regions[i].neighbors.size());
		for (int k = 0; k < regions[i].neighbors.size(); k++)
		{
			regions[i].borderSpIndices[k].clear();
			regions[i].borderPixels[k].clear();
		}
		float regPixels(0);
		float edgePixNum(0);
		float edgeSPNum(0);
		float dx(0), dy(0), rdx(0), rdy(0);
		for (int p = 0; p < regions[i].spIndices.size(); p++)
		{
			int spIdx = regions[i].spIndices[p];
			regPixels += spPoses[spIdx].size();
			int col = spIdx % spWidth;
			int row = spIdx / spWidth;
			if (col < minX)
				minX = col;
			if (col > maxX)
				maxX = col;
			if (row < minY)
				minY = row;
			if (row > maxY)
				maxY = row;
			dx += abs(col*1.0 / spWidth - 0.5);
			dy += abs(row*1.0 / spHeight - 0.5);
			rdx += abs((col - regions[i].cX)*1.0 / spWidth);
			rdy += abs((row - regions[i].cY)*1.0 / spHeight);
			if (row == 0 || row == spHeight - 1 || col == 0 || col == spWidth - 1)
			{
				edgeSPNum++;
				for (size_t k = 0; k < spPoses[spIdx].size(); k++)
				{
					uint2 pixelPos = spPoses[spIdx][k];
					if (pixelPos.x == 0 || pixelPos.x == width - 1 || pixelPos.y == 0 || pixelPos.y == height - 1)
					{
						regions[i].borderEdgePixels.push_back(cv::Point(pixelPos.x, pixelPos.y));
						edgePixNum++;
					}
				}

			}
			//std::cout << "p="<<p << "\n"; 

			std::vector<int> spNeighbors = computer->GetNeighbors4(spIdx);
			for (size_t n = 0; n < spNeighbors.size(); n++)
			{
				//std::cout << "n=" << n << "\n";
				for (size_t j = 0; j < regions[i].neighbors.size(); j++)
				{
					//std::cout << "j=" << j << "\n";
					int regId = regions[i].neighbors[j];
					int labelJ = nLabels[regions[regId].spIndices[0]];
					if (nLabels[spNeighbors[n]] == labelJ)
					{
						regions[i].borders[j]++;
						if (std::find(regions[i].borderSpIndices[j].begin(),
							regions[i].borderSpIndices[j].end(),
							spIdx) == regions[i].borderSpIndices[j].end())
						{
							regions[i].borderSpIndices[j].push_back(spIdx);
							//locate pixels  on the border
							for (size_t pix = 0; pix < spPoses[spIdx].size(); pix++)
							{
								uint2 pixelu2 = spPoses[spIdx][pix];
								cv::Point pixel(pixelu2.x, pixelu2.y);
								for (size_t pn = 0; pn < 4; pn++)
								{
									int y = pixel.y + dy4[pn];
									if (y < 0 || y >= height)
										continue;
									int x = pixel.x + dx4[pn];
									if (x >= 0 && x < width)
									{
										if (nLabels[labels[y*width + x]] == labelJ)
										{
											regions[i].borderPixels[j].push_back(pixel);
											regions[i].borderPixelNum[j]++;
										}
									}
								}
							}
						}

					}
				}
			}
		}
		regions[i].regCircum = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), edgePixNum);
		regions[i].edgeSpNum = edgeSPNum;
		regions[i].edgePixNum = edgePixNum;
		regions[i].pixels = regPixels;
		regions[i].ad2c = make_float2(dx / regions[i].size, dy / regions[i].size);
		regions[i].rad2c = make_float2(rdx / regions[i].size, rdy / regions[i].size);
		minX -= 1;
		minY -= 1;
		maxX += 1;
		maxY += 1;
		minX = std::max(0.f, minX);
		minY = std::max(0.f, minY);
		maxX = std::min(spWidth - 1.0f, maxX);
		maxY = std::min(spHeight - 1.f, maxY);
		float bwidth = maxX - minX + 1;
		float bheight = maxY - minY + 1;
		regions[i].spBbox = cv::Rect(minX, minY, bwidth, bheight);
		regions[i].compactness = std::min(bwidth, bheight) / std::max(bwidth, bheight);
		int step = computer->GetSuperpixelStep();
		std::vector<cv::Point> borderPixels;
		GetRegionBorder(regions[i], borderPixels);
		regions[i].Bbox = cv::boundingRect(borderPixels);

	}
}
void UpdateRegionInfo(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, const cv::Mat& edgeMap, std::vector<SPRegion>& regions)
{

	static int dx4[] = { 1, 0, -1, 0 };
	static int dy4[] = { 0, 1, 0, -1 };
	int * labels;
	SLICClusterCenter* centers;
	int num;
	computer->GetSuperpixelResult(num, labels, centers);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);
	int spWidth = computer->GetSPWidth();
	int spHeight = computer->GetSPHeight();

#pragma omp parallel for
	for (int i = 0; i < regions.size(); i++)
	{
		//std::cout << i << "\n";
		float minX(spWidth), maxX(0), minY(spHeight), maxY(0);
		if (regions[i].size == 0)
			continue;
		int labelI = nLabels[regions[i].spIndices[0]];
		regions[i].borders.resize(regions[i].neighbors.size());
		regions[i].borderPixelNum.resize(regions[i].neighbors.size());


		regions[i].borderSpIndices.resize(regions[i].neighbors.size());
		regions[i].borderPixels.resize(regions[i].neighbors.size());
		regions[i].edgeness.resize(regions[i].borderPixels.size());

		memset(&regions[i].edgeness[0], 0, sizeof(float)*regions[i].edgeness.size());
		memset(&regions[i].borders[0], 0, sizeof(int)*regions[i].borders.size());
		memset(&regions[i].borderPixelNum[0], 0, sizeof(int)*regions[i].borderPixelNum.size());
		for (int k = 0; k < regions[i].neighbors.size(); k++)
		{
			regions[i].borderSpIndices[k].clear();
			regions[i].borderPixels[k].clear();
		}
		float regPixels(0);
		float edgePixNum(0);
		float edgeSPNum(0);
		float dx(0), dy(0), rdx(0), rdy(0);
		for (int p = 0; p < regions[i].spIndices.size(); p++)
		{
			int spIdx = regions[i].spIndices[p];
			regPixels += spPoses[spIdx].size();
			int col = spIdx % spWidth;
			int row = spIdx / spWidth;
			if (col < minX)
				minX = col;
			if (col > maxX)
				maxX = col;
			if (row < minY)
				minY = row;
			if (row > maxY)
				maxY = row;
			dx += abs(col*1.0 / spWidth - 0.5);
			dy += abs(row*1.0 / spHeight - 0.5);
			rdx += abs((col - regions[i].cX)*1.0 / spWidth);
			rdy += abs((row - regions[i].cY)*1.0 / spHeight);
			if (row == 0 || row == spHeight - 1 || col == 0 || col == spWidth - 1)
			{
				bool flag(false);
				for (size_t k = 0; k < spPoses[spIdx].size(); k++)
				{
					uint2 pixelPos = spPoses[spIdx][k];
					if (pixelPos.x == 0 || pixelPos.x == width - 1 || pixelPos.y == 0 || pixelPos.y == height - 1)
					{
						flag = true;
						regions[i].borderEdgePixels.push_back(cv::Point(pixelPos.x, pixelPos.y));
						edgePixNum++;
					}
				}
				if (flag)
					edgeSPNum++;
			}
			//std::cout << "p="<<p << "\n"; 

			std::vector<int> spNeighbors = computer->GetNeighbors4(spIdx);
			for (size_t n = 0; n < spNeighbors.size(); n++)
			{
				//std::cout << "n=" << n << "\n";
				for (size_t j = 0; j < regions[i].neighbors.size(); j++)
				{
					//std::cout << "j=" << j << "\n";
					int regId = regions[i].neighbors[j];
					int labelJ = nLabels[regions[regId].spIndices[0]];
					if (nLabels[spNeighbors[n]] == labelJ)
					{
						regions[i].borders[j]++;
						if (std::find(regions[i].borderSpIndices[j].begin(),
							regions[i].borderSpIndices[j].end(),
							spIdx) == regions[i].borderSpIndices[j].end())
						{
							regions[i].borderSpIndices[j].push_back(spIdx);
							//locate pixels  on the border
							for (size_t pix = 0; pix < spPoses[spIdx].size(); pix++)
							{
								//std::cout << pix << "\n";
								uint2 pixelu2 = spPoses[spIdx][pix];
								cv::Point pixel(pixelu2.x, pixelu2.y);
								for (size_t pn = 0; pn < 4; pn++)
								{
									int y = pixel.y + dy4[pn];
									if (y < 0 || y >= height)
										continue;
									int x = pixel.x + dx4[pn];
									if (x >= 0 && x < width)
									{
										if (nLabels[labels[y*width + x]] == labelJ)
										{
											regions[i].borderPixels[j].push_back(pixel);
											regions[i].borderPixelNum[j]++;

											int idx = (pixel.y*width + pixel.x) * 4;
											float* edgeness = (float*)(edgeMap.data + idx);
											regions[i].edgeness[j] += *edgeness;
										}
									}
								}
							}
						}

					}
				}
			}
		}
		regions[i].regCircum = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), edgePixNum);
		regions[i].edgeSpNum = edgeSPNum;
		regions[i].edgePixNum = edgePixNum;
		regions[i].pixels = regPixels;
		regions[i].ad2c = make_float2(dx / regions[i].size, dy / regions[i].size);
		regions[i].rad2c = make_float2(rdx / regions[i].size, rdy / regions[i].size);
		minX -= 1;
		minY -= 1;
		maxX += 1;
		maxY += 1;
		minX = std::max(0.f, minX);
		minY = std::max(0.f, minY);
		maxX = std::min(spWidth - 1.0f, maxX);
		maxY = std::min(spHeight - 1.f, maxY);
		float bwidth = maxX - minX + 1;
		float bheight = maxY - minY + 1;
		regions[i].spBbox = cv::Rect(minX, minY, bwidth, bheight);
		regions[i].compactness = std::min(bwidth, bheight) / std::max(bwidth, bheight);
		int step = computer->GetSuperpixelStep();
		std::vector<cv::Point> borderPixels;
		GetRegionBorder(regions[i], borderPixels);
		regions[i].Bbox = cv::boundingRect(borderPixels);

	}

}
void UpdateRegionInfo(int width, int height, SuperpixelComputer* computer, const cv::Mat& gradMap, const cv::Mat& scaleMap, const cv::Mat& edgemap, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int * segment)
{
	//GetRegionSegment(width, height, computer, nLabels, segment);
	//GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	GetRegionPixelBorder(width, height, computer, nLabels, regions);

	GetRegionEdgeness(edgemap, regions);
	/*cv::Mat focus;
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);
	CalRegionFocusness(gradMap, scaleMap, edgemap, spPoses, regions, focus);
	cv::imshow("focus", focus);
	cv::waitKey();*/
}

void GetRegionSegment(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, int* segmet)
{
	int * labels;
	SLICClusterCenter* centers;
	int num;
	computer->GetSuperpixelResult(num, labels, centers);
	//#pragma omp parallel for
	for (int i = 0; i < _height; i++)
	{

		for (int j = 0; j < _width; j++)
		{
			int idx = i*_width + j;

			segmet[idx] = nLabels[labels[idx]];
		}
	}
}
void GetRegionSegment(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, cv::Mat& segment)
{
	int * labels;
	SLICClusterCenter* centers;
	int num;
	computer->GetSuperpixelResult(num, labels, centers);
	segment = cv::Mat::zeros(_height, _width, CV_32S);
	std::vector<int> rlabels;
	std::vector<int> rId;

	for (int i = 0; i < _height; i++)
	{
		int* ptr = segment.ptr<int>(i);
		for (int j = 0; j < _width; j++)
		{
			int idx = i*_width + j;
			/*std::vector<int>::iterator itr = std::find(rlabels.begin(), rlabels.end(), nLabels[labels[idx]]);
			if (itr == rlabels.end())
			{
			rlabels.push_back(nLabels[labels[idx]]);
			rId.push_back(rlabels.size());
			ptr[j] = rlabels.size();
			}
			else
			{
			ptr[j] = rId[itr - rlabels.begin()];
			}*/
			ptr[j] = nLabels[labels[idx]];
		}
	}
}
void GetRegionMap(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, cv::Mat& mask, int flag, bool text)
{
	int * labels;
	SLICClusterCenter* centers;
	int num;
	computer->GetSuperpixelResult(num, labels, centers);
	int *pixSeg = new int[_width*_height];
	for (int i = 0; i < _height; i++)
	{
		for (int j = 0; j < _width; j++)
		{
			int idx = i*_width + j;
			pixSeg[idx] = nLabels[labels[idx]];
		}
	}
	GetRegionMap(_width, _height, computer, pixSeg, regions, mask, flag, text);
	delete[] pixSeg;
}
void GetRegionMap(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, std::vector<uint2>& regParis, cv::Mat& mask, int flag)
{
	mask.create(_height, _width, CV_8UC3);
	int * labels;
	SLICClusterCenter* centers;
	int _spSize;
	computer->GetSuperpixelResult(_spSize, labels, centers);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	int *pixSeg = new int[_width*_height];
	for (int i = 0; i < _height; i++)
	{
		for (int j = 0; j < _width; j++)
		{
			int idx = i*_width + j;
			pixSeg[idx] = nLabels[labels[idx]];
		}
	}
	std::vector<int> color(_spSize);
	CvRNG rng = cvRNG(cvGetTickCount());
	for (int i = 0; i < _spSize; i++)
		color[i] = cvRandInt(&rng);

	for (int i = 0; i < regions.size(); i++)
	{

		for (int j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[i].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[i].spIndices[j]][k].x;
				int r = spPoses[regions[i].spIndices[j]][k].y;
				if (flag == 0)
				{
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = (uchar)(regions[i].color.x);
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = (uchar)(regions[i].color.y);
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = (uchar)(regions[i].color.z);
				}
				else
				{
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = (uchar)(color[i]) & 255;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = (uchar)(color[i] >> 8) & 255;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = (uchar)(color[i] >> 16) & 255;
				}


			}
		}
	}
	for (size_t i = 0; i < regParis.size(); i++)
	{
		int ri = regParis[i].x;
		int rj = regParis[i].y;
		for (int j = 0; j < regions[ri].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[ri].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[ri].spIndices[j]][k].x;
				int r = spPoses[regions[ri].spIndices[j]][k].y;

				{
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = (uchar)(color[i]) & 255;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = (uchar)(color[i] >> 8) & 255;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = (uchar)(color[i] >> 16) & 255;
				}

			}
		}
		for (int j = 0; j < regions[rj].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[rj].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[rj].spIndices[j]][k].x;
				int r = spPoses[regions[rj].spIndices[j]][k].y;

				{
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = (uchar)(((color[i]) & 255)*0.8);
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = (uchar)(((color[i] >> 8) & 255)*0.8);
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = (uchar)(((color[i] >> 16) & 255)*0.8);
				}

			}
		}
	}
	delete[] pixSeg;
}
void GetRegionSaliencyMap(std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, std::vector<float>& regSalScores)
{
	for (int id = 0; id < regInfos.size() - 1; id++)
	{
		int i = regInfos[id].id;

		//float weight = regInfos[id].contrast * exp(-9.0*sqr(regInfos[id].ad2c));

		float weight = regInfos[id].contrast*exp(-6.0*(sqr(regInfos[id].ad2c) + regInfos[id].borderRatio));

		regSalScores.push_back(weight);
	}
}

void GetRegionSaliencyMap(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, int candiRegSize, cv::Mat& mask)
{
	int bkgRegId = regInfos[regInfos.size() - 1].id;
	mask = cv::Mat::zeros(_height, _width, CV_32F);
	int * labels;
	SLICClusterCenter* centers;
	int _spSize;
	computer->GetSuperpixelResult(_spSize, labels, centers);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	//std::cout << "Get saliency map-------------\n";
	/*std::vector<double> saliencyScore;
	for (size_t i = 0; i < regInfos.size() - 1; i++)
	{
	saliencyScore.push_back(regInfos[i].RegionSaliency());
	}
	double maxSal = exp(-0.1*regInfos.size());
	double minSal = maxSal / (regInfos.size()-1);
	cv::normalize(saliencyScore, saliencyScore, minSal, maxSal, CV_MINMAX);*/

	for (int id = 0; id < regInfos.size() - 1; id++)
	{
		int i = regInfos[id].id;


		float weight = regInfos[id].contrast * exp(-9.0*sqr(regInfos[id].ad2c));


		//std::cout << "\treg " << i << " contrast " << regInfos[id].contrast << " ad2c " << exp(-9.0*sqr(regInfos[id].ad2c)) << std::endl;
		for (int j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[i].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[i].spIndices[j]][k].x;
				int r = spPoses[regions[i].spIndices[j]][k].y;

				*(float*)(mask.data + (r*_width + c) * 4) = weight;
				//((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = 0xff;



			}
		}

	}
	//mask.mul(mask, (1 - (regInfos.size() - 1.0) / candiRegSize));
	//cv::normalize(mask, mask, 0, 1, CV_MINMAX, CV_32F);
	//mask.convertTo(mask, CV_8U, 255);

}

void GetRegionMap(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, std::vector<int>& flagRegs, cv::Mat& mask)
{
	mask.create(_height, _width, CV_8UC3);
	int * labels;
	SLICClusterCenter* centers;
	int _spSize;
	computer->GetSuperpixelResult(_spSize, labels, centers);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	int *pixSeg = new int[_width*_height];
	for (int i = 0; i < _height; i++)
	{
		for (int j = 0; j < _width; j++)
		{
			int idx = i*_width + j;
			pixSeg[idx] = nLabels[labels[idx]];
		}
	}
	std::vector<int> color(_spSize);
	CvRNG rng = cvRNG(cvGetTickCount());
	for (int i = 0; i < _spSize; i++)
		color[i] = cvRandInt(&rng);
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[i].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[i].spIndices[j]][k].x;
				int r = spPoses[regions[i].spIndices[j]][k].y;

				{
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = (uchar)(regions[i].color.x);
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = (uchar)(regions[i].color.y);
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = (uchar)(regions[i].color.z);
				}

			}
		}
	}
	for (size_t i = 0; i < flagRegs.size(); i++)
	{
		int ri = flagRegs[i];


		for (int k = 0; k < spPoses[ri].size(); k++)
		{
			int c = spPoses[ri][k].x;
			int r = spPoses[ri][k].y;
			if (i == 0)
			{
				((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = 0xff;
				((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = 0;
				((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = 0;
			}
			else
			{
				((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = 0xff;
				((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = 0;
				((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = 0;
			}



		}


	}
	delete[] pixSeg;
}
void GetRegionMap(int widht, int height, SuperpixelComputer* computer, int* segmented, std::vector<SPRegion>& regions, cv::Mat& mask, int flag, bool txtflag)
{
	int _spWidth = computer->GetSPWidth();
	int _spHeight = computer->GetSPHeight();
	int _spSize = _spWidth*_spHeight;
	mask.create(height, widht, CV_8UC3);
	std::vector<int> color(_spSize);
	CvRNG rng = cvRNG(cvGetTickCount());
	for (int i = 0; i < _spSize; i++)
		color[i] = cvRandInt(&rng);

	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);


	for (int i = 0; i < regions.size(); i++)
	{

		for (int j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[i].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[i].spIndices[j]][k].x;
				int r = spPoses[regions[i].spIndices[j]][k].y;
				if (flag == 0)
				{
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = (uchar)(regions[i].color.x);
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = (uchar)(regions[i].color.y);
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = (uchar)(regions[i].color.z);
				}
				else
				{
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = (uchar)(color[i]) & 255;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = (uchar)(color[i] >> 8) & 255;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = (uchar)(color[i] >> 16) & 255;
				}
			}
		}
	}
	if (txtflag)
	{
		char text[20];
		for (size_t i = 0; i < regions.size(); i++)
		{
			if (regions[i].size > 0)
			{
				sprintf(text, "%d", i);
				int x = regions[i].cX * 16;
				int y = regions[i].cY * 16;
				x = x >= widht ? widht - 1 : x;
				y = y >= height ? height - 1 : y;
				cv::putText(mask, text, cv::Point(x, y), CV_FONT_ITALIC, 1, CV_RGB(255, 215, 0));
			}
		}
	}


}
void GetRegionMap(int _width, int _height, SuperpixelComputer* computer, int* segmented, std::vector<float4>& regColors, cv::Mat& mask)
{
	int _spWidth = computer->GetSPWidth();
	int _spHeight = computer->GetSPHeight();
	int _spSize = _spWidth*_spHeight;
	mask.create(_height, _width, CV_8UC3);
	std::vector<int> color(_spSize);
	CvRNG rng = cvRNG(cvGetTickCount());
	for (int i = 0; i < _spSize; i++)
		color[i] = cvRandInt(&rng);

	int * labels;
	SLICClusterCenter* centers;
	int num(0);
	computer->GetSuperpixelResult(num, labels, centers);


	for (int i = 0; i < _height; i++)
	{
		for (int j = 0; j < _width; j++)
		{
			int cl = segmented[i*_width + j];
			//std::cout << i << "," << j << " " << cl << "\n";
			if (cl >= 0 & cl < regColors.size())
			{
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = (uchar)(color[cl]) & 255;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = (uchar)(color[cl] >> 8) & 255;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = (uchar)(color[cl] >> 16) & 255;

				/*((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = (uchar)(regColors[cl].x);
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = (uchar)(regColors[cl].y);
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = (uchar)(regColors[cl].z);*/
			}
			else
			{
				//std::cout << i << "," << j << " " << cl << "\n";
				int idx = 0;
				/*((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = (color[idx]) & 255;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = (color[idx] >> 8) & 255;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = (color[idx] >> 16) & 255;*/
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 0] = 255;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 1] = 0;
				((uchar *)(mask.data + i*mask.step.p[0]))[j*mask.step.p[1] + 2] = 0;
			}


		}
	}
}
struct RegionInfomation
{
	int id;
	int size;
	std::vector<int> neighbors;
	int cX;
	int cY;
};

void GetSaliencyMap(int width, int height,
	SuperpixelComputer* _SPComputer,
	int* segmented,
	std::vector<int>& regSizes,
	std::vector < std::vector<int>>& regIndices,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions,
	cv::Mat& mask)
{


	int * labels;
	SLICClusterCenter* centers;
	int num(0);
	_SPComputer->GetSuperpixelResult(num, labels, centers);
	int _spWidth = _SPComputer->GetSPWidth();
	int _spHeight = _SPComputer->GetSPHeight();

	//Region PIF
	//calculate the inhomogenity of regions
	int K = 3;
	std::vector<float> regPIF(regSizes.size(), 0);
	for (size_t i = 0; i < regSizes.size(); i++)
	{
		int center = std::accumulate(regIndices[i].begin(), regIndices[i].end(), 0);
		center = 1.0*center / regSizes[i];
		int x(0), y(0);
		int step = 1;
		for (int r = 0; r < regIndices[i].size(); r++)
		{
			int xr = regIndices[i][r] % _spWidth;
			int yr = regIndices[i][r] / _spWidth;
			x += xr;
			y += yr;
		}
		/*int x = center%_spWidth;
		int y = center / _spWidth;*/
		x = (x*1.0 / regSizes[i] + 0.5);
		y = (y*1.0 / regSizes[i] + 0.5);
		std::vector<int> ulabel;
		float c(0);
		for (int m = y - K*step; m <= y + K*step; m += step)
		{
			if (m<0 || m>_spHeight - 1)
				continue;
			for (int n = x - K*step; n <= x + K*step; n += step)
			{
				if (n<0 || n>_spWidth - 1)
					continue;
				c++;
				int idx = m*_spWidth + n;
				if (std::find(ulabel.begin(), ulabel.end(), newLabels[idx]) == ulabel.end())
					ulabel.push_back(newLabels[idx]);;
			}
		}
		regPIF[i] = ulabel.size() / c;

	}
	mask.create(height, width, CV_8U);
	std::vector<int>::iterator itr = max_element(regSizes.begin(), regSizes.end());
	int maxId = itr - regSizes.begin();
	int maxSize = *(max_element(regSizes.begin(), regSizes.end()));
	int avgSize = std::accumulate(regSizes.begin(), regSizes.end(), 0) / regSizes.size();
	float threshold = 50;
	for (int i = 0; i < _spHeight; i++)
	{
		for (int j = 0; j < _spWidth; j++)
		{
			int label = i*_spWidth + j;
			//int x = int(centers[label].xy.x + 0.5);
			//int y = int(centers[label].xy.y + 0.5);
			//int idx = x + y*_width;
			int regLabel = newLabels[label];
			float saliency = (1 - regSizes[regLabel] * 1.0 / maxSize);
			if (regLabel == maxId)
			{
				for (size_t i = 0; i < spPoses[label].size(); i++)
				{
					int index = spPoses[label][i].y* width + spPoses[label][i].x;
					mask.data[index] = (uchar)(regPIF[regLabel] * 255 * saliency);
					//mask.data[index] = (uchar)(255 * regPIF[regLabel]);
				}
			}
			else
			{
				for (size_t i = 0; i < spPoses[label].size(); i++)
				{
					int index = spPoses[label][i].y* width + spPoses[label][i].x;
					mask.data[index] = (uchar)(regPIF[regLabel] * 255 * saliency);

				}
			}


		}
	}
	//cv::threshold(mask, mask, 100, 255, CV_THRESH_BINARY);
}

//merge region i to region nRegId
void MergeRegion(int i, int nRegId,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions)
{

	int size0 = regions[nRegId].size;
	int size1 = regions[i].size;
	regions[nRegId].color = (regions[nRegId].color * size0 + regions[i].color * size1)*(1.0 / (size0 + size1));
	regions[nRegId].size = size0 + size1;
	for (size_t j = 0; j < regions[i].spIndices.size(); j++)
	{
		regions[nRegId].spIndices.push_back(regions[i].spIndices[j]);
		newLabels[regions[i].spIndices[j]] = regions[nRegId].id;
	}
	for (int b = 0; b < regions[nRegId].colorHist.size(); b++)
	{
		regions[nRegId].colorHist[b] += regions[i].colorHist[b];
	}
	for (int b = 0; b < regions[nRegId].hog.size(); b++)
	{
		regions[nRegId].hog[b] += regions[i].hog[b];
	}
	regions[i].size = 0;
}


//merge region i to region nRegId
void MergeRegion(int i, int nRegId,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions,
	std::vector<std::vector<int>>& regNeighbors)
{
	int regId = regions[i].id;

	for (size_t n = 0; n < regNeighbors[i].size(); n++)
	{
		SPRegion& reg = regions[regNeighbors[i][n]];
		int Idx = regNeighbors[i][n];
		//对i的所有邻居n，将在n的邻居中删除i
		std::vector<int>::iterator itr = std::find(regNeighbors[Idx].begin(),
			regNeighbors[Idx].end(), i);
		if (itr != regNeighbors[Idx].end())
			regNeighbors[Idx].erase(itr);
		//在n的邻居中加入regId
		if (reg.id != nRegId && std::find(regNeighbors[Idx].begin(), regNeighbors[Idx].end(), nRegId) == regNeighbors[Idx].end())
			regNeighbors[Idx].push_back(nRegId);
		//在合并后的区域nRegId的邻居中加入n
		if (reg.id != nRegId && std::find(regNeighbors[nRegId].begin(), regNeighbors[nRegId].end(), reg.id) == regNeighbors[nRegId].end())
			regNeighbors[nRegId].push_back(reg.id);
	}
	regNeighbors[i].clear();
	int size0 = regions[nRegId].size;
	int size1 = regions[regions[i].id].size;
	regions[nRegId].color = (regions[nRegId].color * size0 + regions[regId].color * size1)*(1.0 / (size0 + size1));
	regions[nRegId].size = size0 + size1;
	for (size_t j = 0; j < regions[regId].spIndices.size(); j++)
	{
		regions[nRegId].spIndices.push_back(regions[regId].spIndices[j]);
		newLabels[regions[regId].spIndices[j]] = nRegId;
	}
	for (int b = 0; b < regions[nRegId].colorHist.size(); b++)
	{
		regions[nRegId].colorHist[b] += regions[i].colorHist[b];
	}
	for (int b = 0; b < regions[nRegId].hog.size(); b++)
	{
		regions[nRegId].hog[b] += regions[i].hog[b];
	}
	regions[i].size = 0;
}
void MergeRegionHist(const SPRegion& reg1, SPRegion& reg2)
{
	int size0 = reg2.size;
	int size1 = reg1.size;
	int histSize = reg1.colorHist.size();
	/*#pragma omp parallel for*/
	for (int b = 0; b < histSize; b++)
	{
		reg2.colorHist[b] = reg2.colorHist[b] * size0 + size1*reg1.colorHist[b];
	}
	cv::normalize(reg2.colorHist, reg2.colorHist, 1, 0, cv::NORM_L1);

	//#pragma omp parallel for
	//	for (int b = 0; b < reg2.hog.size(); b++)
	//	{
	//		reg2.hog[b] = (reg2.hog[b] + reg1.hog[b]);
	//	}
	//	//cv::normalize(reg2.hog, reg2.hog, 1, 0, cv::NORM_L1);
	//
	//#pragma omp parallel for
	//	for (int b = 0; b < reg2.lbpHist.size(); b++)
	//	{
	//		reg2.lbpHist[b] = reg2.lbpHist[b] * size0 + size1*reg1.lbpHist[b];
	//	}
	//	cv::normalize(reg2.lbpHist, reg2.lbpHist, 1, 0, cv::NORM_L1);
}

void MergeRegions(int i, int j,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions)
{
	if (!isNeighbor(regions, i, j))
	{
		regions[j].regFlag = true;
		//计算每个区域的Fillness，合并后取平均
		std::vector<cv::Point>borders;
		GetRegionBorder(regions[i], borders);
		cv::vector<cv::Point> hull;
		cv::convexHull(cv::Mat(borders), hull, false);
		cv::vector<cv::vector<cv::Point>> convexContour;  // Convex hull contour points   
		convexContour.push_back(hull);
		float area = cv::contourArea(convexContour[0]);
		float fillnessI = regions[i].pixels / area;

		hull.clear();
		convexContour.clear();
		GetRegionBorder(regions[j], borders);
		cv::convexHull(cv::Mat(borders), hull, false);

		convexContour.push_back(hull);
		area = cv::contourArea(convexContour[0]);
		float fillnessJ = regions[j].pixels / area;
		//
		regions[j].filleness = (fillnessI + fillnessJ) / 2;

	}
	typedef std::vector<int>::iterator IntVecIter;
	/*std::cout << "-----------------------------\n";
	std::cout << "merge " << i << " to " << j << "\n";*/
	/*std::cout << "neighbors of " << i << "\n";
	for (size_t n = 0; n < regions[i].neighbors.size(); n++)
	{
	std::cout << regions[i].neighbors[n] << " ";
	}
	std::cout << "\n";*/
	for (size_t n = 0; n < regions[i].neighbors.size(); n++)
	{
		int Idx = regions[i].neighbors[n];
		//对i的所有邻居reg
		SPRegion& reg = regions[Idx];
		//std::cout << "neighbor " << Idx << "\n";

		if (reg.id == j)
		{
			//std::cout << "is the neighbor of " << i << std::endl;
			//若j和i是邻居,删除j中关于与i的边界的信息
			IntVecIter itr = std::find(reg.neighbors.begin(),
				reg.neighbors.end(), i);
			if (itr != reg.neighbors.end())
			{
				//std::cout << "remove borders between" << i << " and " << j << std::endl;
				int id = itr - reg.neighbors.begin();
				reg.neighbors.erase(reg.neighbors.begin() + id);
				reg.borderPixelNum.erase(reg.borderPixelNum.begin() + id);
				reg.borderPixels.erase(reg.borderPixels.begin() + id);
				reg.borders.erase(reg.borders.begin() + id);
				reg.borderSpIndices.erase(reg.borderSpIndices.begin() + id);
				reg.edgeness.erase(reg.edgeness.begin() + id);
			}
		}
		else if (!isNeighbor(regions, Idx, j))
		{
			//std::cout << "is not the neighbor of " << j << std::endl;
			//若reg不是j的邻居,更新邻居信息,将i改为j
			IntVecIter itr = std::find(reg.neighbors.begin(),
				reg.neighbors.end(), i);
			if (itr != reg.neighbors.end())
			{

				int id = itr - reg.neighbors.begin();
				reg.neighbors[id] = j;
				//std::cout << "add n to " << j << "'s neighbor\n";
				//在合并后的区域nRegId的邻居中加入n
				regions[j].neighbors.push_back(reg.id);
				regions[j].edgeness.push_back(reg.edgeness[id]);
				regions[j].borderPixelNum.push_back(reg.borderPixelNum[id]);
				regions[j].borders.push_back(reg.borders[id]);
				regions[j].borderPixels.push_back(reg.borderPixels[id]);
				regions[j].borderSpIndices.push_back(reg.borderSpIndices[id]);
			}

		}
		else
		{
			//std::cout << "is the neighbor of " << j <<" too "<< std::endl;
			//reg同时也是j的邻居
			IntVecIter itr = std::find(reg.neighbors.begin(), reg.neighbors.end(), i);
			int Idx_i = itr - reg.neighbors.begin();

			IntVecIter jItr = std::find(reg.neighbors.begin(), reg.neighbors.end(), j);
			int Idx_j = jItr - reg.neighbors.begin();

			IntVecIter nItr = std::find(regions[j].neighbors.begin(), regions[j].neighbors.end(), Idx);
			int Idx_n = nItr - regions[j].neighbors.begin();
			if (itr != reg.neighbors.end() && jItr != reg.neighbors.end())
			{

				regions[j].borderPixelNum[Idx_n] += reg.borderPixelNum[Idx_i];
				regions[j].edgeness[Idx_n] += reg.edgeness[Idx_i];
				regions[j].borders[Idx_n] += reg.borders[Idx_i];
				reg.edgeness[Idx_j] += reg.edgeness[Idx_i];
				reg.borderPixelNum[Idx_j] += reg.borderPixelNum[Idx_i];
				reg.borders[Idx_j] += reg.borders[Idx_i];
				for (size_t m = 0; m < reg.borderPixels[Idx_i].size(); m++)
				{
					reg.borderPixels[Idx_j].push_back(reg.borderPixels[Idx_i][m]);
					regions[j].borderPixels[Idx_n].push_back(reg.borderPixels[Idx_i][m]);
				}

				for (size_t m = 0; m < reg.borderSpIndices[Idx_i].size(); m++)
				{
					reg.borderSpIndices[Idx_j].push_back(reg.borderSpIndices[Idx_i][m]);
					regions[j].borderSpIndices[Idx_n].push_back(reg.borderSpIndices[Idx_i][m]);

				}



				reg.neighbors.erase(itr);
				reg.edgeness.erase(reg.edgeness.begin() + Idx_i);
				reg.borderPixelNum.erase(reg.borderPixelNum.begin() + Idx_i);
				reg.borderSpIndices.erase(reg.borderSpIndices.begin() + Idx_i);
				reg.borders.erase(reg.borders.begin() + Idx_i);
				reg.borderPixels.erase(reg.borderPixels.begin() + Idx_i);

			}


		}


	}



	regions[i].neighbors.clear();
	int size0 = regions[j].size;
	int size1 = regions[i].size;
	if (regions[i].pixels != 0)
		regions[j].color = (regions[j].color * size0 + regions[i].color * size1)*(1.0 / (size0 + size1));
	regions[j].cX = (regions[j].cX * size0 + regions[i].cX * size1)*(1.0 / (size0 + size1));
	regions[j].cY = (regions[j].cY * size0 + regions[i].cY * size1)*(1.0 / (size0 + size1));

	for (size_t s = 0; s < regions[i].spIndices.size(); s++)
	{
		regions[j].spIndices.push_back(regions[i].spIndices[s]);
		newLabels[regions[i].spIndices[s]] = j;
	}
	MergeRegionHist(regions[i], regions[j]);
	//for (int b = 0; b < regions[j].colorHist.size(); b++)
	//{
	//	regions[j].colorHist[b] = regions[j].colorHist[b] * size0 + size1*regions[i].colorHist[b];
	//}
	//cv::normalize(regions[j].colorHist, regions[j].colorHist, 1, 0, cv::NORM_L1);
	//for (int b = 0; b < regions[j].hog.size(); b++)
	//{
	//	regions[j].hog[b] = (regions[j].hog[b] + regions[i].hog[b]);
	//}
	////cv::normalize(regions[j].hog, regions[j].hog, 1, 0, cv::NORM_L1);

	//for (int b = 0; b < regions[j].lbpHist.size(); b++)
	//{
	//	regions[j].lbpHist[b] = regions[j].lbpHist[b] * size0 + size1*regions[i].lbpHist[b];
	//}
	//cv::normalize(regions[j].lbpHist, regions[j].lbpHist, 1, 0, cv::NORM_L1);
	regions[j].Bbox = MergeBox(regions[i].Bbox, regions[j].Bbox);
	regions[j].spBbox = MergeBox(regions[i].spBbox, regions[j].spBbox);
	regions[j].size = size0 + size1;

	regions[j].compactness = std::min(regions[j].Bbox.width, regions[j].Bbox.height) *1.0 / std::max(regions[j].Bbox.width, regions[j].Bbox.height);
	/*for (size_t b = 0; b < regions[i].borderEdgePixels.size(); b++)
	{
	regions[j].borderEdgePixels.push_back(regions[i].borderEdgePixels[b]);
	}*/
	regions[j].borderEdgePixels.resize(regions[j].edgePixNum + regions[i].edgePixNum);
	if (regions[i].edgePixNum > 0)
		memcpy(&regions[j].borderEdgePixels[regions[j].edgePixNum], &regions[i].borderEdgePixels[0], sizeof(cv::Point)*regions[i].edgePixNum);
	regions[j].edgePixNum += regions[i].edgePixNum;
	regions[j].edgeSpNum += regions[i].edgeSpNum;
	regions[j].pixels += regions[i].pixels;
	regions[j].ad2c.x = (regions[i].ad2c.x*size1 + regions[j].ad2c.x*size0) / (regions[j].size);
	regions[j].ad2c.y = (regions[i].ad2c.y*size1 + regions[j].ad2c.y*size0) / (regions[j].size);
	regions[j].rad2c.x = (regions[i].rad2c.x*size1 + regions[j].rad2c.x*size0) / (regions[j].size);
	regions[j].rad2c.y = (regions[i].rad2c.y*size1 + regions[j].rad2c.y*size0) / (regions[j].size);
	regions[j].regCircum = std::accumulate(regions[j].borderPixelNum.begin(), regions[j].borderPixelNum.end(), regions[j].edgePixNum);
	regions[i].size = 0;
	regions[i].spIndices.clear();


}

//merge region i to region nRegId
void MergeRegion(int i, int nRegId,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<int>& regSizes,
	std::vector<std::vector<int>>& regIndices,
	std::vector<float4>& regColors,
	std::vector<SPRegion>& regions,
	std::vector<std::vector<int>>& regNeighbors)
{
	int regId = regions[i].id;
	regions[i].size = 0;
	for (size_t n = 0; n < regNeighbors[i].size(); n++)
	{
		SPRegion& reg = regions[regNeighbors[i][n]];
		int idx = regNeighbors[i][n];
		//对i的所有邻居n，将在n的邻居中删除i
		std::vector<int>::iterator itr = std::find(regNeighbors[idx].begin(),
			regNeighbors[idx].end(), i);
		if (itr != regNeighbors[idx].end())
			regNeighbors[idx].erase(itr);
		//在n的邻居中加人regId
		if (reg.id != nRegId && std::find(regNeighbors[idx].begin(), regNeighbors[idx].end(), nRegId) == regNeighbors[idx].end())
			regNeighbors[idx].push_back(nRegId);
		//在合并后的区域nRegId的邻居中加入n
		if (reg.id != nRegId && std::find(regNeighbors[nRegId].begin(), regNeighbors[nRegId].end(), reg.id) == regNeighbors[nRegId].end())
			regNeighbors[nRegId].push_back(reg.id);
	}
	regNeighbors[i].clear();
	int size0 = regSizes[nRegId];
	int size1 = regSizes[regions[i].id];
	regColors[nRegId] = (regColors[nRegId] * size0 + regColors[regId] * size1)*(1.0 / (size0 + size1));
	regSizes[nRegId] += regSizes[regions[i].id];
	regSizes[regions[i].id] = 0;
	regions[nRegId].size = regSizes[nRegId];
	regions[nRegId].color = regColors[nRegId];

	for (size_t j = 0; j < regIndices[regId].size(); j++)
	{
		regIndices[nRegId].push_back(regIndices[regId][j]);
		newLabels[regIndices[regId][j]] = nRegId;
	}
	regIndices[regId].clear();
}

int HandleHoleDemo(int width, int height, int i, SuperpixelComputer* computer, std::vector<std::vector<uint2>>& spPoses, std::vector<int>& newLabels, std::vector<SPRegion>& regions)
{
	int ret = 0;
	int regId = regions[i].id;
	//处理空洞区域，将其合并到最接近的邻居中
	float minDist(1e10);
	int minId(-1);
	std::vector<int> INeighbors;
	for (size_t n = 0; n < regions[i].neighbors.size(); n++)
	{
		int nid = regions[i].neighbors[n];
		INeighbors.push_back(nid);
		float dist = L1Distance(regions[i].color, regions[nid].color);
		if (dist < minDist)
		{
			minDist = dist;
			minId = n;
		}
	}

	int j = regions[i].neighbors[minId];
	//std::cout << "merge regions " << i << " " << j << std::endl;
	std::vector<uint2> spParis;
	spParis.push_back(make_uint2(i, j));
	cv::Mat mask;
	GetRegionMap(width, height, computer, newLabels, regions, spParis, mask);
	char name[25];
	sprintf(name, "Mergeing%d_%d.jpg", i, j);
	cv::imwrite(name, mask);
	MergeRegions(i, j, newLabels, spPoses, regions);
	GetRegionMap(width, height, computer, newLabels, regions, mask);
	sprintf(name, "Merged%d_%d.jpg", i, j);
	cv::imwrite(name, mask);
	ret++;
	//检查合并后的邻居是不是邻居变少了
	int minN = 2;
	for (size_t n = 0; n < INeighbors.size(); n++)
	{

		int nSize = regions[INeighbors[n]].neighbors.size();
		std::cout << INeighbors[n] << " neighbors " << nSize << std::endl;
		if (regions[INeighbors[n]].size > 0 && regions[INeighbors[n]].size < HoleSize && nSize > 0 && nSize < minN)
		{
			std::cout << "	handle hole " << INeighbors[n] << std::endl;
			ret += HandleHoleDemo(width, height, INeighbors[n], computer, spPoses, newLabels, regions);
		}
	}
	return ret;
}
void HandleHoles(int idx, int width, int height, const char* outPath, SuperpixelComputer* computer, std::vector<SPRegion>& regions, std::vector<int>& newLabels, int holeNThreshold, int holeSizeThreshold, bool debug = false)
{
	float cw(0.5), shw(0.3), ew(0.2);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);
	cv::Mat mask;
	char name[200];

	int holeSize = idx > 10 ? 3 : 0;

	for (size_t i = 0; i < regions.size(); i++)
	{
		if ((regions[i].size > 0 && regions[i].size < holeSizeThreshold && regions[i].neighbors.size() <= holeNThreshold) ||
			(regions[i].size < holeSize && regions[i].size > 0))
		{
			//std::cout << "neighbors size " << regions[i].neighbors.size() << "\n";
			float minDist(1e10);
			int minId(-1);
			std::vector<int> INeighbors;
			for (size_t n = 0; n < regions[i].neighbors.size(); n++)
			{
				int nid = regions[i].neighbors[n];

				double colorDist = cv::compareHist(regions[i].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);

				//double dist = RegionDist(regions[i], regions[n]);
				float borderLen = regions[i].borderPixelNum[n];
				/*float borderLenI = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), 0);
				float borderLenN = std::accumulate(regions[n].borderPixelNum.begin(), regions[n].borderPixelNum.end(), 0);*/
				float borderLenI = regions[i].regCircum;
				float borderLenN = regions[nid].regCircum;
				double shapeDist = 1 - (borderLen) / (ZERO + std::min(borderLenI, borderLenN));

				double edgeness = regions[i].edgeness[n] / (regions[i].borderPixelNum[n] + ZERO);
				//double edgeness2 = regions[i].edgeness[n] / regions[i].borders[n];
				float dist = colorDist*cw + shapeDist*shw + edgeness*ew;
				//std::cout << colorDist << "," << shapeDist << "," << edgeness << "\n";
				if (dist < minDist)
				{
					minDist = dist;
					minId = n;
				}
			}
			//std::cout << "minId "<<minId << "\n";
			int j = regions[i].neighbors[minId];
			if (debug)
			{
				//std::cout << "merge regions " << i << " " << j << std::endl;
				std::vector<uint2> spParis;
				spParis.push_back(make_uint2(i, j));

				GetRegionMap(width, height, computer, newLabels, regions, spParis, mask);
				sprintf(name, "%s%dregMergeH%d_%d.jpg", outPath, idx, i, j);
				cv::imwrite(name, mask);
			}

			MergeRegions(i, j, newLabels, spPoses, regions);

			if (debug)
			{

				GetRegionMap(width, height, computer, newLabels, regions, mask, 0, false);
				sprintf(name, "%s%dregMergeHF%d_%d.jpg", outPath, idx, i, j);
				cv::imwrite(name, mask);

			}

		}

	}
}

void MFHoleHandling(int width, int height, const char* outDir, SuperpixelComputer* computer, std::vector<SPRegion>& regions, std::vector<int>& newLabels, float holeNThreshold, float holeSizeThreshold, bool debug = false)
{
	char outPath[200];
	sprintf(outPath, "%s\\HoleHandling\\", outDir);

	float cw(0.5), shw(0.3), ew(0.2);
	int idx(0);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);
	std::vector<SPRegion>::iterator itr;
	itr = std::find_if(regions.begin(), regions.end(), RegionSizeSmall(holeSizeThreshold));
	while (itr != regions.end())
	{
		SPRegion region = *itr;
		int i = region.id;
		double minDist(1);
		int minId(0);

		for (size_t n = 0; n < regions[i].neighbors.size(); n++)
		{
			int nid = regions[i].neighbors[n];
			//double colorDist = L1Distance(regions[i].color, regions[nid].color);
			//double colorDist = cv::compareHist(regions[i].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);
			float colorDist = RegionColorDist(regions[i], regions[nid]);
			//double dist = RegionDist(regions[i], regions[n]);
			float borderLen = regions[i].borderPixelNum[n];
			/*float borderLenI = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), 0);
			float borderLenN = std::accumulate(regions[n].borderPixelNum.begin(), regions[n].borderPixelNum.end(), 0);*/
			float borderLenI = regions[i].regCircum;
			float borderLenN = regions[nid].regCircum;
			double shapeDist = 1 - (borderLen) / (ZERO + std::min(borderLenI, borderLenN));

			double edgeness = regions[i].edgeness[n] / (regions[i].borderPixelNum[n] + ZERO);
			//double edgeness2 = regions[i].edgeness[n] / (regions[i].borders[n] + ZERO);
			float dist = colorDist*cw + shapeDist*shw + edgeness*ew;
			if (dist < minDist)
			{
				minDist = dist;
				minId = n;
			}
		}

		int j = regions[i].neighbors[minId];
		if (debug)
		{
			CreateDir(outPath);
			char name[300];
			cv::Mat mask;
			//std::cout << "merge regions " << i << " " << j << std::endl;
			std::vector<uint2> spParis;
			spParis.push_back(make_uint2(i, j));

			GetRegionMap(width, height, computer, newLabels, regions, spParis, mask);

			sprintf(name, "%s%dregMergeH%d_%d.jpg", outPath, idx, i, j);
			cv::imwrite(name, mask);
		}
		/*	double colorDist = L1Distance(regions[i].color, regions[j].color) / 255;
		double confidence = exp(-v);
		double wdist = confidence*minDist + (1 - confidence)*colorDist;*/
		MergeRegions(j, i, newLabels, spPoses, regions);

		if (debug)
		{
			char name[300];
			cv::Mat mask;
			GetRegionMap(width, height, computer, newLabels, regions, mask, 0, false);
			sprintf(name, "%s%dregMergeHF%d_%d.jpg", outPath, idx, i, j);
			cv::imwrite(name, mask);

		}
		itr = std::find_if(regions.begin(), regions.end(), RegionSizeSmall(holeSizeThreshold));
		idx++;
	}

}
void SmartHoleHandling(int width, int height, const char* outDir, SuperpixelComputer* computer, std::vector<SPRegion>& regions, std::vector<int>& newLabels, int holeNThreshold, int holeSizeThreshold, bool debug = false)
{
	char outPath[200];
	sprintf(outPath, "%s\\HoleHandling\\", outDir);

	float cw(0.5), shw(0.3), ew(0.2);
	int idx(0);
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);
	for (size_t i = 0; i < regions.size(); i++)
	{

		if (regions[i].size > 0 && regions[i].size < holeSizeThreshold)
		{
			idx++;

			std::list<int> stack;
			for (size_t k = 0; k < regions[i].neighbors.size(); k++)
			{
				stack.push_back(regions[i].neighbors[k]);
			}

			//merge all the hole regions that connected to i into i
			while (!stack.empty())
			{
				int id = *stack.begin();
				stack.pop_front();
				if (regions[id].size > 0 && regions[id].size < holeNThreshold)
				{
					for (size_t k = 0; k < regions[id].neighbors.size(); k++)
					{
						stack.push_back(regions[id].neighbors[k]);
					}
					//stack.push_back(nid);
					MergeRegions(id, i, newLabels, spPoses, regions);
				}
			}
			//if this region is still  a hole, merge it to the closet neighbor		
			if (regions[i].size>0 && regions[i].size < holeSizeThreshold)
			{
				float minDist(1e10);
				int minId(0);
				std::vector<int> INeighbors;
				for (size_t n = 0; n < regions[i].neighbors.size(); n++)
				{
					int nid = regions[i].neighbors[n];

					double dist = RegionColorDist(regions[i], regions[nid]);
					//cv::compareHist(regions[i].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);

					////double dist = RegionDist(regions[i], regions[n]);
					//float borderLen = regions[i].borderPixelNum[n];
					///*float borderLenI = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), 0);
					//float borderLenN = std::accumulate(regions[n].borderPixelNum.begin(), regions[n].borderPixelNum.end(), 0);*/
					//float borderLenI = regions[i].regCircum;
					//float borderLenN = regions[n].regCircum;
					//double shapeDist = 1 - (borderLen) / (ZERO + std::min(borderLenI, borderLenN));

					//double edgeness = regions[i].edgeness[n] / (regions[i].borderPixelNum[n] + ZERO);
					////double edgeness2 = regions[i].edgeness[n] / (regions[i].borders[n] + ZERO);
					//float dist = colorDist*cw + shapeDist*shw + edgeness*ew;
					if (dist < minDist)
					{
						minDist = dist;
						minId = n;
					}
				}

				int j = regions[i].neighbors[minId];
				if (debug)
				{
					CreateDir(outPath);
					char name[300];
					cv::Mat mask;
					//std::cout << "merge regions " << i << " " << j << std::endl;
					std::vector<uint2> spParis;
					spParis.push_back(make_uint2(i, j));

					GetRegionMap(width, height, computer, newLabels, regions, spParis, mask);

					sprintf(name, "%s%dregMergeH%d_%d.jpg", outPath, idx, i, j);
					cv::imwrite(name, mask);
				}
				/*	double colorDist = L1Distance(regions[i].color, regions[j].color) / 255;
				double confidence = exp(-v);
				double wdist = confidence*minDist + (1 - confidence)*colorDist;*/
				MergeRegions(j, i, newLabels, spPoses, regions);

				if (debug)
				{
					char name[300];
					cv::Mat mask;
					GetRegionMap(width, height, computer, newLabels, regions, mask, 0, false);
					sprintf(name, "%s%dregMergeHF%d_%d.jpg", outPath, idx, i, j);
					cv::imwrite(name, mask);

				}

			}
		}
	}
}


int HandleHole(int i, std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions)
{
	const int HoleSize = 10;
	int ret = 0;
	int regId = regions[i].id;
	//处理空洞区域或者小区域将其合并到最接近的邻居中
	float minDist(1e10);
	int minId(0);
	std::vector<int> INeighbors;
	int nid = regions[i].neighbors[0];
	INeighbors.push_back(nid);
	minDist = L1Distance(regions[i].color, regions[nid].color);

	for (size_t n = 1; n < regions[i].neighbors.size(); n++)
	{
		int nid = regions[i].neighbors[n];
		INeighbors.push_back(nid);
		float dist = L1Distance(regions[i].color, regions[nid].color);
		if (dist < minDist)
		{
			minDist = dist;
			minId = n;
		}
	}

	int j = regions[i].neighbors[minId];
	//std::cout << "merge regions " << i << " " << j << std::endl;
	MergeRegions(i, j, newLabels, spPoses, regions);
	ret++;
	//检查合并后的邻居是不是邻居变少了
	int minN = 2;
	for (size_t n = 0; n < INeighbors.size(); n++)
	{

		int nSize = regions[INeighbors[n]].neighbors.size();
		//std::cout << INeighbors[n] << " neighbors " << nSize << std::endl;
		if ((regions[INeighbors[n]].size > 0 && nSize > 0 && nSize <= minN && regions[INeighbors[n]].size < HoleSize))
		{
			//std::cout << "	handle hole " << INeighbors[n] << std::endl;
			ret += HandleHole(INeighbors[n], newLabels, spPoses, regions);
		}
	}
	return ret;
}
int HandleHole(int i, std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions,
	std::vector<std::vector<int>>& regNeighbors)
{
	int ret = 0;
	int regId = regions[i].id;
	//处理空洞区域，将其合并到最接近的邻居中
	float minDist(1e10);
	int minId(-1);
	std::vector<int> INeighbors;
	for (size_t n = 0; n < regNeighbors[i].size(); n++)
	{
		int nid = regNeighbors[i][n];
		INeighbors.push_back(nid);
		float dist = L1Distance(regions[i].color, regions[nid].color);
		if (dist < minDist)
		{
			minDist = dist;
			minId = n;
		}
	}
	int j = regNeighbors[i][minId];
	MergeRegion(i, j, newLabels, spPoses, regions, regNeighbors);
	ret++;
	//检查合并后的邻居是不是邻居变少了
	int minN = 4;
	for (size_t n = 0; n < INeighbors.size(); n++)
	{
		int nSize = regNeighbors[INeighbors[n]].size();
		if (nSize>0 && nSize < minN)
		{
			ret += HandleHole(INeighbors[n], newLabels, spPoses, regions, regNeighbors);
		}
	}
	return ret;
}

//handle hole
int HandleHole(int i, std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<int>& regSizes,
	std::vector<std::vector<int>>& regIndices,
	std::vector<float4>& regColors,
	std::vector < std::vector<int>>& regNeighbors,
	std::vector<SPRegion>& regions)
{
	int ret = 0;
	int regId = regions[i].id;
	//处理空洞区域，将其合并到最接近的邻居中
	float minDist(1e10);
	int minId(-1);
	std::vector<int> INeighbors;
	for (size_t n = 0; n < regNeighbors[i].size(); n++)
	{
		INeighbors.push_back(regNeighbors[i][n]);
		float dist = L1Distance(regions[i].color, regColors[regNeighbors[i][n]]);
		if (dist < minDist)
		{
			minDist = dist;
			minId = n;
		}
	}
	int j = regNeighbors[i][minId];
	MergeRegion(i, j, newLabels, spPoses, regSizes, regIndices, regColors, regions, regNeighbors);
	ret++;
	//检查合并后的邻居是不是邻居变少了
	int minN = 4;
	for (size_t n = 0; n < INeighbors.size(); n++)
	{
		int nSize = regNeighbors[INeighbors[n]].size();
		if (nSize>0 && nSize < minN)
		{
			ret += HandleHole(INeighbors[n], newLabels, spPoses, regSizes, regIndices, regColors, regNeighbors, regions);
		}
	}
	return ret;
}



void RegionAnalysis(int width, int height, SuperpixelComputer* computer, int* segmented,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions,
	std::vector<std::vector<int>>& regNeighbors)
{
	static const int dx4[] = { -1, 0, 1, 0 };
	static const int dy4[] = { 0, -1, 0, 1 };
	int spWidth, spHeight, spSize;
	spWidth = computer->GetSPWidth();
	spHeight = computer->GetSPHeight();
	int * labels;
	SLICClusterCenter* centers;
	computer->GetSuperpixelResult(spSize, labels, centers);
	//handel holes
	//std::cout << "handle holes\n";
	int RegionNum = regions.size();
	regNeighbors.resize(RegionNum);
	//#pragma omp parallel for
	for (int i = 0; i < RegionNum; i++)
	{
		if (regions[i].size > 0)
		{

			int x(0), y(0);
			int step = 1;
			for (int j = 0; j < regions[i].spIndices.size(); j++)
			{
				int xr = regions[i].spIndices[j] % spWidth;
				int yr = regions[i].spIndices[j] / spWidth;
				x += xr;
				y += yr;
				int spLabel = regions[i].spIndices[j];
				int spX = spLabel%spWidth;
				int spY = spLabel / spWidth;
				for (size_t n = 0; n < 4; n++)
				{
					int nx = spX + dx4[n];
					int ny = spY + dy4[n];
					if (nx < 0 || nx >= spWidth || ny < 0 || ny >= spHeight)
						continue;
					int nid = nx + ny*spWidth;
					/*	int npx = centers[nid].xy.x + 0.5;
					int npy = centers[nid].xy.y + 0.5;
					int label = segmented[npx + npy*width];*/
					int label = newLabels[nid];
					if (label != i &&
						std::find(regNeighbors[i].begin(), regNeighbors[i].end(), label) == regNeighbors[i].end())
						regNeighbors[i].push_back(label);
				}
			}
			/*int x = center%_spWidth;
			int y = center / _spWidth;*/
			x = (x*1.0 / regions[i].size + 0.5);
			y = (y*1.0 / regions[i].size + 0.5);
			regions[i].cX = x;
			regions[i].cY = y;

		}

	}
	std::vector<int> holeRegId;
	int minNeighbors(4);
	int minSize = 10;
	//对于所有邻居数小于minSize的区域，将其与其他小区域一起合并到最大的区域
	int count(0);
	//#pragma omp parallel for
	for (int i = 0; i < regions.size(); i++)
	{
		if (regions[i].size == 0 || regNeighbors[i].size() == 0)
			continue;
		//std::cout << i << "\n";
		if (regNeighbors[i].size() < minNeighbors)
		{
			int regId = regions[i].id;
			//处理空洞区域，将其合并到最接近的邻居中
			count += HandleHole(i, newLabels, spPoses, regions, regNeighbors);
		}
	}
	std::sort(regions.begin(), regions.end(), RegionSizeCmp());
	int nsize = std::find_if(regions.begin(), regions.end(), RegionSizeZero()) - regions.begin();
	regions.resize(nsize);
	//std::cout << "after hole filling " << nsize << "\n";

	//std::cout << "handle occolussions\n";
	//handel occolussion	
	//float cthres = 3;
	//
	///*while (nsize > 5)*/
	//{
	//	std::sort(regions.begin(), regions.end(), RegionColorCmp());
	//	
	//	std::vector<float> grad;
	//	for (size_t i = 0; i < regions.size() - 1; i++)
	//	{
	//		grad.push_back(L1Distance(regions[i].color, regions[i + 1].color));

	//	}
	//	//std::cout << "calc grad\n";
	//	std::vector<std::vector<int>> merge;
	//	int ii = 0;
	//	while (ii < grad.size())
	//	{
	//		std::vector<int> ids;
	//		while ( ii < grad.size() && grad[ii] < cthres)
	//		{
	//			ids.push_back(ii);
	//			ii++;
	//		}
	//		if (ids.size() > 0)
	//			merge.push_back(ids);
	//		ii++;
	//	}
	//	//std::cout << "begin merge\n";
	//	for (size_t i = 0; i < merge.size(); i++)
	//	{
	//		for (size_t j = 0; j < merge[i].size(); j++)
	//		{
	//			//std::cout << "merge " << merge[i][j] << " with " << merge[i][j] + 1 << "\n";
	//			MergeRegion(merge[i][j], merge[i][j] + 1, newLabels, spPoses, regions);
	//		}
	//	}

	//	std::sort(regions.begin(), regions.end(), RegionSizeCmp());
	//	nsize = std::find_if(regions.begin(), regions.end(), RegionSizeZero()) - regions.begin();
	//	regions.resize(nsize);
	//	cthres += 10;
	//}
	for (size_t i = 0; i < regions.size(); i++)
	{
		float d(0);
		for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		{
			int y = regions[i].spIndices[j] / spWidth;
			int x = regions[i].spIndices[j] % spWidth;
			d += abs(x - regions[i].cX) + abs(y - regions[i].cY);
		}
		regions[i].moment = d;
	}


#pragma omp parallel for
	for (int i = 0; i < regions.size(); i++)
	{
		cv::normalize(regions[i].colorHist, regions[i].colorHist, 1, 0, cv::NORM_L1);
		cv::normalize(regions[i].hog, regions[i].hog, 1, 0, cv::NORM_L1);

	}

	//重新排序和计算邻居
	for (size_t i = 0; i < regions.size(); i++)
	{
		regions[i].id = i;
		for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		{
			newLabels[regions[i].spIndices[j]] = i;
		}

	}

#pragma omp parallel for
	/*for (int i = 0; i < newLabels.size(); i++)
	{

	for (int j = 0; j < spPoses[i].size(); j++)
	segmented[spPoses[i][j].x + spPoses[i][j].y*width] = newLabels[i];
	}*/
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int idx = i*width + j;
			segmented[idx] = newLabels[labels[idx]];
		}
	}
	regNeighbors.resize(regions.size());
	for (size_t i = 0; i < regions.size(); i++)
	{
		regNeighbors[i].clear();
		for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		{
			int spLabel = regions[i].spIndices[j];
			int spX = spLabel%spWidth;
			int spY = spLabel / spWidth;
			for (size_t n = 0; n < 4; n++)
			{
				int nx = spX + dx4[n];
				int ny = spY + dy4[n];
				if (nx < 0 || nx >= spWidth || ny < 0 || ny >= spHeight)
					continue;
				int nid = nx + ny*spWidth;
				/*int npx = centers[nid].xy.x + 0.5;
				int npy = centers[nid].xy.y + 0.5;
				int label = segmented[npx + npy*width];*/
				int label = newLabels[nid];
				if (label != i &&
					std::find(regNeighbors[i].begin(), regNeighbors[i].end(), label) == regNeighbors[i].end())
					regNeighbors[i].push_back(label);
			}
		}
	}


}

void RegionAnalysis(int width, int height, SuperpixelComputer* computer, int* segmented,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<int>& regSizes,
	std::vector<std::vector<int>>& regIndices,
	std::vector<float4>& regColors,
	std::vector<SPRegion>& regions)
{
	static const int dx4[] = { -1, 0, 1, 0 };
	static const int dy4[] = { 0, -1, 0, 1 };
	int spWidth, spHeight, spSize;
	spWidth = computer->GetSPWidth();
	spHeight = computer->GetSPHeight();
	int * labels;
	SLICClusterCenter* centers;
	computer->GetSuperpixelResult(spSize, labels, centers);

	int RegionNum = regSizes.size();
	std::vector<std::vector<int>> RegNeighbors(RegionNum);
	//#pragma omp parallel for
	for (int i = 0; i < RegionNum; i++)
	{
		if (regSizes[i] > 0)
		{
			SPRegion reg;
			reg.color = regColors[i];
			reg.id = i;
			reg.size = regSizes[i];
			int x(0), y(0);
			int step = 1;
			for (int j = 0; j < regIndices[i].size(); j++)
			{
				int xr = regIndices[i][j] % spWidth;
				int yr = regIndices[i][j] / spWidth;
				x += xr;
				y += yr;
				int spLabel = regIndices[i][j];
				int spX = spLabel%spWidth;
				int spY = spLabel / spWidth;
				for (size_t n = 0; n < 4; n++)
				{
					int nx = spX + dx4[n];
					int ny = spY + dy4[n];
					if (nx < 0 || nx >= spWidth || ny < 0 || ny >= spHeight)
						continue;
					int nid = nx + ny*spWidth;
					int npx = centers[nid].xy.x + 0.5;
					int npy = centers[nid].xy.y + 0.5;
					int label = segmented[npx + npy*width];
					if (label != i &&
						std::find(RegNeighbors[i].begin(), RegNeighbors[i].end(), label) == RegNeighbors[i].end())
						RegNeighbors[i].push_back(label);
				}
			}
			/*int x = center%_spWidth;
			int y = center / _spWidth;*/
			x = (x*1.0 / regSizes[i] + 0.5);
			y = (y*1.0 / regSizes[i] + 0.5);
			reg.cX = x;
			reg.cY = y;
			regions.push_back(reg);
		}

	}
	std::vector<int> holeRegId;
	int minNeighbors(4);
	int minSize = 10;
	//对于所有邻居数小于minSize的区域，将其与其他小区域一起合并到最大的区域
	int count(0);
	//#pragma omp parallel for
	for (int i = 0; i < regions.size(); i++)
	{
		if (regions[i].size == 0 || RegNeighbors[i].size() == 0)
			continue;
		//std::cout << i << "\n";
		if (RegNeighbors[i].size() < minNeighbors)
		{
			int regId = regions[i].id;
			//处理空洞区域，将其合并到最接近的邻居中
			count += HandleHole(i, newLabels, spPoses, regSizes, regIndices, regColors, RegNeighbors, regions);


		}
	}

	std::sort(regions.begin(), regions.end(), RegionSizeCmp());
	int nsize = std::find_if(regions.begin(), regions.end(), RegionSizeZero()) - regions.begin();
	regions.resize(nsize);

	std::vector<int>nRegSizes;
	std::vector<float4> nRegColors;
	std::vector<std::vector<int>> nRegIndices;
	int k = 0;
	for (size_t i = 0; i < regions.size(); i++)
	{
		int id = regions[i].id;
		nRegSizes.push_back(regSizes[id]);
		nRegColors.push_back(regColors[id]);
		nRegIndices.push_back(regIndices[id]);
		for (size_t j = 0; j < regIndices[id].size(); j++)
		{
			newLabels[regIndices[id][j]] = k;
		}
		k++;

	}
	std::swap(nRegSizes, regSizes);
	std::swap(nRegColors, regColors);
	std::swap(nRegIndices, regIndices);


	for (int i = 0; i < newLabels.size(); i++)
	{
		//#pragma omp parallel for
		for (int j = 0; j < spPoses[i].size(); j++)
			segmented[spPoses[i][j].x + spPoses[i][j].y*width] = newLabels[i];
	}


	/*for (size_t i = 0; i < regions.size(); i++)
	{
	float d(0);
	for (size_t j = 0; j < regIndices[regions[i].id].size(); j++)
	{
	int y = regIndices[regions[i].id][j] / spWidth;
	int x = regIndices[regions[i].id][j] % spWidth;
	d += abs(x - regions[i].cX) + abs(y - regions[i].cY);
	}
	regions[i].moment = d;
	}*/

}


void GetSaliencyMap(int width, int height, SuperpixelComputer* computer,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions,
	cv::Mat& mask)
{
	int * labels;
	SLICClusterCenter* centers;
	int num(0);
	computer->GetSuperpixelResult(num, labels, centers);
	int _spWidth = computer->GetSPWidth();
	int _spHeight = computer->GetSPHeight();

	//Region PIF
	//calculate the inhomogenity of regions
	int K = 3;
	std::vector<float> regPIF(regions.size(), 0);
	for (size_t i = 0; i < regions.size(); i++)
	{
		int step = 1;
		int y(regions[i].cY);
		int x(regions[i].cX);
		std::vector<int> ulabel;
		float c(0);
		for (int m = y - K*step; m <= y + K*step; m += step)
		{
			if (m<0 || m>_spHeight - 1)
				continue;
			for (int n = x - K*step; n <= x + K*step; n += step)
			{
				if (n<0 || n>_spWidth - 1)
					continue;
				c++;
				int idx = m*_spWidth + n;
				if (std::find(ulabel.begin(), ulabel.end(), newLabels[idx]) == ulabel.end())
					ulabel.push_back(newLabels[idx]);;
			}
		}
		regPIF[i] = ulabel.size() / c;

	}
	float momentSum(0);
	for (int i = 0; i < regions.size(); i++)
	{

		momentSum += regions[i].moment;
	}
	mask.create(height, width, CV_8U);
	for (int i = 0; i < regions.size(); i++)
	{
		float sizeSal = (1 - 1.0*regions[i].size / regions[0].size);
		float momentSal = 1 - regions[i].moment / momentSum;
		for (int j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[i].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[i].spIndices[j]][k].x;
				int r = spPoses[regions[i].spIndices[j]][k].y;
				*(mask.data + r*width + c) = (unsigned char)(255.0*momentSal*sizeSal*regPIF[i]);
			}
		}

	}


	//cv::threshold(mask, mask, 100, 255, CV_THRESH_BINARY);
}

void GetContrastMap(int width, int height, SuperpixelComputer* computer, std::vector<int>& newLabels, std::vector<std::vector<uint2>>& spPoses, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, cv::Mat& mask)
{

	std::sort(regions.begin(), regions.end(), RegionWSizeCmp());

	/*int x1(255), y1(284), x2(466), y2(137);
	int reg1 = segmented[x1 + y1*_width];
	int reg2 = segmented[x2 + y2*_width];*/
	int spWidth = computer->GetSPWidth();
	int spHeight = computer->GetSPHeight();

	int bgReg(0);
	float bgSize(0);
	int totalSize = computer->GetSPHeight()*computer->GetSPWidth();
	for (size_t i = 0; i < regions.size(); i++)
	{

		bgSize += regions[i].size;
		bgReg++;
		if (bgSize > 0.65*totalSize)
			break;

	}

	//std::cout <<"bgreg = " <<bgReg << "\n";
	mask.create(height, width, CV_32F);
	adjacency_list_t adjacency_list(regions.size());
	for (size_t i = 0; i < regions.size(); i++)
	{
		for (size_t j = 0; j < regNeighbors[i].size(); j++)
		{
			int n = regNeighbors[i][j];
			float dist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
			adjacency_list[i].push_back(neighbor(n, 1));
		}

	}
	std::vector<std::vector<weight_t>> min_distances(bgReg);
	std::vector<std::vector<vertex_t>> previouses(bgReg);
	for (size_t i = 0; i < bgReg; i++)
	{
		DijkstraComputePaths(i, adjacency_list, min_distances[i], previouses[i]);
	}


	float M = 10;
	for (size_t i = 0; i < regions.size(); i++)
	{

		int minId(0);
		double minDist(regions.size());
		for (size_t k = 0; k < bgReg; k++)
		{
			if (min_distances[k][i] == std::numeric_limits<double>::infinity())
			{
				std::cout << "infinity\n";

			}
			if (min_distances[k][i] < minDist)
			{
				minDist = min_distances[k][i];
				minId = k;
			}
		}

		//if (i == 5)
		//{
		//	float dist = min_distances[minId][i];
		//	std::list<vertex_t> path = DijkstraGetShortestPathTo(i, previouses[minId]);
		//	std::cout << "Path : ";
		//	std::copy(path.begin(), path.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
		//	for (std::list<vertex_t>::iterator itr = path.begin(); itr != path.end(); itr++)
		//	{
		//		int r = *itr;
		//		for (size_t j = 0; j < regions[r].spIndices.size(); j++)
		//		{
		//			for (size_t s = 0; s < spPoses[regions[r].spIndices[j]].size(); s++)
		//			{
		//				uint2 xy = spPoses[regions[r].spIndices[j]][s];

		//				int idx = xy.x + xy.y*width;
		//				//mask.data[idx] = minDist * 255;
		//				mask.data[idx] = (regions[r].color.x + regions[r].color.y + regions[r].color.z) / 3;
		//			}
		//		}
		//	}
		//	for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		//	{
		//		for (size_t s = 0; s < spPoses[regions[i].spIndices[j]].size(); s++)
		//		{
		//			uint2 xy = spPoses[regions[i].spIndices[j]][s];

		//			int idx = xy.x + xy.y*width;
		//			//mask.data[idx] = minDist * 255;
		//			mask.data[idx] = 255;
		//		}
		//	}
		//	for (size_t j = 0; j < regions[minId].spIndices.size(); j++)
		//	{
		//		for (size_t s = 0; s < spPoses[regions[minId].spIndices[j]].size(); s++)
		//		{
		//			uint2 xy = spPoses[regions[minId].spIndices[j]][s];

		//			int idx = xy.x + xy.y*width;
		//			//mask.data[idx] = minDist * 255;
		//			mask.data[idx] = 128;
		//		}
		//	}
		//	std::cout << std::endl;
		//	return;
		//}
		//else
		//{
		//	continue;
		//}

		//寻找距离最近的背景区域（距离背景区域边缘）
		/*std::vector<float> Dy;
		std::vector<float> Dx;
		for (size_t k = 0; k < bgReg; k++)
		{
		float dx = (regions[k].cX - regions[i].cX) / M;
		float dy = (regions[k].cY - regions[i].cY) / M;

		Dx.push_back(dx);
		Dy.push_back(dy);
		}
		int step = 1;

		while (step < M)
		{
		for (size_t j = 0; j < bgReg; j++)
		{
		int  x = regions[i].cX + Dx[j] * step;
		int y = regions[i].cY + Dy[j] * step;
		if (x>0 && x < spWidth && y>0 && y < spHeight)
		{
		if (newLabels[x + y*spWidth] == regions[j].id)
		{
		minId = j;
		break;
		}

		}
		}
		step++;
		if (minId >= 0)
		break;
		}*/

		if (minDist == std::numeric_limits<double>::infinity())
		{
			minDist = 3;
		}
		//std::cout << "minId = " << minId << "\n";
		//float nBgColorDist = L1Distance(regions[i].color, regions[minId].color)/255;
		float nBgColorDist = cv::compareHist(regions[i].colorHist, regions[minId].colorHist, CV_COMP_BHATTACHARYYA);
		//float nBgHogDist = cv::compareHist(regions[i].hog, regions[minId].hog, CV_COMP_BHATTACHARYYA);
		//float nBgColorDist = maxDist;

		float sal = nBgColorDist*exp(-9.0 * (sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y)));
		for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (size_t s = 0; s < spPoses[regions[i].spIndices[j]].size(); s++)
			{
				uint2 xy = spPoses[regions[i].spIndices[j]][s];

				int idx = xy.x + xy.y*width;
				//mask.at<cv::Vec3b>(xy.y, xy.x) = color;
				mask.at<float>(xy.y, xy.x) = sal;
				//*(float*)(mask.data + idx * 4) = minDist;
				//mask.at<float>(xy.y, xy.x) = (regions[minId].color.x + regions[minId].color.y + regions[minId].color.z) / 3 / 255;
			}
		}
	}
	normalize(mask, mask, 0, 1, cv::NORM_MINMAX, CV_32F);
	/*double min, max;
	cv::minMaxLoc(mask, &min, &max);*/
	//cv::threshold(mask, mask, 1.5, 255, CV_THRESH_BINARY);
	mask.convertTo(mask, CV_8U, 255);

}

//@idxImg 每个像素量化后的颜色id
//@colorNum 量化后的颜色数
void BuildQHistorgram(const cv::Mat& idxImg, int colorNum, SuperpixelComputer* computer, HISTOGRAMS& colorHists)
{
	colorHists.clear();
	int _width = idxImg.cols;
	int _height = idxImg.rows;

	//计算每个超像素与周围超像素的差别
	int spHeight = computer->GetSPHeight();
	int spWidth = computer->GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int _spSize(0);
	computer->GetSuperpixelResult(_spSize, labels, centers);
	colorHists.resize(_spSize);

	//每个超像素中包含的像素以及位置
	std::vector<std::vector<uint2>> _spPoses;
	computer->GetSuperpixelPoses(_spPoses);

	for (int i = 0; i < _spSize; i++)
	{
		colorHists[i].resize(colorNum);
		memset(&colorHists[i][0], 0, sizeof(float)*colorNum);
		for (int j = 0; j < _spPoses[i].size(); j++)
		{
			int c = _spPoses[i][j].x;
			int r = _spPoses[i][j].y;
			int idx = c + r*_width;
			int* cidx = (int*)(idxImg.data + idx * 4);
			colorHists[i][*cidx]++;
		}
		cv::normalize(colorHists[i], colorHists[i], 1, 0, cv::NORM_L1);
	}
}
float RegionContrast(const cv::Mat&img, const cv::Mat& mask, int colorSpace)
{

	cv::Mat fimg, labImg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	cv::cvtColor(fimg, labImg, CV_BGR2Lab);

	int _colorBins(16);

	int _totalColorBins = _colorBins*_colorBins*_colorBins;

	HISTOGRAMS _colorHists;
	_colorHists.resize(2);

	float _colorSteps[3], _colorMins[3];
	if (colorSpace)
	{
		//lab color space
		float labMax[3] = { 100.f, 98.2352f, 94.4758f };
		float labMin[3] = { 0, -87, -108 };
		for (size_t i = 0; i < 3; i++)
		{
			_colorMins[i] = labMin[i];
			_colorSteps[i] = (labMax[i] - labMin[i]) / _colorBins;
		}
	}
	else
	{
		//rgb color space
		_colorSteps[0] = _colorSteps[1] = _colorSteps[2] = 1.0 / _colorBins;
		_colorMins[0] = _colorMins[1] = _colorMins[2] = 0;
	}
	for (int i = 0; i < 2; i++)
	{
		_colorHists[i].resize(_totalColorBins);
		memset(&_colorHists[i][0], 0, sizeof(float)*_totalColorBins);
	}
#pragma omp parallel for
	for (int m = 0; m < img.rows; m++)
	{
		cv::Vec3f* colorPtr;
		if (colorSpace)
		{
			colorPtr = labImg.ptr<cv::Vec3f>(m);
		}
		else
		{
			colorPtr = fimg.ptr<cv::Vec3f>(m);
		}
		for (int n = 0; n < img.cols; n++)
		{

			int  bin = 0;
			int s = 1;
			for (int c = 0; c < 3; c++)
			{
				bin += s*std::min<float>(floor((colorPtr[n][c] - _colorMins[c]) / _colorSteps[c]), _colorBins - 1);
				s *= _colorBins;
			}
			if (mask.data[m*img.cols + n] == 0xff)
				_colorHists[0][bin] ++;
			else
				_colorHists[1][bin] ++;
		}


	}

	for (int i = 0; i < 2; i++)
	{
		cv::normalize(_colorHists[i], _colorHists[i], 1, 0, cv::NORM_L1);
	}
	return cv::compareHist(_colorHists[0], _colorHists[1], CV_COMP_BHATTACHARYYA);
}
void BuildHistogram(const cv::Mat& img, SuperpixelComputer* computer, HISTOGRAMS& _colorHists, HISTOGRAMS& _HOGs, int colorSpace)
{
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

	//计算每个超像素与周围超像素的差别
	int spHeight = computer->GetSPHeight();
	int spWidth = computer->GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int _spSize(0);
	computer->GetSuperpixelResult(_spSize, labels, centers);
	_colorHists.resize(_spSize);
	_HOGs.resize(_spSize);
	//每个超像素中包含的像素以及位置
	std::vector<std::vector<cv::Vec3b>> pixels(_spSize);
	std::vector<std::vector<uint2>> _spPoses;
	computer->GetSuperpixelPoses(_spPoses);

	int _colorBins(16);
	int _hogBins(4);
	int _totalColorBins = _colorBins*_colorBins*_colorBins;
	int _hogStep = 360.0 / _hogBins;

	float _colorSteps[3], _colorMins[3];
	if (colorSpace)
	{
		//lab color space
		float labMax[3] = { 100.f, 98.2352f, 94.4758f };
		float labMin[3] = { 0, -87, -108 };
		for (size_t i = 0; i < 3; i++)
		{
			_colorMins[i] = labMin[i];
			_colorSteps[i] = (labMax[i] - labMin[i]) / _colorBins;
		}
	}
	else
	{
		//rgb color space
		_colorSteps[0] = _colorSteps[1] = _colorSteps[2] = 1.0 / _colorBins;
		_colorMins[0] = _colorMins[1] = _colorMins[2] = 0;
	}




	for (int i = 0; i < _spSize; i++)
	{
		_colorHists[i].resize(_totalColorBins);
		_HOGs[i].resize(_hogBins);
		for (size_t k = 0; k < _hogBins; k++)
		{
			_HOGs[i][k] = 0.1f;
		}
		memset(&_colorHists[i][0], 0, sizeof(float)*_totalColorBins);
	}

#pragma omp parallel for
	for (int i = 0; i < _spSize; i++)
	{
		for (int j = 0; j < _spPoses[i].size(); j++)
		{
			int n = _spPoses[i][j].x;
			int m = _spPoses[i][j].y;
			float* magPtr = _magImg.ptr<float>(m);
			float* angPtr = _angImg.ptr<float>(m);
			cv::Vec3f* colorPtr;
			if (colorSpace)
			{
				colorPtr = labImg.ptr<cv::Vec3f>(m);
			}
			else
			{
				colorPtr = fimg.ptr<cv::Vec3f>(m);
			}


			int bin = std::min<float>(floor(angPtr[n] / _hogStep), _hogBins - 1);
			_HOGs[i][bin] += magPtr[n];
			bin = 0;
			int s = 1;
			for (int c = 0; c < 3; c++)
			{
				bin += s*std::min<float>(floor((colorPtr[n][c] - _colorMins[c]) / _colorSteps[c]), _colorBins - 1);

				s *= _colorBins;
			}
			_colorHists[i][bin] ++;
		}
		cv::normalize(_colorHists[i], _colorHists[i], 1, 0, cv::NORM_L1);
		cv::normalize(_HOGs[i], _HOGs[i], 1, 0, cv::NORM_L1);
	}
}

void BuildHistogram(const cv::Mat& img, SuperpixelComputer* computer, HISTOGRAMS& _colorHists, HISTOGRAMS& _HOGs, HISTOGRAMS& lbpHists, int colorSpace)
{
	cv::Mat fimg, gray, labImg, lbpImg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	cv::cvtColor(fimg, labImg, CV_BGR2Lab);
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
	cv::Mat dx, dy, _angImg, _magImg;
	cv::Scharr(gray, dx, CV_32F, 1, 0);
	cv::Scharr(gray, dy, CV_32F, 0, 1);
	cv::cartToPolar(dx, dy, _magImg, _angImg, true);
	LBPGRAY(gray, lbpImg);



	int _width = img.cols;
	int _height = img.rows;

	//计算每个超像素与周围超像素的差别
	int spHeight = computer->GetSPHeight();
	int spWidth = computer->GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int _spSize(0);
	computer->GetSuperpixelResult(_spSize, labels, centers);
	_colorHists.resize(_spSize);
	_HOGs.resize(_spSize);
	lbpHists.resize(_spSize);

	//每个超像素中包含的像素以及位置
	std::vector<std::vector<cv::Vec3b>> pixels(_spSize);
	std::vector<std::vector<uint2>> _spPoses;
	computer->GetSuperpixelPoses(_spPoses);
	int lbpBinSize = 59;
	int _colorBins(16);
	int _hogBins(4);
	int _totalColorBins = _colorBins*_colorBins*_colorBins;
	int _hogStep = 360.0 / _hogBins;

	float _colorSteps[3], _colorMins[3];
	if (colorSpace)
	{
		//lab color space
		float labMax[3] = { 100.f, 127.f, 127.f };
		float labMin[3] = { 0, -127, -127 };
		for (size_t i = 0; i < 3; i++)
		{
			_colorMins[i] = labMin[i];
			_colorSteps[i] = (labMax[i] - labMin[i]) / _colorBins;
		}
	}
	else
	{
		//rgb color space
		_colorSteps[0] = _colorSteps[1] = _colorSteps[2] = 1.0 / _colorBins;
		_colorMins[0] = _colorMins[1] = _colorMins[2] = 0;
	}



#pragma omp parallel for
	for (int i = 0; i < _spSize; i++)
	{
		_colorHists[i].resize(_totalColorBins);
		_HOGs[i].resize(_hogBins);
		lbpHists[i].resize(lbpBinSize);
		for (size_t k = 0; k < _hogBins; k++)
		{
			_HOGs[i][k] = 0.1f;
		}
		memset(&_colorHists[i][0], 0, sizeof(float)*_totalColorBins);
		memset(&lbpHists[i][0], 0, sizeof(float)*lbpBinSize);
	}

#pragma omp parallel for
	for (int i = 0; i < _spSize; i++)
	{

		for (int j = 0; j < _spPoses[i].size(); j++)
		{
			int n = _spPoses[i][j].x;
			int m = _spPoses[i][j].y;
			float* magPtr = _magImg.ptr<float>(m);
			float* angPtr = _angImg.ptr<float>(m);
			cv::Vec3f* colorPtr;
			if (colorSpace)
			{
				colorPtr = labImg.ptr<cv::Vec3f>(m);
			}
			else
			{
				colorPtr = fimg.ptr<cv::Vec3f>(m);
			}


			int bin = std::min<float>(floor(angPtr[n] / _hogStep), _hogBins - 1);
			_HOGs[i][bin] += magPtr[n];
			bin = 0;
			int s = 1;
			for (int c = 0; c < 3; c++)
			{
				bin += s*std::min<float>(floor((colorPtr[n][c] - _colorMins[c]) / _colorSteps[c]), _colorBins - 1);

				s *= _colorBins;
			}
			_colorHists[i][bin] ++;

			int idx = n + m*_width;
			uchar ptr = *(uchar*)(lbpImg.data + idx);
			lbpHists[i][ptr]++;
		}
		cv::normalize(_colorHists[i], _colorHists[i], 1, 0, cv::NORM_L1);
		cv::normalize(_HOGs[i], _HOGs[i], 1, 0, cv::NORM_L1);
		cv::normalize(lbpHists[i], lbpHists[i], 1, 0, cv::NORM_L1);
	}
}

void PickSaliencyRegion(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& salMap)
{

	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	//GetRegionMap(width, height, computer, nLabels, regions, salMap);
	salMap.create(height, width, CV_32F);
	salMap = cv::Scalar(0);

	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{

			for (size_t j = 0; j < regions[i].spIndices.size(); j++)
			{
				int spIdx = regions[i].spIndices[j];
				for (size_t k = 0; k < spPoses[spIdx].size(); k++)
				{
					int c = spPoses[regions[i].spIndices[j]][k].x;
					int r = spPoses[regions[i].spIndices[j]][k].y;
					*((float *)(salMap.data + (r*width + c) * 4)) = regions[i].edgeSpNum;

				}
			}
		}

	}
	cv::normalize(salMap, salMap, 1, 0, cv::NORM_MINMAX);
	salMap.convertTo(salMap, CV_8U, 255.0);
}
float PickMostSaliencyRegions(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& mask, cv::Mat& dbgMap)
{
	dbgMap = cv::Mat::zeros(height, width, CV_8UC3);
	mask = cv::Mat::zeros(height, width, CV_8U);
	float salPixels(0);

	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);
	cv::vector<cv::Point> borders;
	struct RegPrior
	{
		RegPrior(int Id, float Prior) :id(Id), prior(Prior){}
		int id;
		float prior;
	};
	struct RegPriorComparer
	{
		bool operator()(RegPrior & a, RegPrior& b)
		{
			return a.prior < b.prior;
		}
	};
	float edgeSPNum = 2 * (computer->GetSPWidth() + computer->GetSPHeight());
	std::vector<RegPrior> regs;
	std::cout << "picking most saliency region...\n ";
	float regNum(0);
	//选择距离中心最近，面积最大，且不包含边缘的区域
	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{
			regNum++;
			float contrast(0);
			float neighborEdge(0);
			for (size_t j = 0; j < regions[i].neighbors.size(); j++)
			{
				neighborEdge += regions[regions[i].neighbors[j]].edgeSpNum;


			}
			if (neighborEdge < 1e-5)
			{
				contrast = 1;
			}
			else
			{
				for (size_t j = 0; j < regions[i].neighbors.size(); j++)
				{
					float w = regions[regions[i].neighbors[j]].edgeSpNum / (neighborEdge + 1e-5);
					contrast += w*cv::compareHist(regions[i].colorHist, regions[regions[i].neighbors[j]].colorHist, CV_COMP_BHATTACHARYYA);

				}
			}

			contrast = 1 - contrast;
			//contrast /= regions[i].neighbors.size();
			float ad2c = sqrt(sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y));
			float uad2c = 0.2;
			float urelSize = 0.2;
			float k2 = 1;
			//float relSize = 1-1 / sqrt(M_2_PI*k2) *exp(-sqr(regions[i].size *1.0/ computer->GetSuperpixelSize() - urelSize) / 2/k2);
			float relSize = regions[i].size*1.0 / computer->GetSuperpixelSize();
			relSize = relSize < 0.05 ? 1 : 0;
			float edgeNum = regions[i].edgeSpNum / edgeSPNum;

			float uedgeNum = 0.1;
			//float p1 = exp(-sqr(ad2c - uad2c) / 2);
			//float p2 = exp(-sqr(relSize - urelSize) / 2);
			//float p3 = exp(-sqr(edgeNum - uedgeNum) / 2);
			////float rad2c = sqrt(sqr(regions[i].rad2c.x) + sqr(regions[i].rad2c.y));
			float prior = (contrast + ad2c + edgeNum + relSize) / 4;
			//float prior = p1 + p2 + p3;
			//std::cout << "region " << i << ":\n relSize=" << relSize << ",contrast=" << contrast << ",ad2c=" << ad2c << ",edgeNum=" << edgeNum << "\n";
			//std::cout << "region " << i << " "<<prior << "\n";
			regs.push_back(RegPrior(i, prior));
		}
	}
	std::sort(regs.begin(), regs.end(), RegPriorComparer());
	//cv::Mat fmask = cv::Mat::zeros(height, width, CV_32F);
	//for (int i = 0; i < regions.size(); i++)
	//{
	//	

	//	for (int j = 0; j < regions[i].spIndices.size(); j++)
	//	{
	//		for (int k = 0; k < spPoses[regions[i].spIndices[j]].size(); k++)
	//		{
	//			int c = spPoses[regions[i].spIndices[j]][k].x;
	//			int r = spPoses[regions[i].spIndices[j]][k].y;
	//			*((float *)(fmask.data + (r*width+c)*4) )= regs[i].prior;
	//		}
	//	}

	//}
	////cv::normalize(fmask, fmask, 0, 1, CV_MINMAX);
	//fmask.convertTo(fmask, CV_8U, 255);
	////cv::threshold(fmask, fmask, 128, 255, CV_THRESH_BINARY);
	//
	//cv::imshow("fmask", fmask);
	//cv::waitKey(0);
	std::vector<int> salReg;
	int threshold = 2;
	/*if (regNum >= 15)
	threshold = 5;
	else if (regNum > 10)
	threshold = 4;
	else if (regNum > 4)
	threshold = 3;
	else if (regNum <= 3)
	threshold = 1;*/

	salReg.push_back(regs[0].id);
	float salAd2cX = regions[regs[0].id].ad2c.x;
	float salAd2cY = regions[regs[0].id].ad2c.y;

	if (regNum > 2)
	{
		if (std::find(regions[regs[0].id].neighbors.begin(), regions[regs[0].id].neighbors.end(), regs[1].id) != regions[regs[0].id].neighbors.end())
		{
			if (regs[0].prior / regs[1].prior > 0.8)
			{
				std::cout << "select two regions" << regs[0].prior << ">" << regs[1].prior << std::endl;
				salReg.push_back(regs[1].id);
			}

		}

	}

	//for (size_t i = 0; i < 1; i++)
	//{
	//	
	//	salReg.push_back(regs[i].id);
	//}

	float salEdgeSPNum(0);

	for (int i = 0; i < regions.size(); i++)
	{
		bool flag(false);
		if (std::find(salReg.begin(), salReg.end(), i) != salReg.end())
		{
			flag = true;
			for (size_t j = 0; j < regions[i].borderPixels.size(); j++)
			{
				for (size_t k = 0; k < regions[i].borderPixels[j].size(); k++)
				{
					borders.push_back(cv::Point(regions[i].borderPixels[j][k].x, regions[i].borderPixels[j][k].y));
				}
			}
			for (size_t j = 0; j < regions[i].borderEdgePixels.size(); j++)
			{
				borders.push_back(regions[i].borderEdgePixels[j]);
			}
			salEdgeSPNum += regions[i].edgeSpNum;
		}

		for (int j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[i].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[i].spIndices[j]][k].x;
				int r = spPoses[regions[i].spIndices[j]][k].y;
				if (!flag)
				{

					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = (uchar)(regions[i].color.x);
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = (uchar)(regions[i].color.y);
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = (uchar)(regions[i].color.z);
				}
				else
				{
					salPixels++;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1]] = 255;
					if (i == regs[0].id)
					{
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = 0;
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = 255;
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = 0;
					}
					else
					{
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = 0;
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = 0;
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = 255;
					}

				}
			}
		}

	}


	cv::vector<cv::Point> hull;
	cv::convexHull(cv::Mat(borders), hull, false);
	cv::vector<cv::vector<cv::Point>> convexContour;  // Convex hull contour points   
	convexContour.push_back(hull);
	//approxPolyDP(hull, convexContour, 0.001, true);
	//for (int i = 0; i < convexContour.size(); i++)
	//{
	//	drawContours(mask, convexContour, i, cv::Scalar(0, 0, 255), 1, 8);
	//	//drawContours(contoursRst, convexContour, i, cv::Scalar(255, 255, 255), CV_FILLED);
	//}
	float area = cv::contourArea(convexContour[0]);
	float fill = salPixels / area;
	float edgeR = salEdgeSPNum / edgeSPNum;
	float relSize = salPixels / width / height;
	float fillMinThreshold = 0.2;
	float edgeMaxThreshold = 0.16;
	float sizeMinThreshold = 0.05;
	float sizeMaxThreshold = 0.75;

	std::cout << "F " << fill << " E " << edgeR << " S " << relSize << "\n";
	if (fill > fillMinThreshold && edgeR < edgeMaxThreshold && relSize > sizeMinThreshold && relSize < sizeMaxThreshold)
		return fill + 1 - edgeR + relSize;
	else
		return 0;


}

void PickSaliencyRegion(int width, int height, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, cv::Mat& salMap, float ratio)
{
	int zeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int regSize = regions.size() - zeroReg;
	//从所有区域中选择ratio个显著性区域（按照区域包含边界像素的多少进行排序）
	std::vector<SPRegion> tmpRegions(regions);
	std::sort(tmpRegions.begin(), tmpRegions.end(), RegionWSizeDescCmp());
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	//GetRegionMap(width, height, computer, nLabels, regions, salMap);
	salMap.create(height, width, CV_8U);
	salMap = cv::Scalar(0);
	int n = (int)(regSize*ratio + 0.5);
	int salReg(0);
	for (size_t i = 0; i < tmpRegions.size() && salReg < n; i++)
	{
		if (tmpRegions[i].size > 0)
		{
			salReg++;
			for (size_t j = 0; j < tmpRegions[i].spIndices.size(); j++)
			{
				int spIdx = tmpRegions[i].spIndices[j];
				for (size_t k = 0; k < spPoses[spIdx].size(); k++)
				{
					int c = spPoses[tmpRegions[i].spIndices[j]][k].x;
					int r = spPoses[tmpRegions[i].spIndices[j]][k].y;
					((uchar *)(salMap.data + r*salMap.step.p[0]))[c*salMap.step.p[1] + 0] = 255;

				}
			}
		}

	}
}

void HandleOcculusion(const cv::Mat& img, SuperpixelComputer& computer, const char* outPath, std::vector<int>& newLabels, std::vector<RegionSalInfo>& regInfos, std::vector<SPRegion>& regions, int * segment, bool debug = false)
{
	float maxColorDist(0), minColorDist(255);
	float maxAd2cDist(0);
	float avgColorDist(0);
	int avgColorDistSize(0);
	float avgAd2cDis(0);
	float minDist(0);
	float threshold(0.4);
	int idx(0);
	std::vector<RegDist> RegDists;
	for (size_t i = 0; i < regInfos.size() - 1; i++)
	{
		int id = regInfos[i].id;
		for (size_t j = i + 1; j < regInfos.size(); j++)
		{
			int nid = regInfos[j].id;

			//float colorDist = cv::compareHist(regions[id].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);
			float colorDist = RegionColorDist(regions[id], regions[nid]);
			//rd.edgeness = regions[id].borders[j] * 1.0 / std::min(regions[id].bor);
			avgColorDist += colorDist;
			avgColorDistSize++;
			if (colorDist > maxColorDist)
				maxColorDist = colorDist;
			if (colorDist < minColorDist)
				minColorDist = colorDist;
			if ((regions[id].edgeSpNum < 1 && regions[nid].edgeSpNum < 1) ||
				(regions[id].edgeSpNum > 1 && regions[nid].edgeSpNum > 1) ||
				isNeighbor(regions, id, nid))
			{
				RegDist rd;
				float ad2cI = sqrt(sqr(regions[id].ad2c.x) + sqr(regions[id].ad2c.y));
				float ad2cJ = sqrt(sqr(regions[nid].ad2c.x) + sqr(regions[nid].ad2c.y));
				float ad2cDis = abs(ad2cI - ad2cJ);
				rd.id = i;
				rd.sRid = id;
				rd.bRid = nid;
				rd.colorDist = colorDist;
				rd.edgeness = ad2cDis;
				RegDists.push_back(rd);
			}
			/*	;
			rd.edgeness = ad2cDis;
			avgAd2cDis += ad2cDis;*/



		}
	}
	avgColorDist /= avgColorDistSize;
	avgAd2cDis /= avgColorDistSize;
	float wc = 1;
	float wa = 0.5;
	threshold = avgColorDist*0.8;
	if (debug)
	{
		std::cout << "avgColorDist = " << avgColorDist << "\n";
		std::cout << "threshold=" << threshold << "\n";

	}
	/*for (int i = 0; i < RegDists.size(); i++)
	{
	RegDists[i].colorDist = (RegDists[i].colorDist - minColorDist) / (maxColorDist - minColorDist);

	}*/

	std::vector<std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);

	char name[200];
	cv::Mat rmask;
	int regSize = regInfos.size();


	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(wc, wa, 0, 0));
	minDist = 0;
	char outPutPath[300];
	if (debug)
	{

		sprintf(outPutPath, "%s\\OccHandling\\", outPath);
		CreateDir(outPutPath);
	}
	//处理遮挡
	while (regSize > 2 && minDist < threshold)
	{
		minDist = wc*RegDists[0].colorDist + wa*RegDists[0].edgeness;
		if (minDist > threshold)
		{
			if (debug)
			{
				std::cout << minDist << " > " << threshold << "occ handel stop\n";
			}


			break;
		}



		if (debug)
		{
			std::vector<uint2> pair;
			pair.push_back(make_uint2(RegDists[0].sRid, RegDists[0].bRid));
			GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, pair, rmask);
			sprintf(name, "%s%dHOccMerge0_%d_%d_Region_%2d_%d.jpg", outPutPath, idx, RegDists[0].sRid, RegDists[0].bRid, (int)(minDist * 100), regSize);

			cv::imwrite(name, rmask);
		}

		MergeRegions(RegDists[0].sRid, RegDists[0].bRid, newLabels, spPoses, regions);

		if (debug)
		{
			GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);

			sprintf(name, "%s%dHOccMerge1_%d_%d_Region_%2d_%d.jpg", outPutPath, idx, RegDists[0].sRid, RegDists[0].bRid, (int)(minDist * 100), regSize - 1);
			cv::imwrite(name, rmask);

		}

		regInfos.erase(RegDists[0].id + regInfos.begin());
		RegDists.clear();
		for (size_t i = 0; i < regInfos.size() - 1; i++)
		{
			int id = regInfos[i].id;
			for (size_t j = i + 1; j < regInfos.size(); j++)
			{
				int nid = regInfos[j].id;
				if ((regions[id].edgeSpNum < 1 && regions[nid].edgeSpNum < 1) ||
					(regions[id].edgeSpNum > 1 && regions[nid].edgeSpNum > 1))
				{
					RegDist rd;
					rd.id = i;
					rd.sRid = id;
					rd.bRid = nid;
					//rd.colorDist = cv::compareHist(regions[id].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);
					rd.colorDist = RegionColorDist(regions[id], regions[nid]);
					float ad2cI = sqrt(sqr(regions[id].ad2c.x) + sqr(regions[id].ad2c.y));
					float ad2cJ = sqrt(sqr(regions[nid].ad2c.x) + sqr(regions[nid].ad2c.y));
					float ad2cDis = abs(ad2cI - ad2cJ);
					rd.edgeness = ad2cDis;
					RegDists.push_back(rd);
				}
			}
		}
		if (RegDists.size() == 0)
		{
			std::cout << "regdists == 0 occhandle stop\n";
			break;
		}

		std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(wc, wa, 0, 0));
		idx++;
		regSize--;
		//UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, regions, segment);
		//RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, debug);
	}

	if (debug)
	{
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, 1);
		sprintf(name, "%sOccHandled_%d.jpg", outPath, regSize);
		cv::imwrite(name, rmask);
	}

}
float RegionBorderLocalContrast(SuperpixelComputer& computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int trid, HISTOGRAMS& colorHist)
{
	std::vector<int> borderSPs;

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
	return dist;
}
float RegionBoxLocalContrast(SuperpixelComputer& computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int rid, HISTOGRAMS& colorHist, float theta)
{
	float spWidth = computer.GetSPWidth();
	float spHeight = computer.GetSPHeight();
	cv::Rect spBbox = regions[rid].spBbox;
	cv::Rect _spBbox = spBbox;
	float hw = spBbox.width*1.0 / spBbox.height;
	float w = sqrt(regions[rid].size / theta*hw);
	cv::Rect oRect = regions[rid].Bbox;
	if (w > spBbox.width)
	{


		_spBbox.x = spBbox.x - (w - spBbox.width) / 2 + 0.5;
		if (_spBbox.x < 0)
			_spBbox.x = 0;
		if (_spBbox.x > spWidth)
			_spBbox.x = spWidth - 1;

		_spBbox.y = spBbox.y - (w - spBbox.width) / 2 / hw + 0.5;
		if (_spBbox.y < 0)
			_spBbox.y = 0;
		if (_spBbox.y > spHeight)
			_spBbox.y = spHeight - 1;

		if (_spBbox.x + w > spWidth)
			_spBbox.width = spWidth - _spBbox.x + 0.5;
		else
			_spBbox.width = w + 0.5;

		int  _spHeight = _spBbox.width / hw + 0.5;
		if (_spBbox.y + _spHeight > spHeight)
			_spBbox.height = spHeight - _spBbox.y + 0.5;
		else
			_spBbox.height = _spHeight;

		int step = computer.GetSuperpixelStep();
		float minX = (_spBbox.x - 0.5)*step;
		float maxX = (_spBbox.x + _spBbox.width + 1)*step;
		float maxY = (_spBbox.y + _spBbox.height + 1)*step;
		float minY = (_spBbox.y - 0.5)*step;
		minX = std::max(0.f, minX);
		minY = std::max(0.f, minY);


		float width = maxX - minX;
		float height = maxY - minY;

		maxX = std::min(width*1.f, maxX);
		maxY = std::min(height*1.f, maxY);
		oRect = cv::Rect(minX, minY, width, height);
	}

	spBbox = _spBbox;
	std::vector<float> boxBorderHist(colorHist[0].size(), 0);
	for (size_t m = spBbox.x; m < spBbox.x + spBbox.width; m++)
	{
		for (size_t n = spBbox.y; n < spBbox.y + spBbox.height; n++)
		{
			int id = n*spWidth + m;
			if (id < computer.GetSuperpixelSize() && nLabels[id] != regions[rid].id)
			{
				for (size_t k = 0; k < colorHist[id].size(); k++)
				{
					boxBorderHist[k] += colorHist[id][k];
				}
			}
		}

	}
	cv::normalize(boxBorderHist, boxBorderHist, 1, 0, cv::NORM_L1);
	double boxdist = cv::compareHist(boxBorderHist, regions[rid].colorHist, CV_COMP_BHATTACHARYYA);
	return boxdist;
}
void GetSaliencyProposalMap(const char* outPath, const cv::Mat& img, std::vector<SPRegion>& regions, SuperpixelComputer& computer, HISTOGRAMS& colorHist, std::vector<int>& nLabels, std::vector<PropScore>& propScores, std::vector<cv::Mat>& proposals)
{
	int _width = img.cols;
	int _height = img.rows;
	int spHeight = computer.GetSPHeight();
	int spWidth = computer.GetSPWidth();
	int trid = 1;


	float boxdist = RegionBoxLocalContrast(computer, nLabels, regions, trid, colorHist);
	//std::cout <<"dist"<< dist << "\n";
	double fillness = regions[1].pixels / regions[1].Bbox.area();
	double size = regions[1].pixels / _width / _height;
	double compactness = std::min(regions[1].Bbox.width, regions[1].Bbox.height)*1.0 / std::max(regions[1].Bbox.width, regions[1].Bbox.height);
	float weightF = exp(-sqr(fillness - meanFillness) / thetaFill);
	//float weightF = fillness > 0.15 ? 1 : 0;
	float weightS = exp(-sqr(size - meanRelSize) / thetaSize);
	float weightO = boxdist > 0.9 ? 0.9 : boxdist;
	float weight = weightF * weightS + weightO;
	PropScore ps;
	ps.id = propScores.size();
	ps.score = weight;
	propScores.push_back(ps);

	/*fillnessVec.push_back(fillness);
	compactnessVec.push_back(compactness);
	objectVec.push_back(boxdist);*/

	char imgName[300];
	char outDir[200];
	sprintf(outDir, "%s\\saliency\\", outPath);
	CreateDir(outDir);
	//CalRegionFocusness(gradMap, scaleMap, edgeMap, _spPoses, regions, focus);
	sprintf(imgName, "%s%dSaliency_%d_%d_%d.png", outDir, propScores.size(), (int)(weightF * 100), (int)(weightS * 100), (int)(boxdist * 100));
	cv::imwrite(imgName, img);

	/*cv::Mat rmask;
	ShowRegionBorder(img, computer, nLabels, regions, 1,rmask);*/


}




void Saliency(const char* outPath, const char* imgName, std::vector<PropScore>& propScores, std::vector<SPRegion>& candidateRegs, std::vector<std::vector<int>>& proposalIds, std::vector<cv::Mat>& proposals, cv::Mat& saliencyMap)
{

	std::sort(propScores.begin(), propScores.end(), PropScoreCmp());
	if (proposals.size() == 1)
	{
		saliencyMap = proposals[0];
	}
	else
	{
		float weightSum = 0;
		for (size_t i = 0; i < propScores.size() / 2; i++)
		{
			weightSum += propScores[i].score;
		}
		for (size_t i = 0; i < propScores.size() / 2; i++)
		{
			cv::addWeighted(proposals[propScores[i].id], propScores[i].score / weightSum, saliencyMap, 1, 0, saliencyMap, CV_32F);
		}

	}

	cv::GaussianBlur(saliencyMap, saliencyMap, cv::Size(3, 3), 0);
	cv::normalize(saliencyMap, saliencyMap, 0, 255, CV_MINMAX, CV_8U);
	char fileName[200];
	sprintf(fileName, "%s\\%s_RMN.png", outPath, imgName);
	cv::imwrite(fileName, saliencyMap);
	cv::Mat salMax = proposals[propScores[0].id];
	cv::normalize(salMax, salMax, 0, 255, CV_MINMAX);
	sprintf(fileName, "%s\\%s_MAX.png", outPath, imgName);
	cv::imwrite(fileName, salMax);
	cv::threshold(saliencyMap, saliencyMap, 128, 255, CV_THRESH_BINARY);

}
void Saliency(const char* outPath, const char* imgName, std::vector<PropScore>& propScores, std::vector<cv::Mat>& proposals, cv::Mat& saliencyMap)
{
	saliencyMap = cv::Mat::zeros(proposals[0].size(), CV_32F);
	std::sort(propScores.begin(), propScores.end(), PropScoreCmp());
	if (proposals.size() == 1)
	{
		saliencyMap = proposals[0];
	}
	else
	{
		float weightSum = 0;
		for (size_t i = 0; i < propScores.size() / 2; i++)
		{
			weightSum += propScores[i].score;
		}
		for (size_t i = 0; i < propScores.size() / 2; i++)
		{
			cv::addWeighted(proposals[propScores[i].id], propScores[i].score / weightSum, saliencyMap, 1, 0, saliencyMap, CV_32F);
		}
		cv::normalize(saliencyMap, saliencyMap, 0, 255, CV_MINMAX, CV_8U);
	}

	char fileName[200];
	sprintf(fileName, "%s\\saliency\\%s_RM.png", outPath, imgName);
	cv::imwrite(fileName, saliencyMap);
	cv::threshold(saliencyMap, saliencyMap, 128, 255, CV_THRESH_BINARY);



}

void GCOptimization(const cv::Mat& img, const cv::Mat& salImg, cv::Mat& optRst)
{
	SuperpixelComputer* spComputer = new SuperpixelComputer(img.cols, img.rows, 5);
	spComputer->ComputeSuperpixel(img);
	SLICClusterCenter* centers;
	int spNum, *labels;
	spComputer->GetSuperpixelResult(spNum, labels, centers);
	std::vector<std::vector<cv::Point> > spPoints;
	spComputer->GetSuperpixelPoints(spPoints);
	typedef Graph<float, float, float> GraphType;
	GraphType *g;
	g = new GraphType(/*estimated # of nodes*/ spNum, /*estimated # of edges*/ spNum * 2);
	for (size_t i = 0; i < spNum; i++)
	{
		g->add_node();
		float num(0);
		for (size_t j = 0; j < spPoints[i].size(); j++)
		{

			num += salImg.at<float>(spPoints[i][j]);
		}

	}
}

void GetRegionBorder(SPRegion& reg, std::vector<cv::Point>& borders)
{
	int innerSize = std::accumulate(reg.borderPixelNum.begin(), reg.borderPixelNum.end(), 0);

	borders.resize(reg.borderEdgePixels.size() + innerSize);
	int cPtr(0);
	for (size_t j = 0; j < reg.borderPixels.size(); j++)
	{
		int size = sizeof(cv::Point)*reg.borderPixelNum[j];
		if (size > 0)
		{
			memcpy(&borders[cPtr], &reg.borderPixels[j][0], size);
			cPtr += reg.borderPixelNum[j];
		}


	}
	if (reg.borderEdgePixels.size() > 0)
		memcpy(&borders[cPtr], &reg.borderEdgePixels[0], sizeof(cv::Point)*reg.borderEdgePixels.size());
}
void ShowRegionBorder(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, int regId, cv::Mat rmask)
{
	int width = img.cols;
	int height = img.rows;
	GetRegionMap(width, height, &computer, newLabels, regions, rmask, 0, false);
	std::vector<cv::Point> borders;
	GetRegionBorder(regions[regId], borders);
	cv::Mat rsmask = rmask.clone();
	for (size_t i = 0; i < borders.size(); i++)
	{
		cv::circle(rsmask, borders[i], 1, cv::Scalar(255, 0, 0));

	}
	char text[20];
	sprintf(text, "%d", regId);
	int x = regions[regId].cX * 16;
	int y = regions[regId].cY * 16;
	x = x >= width ? width - 1 : x;
	y = y >= height ? height - 1 : y;
	cv::putText(rsmask, text, cv::Point(x, y), CV_FONT_ITALIC, 1, CV_RGB(255, 215, 0));
	cv::rectangle(rsmask, regions[regId].Bbox, cv::Scalar(0, 0, 255));
	cv::imshow("RegionBorder", rsmask);
	cv::waitKey(0);

}
void ShowRegionBorder(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, cv::Mat rmask)
{
	int width = img.cols;
	int height = img.rows;
	GetRegionMap(width, height, &computer, newLabels, regions, rmask, 0, false);

	for (size_t r = 0; r < regInfos.size(); r++)
	{
		int rid = regInfos[r].id;
		std::vector<cv::Point> borders;
		GetRegionBorder(regions[rid], borders);
		cv::Mat rsmask = rmask.clone();
		for (size_t i = 0; i < borders.size(); i++)
		{
			cv::circle(rsmask, borders[i], 1, cv::Scalar(255, 0, 0));

		}



		cv::Rect cbox = cv::boundingRect(borders);


		char text[20];
		sprintf(text, "%d", regInfos[r].id);
		int x = regions[regInfos[r].id].cX * 16;
		int y = regions[regInfos[r].id].cY * 16;
		x = x >= width ? width - 1 : x;
		y = y >= height ? height - 1 : y;
		cv::putText(rsmask, text, cv::Point(x, y), CV_FONT_ITALIC, 1, CV_RGB(255, 215, 0));
		cv::rectangle(rsmask, regions[regInfos[r].id].Bbox, cv::Scalar(0, 0, 255));

		cv::rectangle(rsmask, cbox, cv::Scalar(0, 255, 255));
		cv::imshow("test", rsmask);
		cv::waitKey(0);
	}
}

//combine region i to region j
//将小区域(像素数不足)合并到其它区域
void CombineRegion(int i, int j, std::vector<SPRegion>& regions, std::vector<int>& newLabels)
{
	SPRegion region;

	int size0 = regions[j].size;
	int size1 = regions[i].size;
	if (regions[i].pixels != 0)
		regions[j].color = (regions[j].color * size0 + regions[i].color * size1)*(1.0 / (size0 + size1));
	regions[j].cX = (regions[j].cX * size0 + regions[i].cX * size1)*(1.0 / (size0 + size1));
	regions[j].cY = (regions[j].cY * size0 + regions[i].cY * size1)*(1.0 / (size0 + size1));
	for (size_t s = 0; s < regions[i].spIndices.size(); s++)
	{
		regions[j].spIndices.push_back(regions[i].spIndices[s]);
		newLabels[regions[i].spIndices[s]] = j;
	}
	for (int b = 0; b < regions[j].colorHist.size(); b++)
	{
		regions[j].colorHist[b] = regions[j].colorHist[b] * size0 + size1*regions[i].colorHist[b];
	}
	cv::normalize(regions[j].colorHist, regions[j].colorHist, 1, 0, cv::NORM_L1);
	for (int b = 0; b < regions[j].hog.size(); b++)
	{
		regions[j].hog[b] = (regions[j].hog[b] + regions[i].hog[b]);
	}
	//cv::normalize(regions[j].hog, regions[j].hog, 1, 0, cv::NORM_L1);

	for (int b = 0; b < regions[j].lbpHist.size(); b++)
	{
		regions[j].lbpHist[b] = regions[j].lbpHist[b] * size0 + size1*regions[i].lbpHist[b];
	}
	if (regions[j].lbpHist.size() > 0)
		cv::normalize(regions[j].lbpHist, regions[j].lbpHist, 1, 0, cv::NORM_L1);

	regions[j].pixels = regions[i].pixels + regions[j].pixels;

	for (size_t n = 0; n < regions[i].neighbors.size(); n++)
	{

		int nid = regions[i].neighbors[n];
		if (regions[nid].size == 0)
			continue;
		std::vector<int>::iterator itr = std::find(regions[nid].neighbors.begin(), regions[nid].neighbors.end(), i);
		if (itr != regions[nid].neighbors.end())
		{
			if (nid != j)
			{
				if (std::find(regions[nid].neighbors.begin(), regions[nid].neighbors.end(), j) == regions[nid].neighbors.end())
					regions[nid].neighbors[itr - regions[nid].neighbors.begin()] = j;
				else
					regions[nid].neighbors.erase(itr);
				if (std::find(regions[j].neighbors.begin(), regions[j].neighbors.end(), nid) == regions[j].neighbors.end())
					regions[j].neighbors.push_back(nid);
			}
			else
			{
				regions[nid].neighbors.erase(itr);
			}
		}



	}
	regions[i].size = 0;
	regions[i].spIndices.clear();
	regions[j].size = size0 + size1;
}

void SaliencyGuidedRegionGrowing(const char* workingPath, const char* imgFolder, const char* outputPath, const char* imgName, const cv::Mat& img, const cv::Mat& edgeMap, SuperpixelComputer& computer, cv::Mat& salMap, int regThreshold, bool debug)
{
	char outPath[300];
	char fileName[300];
	sprintf(outPath, "%s\\%s\\", outputPath, imgName);
	//if (debug)
	{
		CreateDir(outPath);

	}
	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);

	int width = img.cols, height = img.rows;
	int spWidth = computer.GetSPWidth(), spHeight = computer.GetSPHeight();
	int spSize(spWidth*spHeight);
	//build historgram
	HISTOGRAMS colorHist, gradHist, lbpHist, hColorHist;
	//BuildHistogram(img, &computer, colorHist, gradHist);

	nih::Timer timer;
#if (TEST_SPEED)
	{

		timer.start();

	}
#endif

	//BuildHistogram(img, &computer, colorHist, gradHist, lbpHist, 0);
	cv::Mat idx1i, _color3f, _colorNum;
	double ratio = 0.95;
	const int clrNums[3] = { 12, 12, 12 };
	cv::Mat fimg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	int num = Quantize(fimg, idx1i, _color3f, _colorNum, ratio, clrNums);
	BuildQHistorgram(idx1i, num, &computer, colorHist);
	gradHist.resize(spSize);
	cv::cvtColor(_color3f, _color3f, CV_BGR2Lab);
	cv::Mat_<float> cDistCache1f = cv::Mat::zeros(_color3f.cols, _color3f.cols, CV_32F); {
		cv::Vec3f* pColor = (cv::Vec3f*)_color3f.data;
		for (int i = 0; i < cDistCache1f.rows; i++)
			for (int j = i + 1; j < cDistCache1f.cols; j++)
			{

				float distL = (pColor[i][0] - pColor[j][0]);
				float distA = (pColor[i][1] - pColor[j][1]);
				float distB = (pColor[i][2] - pColor[j][2]);
				float dist = sqrt(sqr(distL) + sqr(distA) + sqr(distB));
				cDistCache1f[i][j] = cDistCache1f[j][i] = dist;
			}
	}
	double minDist, maxDist;
	cv::minMaxLoc(cDistCache1f, &minDist, &maxDist);
	gColorDist = cDistCache1f;
	gMaxDist = maxDist;

	PrepareQCSimMatrix(num);

#if (TEST_SPEED)
	{
		timer.stop();
		std::cout << "build histogram " << timer.seconds() * 1000 << "ms\n";

	}
#endif
	/*float avgQDist(0), avgDist(0); float avgQHR(0);
	int count(0);
	for (int i = 0; i < spSize; i++)
	{
	std::vector<int> neighbors = computer.GetNeighbors4(i);
	for (int j = 0; j < neighbors.size(); j++)
	{
	count++;
	float qdist = RegionColorDist(colorHist[i], colorHist[j])/maxDist;
	float dist = cv::compareHist(hColorHist[i], hColorHist[j],CV_COMP_BHATTACHARYYA);
	avgQDist += qdist;
	avgDist += dist;
	avgQHR += qdist/(dist+1e-6);
	}
	}
	avgQDist /= count;
	avgDist /= count;
	avgQHR /= count;
	std::cout << "histogram dist: " << avgQDist << "," << avgDist << "," << avgQHR << "\n";*/
#if(TEST_SPEED)
	{
		timer.start();
	}
#endif
	int * labels(NULL);
	SLICClusterCenter* centers(NULL);
	computer.GetSuperpixelResult(spSize, labels, centers);
	std::vector<std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);
	std::vector<int> newLabels(spSize);
	std::vector<SPRegion> regions;


	//init regions 
	regions.resize(spSize);
#pragma omp parallel for	
	for (int i = 0; i < spSize; i++)
	{
		newLabels[i] = i;

		regions[i].cX = i%spWidth;
		regions[i].cY = i / spWidth;
		regions[i].color = centers[i].rgb;
		regions[i].colorHist = colorHist[i];
		regions[i].hog = gradHist[i];
		//regions[i].lbpHist = lbpHist[i];
		regions[i].size = 1;
		regions[i].dist = 0;
		regions[i].id = i;
		regions[i].neighbors = computer.GetNeighbors4(i);
		regions[i].spIndices.push_back(i);
		regions[i].pixels = spPoses[i].size();
	}

	int* segment = new int[img.cols*img.rows];


	int minPixels = sqr(computer.GetSuperpixelStep()) / 2;
	//处理边缘
	//top
	for (size_t i = 0; i < spWidth; i++)
	{
		if (regions[i].size > 0 && regions[i].pixels < minPixels)
		{
			CombineRegion(i, i + spWidth, regions, newLabels);
		}
	}
	//bottom
	for (size_t i = spWidth*(spHeight - 1); i <= spWidth*spHeight - 1; i++)
	{
		if (regions[i].size > 0 && regions[i].pixels < minPixels)
		{
			CombineRegion(i, i - spWidth, regions, newLabels);
		}
	}
	//left
	for (size_t i = 0; i <= spWidth*(spHeight - 1); i += spWidth)
	{
		if (regions[i].size > 0 && regions[i].pixels < minPixels)
		{
			CombineRegion(i, i + 1, regions, newLabels);
		}
	}
	//right
	for (size_t i = spWidth - 1; i < spHeight*spWidth - 1; i += spWidth)
	{
		if (regions[i].size > 0 && regions[i].pixels < minPixels)
		{
			CombineRegion(i, i - 1, regions, newLabels);
		}
	}
#if (TEST_SPEED)
	{
		timer.stop();
		std::cout << "init regions  " << timer.seconds() * 1000 << "ms\n";

	}
#endif


#if(TEST_SPEED)
	{
		timer.start();
	}
#endif

	UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, edgeMap, regions);


#if (TEST_SPEED)
	{
		timer.stop();
		std::cout << "Init Region info " << timer.seconds() * 1000 << "ms\n";

	}
#endif


	int ZeroReg, RegSize(regions.size());
	cv::Mat rmask;

	ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	RegSize = regions.size() - ZeroReg;
	if (debug)
	{
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, 0, false);

		sprintf(fileName, "%sInitRegion_%d.jpg", outPath, RegSize);
		cv::imwrite(fileName, rmask);
	}

#if(TEST_SPEED)
	{
		timer.start();
	}
#endif
	/*Queue RegNPairs;
	std::vector<float> regAges;
	PrepareForRegionGrowing(spSize, regions, RegNPairs, regAges);*/
	int iter(1);
	int validSize = RegSize;
	float holeSize(5);
	while (RegSize > regThreshold)
	{

		int needToMerge = RegSize * 0.2;
		RegionGrowing(iter, img, outPath, edgeMap, computer, newLabels, regions, needToMerge, debug);
		//FastRegionGrowing(iter, img, outPath, computer, RegNPairs, regAges, newLabels, regions, needToMerge, debug);

		ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		RegSize = regions.size() - ZeroReg;
		holeSize = iter / 6.0;
		int smallReg = std::count_if(regions.begin(), regions.end(), RegionSizeSmall(holeSize));
		validSize = RegSize - smallReg;
		if (validSize < regThreshold)
			MFHoleHandling(width, height, outPath, &computer, regions, newLabels, HoleNeighborsNum, holeSize, debug);

		ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		RegSize = regions.size() - ZeroReg;
		iter++;
	}

#if (TEST_SPEED)
	{
		timer.stop();
		std::cout << "1st region growing " << timer.seconds() * 1000 << "ms\n";

	}
#endif


	if (debug)
	{
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, 0, false);
		ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		RegSize = regions.size() - ZeroReg;
		sprintf(fileName, "%sMerge1_%d.jpg", outPath, RegSize);
		cv::imwrite(fileName, rmask);
		/*cv::Mat vmap = cv::Mat::zeros(img.size(), CV_32F);
		for (size_t i = 0; i < regions.size(); i++)
		{
		double v = exp(-HistogramVariance(regions[i].colorHist));
		for (int j = 0; j < regions[i].spIndices.size(); j++)
		{
		for (int k = 0; k < spPoses[regions[i].spIndices[j]].size(); k++)
		{
		int c = spPoses[regions[i].spIndices[j]][k].x;
		int r = spPoses[regions[i].spIndices[j]][k].y;
		*(float*)(vmap.data + (r*img.cols + c) * 4) = v;
		}
		}
		}
		cv::normalize(vmap, vmap, 255,0, cv::NORM_MINMAX,CV_8U);
		sprintf(fileName, "%sMerge1V_%d.jpg", outPath, RegSize);
		cv::imwrite(fileName, vmap);*/
	}
	SmartHoleHandling(width, height, outPath, &computer, regions, newLabels, HoleNeighborsNum, HoleSize, debug);
	//MFHoleHandling(width, height, outPath, &computer, regions, newLabels, HoleNeighborsNum, holeSize, debug);

	ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	RegSize = regions.size() - ZeroReg;
	if (debug)
	{
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, 0, false);
		ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		RegSize = regions.size() - ZeroReg;
		sprintf(fileName, "%sMerge2_%d.jpg", outPath, RegSize);
		cv::imwrite(fileName, rmask);

	}

	if (debug)
	{
		cv::Mat focus;
		CalRegionFocusness(gray, edgeMap, spPoses, regions, focus);
		cv::Mat focusMap;
		focus.convertTo(focusMap, CV_8U, 255);
		sprintf(fileName, "%sFocus.jpg", outPath);
		cv::imwrite(fileName, focusMap);
	}



#if(TEST_SPEED)
	{
		timer.start();
	}
#endif
	std::vector<RegionSalInfo> regInfos;

	//RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, debug);
	RegionSaliency(img.cols, img.rows, colorHist, outPath, &computer, newLabels, regions, regInfos, debug);
	if (regInfos.size() > 2)
		HandleOcculusion(img, computer, outPath, newLabels, regInfos, regions, segment, debug);

	//处理图像有边框的情况,这时边框的占据了全部border造成背景的borderratio为0
	//将边框合并到最大的区域的邻居
	/*if (regInfos.size() ==3)
	{
	for (int i = 0; i < regInfos.size(); i++)
	{
	int id = regInfos[i].id;
	if (regions[id].edgeSpNum / ((spWidth + spHeight) * 2 - 4) > 0.99)
	{
	float maxSize(0);
	int mId(0);

	for (int n = 0; n < regions[id].neighbors.size(); n++)
	{
	int nid = regions[id].neighbors[n];
	if (regions[nid].size > maxSize)
	{
	maxSize = regions[nid].size;
	mId = nid;
	}
	}
	MergeRegions(id, mId, newLabels, spPoses, regions);
	}
	}
	}*/



	/*std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());

	int bkgRegId = regInfos[regInfos.size() - 1].id;
	for (size_t i = 0; i < regInfos.size()-1; i++)
	{
	int id = regInfos[i].id;
	regInfos[i].neighbors.clear();
	int nonBkgNeighbor(0);
	for (size_t j = 0; j < regions[id].neighbors.size(); j++)
	{
	int regId = regions[id].neighbors[j];
	if (regId == bkgRegId)
	break;
	else
	nonBkgNeighbor++;


	}
	if (nonBkgNeighbor >0 && nonBkgNeighbor<3)
	{

	for (int n = 0; n < regInfos.size(); n++)
	{
	std::vector<int> neighbors = regions[regInfos[n].id].neighbors;
	if (std::find(neighbors.begin(), neighbors.end(), id) != neighbors.end())
	{

	MergeRegions(id, regInfos[n].id, newLabels, spPoses, regions);
	regInfos.erase(regInfos.begin() + i);
	}
	}
	}
	}
	if (debug)
	{
	char name[300];
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, 1);
	sprintf(name, "%sHoleHandled_%d.jpg", outPath, regInfos.size());
	cv::imwrite(name, rmask);
	}*/
#if (TEST_SPEED)
	{
		timer.stop();
		std::cout << "HandleOcculusion " << timer.seconds() * 1000 << "ms\n";

	}
#endif

	//ShowRegionBorder(img, computer, newLabels, regions, regInfos, rmask);

	if (debug)
	{
		/*
		std::cout << "After Occulusion handling" << regInfos.size() << " regions remains\n";
		std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());
		for (size_t i = 0; i < regInfos.size(); i++)
		std::cout << regInfos[i] << "\n";
		*/

		/*for (size_t i = 0; i < regInfos.size(); i++)
		{
		int ri = regInfos[i].id;

		double variance = HistogramVariance(regions[ri].colorHist);
		std::cout << "------------------------------\n";
		std::cout<<ri<< " hist v = " << variance<<"\n";
		for (size_t j = 0; j < regInfos.size(); j++)
		{
		if (i == j)
		continue;
		int ji = regInfos[j].id;
		double colorDist = L1Distance(regions[ri].color, regions[ji].color)/255;
		double histDist = cv::compareHist(regions[ri].colorHist, regions[ji].colorHist, CV_COMP_BHATTACHARYYA);
		std::cout << ri << "," << ji << " color dist "<<colorDist << ", hist dist " << histDist << "\n";
		double textureDist = cv::compareHist(regions[ri].lbpHist, regions[ji].lbpHist, CV_COMP_BHATTACHARYYA);
		double hogDist = cv::compareHist(regions[ri].hog, regions[ji].hog, CV_COMP_BHATTACHARYYA);
		std::cout << " txture dist " << colorDist << ", hog dist " << hogDist << "\n";
		}
		}*/
	}

	std::vector<PropScore> propScores;

	float totalWeights(0);
	float borderRatio = regInfos[regInfos.size() - 1].borderRatio;
	std::vector<int> bgRegIds;
	float rmThreshold = 0.75;



	char spath[200];
	if (debug)
	{


		sprintf(spath, "%s//saliency//", outPath);
		CreateDir(spath);

	}

#if(TEST_SPEED)
	{
		timer.start();
	}
#endif

	std::vector<SPRegion> candiRegions;
	std::vector<RegionSalInfo> candiRegSalInfo(regInfos);
	for (size_t i = 0; i < regInfos.size(); i++)
	{
		candiRegions.push_back(regions[regInfos[i].id]);
		candiRegions[i].regContrast = regInfos[i].contrast;
	}
	if (regInfos.size() == 2)
	{
		salMap = cv::Mat::zeros(img.size(), CV_8U);
		for (int j = 0; j < candiRegions[0].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[candiRegions[0].spIndices[j]].size(); k++)
			{
				int c = spPoses[candiRegions[0].spIndices[j]][k].x;
				int r = spPoses[candiRegions[0].spIndices[j]][k].y;

				*(uchar*)(salMap.data + (r*width + c)) = 255;

			}
		}
		sprintf(fileName, "%s\\%s_RMF.png", outputPath, imgName);
		cv::imwrite(fileName, salMap);
		sprintf(fileName, "%s\\%s_RMN.png", outputPath, imgName);
		cv::imwrite(fileName, salMap);
		cv::threshold(salMap, salMap, 127, 255, CV_THRESH_BINARY);
		return;
	}
	//RegionSaliency(img.cols, img.rows, colorHist, outPath, &computer, newLabels, regions, regInfos, debug);
	//int bkgId = regInfos[regInfos.size() - 1].id;
	//cv::Mat bkgSal = cv::Mat::zeros(height, width, CV_32F);
	//for (size_t i = 0; i < regions[bkgId].spIndices.size(); i++)
	//{
	//	for (int k = 0; k < spPoses[regions[bkgId].spIndices[i]].size(); k++)
	//	{
	//		int c = spPoses[regions[bkgId].spIndices[i]][k].x;
	//		int r = spPoses[regions[bkgId].spIndices[i]][k].y;
	//		//*(float*)(bkgSal.data + 4 * (r*width + c)) = cv::compareHist(colorHists[regions[bkgId].spIndices[i]], bkgHist, CV_COMP_BHATTACHARYYA);
	//		*(float*)(bkgSal.data + 4 * (r*width + c)) = L1Distance(regions[bkgId].color, centers[regions[bkgId].spIndices[i]].rgb);
	//	}
	//}
	//cv::normalize(bkgSal, bkgSal, 0, 0.49, CV_MINMAX, CV_32F);
	//cv::threshold(bkgSal, bkgSal, 0.28, 0.49, CV_THRESH_TOZERO);
	//
	//bkgSal.convertTo(bkgSal, CV_8U, 255);
	//cv::imshow("bkg sal", bkgSal);
	/*cv::waitKey();*/

	std::vector<std::vector<int>> proposalIds;
	std::vector<std::vector<int>> cproposalIds;
	std::vector<std::vector<float>> regScores;
	std::vector<cv::Mat> proposals;
	char outDir[200];
	sprintf(outDir, "%s\\saliency\\", outPath);
	CreateDir(outDir);

	sprintf(fileName, "%sbox.txt", outDir);
	std::ofstream outFile(fileName);

	while (1)
	{

		if (RegionSaliency(img.cols, img.rows, colorHist, outPath, &computer, newLabels, regions, regInfos, debug))
		{

			std::vector<int> ids;
			std::vector<int> cids;
			std::vector<float> scores;
			std::vector<float> fgCHist(colorHist[0].size(), 0);
			std::vector<float> boxBorderHist(colorHist[0].size(), 0);
			HISTOGRAM bkgHist = regions[regInfos.size() - 1].colorHist;
			float4 fgAvgColor = make_float4(0, 0, 0, 0);
			float4 boxBorderAvgColor = make_float4(0, 0, 0, 0);
			float fgSpNum(0);
			float boxBorderSpNum(0);
			float pixels(0);
			cv::Rect box = regions[regInfos[0].id].Bbox;
			cv::Rect spBbox = regions[regInfos[0].id].spBbox;
			for (size_t i = 0; i < regInfos.size() - 1; i++)
			{
				int id = regInfos[i].id;
				ids.push_back(id);
				float weight = regInfos[i].contrast;
				scores.push_back(weight);
				int cid = std::find_if(candiRegions.begin(), candiRegions.end(), RegionIdLocate(id)) - candiRegions.begin();
				cids.push_back(cid);
				for (size_t j = 0; j < colorHist[0].size(); j++)
				{
					fgCHist[j] += regions[id].colorHist[j];

				}
				fgAvgColor = fgAvgColor + regions[id].color*regions[id].size;
				fgSpNum += regions[id].size;
				pixels += regions[regInfos[i].id].pixels;
				box = MergeBox(box, regions[regInfos[i].id].Bbox);
				spBbox = MergeBox(spBbox, regions[regInfos[i].id].spBbox);
			}

			for (size_t m = spBbox.x; m < spBbox.x + spBbox.width; m++)
			{
				for (size_t n = spBbox.y; n < spBbox.y + spBbox.height; n++)
				{
					int id = n*spWidth + m;
					if (id < computer.GetSuperpixelSize() && newLabels[id] == regInfos[regInfos.size() - 1].id)
					{
						boxBorderSpNum++;
						boxBorderAvgColor = boxBorderAvgColor + centers[id].rgb;
						for (size_t k = 0; k < colorHist[id].size(); k++)
						{
							boxBorderHist[k] += colorHist[id][k];
						}
					}
				}

			}
			boxBorderAvgColor = boxBorderAvgColor*(1.0 / boxBorderSpNum);
			fgAvgColor = fgAvgColor*(1.0 / fgSpNum);

			cv::normalize(fgCHist, fgCHist, 1, 0, cv::NORM_L1);
			cv::normalize(boxBorderHist, boxBorderHist, 1, 0, cv::NORM_L1);
			double boxdist = cv::compareHist(boxBorderHist, fgCHist, CV_COMP_BHATTACHARYYA);
			//double boxdist = RegionColorDist(boxBorderHist, fgCHist, boxBorderAvgColor, fgAvgColor);
			proposalIds.push_back(ids);
			cproposalIds.push_back(cids);
			regScores.push_back(scores);

			double fillness = pixels / box.area();
			double size = pixels / width / height;
			double compactness = std::min(box.width, box.height)*1.0 / std::max(box.width, box.height);
			float weightF = exp(-sqr(fillness - meanFillness) / thetaFill);
			//float weightF = fillness > 0.15 ? 1 : 0;
			float weightS = exp(-sqr(size - meanRelSize) / thetaSize);
			float weightO = boxdist;
			float weight = 0.05*weightF + 0.65*weightS + 0.3*weightO;


			PropScore ps;
			ps.id = proposals.size();
			ps.score = weight;
			totalWeights += weight;
			//CalRegionFocusness(gradMap, scaleMap, edgeMap, _spPoses, regions, focus);
			GetRegionSaliencyMap(width, height, &computer, newLabels, regions, regInfos, candiRegions.size(), rmask);
			proposals.push_back(rmask.clone());
			/*if (!debug)*/
			{
				char imgName[300];

				rmask.convertTo(rmask, CV_8U, 255);
				sprintf(imgName, "%s%dSaliency_%d_%d_%d.png", outDir, regInfos.size(), (int)(weightF * 100), (int)(weightS * 100), (int)(boxdist * 100));
				cv::imwrite(imgName, rmask);

			}

			propScores.push_back(ps);

			outFile << propScores.size() << "\t" << box.x << "\t" << box.y << "\t" << box.width << "\t" << box.height << "\n";
			/*proposals.push_back(salPropose.clone());
			std::vector<int> nLabels(spSize);
			memset(&nLabels[0], sizeof(int)*spSize, 0);
			std::vector<SPRegion> tmpRegions;
			SPRegion reg0;
			reg0 = regions[regInfos[regInfos.size() - 1].id];
			reg0.id = 0;
			SPRegion reg1;
			reg1.id = 1;
			std::vector<int> salIds;
			float4 c1;
			c1 = make_float4(0, 0, 0, 0);
			std::vector<float> fgHist(colorHist[0].size(), 0);
			for (size_t i = 0; i < regInfos.size() - 1; i++)
			{
			SPRegion& region = regions[regInfos[i].id];
			reg1.size += region.size;
			c1 = c1 + region.color*region.size;
			for (size_t j = 0; j < fgHist.size(); j++)
			{
			fgHist[j] += region.colorHist[j];
			}
			for (size_t j = 0; j < region.size; j++)
			{
			reg1.spIndices.push_back(region.spIndices[j]);
			nLabels[region.spIndices[j]] = 1;
			}


			}
			reg0.neighbors.clear();
			reg1.color = c1*(1.0 / reg1.size);

			cv::normalize(fgHist, fgHist, 1, 0, cv::NORM_L1);
			reg1.colorHist = fgHist;
			reg0.neighbors.push_back(1);
			reg1.neighbors.push_back(0);
			tmpRegions.push_back(reg0);
			tmpRegions.push_back(reg1);
			UpdateRegionInfo(width, height, &computer, nLabels, tmpRegions, segment);
			GetSaliencyProposalMap(outPath, salPropose, tmpRegions, computer, colorHist, nLabels, propScores, proposals);*/
		}

		if (regInfos.size() == 2)
			break;
		SalGuidedRegMergion(img, (char*)outPath, regInfos, computer, newLabels, regions, debug);


	}
	outFile.close();
	salMap = cv::Mat::zeros(img.size(), CV_32F);
	std::sort(propScores.begin(), propScores.end(), PropScoreCmp());
	for (size_t i = 0; i < propScores.size() / 2; i++)
	{
		for (size_t j = 0; j < cproposalIds[propScores[i].id].size(); j++)
		{
			float weight = propScores[i].score * regScores[propScores[i].id][j];
			candiRegions[cproposalIds[propScores[i].id][j]].regSalScore += weight;
			totalWeights += propScores[i].score;
		}

	}

	for (size_t i = 0; i < candiRegions.size(); i++)
	{
		float sal = candiRegions[i].regSalScore / totalWeights*exp(-9.0*(sqr(candiRegions[i].ad2c.x) + sqr(candiRegions[i].ad2c.y)));
		//
		candiRegions[i].regSalScore = sal;
		for (int j = 0; j < candiRegions[i].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[candiRegions[i].spIndices[j]].size(); k++)
			{
				int c = spPoses[candiRegions[i].spIndices[j]][k].x;
				int r = spPoses[candiRegions[i].spIndices[j]][k].y;

				*(float*)(salMap.data + (r*width + c) * 4) = sal;

			}
		}
	}

	cv::GaussianBlur(salMap, salMap, cv::Size(3, 3), 0);
	cv::normalize(salMap, salMap, 0, 255, CV_MINMAX, CV_8U);

	sprintf(fileName, "%s\\%s_RMF.png", outputPath, imgName);
	cv::imwrite(fileName, salMap);

	//smooth by neighbors
	salMap = cv::Mat::zeros(img.size(), CV_32F);
	float lmd = 0.8;
	std::sort(candiRegions.begin(), candiRegions.end(), RegionSalscoreCmp());
	for (size_t i = 0; i < candiRegions.size() - 1; i++)
	{
		if (candiRegions[i].regSalScore < 1e-6)
			continue;
		float sumScore(0), sumContrast(1e-6);
		candiRegions[i].regContrast;
		for (size_t j = 0; j < candiRegions[i].neighbors.size(); j++)
		{


			int nid = std::find_if(candiRegions.begin(), candiRegions.end(), RegionIdLocate(candiRegions[i].neighbors[j])) - candiRegions.begin();
			if (nid != candiRegions.size() - 1)
			{
				float contrastDist = candiRegions[nid].regContrast - candiRegions[i].regContrast;

				sumScore += candiRegions[nid].regSalScore*exp(-sqr(contrastDist)*9.0);
				sumContrast += 1;
			}

		}
		//std::cout << candiRegions[i].id << "contrast " << candiRegions[i].regContrast << " score " << candiRegions[i].regSalScore << " update to ";
		candiRegions[i].regSalScore = lmd*candiRegions[i].regSalScore + (1 - lmd)*sumScore / sumContrast;
		//std::cout << candiRegions[i].regSalScore << "\n";
	}
	for (size_t i = 0; i < candiRegions.size(); i++)
	{
		for (int j = 0; j < candiRegions[i].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[candiRegions[i].spIndices[j]].size(); k++)
			{
				int c = spPoses[candiRegions[i].spIndices[j]][k].x;
				int r = spPoses[candiRegions[i].spIndices[j]][k].y;

				*(float*)(salMap.data + (r*width + c) * 4) = candiRegions[i].regSalScore;

			}
		}
	}
	cv::GaussianBlur(salMap, salMap, cv::Size(3, 3), 0);
	cv::normalize(salMap, salMap, 0, 255, CV_MINMAX, CV_8U);

	sprintf(fileName, "%s\\%s_RMS.png", outputPath, imgName);
	cv::imwrite(fileName, salMap);
	cv::threshold(salMap, salMap, 127, 255, CV_THRESH_BINARY);
	sprintf(fileName, "%s\\%s_RMSB.png", outputPath, imgName);
	cv::imwrite(fileName, salMap);

	salMap = cv::Mat::zeros(img.size(), CV_32F);
	//Saliency(outPath, imgName, propScores, proposals, salMap);
	Saliency(outputPath, imgName, propScores, candiRegions, proposalIds, proposals, salMap);
	//cv::add(salMap, bkgSal, salMap);
	cv::threshold(salMap, salMap, 127, 255, CV_THRESH_BINARY);
#if (TEST_SPEED)
	{
		timer.stop();
		std::cout << "SaliencyGuided Region mergion " << timer.seconds() * 1000 << "ms\n";

	}
#endif

}
void IterativeRegionGrowing(const cv::Mat& img, const cv::Mat& edgeMap, const char* imgName, const char* outPutPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, cv::Mat& saliencyRst, int regThreshold, bool debug)
{

	char outPath[300];
	sprintf(outPath, "%s%s\\", outPutPath, imgName);
	if (debug)
	{
		CreateDir(outPath);

	}
	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	/*cv::Mat scaleMap,gradMap;
	CalScale(gray, scaleMap);
	cv::Mat scaleMapS;
	cv::normalize(scaleMap, scaleMapS, 255, 0, CV_MINMAX, CV_8U);
	cv::imshow("scaleMap", scaleMapS);
	cv::waitKey();
	DogGradMap(gray, gradMap);*/
	int width = img.cols, height = img.rows;
	int spWidth = computer.GetSPWidth(), spHeight = computer.GetSPHeight();
	int spSize(spWidth*spHeight);
	//build historgram
	HISTOGRAMS colorHist, gradHist, lbpHist;
	//BuildHistogram(img, &computer, colorHist, gradHist);

	BuildHistogram(img, &computer, colorHist, gradHist, lbpHist);
	/*cv::Mat idx1i, _color3f, _colorNum;
	double ratio = 0.95;
	const int clrNums[3] = { 12, 12, 12 };
	cv::Mat fimg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	int num = Quantize(fimg, idx1i, _color3f, _colorNum, ratio, clrNums);
	BuildQHistorgram(idx1i, num, &computer, colorHist);
	gradHist.resize(spSize);
	cv::cvtColor(_color3f, _color3f, CV_BGR2Lab);
	cv::Mat_<float> cDistCache1f = cv::Mat::zeros(_color3f.cols, _color3f.cols, CV_32F); {
	cv::Vec3f* pColor = (cv::Vec3f*)_color3f.data;
	for (int i = 0; i < cDistCache1f.rows; i++)
	for (int j = i + 1; j < cDistCache1f.cols; j++)
	{
	float dist = vecDist<float, 3>(pColor[i], pColor[j]);
	cDistCache1f[i][j] = cDistCache1f[j][i] = vecDist<float, 3>(pColor[i], pColor[j]);
	}
	}*/

	int * labels(NULL);
	SLICClusterCenter* centers(NULL);
	computer.GetSuperpixelResult(spSize, labels, centers);
	std::vector<std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);
	newLabels.resize(spSize);
	//init regions 
	for (int i = 0; i < spSize; i++)
	{
		newLabels[i] = i;
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

	//处理边缘
	if (spPoses[spWidth - 1].size() < 10)
	{
		for (size_t i = 0; i < spHeight; i++)
		{
			MergeRegions(spWidth*i - 1, spWidth*i - 2, newLabels, spPoses, regions);
		}
	}
	if (spPoses[(spHeight - 1)*spWidth].size() < 10)
	{
		for (size_t i = 0; i < spWidth; i++)
		{
			MergeRegions((spHeight - 1)*spWidth + i, (spHeight - 2)*spWidth + i, newLabels, spPoses, regions);
		}
	}


	int* segment = new int[img.cols*img.rows];


	int ZeroReg, RegSize(regions.size());

	//for (size_t i = 0; i < regions.size(); i++)
	//{
	//	if ((regions[i].size > 0 && regions[i].neighbors.size() <= 2 && regions[i].size < HoleSize) ||
	//		regions[i].size == 1)
	//	{
	//		int regId = regions[i].id;
	//		//处理空洞区域，将其合并到最接近的邻居中
	//		HandleHole(i, newLabels, spPoses, regions);
	//		//HandleHoleDemo(width, height, i, &computer, spPoses, newLabels, regions);
	//	}
	//}

	int iter(0);
	while (RegSize > regThreshold)
	{
		UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, edgeMap, regions);


		//UpdateRegionInfo(img.cols, img.rows, &computer, gradMap, scaleMap, edgeMap, newLabels, regions, segment);
		int needToMerge = (RegSize - regThreshold) / 2;
		needToMerge = std::max(1, needToMerge);
		RegionGrowing(iter++, img, outPath, edgeMap, computer, newLabels, regions, needToMerge, debug);


		//for (size_t i = 0; i < regions.size(); i++)
		//{
		//	if (regions[i].size > 0 && regions[i].neighbors.size() <= HoleNeighborsNum && regions[i].size < HoleSize)
		//	{
		//		int regId = regions[i].id;
		//		//处理空洞区域，将其合并到最接近的邻居中
		//		HandleHole(i, newLabels, spPoses, regions);
		//		//HandleHoleDemo(width, height, i, &computer, spPoses, newLabels, regions);
		//	}
		//}
		ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		RegSize = regions.size() - ZeroReg;
	}
	cv::Mat sal1, sal2;
	char name[200];
	cv::Mat rmask;
	if (debug)
	{
		CreateDir((char*)outPath);

		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
		sprintf(name, "%sMerge1_%d.jpg", outPath, RegSize);
		cv::imwrite(name, rmask);

	}
	SmartHoleHandling(width, height, outPath, &computer, regions, newLabels, HoleNeighborsNum, HoleSize, true);
	//int holeRegNum(0);
	//for (size_t i = 0; i < regions.size(); i++)
	//{
	//	if ((regions[i].size > 0 && regions[i].neighbors.size() <= HoleNeighborsNum && regions[i].size < HoleSize) ||
	//		(regions[i].size < 5 && regions[i].size > 0))
	//	{
	//		int regId = regions[i].id;
	//		//处理空洞区域，将其合并到最接近的邻居中
	//		holeRegNum += HandleHole(i, newLabels, spPoses, regions);
	//		//HandleHoleDemo(width, height, i, &computer, spPoses, newLabels, regions);
	//	}
	//}
	ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	RegSize = regions.size() - ZeroReg;
	//RegSize -= holeRegNum;

	//PickSaliencyRegion(img.cols, img.rows, &computer, newLabels, regions, sal1, 0.6);
	if (debug)
	{
		cv::Mat rmask;
		char name[200];
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
		sprintf(name, "%sMerge2_%d.jpg", outPath, RegSize);
		cv::imwrite(name, rmask);
		sal1.convertTo(rmask, CV_8U, 1.0);
		sprintf(name, "%sSal1_%d.jpg", outPath, RegSize);
		cv::imwrite(name, rmask);
	}

	float maxSaliency(0);
	float maxContrast(0);

	cv::Mat mask, dbgMap;

	std::vector<float> weights;
	std::vector<cv::Mat> salResults;
	float totalWeights(0);
	std::vector<RegionSalInfo> regInfos;
	UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, edgeMap, regions);
	RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, debug);

	HandleOcculusion(img, computer, outPath, newLabels, regInfos, regions, segment, debug);
	UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, edgeMap, regions);
	//UpdateRegionInfo(img.cols, img.rows, &computer, gradMap, scaleMap, edgeMap, newLabels, regions, segment);
	if (debug)
	{
		RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, debug);
		std::cout << "After Occulusion handling" << regInfos.size() << " regions remains\n";
		std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());
		for (size_t i = 0; i < regInfos.size(); i++)
			std::cout << regInfos[i] << "\n";
	}
	//保存这时的状态
	RegionPartition rp;
	for (size_t i = 0; i < regInfos.size(); i++)
	{
		regInfos[i].neighbors.clear();
		for (size_t j = 0; j < regions[regInfos[i].id].neighbors.size(); j++)
		{
			int regId = regions[regInfos[i].id].neighbors[j];
			for (size_t k = 0; k < regInfos.size(); k++)
			{
				if (regInfos[k].id == regId)
				{
					regInfos[i].neighbors.push_back(k);
					break;
				}
			}
		}
	}
	//adjacency_list_t adjacency_list(regInfos.size());
	//rp.minDistances.resize(regInfos.size());
	//
	//std::vector<std::vector<vertex_t>> previouses(regInfos.size());
	//
	//for (size_t i = 0; i < regInfos.size(); i++)
	//{
	//	rp.regions.push_back(regions[regInfos[i].id]);
	//	for (size_t j = 0; j < regInfos[i].neighbors.size(); j++)
	//	{
	//		int n = regInfos[i].neighbors[j];

	//		adjacency_list[i].push_back(neighbor(n, 1));
	//	}
	//}
	//for (size_t i = 0; i < regInfos.size(); i++)
	//{
	//	DijkstraComputePaths(i, adjacency_list, rp.minDistances[i], previouses[i]);
	//	/*for (size_t j = 0; j <rp.minDistances[i].size() ; j++)
	//	{
	//		std::cout << "min distance " << regInfos[i].id << " to " << regInfos[j].id << " is " << rp.minDistances[i][j] << std::endl;
	//	}*/
	//}

	float borderRatio = regInfos[regInfos.size() - 1].borderRatio;
	std::vector<int> bgRegIds;
	float rmThreshold = 0.75;
	std::vector<cv::Mat> salMaps;

	while (regInfos.size() > 2)
	{
		cv::Mat salMap;
		SalGuidedRegMergion(img, (char*)outPath, regInfos, computer, newLabels, regions, debug);
		UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, edgeMap, regions);
		RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, salMap, debug);
		if ((regInfos[regInfos.size() - 1].borderRatio > 0.8 && regInfos.size() < 8) ||
			regInfos.size() < 5)
			salMaps.push_back(salMap.clone());

		/*if (regInfos.size() < 8 || borderRatio > 0.80)
		salMaps.push_back(salMap.clone());*/
		borderRatio = regInfos[regInfos.size() - 1].borderRatio;
		if (debug)
		{
			std::cout << "After mergion " << regInfos.size() << " regions remains\n";
			std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());
			for (size_t i = 0; i < regInfos.size(); i++)
				std::cout << regInfos[i] << "\n";
		}
	}



	std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, 1);
	sprintf(name, "%s%s_region_%d.jpg", outPutPath, imgName, regInfos.size());
	cv::imwrite(name, rmask);
	char fileName[200];
	char spath[200];
	sprintf(spath, "%s//saliency//", outPath);
	CreateDir(spath);
	for (size_t i = 0; i < salMaps.size(); i++)
	{
		sprintf(fileName, "%s%dSaliency.png", spath, i + 1);
		cv::imwrite(fileName, salMaps[i]);
	}

	while (regInfos.size()>2)
	{
		SalGuidedRegMergion2(img, (char*)outPath, regInfos, computer, newLabels, regions, debug);


		UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, edgeMap, regions);
		cv::Mat salMap;
		RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, salMap, debug);

		salMaps.push_back(salMap.clone());
		/*std::cout << "After merging\n";
		std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());
		for (size_t i = 0; i < regInfos.size(); i++)
		std::cout << regInfos[i] << "\n";*/

	}



	saliencyRst.create(height, width, CV_8U);
	saliencyRst = cv::Scalar(0);

	std::vector<int> salIds;
	int salId = regInfos[0].id;
	salIds.push_back(salId);

	/*int salRegNum = regInfos.size() <= 3 ? 1 : 2;

	for (size_t i = 0; i < regions[salId].neighbors.size(); i++)
	{
	size_t j;
	for (j = 0; j < regInfos.size(); j++)
	{
	if (regInfos[j].id == regions[salId].neighbors[i])
	break;
	}
	if (j < salRegNum)
	salIds.push_back(regions[salId].neighbors[i]);
	}*/

	for (size_t i = 0; i < salIds.size(); i++)
	{
		int regId = salIds[i];
		for (size_t j = 0; j < regions[regId].spIndices.size(); j++)
		{
			int spIdx = regions[regId].spIndices[j];
			for (size_t k = 0; k < spPoses[spIdx].size(); k++)
			{
				int c = spPoses[regions[regId].spIndices[j]][k].x;
				int r = spPoses[regions[regId].spIndices[j]][k].y;
				*(char *)(saliencyRst.data + (r*width + c)) = 0xff;

			}
		}
	}
	/*saliencyRst = cv::Mat::zeros(height, width, CV_32F);
	for (size_t i = 0; i < salMaps.size(); i++)
	{
	cv::add(saliencyRst, salMaps[i], saliencyRst);
	}
	cv::normalize(saliencyRst, saliencyRst, 0, 255, CV_MINMAX, CV_8U);*/
	//cv::threshold(saliencyRst, saliencyRst, regInfos[0].RegionSaliency()*0.96, 255, CV_8U);
	delete[] segment;

	std::sort(regions.begin(), regions.end(), RegionSizeCmp());
	int size = std::find_if(regions.begin(), regions.end(), RegionSizeZero()) - regions.begin();
	regions.resize(size);


	//region neighbors index by region vector index
	regNeighbors.resize(regions.size());
	for (size_t i = 0; i < regions.size(); i++)
	{
		for (size_t j = 0; j < regions[i].neighbors.size(); j++)
		{
			for (size_t k = 0; k < regions.size(); k++)
			{
				if (regions[k].id == regions[i].neighbors[j])
				{
					regNeighbors[i].push_back(k);
					break;
				}
			}
		}
	}
}





void IterativeRegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, int regThreshold)
{
	cv::Mat edgeMap;
	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::Canny(img, edgeMap, 100, 300);
	edgeMap.convertTo(edgeMap, CV_32F, 1.0 / 255);


	int width = img.cols, height = img.rows;
	int spWidth = computer.GetSPWidth(), spHeight = computer.GetSPHeight();
	int spSize(spWidth*spHeight);
	//build historgram
	HISTOGRAMS colorHist, gradHist, lbpHist;
	//BuildHistogram(img, &computer, colorHist, gradHist);

	BuildHistogram(img, &computer, colorHist, gradHist, lbpHist);
	/*cv::Mat idx1i, _color3f, _colorNum;
	double ratio = 0.95;
	const int clrNums[3] = { 12, 12, 12 };
	cv::Mat fimg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	int num = Quantize(fimg, idx1i, _color3f, _colorNum, ratio, clrNums);
	BuildQHistorgram(idx1i, num, &computer, colorHist);
	gradHist.resize(spSize);
	cv::cvtColor(_color3f, _color3f, CV_BGR2Lab);
	cv::Mat_<float> cDistCache1f = cv::Mat::zeros(_color3f.cols, _color3f.cols, CV_32F); {
	cv::Vec3f* pColor = (cv::Vec3f*)_color3f.data;
	for (int i = 0; i < cDistCache1f.rows; i++)
	for (int j = i + 1; j < cDistCache1f.cols; j++)
	{
	float dist = vecDist<float, 3>(pColor[i], pColor[j]);
	cDistCache1f[i][j] = cDistCache1f[j][i] = vecDist<float, 3>(pColor[i], pColor[j]);
	}
	}*/

	int * labels(NULL);
	SLICClusterCenter* centers(NULL);
	computer.GetSuperpixelResult(spSize, labels, centers);

	newLabels.resize(spSize);
	//init regions 
	for (int i = 0; i < spSize; i++)
	{
		newLabels[i] = i;
		SPRegion region;
		region.cX = centers[i].xy.x;
		region.cY = centers[i].xy.y;
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
		//regNeighbors.push_back(computer.GetNeighbors(i));
	}
	int* segment = new int[img.cols*img.rows];
	//GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	//GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	GetRegionPixelBorder(img.cols, img.rows, &computer, newLabels, regions);
	GetRegionEdgeness(edgeMap, regions);
	delete[] segment;
	int ZeroReg, RegSize(regions.size());


	int iter(0);
	while (RegSize > regThreshold)
	{
		RegionGrowing(iter++, img, outPath, edgeMap, computer, newLabels, regions, thresholdF);
		ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		RegSize = regions.size() - ZeroReg;
	}

	/*std::vector<std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);
	cv::normalize(spSaliency, spSaliency, 1, 0, cv::NORM_MINMAX);
	cv::Mat smap(height, width, CV_8U);
	for (size_t i = 0; i < spSize; i++)
	{
	for (size_t j = 0; j < spPoses[i].size(); j++)
	{
	int idx = spPoses[i][j].x + spPoses[i][j].y*width;
	smap.data[idx] = spSaliency[i] * 255;
	}
	}
	cv::imshow("spSaliency", smap);
	cv::waitKey();*/

	std::sort(regions.begin(), regions.end(), RegionSizeCmp());
	int size = std::find_if(regions.begin(), regions.end(), RegionSizeZero()) - regions.begin();
	regions.resize(size);

	//region neighbors index by region vector index
	regNeighbors.resize(size);
	for (size_t i = 0; i < regions.size(); i++)
	{
		for (size_t j = 0; j < regions[i].neighbors.size(); j++)
		{
			for (size_t k = 0; k < regions.size(); k++)
			{
				if (regions[k].id == regions[i].neighbors[j])
				{
					regNeighbors[i].push_back(k);
					break;
				}
			}
		}
	}
}
//iterative region growing
void IterativeRegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF, int regThreshold)
{
	int width = img.cols, height = img.rows;
	int spWidth = computer.GetSPWidth(), spHeight = computer.GetSPHeight();
	int spSize(spWidth*spHeight);
	//build historgram
	HISTOGRAMS colorHist, gradHist;
	BuildHistogram(img, &computer, colorHist, gradHist);

	/*cv::Mat idx1i, _color3f, _colorNum;
	double ratio = 0.95;
	const int clrNums[3] = { 12, 12, 12 };
	cv::Mat fimg;
	img.convertTo(fimg, CV_32FC3, 1.0 / 255);
	int num = Quantize(fimg, idx1i, _color3f, _colorNum, ratio, clrNums);
	BuildQHistorgram(idx1i, num, &computer, colorHist);
	gradHist.resize(spSize);
	cv::cvtColor(_color3f, _color3f, CV_BGR2Lab);
	cv::Mat_<float> cDistCache1f = cv::Mat::zeros(_color3f.cols, _color3f.cols, CV_32F); {
	cv::Vec3f* pColor = (cv::Vec3f*)_color3f.data;
	for (int i = 0; i < cDistCache1f.rows; i++)
	for (int j = i + 1; j < cDistCache1f.cols; j++)
	{
	float dist =  vecDist<float, 3>(pColor[i], pColor[j]);
	cDistCache1f[i][j] = cDistCache1f[j][i] = vecDist<float, 3>(pColor[i], pColor[j]);

	}

	}*/

	int * labels(NULL);
	SLICClusterCenter* centers(NULL);
	computer.GetSuperpixelResult(spSize, labels, centers);

	newLabels.resize(spSize);
	//init regions 
	for (int i = 0; i < spSize; i++)
	{
		newLabels[i] = i;
		SPRegion region;
		region.cX = centers[i].xy.x;
		region.cY = centers[i].xy.y;
		region.color = centers[i].rgb;
		region.colorHist = colorHist[i];
		region.hog = gradHist[i];
		region.size = 1;
		region.dist = 0;
		region.id = i;
		region.neighbors = computer.GetNeighbors4(i);
		region.spIndices.push_back(i);
		regions.push_back(region);
		//regNeighbors.push_back(computer.GetNeighbors(i));
	}
	int* segment = new int[img.cols*img.rows];
	GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	delete[] segment;
	int ZeroReg, RegSize(regions.size());

	while (RegSize > regThreshold)
	{
		RegionGrowing(img, computer, newLabels, regions, thresholdF);
		ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		RegSize = regions.size() - ZeroReg;
	}

	std::sort(regions.begin(), regions.end(), RegionSizeCmp());
	int size = std::find_if(regions.begin(), regions.end(), RegionSizeZero()) - regions.begin();
	regions.resize(size);

}

void RegionGrowing(const cv::Mat& img, const char* outPath, std::vector<float>& spSaliency, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF)
{
	static int idx = 0;
	std::vector<RegDist> RegDists;
	float avgColorDist(0);
	float avgHogDist(0);
	float avgSizeDist(0);
	int sum(0);
	int spSize = computer.GetSuperpixelSize();
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (i < n)
			{
				double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				double hogDist = cv::compareHist(regions[i].hog, regions[n].hog, CV_COMP_BHATTACHARYYA);
				double lbpDist = cv::compareHist(regions[i].lbpHist, regions[n].lbpHist, CV_COMP_BHATTACHARYYA);
				avgColorDist += colorDist;
				avgHogDist += hogDist;
				//double dist = RegionDist(regions[i], regions[n]);
				float borderLen = regions[i].borders[j];
				float borderLenI = std::accumulate(regions[i].borders.begin(), regions[i].borders.end(), 0);
				float borderLenN = std::accumulate(regions[n].borders.begin(), regions[n].borders.end(), 0);
				float pborderLen = regions[i].borderPixelNum[j];
				float pborderLenI = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), 0);
				float pborderLenN = std::accumulate(regions[n].borderPixelNum.begin(), regions[n].borderPixelNum.end(), 0);
				double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
				double pshapeDist = 1 - (pborderLen) / std::min(pborderLenI, pborderLenN);
				double sizeDist = (regions[i].size + regions[n].size)*1.0 / spSize;
				sizeDist = shapeDist;
				avgSizeDist += pshapeDist;
				sum++;
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				float c = 0.6;
				rd.colorDist = colorDist;
				rd.shapeDist = shapeDist;
				rd.sizeDist = sizeDist;
				rd.hogDist = hogDist;
				rd.lbpDist = lbpDist;
				RegDists.push_back(rd);
			}
		}
	}
	avgColorDist /= sum;
	avgHogDist /= sum;
	avgSizeDist /= sum;

	double avgDistSum = avgColorDist + avgHogDist + avgSizeDist;

	//std::cout << idx << ": avgColorDist= " << avgColorDist << ",avgHogDist= " << avgHogDist << ",avgSizeDist= " << avgSizeDist << "\n";
	//std::cout << cw << "," << hw << "," << sw << "\n";
	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(cw, hw, siw, shw));
	int N = std::max(1, (int)(thresholdF*RegDists.size()));

	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);


	std::vector<uint2> regPairs;
	std::vector<int> sIds;
	for (int i = 0; i < N; i++)
	{
		if (regions[RegDists[i].bRid].size > 0 && regions[RegDists[i].sRid].size > 0)
		{
			if (std::find(sIds.begin(), sIds.end(), RegDists[i].sRid) == sIds.end() &&
				std::find(sIds.begin(), sIds.end(), RegDists[i].bRid) == sIds.end())
			{
				sIds.push_back(RegDists[i].sRid);
				sIds.push_back(RegDists[i].bRid);
				regPairs.push_back(make_uint2(RegDists[i].bRid, RegDists[i].sRid));
				//超像素参与区域合并的次数，背景超像素参与次数更多，显著性超像素参与次数更少
				for (int p = 0; p < regions[RegDists[i].bRid].spIndices.size(); p++)
				{
					spSaliency[regions[RegDists[i].bRid].spIndices[p]] ++;
				}
				for (int p = 0; p < regions[RegDists[i].sRid].spIndices.size(); p++)
				{
					spSaliency[regions[RegDists[i].sRid].spIndices[p]] ++;
				}
			}

		}

	}
	cv::Mat mask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);
	char name[100];
	sprintf(name, "%s%dregMergeB.jpg", outPath, idx);
	cv::imwrite(name, mask);
	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}
	int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int RegSize = regions.size() - ZeroReg;
	////handle holes
	//for (int i = 0; i < regions.size(); i++)
	//{
	//	if (regions[i].size == 1 && regions[i].neighbors.size() < 4)
	//	{
	//		int minId = 0;
	//		int nId = regions[i].neighbors[0];
	//		float minDist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	//		for (size_t j = 1; j < regions[i].neighbors.size(); j++)
	//		{
	//			nId = regions[i].neighbors[j];
	//			float dist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	//			if (dist < minDist)
	//			{
	//				minDist = dist;
	//				minId = nId;
	//			}
	//		}
	//		MergeRegions(i, nId, newLabels, spPoses, regions);
	//	}
	//}
	int* segment = new int[img.cols*img.rows];
	//GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	//GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	GetRegionPixelBorder(img.cols, img.rows, &computer, newLabels, regions);
	/*if (idx == 124)*/
	//{
	//	std::vector<std::vector<uint2>> spPoses;
	//	computer.GetSuperpixelPoses(spPoses);
	//	cv::Mat bmask(img.rows, img.cols, CV_8U);
	//	bmask = cv::Scalar(0);
	//	for (size_t i = 0; i < regions.size(); i++)
	//	{
	//		if (regions[i].size == 0)
	//			continue;
	//		for (int j = 0; j < regions[i].borderPixels.size(); j++)
	//		{
	//			std::vector<uint2> borderPixels = regions[i].borderPixels[j];
	//			for (size_t k = 0; k < borderPixels.size(); k++)
	//			{
	//				int id = borderPixels[k].y*img.cols + borderPixels[k].x;
	//				bmask.data[id] = 0xff;
	//			}

	//		}
	//		break;
	//	}
	//	/*	sprintf(name, "%dreg0Border.jpg", idx);
	//	cv::imwrite(name, bmask);*/
	//	cv::imshow("border", bmask);
	//	cv::waitKey(0);
	//}
	delete[] segment;

	cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
	sprintf(name, "%s%dregMergeF_%d.jpg", outPath, idx, RegSize);
	cv::imwrite(name, rmask);
	idx++;
}

void RegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF)
{
	static int idx = 0;
	std::vector<RegDist> RegDists;
	float avgColorDist(0);
	float avgHogDist(0);
	float avgSizeDist(0);
	int sum(0);
	int spSize = computer.GetSuperpixelSize();
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (i < n)
			{
				double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				double hogDist = cv::compareHist(regions[i].hog, regions[n].hog, CV_COMP_BHATTACHARYYA);
				double lbpDist = cv::compareHist(regions[i].lbpHist, regions[n].lbpHist, CV_COMP_BHATTACHARYYA);
				avgColorDist += colorDist;
				avgHogDist += hogDist;
				//double dist = RegionDist(regions[i], regions[n]);
				float borderLen = regions[i].borders[j];
				float borderLenI = std::accumulate(regions[i].borders.begin(), regions[i].borders.end(), 0);
				float borderLenN = std::accumulate(regions[n].borders.begin(), regions[n].borders.end(), 0);
				double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
				double sizeDist = (regions[i].size + regions[n].size)*1.0 / spSize;
				sizeDist = shapeDist;
				avgSizeDist += sizeDist;
				sum++;
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				float c = 0.6;
				rd.colorDist = colorDist;
				rd.sizeDist = sizeDist;
				rd.hogDist = hogDist;
				rd.lbpDist = lbpDist;
				RegDists.push_back(rd);
			}
		}
	}
	avgColorDist /= sum;
	avgHogDist /= sum;
	avgSizeDist /= sum;
	double cw, hw, shw, sw;
	double avgDistSum = avgColorDist + avgHogDist + avgSizeDist;
	cw = 0.5;
	hw = 0.1;
	shw = 0.3;
	sw = 0.1;
	//std::cout << idx << ": avgColorDist= " << avgColorDist << ",avgHogDist= " << avgHogDist << ",avgSizeDist= " << avgSizeDist << "\n";
	//std::cout << cw << "," << hw << "," << sw << "\n";
	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(cw, hw, shw, sw));
	int N = std::max(1, (int)(thresholdF*RegDists.size()));

	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);


	std::vector<uint2> regPairs;
	std::vector<int> sIds;
	for (int i = 0; i < N; i++)
	{
		if (regions[RegDists[i].bRid].size > 0 && regions[RegDists[i].sRid].size > 0)
		{
			if (std::find(sIds.begin(), sIds.end(), RegDists[i].sRid) == sIds.end() &&
				std::find(sIds.begin(), sIds.end(), RegDists[i].bRid) == sIds.end())
			{
				sIds.push_back(RegDists[i].sRid);
				sIds.push_back(RegDists[i].bRid);
				regPairs.push_back(make_uint2(RegDists[i].bRid, RegDists[i].sRid));
			}

		}

	}
	cv::Mat mask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);
	char name[100];
	sprintf(name, "%s%dregMergeB.jpg", outPath, idx);
	cv::imwrite(name, mask);
	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}
	int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int RegSize = regions.size() - ZeroReg;
	////handle holes
	//for (int i = 0; i < regions.size(); i++)
	//{
	//	if (regions[i].size == 1 && regions[i].neighbors.size() < 4)
	//	{
	//		int minId = 0;
	//		int nId = regions[i].neighbors[0];
	//		float minDist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	//		for (size_t j = 1; j < regions[i].neighbors.size(); j++)
	//		{
	//			nId = regions[i].neighbors[j];
	//			float dist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	//			if (dist < minDist)
	//			{
	//				minDist = dist;
	//				minId = nId;
	//			}
	//		}
	//		MergeRegions(i, nId, newLabels, spPoses, regions);
	//	}
	//}
	int* segment = new int[img.cols*img.rows];
	GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);

	delete[] segment;


	cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
	sprintf(name, "%s%dregMergeF_%d.jpg", outPath, idx, RegSize);
	cv::imwrite(name, rmask);
	idx++;

	/*std::sort(regions.begin(), regions.end(), RegionSizeCmp());*/
	//int size = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
}

void RegionGrowing(const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, cv::Mat1f& colorDistCache, float thresholdF)
{
	static int idx = 0;
	std::vector<RegDist> RegDists;
	float avgColorDist(0);
	float avgHogDist(0);
	float avgSizeDist(0);

	float maxColorDist(1e-6);
	float maxShapDist(1e-6);
	int sum(0);
	int spSize = computer.GetSuperpixelSize();
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (i < n)
			{
				//double colorDist = RegionDist(regions[i], regions[n], colorDistCache);
				double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				if (colorDist > maxColorDist)
					maxColorDist = colorDist;
				avgColorDist += colorDist;

				//double dist = RegionDist(regions[i], regions[n]);
				float borderLen = regions[i].borders[j];
				float borderLenI = std::accumulate(regions[i].borders.begin(), regions[i].borders.end(), 0);
				float borderLenN = std::accumulate(regions[n].borders.begin(), regions[n].borders.end(), 0);
				double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
				if (shapeDist > maxShapDist)
					maxShapDist = shapeDist;
				double sizeDist = (regions[i].size + regions[n].size)*1.0 / spSize;
				sizeDist = shapeDist;
				avgSizeDist += sizeDist;
				sum++;
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				float c = 0.6;
				rd.colorDist = colorDist;
				rd.sizeDist = sizeDist;

				RegDists.push_back(rd);
			}
		}
	}
	avgColorDist /= sum;
	avgHogDist /= sum;
	avgSizeDist /= sum;
	double cw, hw, sw;
	double avgDistSum = avgColorDist + avgHogDist + avgSizeDist;
	cw = 1;
	hw = 0;
	sw = 0;
	//std::cout << idx << ": avgColorDist= " << avgColorDist << ",avgHogDist= " << avgHogDist << ",avgSizeDist= " << avgSizeDist << "\n";
	//std::cout << cw << "," << hw << "," << sw << "\n";
	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(cw, hw, sw, 0));
	int N = std::max(1, (int)(thresholdF*RegDists.size()));

	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);


	std::vector<uint2> regPairs;
	std::vector<int> sIds;
	for (int i = 0; i < N; i++)
	{
		if (regions[RegDists[i].bRid].size > 0 && regions[RegDists[i].sRid].size > 0)
		{
			if (std::find(sIds.begin(), sIds.end(), RegDists[i].sRid) == sIds.end() &&
				std::find(sIds.begin(), sIds.end(), RegDists[i].bRid) == sIds.end())
			{
				sIds.push_back(RegDists[i].sRid);
				sIds.push_back(RegDists[i].bRid);
				regPairs.push_back(make_uint2(RegDists[i].bRid, RegDists[i].sRid));
			}

		}

	}
	cv::Mat mask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);
	char name[100];
	sprintf(name, "%s%dregMergeB.jpg", outPath, idx);
	cv::imwrite(name, mask);
	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}
	int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int RegSize = regions.size() - ZeroReg;
	////handle holes
	//for (int i = 0; i < regions.size(); i++)
	//{
	//	if (regions[i].size == 1 && regions[i].neighbors.size() < 4)
	//	{
	//		int minId = 0;
	//		int nId = regions[i].neighbors[0];
	//		float minDist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	//		for (size_t j = 1; j < regions[i].neighbors.size(); j++)
	//		{
	//			nId = regions[i].neighbors[j];
	//			float dist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	//			if (dist < minDist)
	//			{
	//				minDist = dist;
	//				minId = nId;
	//			}
	//		}
	//		MergeRegions(i, nId, newLabels, spPoses, regions);
	//	}
	//}
	int* segment = new int[img.cols*img.rows];
	GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	delete[] segment;

	cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
	sprintf(name, "%s%dregMergeF_%d.jpg", outPath, idx, RegSize);
	cv::imwrite(name, rmask);
	idx++;
}
void RegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF)
{
	static int idx = 0;
	std::vector<RegDist> RegDists;
	float avgColorDist(0);
	float avgHogDist(0);
	float avgSizeDist(0);
	int sum(0);

	int spSize = computer.GetSuperpixelSize();
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (i < n)
			{
				double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				double hogDist = cv::compareHist(regions[i].hog, regions[n].hog, CV_COMP_BHATTACHARYYA);
				avgColorDist += colorDist;
				avgHogDist += hogDist;
				//double dist = RegionDist(regions[i], regions[n]);
				float borderLen = regions[i].borders[j];
				float borderLenI = std::accumulate(regions[i].borders.begin(), regions[i].borders.end(), 0);
				float borderLenN = std::accumulate(regions[n].borders.begin(), regions[n].borders.end(), 0);
				double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
				double sizeDist = (regions[i].size + regions[n].size)*1.0 / spSize;
				double edgeness = regions[i].edgeness[j] / regions[i].borderPixelNum[j];
				sizeDist = shapeDist;
				avgSizeDist += sizeDist;
				sum++;
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				rd.edgeness = edgeness;
				rd.colorDist = colorDist;
				rd.sizeDist = sizeDist;
				rd.shapeDist = shapeDist;
				RegDists.push_back(rd);
			}
		}
	}
	avgColorDist /= sum;
	avgHogDist /= sum;
	avgSizeDist /= sum;

	double avgDistSum = avgColorDist + avgHogDist + avgSizeDist;

	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(cw, hw, shw, siw));
	int N = std::max(1, (int)(thresholdF*RegDists.size()));

	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);


	std::vector<uint2> regPairs;
	std::vector<int> sIds;
	for (int i = 0; i < N; i++)
	{
		if (regions[RegDists[i].bRid].size > 0 && regions[RegDists[i].sRid].size > 0)
		{
			if (std::find(sIds.begin(), sIds.end(), RegDists[i].sRid) == sIds.end() &&
				std::find(sIds.begin(), sIds.end(), RegDists[i].bRid) == sIds.end())
			{
				sIds.push_back(RegDists[i].sRid);
				sIds.push_back(RegDists[i].bRid);
				regPairs.push_back(make_uint2(RegDists[i].bRid, RegDists[i].sRid));
			}

		}

	}
	//cv::Mat mask;
	//GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);
	//char name[100];
	//sprintf(name, "%dregMergeB.jpg", idx);
	//cv::imwrite(name, mask);
	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}
	////handle holes
	//for (int i = 0; i < regions.size(); i++)
	//{
	//	if (regions[i].size == 1 && regions[i].neighbors.size() < 4)
	//	{
	//		int minId = 0;
	//		int nId = regions[i].neighbors[0];
	//		float minDist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	//		for (size_t j = 1; j < regions[i].neighbors.size(); j++)
	//		{
	//			nId = regions[i].neighbors[j];
	//			float dist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	//			if (dist < minDist)
	//			{
	//				minDist = dist;
	//				minId = nId;
	//			}
	//		}
	//		MergeRegions(i, nId, newLabels, spPoses, regions);
	//	}
	//}


	/*cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
	sprintf(name, "%dregMergeF.jpg", idx);
	cv::imwrite(name, rmask);
	idx++;*/


}

void RegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, cv::Mat1f& colorDist, float thresholdF)
{
	static int idx = 0;
	std::vector<RegDist> RegDists;
	int spSize = computer.GetSuperpixelSize();
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (i < n)
			{
				double dist = RegionDist(regions[i], regions[n], colorDist);
				float borderLen = regions[i].borders[j];
				float borderLenI = std::accumulate(regions[i].borders.begin(), regions[i].borders.end(), 0);
				float borderLenN = std::accumulate(regions[n].borders.begin(), regions[n].borders.end(), 0);
				double sizeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				rd.colorDist = dist/*+sizeDist*/;
				RegDists.push_back(rd);
			}
		}
	}

	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer());
	int N = thresholdF*RegDists.size();
	N = 1;
	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);


	std::vector<uint2> regPairs;
	std::vector<int> sIds;
	for (int i = 0; i < N; i++)
	{
		if (regions[RegDists[i].bRid].size > 0 && regions[RegDists[i].sRid].size > 0)
		{
			if (std::find(sIds.begin(), sIds.end(), RegDists[i].sRid) == sIds.end() &&
				std::find(sIds.begin(), sIds.end(), RegDists[i].bRid) == sIds.end())
			{
				sIds.push_back(RegDists[i].sRid);
				sIds.push_back(RegDists[i].bRid);
				regPairs.push_back(make_uint2(RegDists[i].bRid, RegDists[i].sRid));
			}

		}

	}
	cv::Mat mask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);
	char name[100];
	sprintf(name, "%dregMergeB.jpg", idx);
	cv::imwrite(name, mask);
	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}
	//handle hole
	/*for (int i = 0; i < regions.size(); i++)
	{
	if (regions[i].size == 1 && regions[i].neighbors.size() < 4)
	{
	int minId = 0;
	int nId = regions[i].neighbors[0];
	float minDist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	for (size_t j = 1; j < regions[i].neighbors.size(); j++)
	{
	nId = regions[i].neighbors[j];
	float dist = cv::compareHist(regions[i].colorHist, regions[nId].colorHist, CV_COMP_BHATTACHARYYA);
	if (dist < minDist)
	{
	minDist = dist;
	minId = nId;
	}
	}
	MergeRegions(i, nId, newLabels, spPoses, regions);
	}
	}*/

	/*if (idx == 15)
	{
	cv::Mat segment;
	GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	cv::imwrite("segment.png", segment);
	}*/
	int* segment = new int[img.cols*img.rows];
	GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);

	/*if (idx == 124)*/
	//{
	//	std::vector<std::vector<uint2>> spPoses;
	//	computer.GetSuperpixelPoses(spPoses);
	//	cv::Mat bmask(img.rows,img.cols,CV_8U);
	//	bmask = cv::Scalar(0);
	//	for (size_t i = 0; i < regions.size(); i++)
	//	{
	//		if (regions[i].size == 0)
	//			continue;
	//		for (int j = 0; j < regions[i].borderSpIndices.size(); j++)
	//		{
	//			std::vector<int> borderSpIndices = regions[i].borderSpIndices[j];
	//			for (size_t k = 0; k < borderSpIndices.size(); k++)
	//			{
	//				int id = borderSpIndices[k];
	//				std::vector<uint2> poses = spPoses[id];
	//				for (int m = 0; m < poses.size(); m++)
	//				{
	//					bmask.data[poses[m].x + poses[m].y*img.cols] = 0xff;
	//				}
	//			}
	//			
	//		}
	//		break;
	//	}
	//	sprintf(name, "%dreg0Border.jpg", idx);
	//	cv::imwrite(name, bmask);
	///*	cv::imshow("border", bmask);
	//	cv::waitKey(0);*/
	//}
	cv::Mat rmask;
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
	sprintf(name, "%dregMergeF.jpg", idx);
	cv::imwrite(name, rmask);
	idx++;
	delete[] segment;
}

int Quantize(cv::Mat& img3f, cv::Mat &idx1i, cv::Mat &_color3f, cv::Mat &_colorNum, double ratio, const int clrNums[3])
{
	using namespace cv;
	using namespace std;
	float clrTmp[3] = { clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f };
	int w[3] = { clrNums[1] * clrNums[2], clrNums[2], 1 };

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
			idx[x] = (int)(imgData[0] * clrTmp[0])*w[0] + (int)(imgData[1] * clrTmp[1])*w[1] + (int)(imgData[2] * clrTmp[2]);
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
		int maxDropNum = cvRound(rows * cols * (1 - ratio));
		for (int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
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

int RegionGrowingN(int idx, const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug)
{

	std::vector<RegDist> RegDists;
	float avgColorDist(0);
	float avgHogDist(0);
	float avgSizeDist(0);
	float avgRegSize(0);
	int sum(0);
	int spSize = computer.GetSuperpixelSize();
	for (int i = 0; i < regions.size(); i++)
	{

		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (i < n)
			{
				double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				double hogDist = cv::compareHist(regions[i].hog, regions[n].hog, CV_COMP_BHATTACHARYYA);
				double lbpDist = cv::compareHist(regions[i].lbpHist, regions[n].lbpHist, CV_COMP_BHATTACHARYYA);
				avgColorDist += colorDist;
				avgHogDist += hogDist;
				//double dist = RegionDist(regions[i], regions[n]);
				float borderLen = regions[i].borderPixelNum[j];
				/*float borderLenI = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), 0);
				float borderLenN = std::accumulate(regions[n].borderPixelNum.begin(), regions[n].borderPixelNum.end(), 0);*/
				float borderLenI = regions[i].regCircum;
				float borderLenN = regions[n].regCircum;
				double shapeDist = 1 - (borderLen) / (std::min(borderLenI, borderLenN) + ZERO);
				double sizeDist = exp(std::min(regions[i].size, regions[n].size)*1.0 / spSize);
				double edgeness = regions[i].edgeness[j] / regions[i].borderPixelNum[j];
				double edgeness2 = regions[i].edgeness[j] / regions[i].borders[j];
				//std::cout << edgeness << " edgeness2 " << edgeness2 << " ratio " <<edgeness2/edgeness<<"\n";

				avgSizeDist += sizeDist;
				sum++;
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				rd.colorDist = colorDist;
				rd.shapeDist = shapeDist;
				rd.sizeDist = sizeDist;
				rd.hogDist = hogDist;
				rd.lbpDist = lbpDist;
				rd.edgeness = edgeness2;
				RegDists.push_back(rd);
			}
		}
	}
	avgColorDist /= sum;
	avgHogDist /= sum;
	avgSizeDist /= sum;

	double avgDistSum = avgColorDist + avgHogDist + avgSizeDist;

	//std::cout << idx << ": avgColorDist= " << avgColorDist << ",avgHogDist= " << avgHogDist << ",avgSizeDist= " << avgSizeDist << "\n";
	//std::cout << cw << "," << hw << "," << sw << "\n";
	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(cw, hw, shw, siw));

	int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int RegSize = regions.size() - ZeroReg;
	std::vector<uint2> regPairs;
	std::vector<int> sIds;

	for (size_t i = 0; i < RegDists.size(); i++)
	{
		if (RegDists[i].colorDist > thresholdF)
			break;
		if (regions[RegDists[i].bRid].size > 0 && regions[RegDists[i].sRid].size > 0)
		{
			if (std::find(sIds.begin(), sIds.end(), RegDists[i].sRid) == sIds.end() &&
				std::find(sIds.begin(), sIds.end(), RegDists[i].bRid) == sIds.end())
			{
				sIds.push_back(RegDists[i].sRid);
				sIds.push_back(RegDists[i].bRid);
				regPairs.push_back(make_uint2(RegDists[i].bRid, RegDists[i].sRid));
			}

		}

	}


	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);



	char name[200];
	if (debug)
	{
		CreateDir((char*)outPath);
		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);

		sprintf(name, "%s%dregMergeB.jpg", outPath, idx);
		cv::imwrite(name, mask);
	}
	float maxColorDist(0);
	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
		float colorDist = cv::compareHist(regions[regPairs[i].x].colorHist, regions[regPairs[i].y].colorHist, CV_COMP_BHATTACHARYYA);
		if (colorDist > maxColorDist)
			maxColorDist = colorDist;
	}
	ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	RegSize = regions.size() - ZeroReg;



	if (debug)
	{
		cv::Mat rmask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, false, false);
		sprintf(name, "%s%dregMergeF_%d_%2d.jpg", outPath, idx, RegSize, (int)(maxColorDist * 100));
		cv::imwrite(name, rmask);
	}
	//HandleHoles(idx, img.cols, img.rows, (const char*)outPath, &computer, regions, newLabels, HoleNeighborsNum, HoleSize, true);
	idx++;
	return regPairs.size();
}
void PrepareForRegionGrowing(int spSize, std::vector<SPRegion>& regions, Queue& RegNPairs, std::vector<float>& regAges)
{
	int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int RegSize = regions.size() - ZeroReg;
	regAges.resize(regions.size());
	memset(&regAges[0], 0, sizeof(float)*RegSize);

	for (int i = 0; i < regions.size(); i++)
	{
		if (regions[i].size == 0)
			continue;
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (regions[n].size == 0)
				continue;
			if (i < n)
			{

				//double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				double colorDist = RegionColorDist(regions[i], regions[n]);
				double hogDist = cv::compareHist(regions[i].hog, regions[n].hog, CV_COMP_BHATTACHARYYA);
				double lbpDist = cv::compareHist(regions[i].lbpHist, regions[n].lbpHist, CV_COMP_BHATTACHARYYA);

				float borderLen = regions[i].borderPixelNum[j];

				float borderLenI = regions[i].regCircum;
				float borderLenN = regions[n].regCircum;
				double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
				double sizeDist = (regions[i].size + regions[n].size)*1.0 / spSize;

				double edgeness = regions[i].edgeness[j] / regions[i].borderPixelNum[j];
				double edgeness2 = regions[i].edgeness[j] / regions[i].borders[j];


				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				rd.colorDist = colorDist;
				rd.shapeDist = shapeDist;
				rd.sizeDist = sizeDist;
				rd.hogDist = hogDist;
				rd.lbpDist = lbpDist;
				rd.edgeness = edgeness2;
				rd.sRidAge = 0;
				rd.bRidAge = 0;
				RegNPairs.push(rd);
			}
		}
	}
}

void FastRegionGrowing(int iter, const cv::Mat& img, const char* outPath, SuperpixelComputer& computer, Queue& RegNPairs, std::vector<float>& regAges, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug)
{
	int merged = 0;
	std::vector<std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);
	int spSize = computer.GetSuperpixelSize();

	std::vector<uint2> regPairs;
	while (merged < thresholdF &&RegNPairs.size() > 0)
	{
		RegDist rd = RegNPairs.top();
		RegNPairs.pop();

		if (rd.sRidAge >= regAges[rd.sRid] && rd.bRidAge >= regAges[rd.bRid])
		{

			if (debug)
			{

				regPairs.push_back(make_uint2(rd.sRid, rd.bRid));


			}

			//std::cout << "merging "<<rd.sRid << "," << rd.bRid << std::endl;
			MergeRegions(rd.bRid, rd.sRid, newLabels, spPoses, regions);
			//std::cout << "merged " << rd.sRid << "," << rd.bRid << std::endl;
			merged++;
			regAges[rd.bRid]++;
			regAges[rd.sRid]++;

			for (int j = 0; j < regions[rd.sRid].neighbors.size(); j++)
			{
				int i = rd.sRid;
				int n = regions[rd.sRid].neighbors[j];
				if (regions[n].size == 0)
					continue;


				//double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				double colorDist = RegionColorDist(regions[i], regions[n]);
				double hogDist = cv::compareHist(regions[i].hog, regions[n].hog, CV_COMP_BHATTACHARYYA);
				double lbpDist = cv::compareHist(regions[i].lbpHist, regions[n].lbpHist, CV_COMP_BHATTACHARYYA);

				//double dist = RegionDist(regions[i], regions[n]);
				float borderLen = regions[i].borderPixelNum[j];
				/*float borderLenI = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), 0);
				float borderLenN = std::accumulate(regions[n].borderPixelNum.begin(), regions[n].borderPixelNum.end(), 0);*/
				float borderLenI = regions[i].regCircum;
				float borderLenN = regions[n].regCircum;
				double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
				double sizeDist = (regions[i].size + regions[n].size)*1.0 / spSize;

				double edgeness = regions[i].edgeness[j] / regions[i].borderPixelNum[j];
				double edgeness2 = regions[i].edgeness[j] / regions[i].borders[j];
				//std::cout << edgeness << " edgeness2 " << edgeness2 << " ratio " <<edgeness2/edgeness<<"\n";


				RegDist nrd;
				nrd.sRid = std::min(rd.sRid, n);
				nrd.bRid = std::max(rd.sRid, n);
				nrd.sRidAge = regAges[nrd.sRid];
				nrd.bRidAge = regAges[nrd.bRid];
				nrd.colorDist = colorDist;
				nrd.shapeDist = shapeDist;
				nrd.sizeDist = sizeDist;
				nrd.hogDist = hogDist;
				nrd.lbpDist = lbpDist;
				nrd.edgeness = edgeness2;
				RegNPairs.push(nrd);

			}

		}

	}
	if (debug)
	{
		CreateDir((char*)outPath);
		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);
		char name[200];
		sprintf(name, "%s%dregMergeB.jpg", outPath, iter);
		cv::imwrite(name, mask);

		int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		int RegSize = regions.size() - ZeroReg;


		std::vector<uint2> holeRegs;
		float holeThreshold = iter / 6.0;
		for (size_t i = 0; i < regions.size(); i++)
		{
			if (regions[i].size > 0 && regions[i].size < holeThreshold)
			{
				holeRegs.push_back(make_uint2(i, i));
			}

		}
		cv::Mat rmask;
		//GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, false, false);
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, holeRegs, rmask);
		sprintf(name, "%s%dregMergeF_%d_%2d.jpg", outPath, iter, RegSize, RegSize - holeRegs.size());
		cv::imwrite(name, rmask);
	}

}



void RegionGrowing(int idx, const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug)
{
	std::vector<RegDist> RegDists;
	float avgColorDist(0), minColorDist(255), maxColorDist(0);
	float avgShapeDist(0), minShapeDist(1), maxShapeDist(0);
	float avgSizeDist(0), minSizeDist(1), maxSizeDist(0);
	float avgEdgeDist(0), minEdgeDist(1), maxEdgeDist(0);
	float avgRegSize(0);

	int sum(0);
	int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int RegSize = regions.size() - ZeroReg;
	int spSize = computer.GetSuperpixelSize();

	for (int i = 0; i < regions.size(); i++)
	{
		if (regions[i].size == 0)
			continue;
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (regions[n].size == 0)
				continue;
			if (i < n)
			{

				//double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				double colorDist = RegionColorDist(regions[i], regions[n]);
				/*double hogDist = cv::compareHist(regions[i].hog, regions[n].hog, CV_COMP_BHATTACHARYYA);
				double lbpDist = cv::compareHist(regions[i].lbpHist, regions[n].lbpHist, CV_COMP_BHATTACHARYYA);*/
				avgColorDist += colorDist;
				if (colorDist > maxColorDist)
					maxColorDist = colorDist;
				if (colorDist < minColorDist)
					minColorDist = colorDist;
				//avgHogDist += hogDist;
				//double dist = RegionDist(regions[i], regions[n]);
				float borderLen = regions[i].borderPixelNum[j];
				/*float borderLenI = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), 0);
				float borderLenN = std::accumulate(regions[n].borderPixelNum.begin(), regions[n].borderPixelNum.end(), 0);*/
				float borderLenI = regions[i].regCircum;
				float borderLenN = regions[n].regCircum;
				double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
				avgShapeDist += shapeDist;
				if (shapeDist > maxShapeDist)
					maxShapeDist = shapeDist;
				if (shapeDist < minShapeDist)
					minShapeDist = shapeDist;

				double sizeDist = (regions[i].size + regions[n].size)*1.0 / spSize;
				avgSizeDist += sizeDist;
				if (sizeDist > maxSizeDist)
					maxSizeDist = sizeDist;
				else if (sizeDist < minSizeDist)
					minSizeDist = sizeDist;

				double edgeness = regions[i].edgeness[j] / regions[i].borderPixelNum[j];
				double edgeness2 = regions[i].edgeness[j] / regions[i].borders[j];
				//std::cout << edgeness << " edgeness2 " << edgeness2 << " ratio " <<edgeness2/edgeness<<"\n";
				avgEdgeDist += edgeness2;
				if (edgeness > maxEdgeDist)
					maxEdgeDist = edgeness;
				else if (edgeness < minEdgeDist)
					minEdgeDist = edgeness;


				sum++;
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				rd.oColorDist = rd.colorDist = colorDist;
				rd.shapeDist = shapeDist;
				rd.sizeDist = sizeDist;
				/*	rd.hogDist = hogDist;
				rd.lbpDist = lbpDist;*/
				rd.edgeness = edgeness2;
				RegDists.push_back(rd);
			}
		}
	}
	avgColorDist /= sum;
	avgShapeDist /= sum;
	avgEdgeDist /= sum;
	avgSizeDist /= sum;

	//normalize

	//for (int i = 0; i < RegDists.size(); i++)
	//{
	//	RegDists[i].colorDist = (RegDists[i].colorDist - minColorDist) / (maxColorDist - minColorDist);
	//	RegDists[i].shapeDist = (RegDists[i].shapeDist - minShapeDist) / (maxShapeDist - minShapeDist);
	//	RegDists[i].edgeness = RegDists[i].edgeness / avgEdgeDist;// (RegDists[i].edgeness - minEdgeDist) / (maxEdgeDist - minEdgeDist);
	//	RegDists[i].sizeDist = (RegDists[i].sizeDist - minSizeDist) / (maxSizeDist - minSizeDist);

	//}
	//double avgDistSum = avgColorDist + avgHogDist + avgSizeDist;*/

	//std::cout << idx << ": avgColorD= " << avgColorDist << ",avgShapeD= " << avgShapeDist << ",avgSizeD= " << avgSizeDist <<" avgEdgeD= "<<avgEdgeDist<< "\n";
	//std::cout << cw << "," << hw << "," << sw << "\n";
	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(cw, hw, shw, siw));
	//std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(1 / avgColorDist * 3, 1 / avgEdgeDist * 3, 1 / avgShapeDist * 2, 1 / avgSizeDist * 2));
	//std::cout << RegDists[0] << "\n";


	//int N = std::max(1, (int)(thresholdF*RegDists.size()));
	//N = std::min(RegSize - 5, N);
	int N = thresholdF;

	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);

	maxColorDist = 0;
	std::vector<uint2> regPairs;
	std::vector<int> sIds;
	int MCount(0);

	for (int i = 0; i < RegDists.size() && MCount < N; i++)
	{
		if (regions[RegDists[i].bRid].size > 0 && regions[RegDists[i].sRid].size > 0)
		{
			if (std::find(sIds.begin(), sIds.end(), RegDists[i].sRid) == sIds.end() &&
				std::find(sIds.begin(), sIds.end(), RegDists[i].bRid) == sIds.end())
			{
				sIds.push_back(RegDists[i].sRid);
				sIds.push_back(RegDists[i].bRid);
				MCount++;
				if (maxColorDist < RegDists[i].oColorDist)
				{
					maxColorDist = RegDists[i].oColorDist;
				}
				MergeRegions(RegDists[i].bRid, RegDists[i].sRid, newLabels, spPoses, regions);
				if (debug)
					regPairs.push_back(make_uint2(RegDists[i].bRid, RegDists[i].sRid));
			}

		}

	}



	char name[200];
	if (debug)
	{
		CreateDir((char*)outPath);
		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);

		sprintf(name, "%s%dregMergeB.jpg", outPath, idx);
		cv::imwrite(name, mask);
	}



	/*for (int i = 0; i < regPairs.size(); i++)
	{
	MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}*/

	ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	RegSize = regions.size() - ZeroReg;



	if (debug)
	{
		std::vector<uint2> holeRegs;
		float holeThreshold = idx / 6.0;
		for (size_t i = 0; i < regions.size(); i++)
		{
			if (regions[i].size > 0 && regions[i].size < holeThreshold)
			{
				holeRegs.push_back(make_uint2(i, i));
			}

		}
		cv::Mat rmask;
		//GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, false, false);
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, holeRegs, rmask);
		sprintf(name, "%s%dregMergeF_%d_%2d.jpg", outPath, idx, RegSize, RegSize - holeRegs.size());
		cv::imwrite(name, rmask);
	}

	//HandleHoles(idx, img.cols, img.rows, (const char*)outPath, &computer, regions, newLabels, HoleNeighborsNum, HoleSize, true);
	idx++;
	//std::cout <<"max Color dist "<<maxColorDist << "\n"; 
}

void SalGuidedRegMergion2(const cv::Mat& img, const char* path, std::vector<RegionSalInfo>& regSalInfos, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, bool debug)
{
	static int idx = 0;
	char name[200];
	char outpath[200];
	sprintf(outpath, "%s\\SGGrowing2\\", path);

	std::vector<uint2> regPairs;
	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);

	std::sort(regSalInfos.begin(), regSalInfos.end(), RegionSalDescCmp());
	SPRegion salRegion = regions[regSalInfos[0].id];
	float maxSal = regSalInfos[0].RegionSaliency();
	if (debug)
	{
		std::cout << "region info:\n\t" << regSalInfos[0] << "\n\t" << regSalInfos[1] << "\n";
		std::cout << "w/h " << regions[regSalInfos[1].id].compactness << " compactness " << regSalInfos[1].compactness << "\n";
	}

	if (isNeighbor(regions, regSalInfos[0].id, regSalInfos[1].id) && regSalInfos[1].IsSaliency())
	{
		if (debug)
			std::cout << " adding salient region \n\t";
		regPairs.push_back(make_uint2(regSalInfos[0].id, regSalInfos[1].id));
	}
	else
	{
		if (debug)
			std::cout << " adding background region \n\t";
		//std::cout << regSalInfos[regSalInfos.size()-1].id << "\n\t" << regSalInfos[1] << "\n";
		regPairs.push_back(make_uint2(regSalInfos[regSalInfos.size() - 1].id, regSalInfos[1].id));
	}

	if (debug)
	{
		CreateDir(outpath);
		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);

		sprintf(name, "%s%dFGMergeB.jpg", outpath, idx);
		cv::imwrite(name, mask);
	}

	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}


	if (debug)
	{
		cv::Mat rmask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
		sprintf(name, "%s%dFGMergeF_%d.jpg", outpath, idx, regSalInfos.size() - 1);
		cv::imwrite(name, rmask);
	}
	idx++;
}

void SalGuidedRegMergion(const cv::Mat& img, const char* path, std::vector<RegionSalInfo>& regSalInfos, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, RegionPartition& rp, bool debug)
{
	static int idx = 0;
	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);

	std::vector<uint2> regPairs;
	std::vector<uint2> sregPairs;
	std::vector<int> sIds;
	//regPairs.push_back(make_uint2(regSalInfos[0].id, regSalInfos[1].id));
	//std::cout << "SalGuidedRegMergion" << "\n";
	//for (size_t i = 0; i < regSalInfos.size(); i++)
	//	std::cout << "\tregion " << i << ":" << regSalInfos[i] << "\n";

	std::sort(regSalInfos.begin(), regSalInfos.end(), RegionSalDescCmp());
	regPairs.push_back(make_uint2(regSalInfos[regSalInfos.size() - 1].id, regSalInfos[regSalInfos.size() - 2].id));
	/*if (regSalInfos.size() > 4)
	{
	sregPairs.push_back(make_uint2(regSalInfos[0].id, regSalInfos[1].id));
	}*/
	bool flag(true);
	int bkgRegId = regSalInfos[regSalInfos.size() - 1].id;
	for (size_t i = 0; i < rp.bkgRegIds.size(); i++)
	{
		if (rp.regions[rp.bkgRegIds[i]].id == bkgRegId)
		{
			flag = false;
			break;
		}
	}

	if (flag)
	{
		int k(0);
		for (; k < rp.regions.size(); k++)
		{
			if (rp.regions[k].id == bkgRegId)
			{
				break;
			}
		}
		rp.bkgRegIds.push_back(k);
	}

	flag = true;
	bkgRegId = regSalInfos[regSalInfos.size() - 2].id;
	for (size_t i = 0; i < rp.bkgRegIds.size(); i++)
	{
		if (rp.regions[rp.bkgRegIds[i]].id == bkgRegId)
		{
			flag = false;
			break;
		}
	}

	if (flag)
	{
		int k(0);
		for (; k < rp.regions.size(); k++)
		{
			if (rp.regions[k].id == bkgRegId)
			{
				break;
			}
		}
		rp.bkgRegIds.push_back(k);
	}


	char name[200];
	char outpath[200];

	if (debug)
	{
		sprintf(outpath, "%s\\SGGrowing\\", path);
		CreateDir(outpath);
		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);

		sprintf(name, "%s%dBKMergeB.jpg", outpath, idx);
		cv::imwrite(name, mask);
	}

	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}


	if (debug)
	{
		cv::Mat rmask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
		sprintf(name, "%s%dBKMergeF_%d.jpg", outpath, idx, regSalInfos.size() - 1);
		cv::imwrite(name, rmask);
	}
	idx++;

	if (debug && sregPairs.size() > 0)
	{

		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, sregPairs, mask);

		sprintf(name, "%s%dSMergeB.jpg", outpath, idx);
		cv::imwrite(name, mask);

		for (int i = 0; i < sregPairs.size(); i++)
		{
			MergeRegions(sregPairs[i].x, sregPairs[i].y, newLabels, spPoses, regions);
		}


		if (debug)
		{
			cv::Mat rmask;
			GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
			sprintf(name, "%s%dSMergeF_%d.jpg", outpath, idx, regSalInfos.size() - 2);
			cv::imwrite(name, rmask);
		}
		idx++;
	}
}

void SalGuidedRegMergion(const cv::Mat& img, const char* path, std::vector<RegionSalInfo>& regSalInfos, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, bool debug)
{
	static int idx = 0;
	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);

	std::vector<uint2> regPairs;
	std::vector<uint2> sregPairs;
	std::vector<int> sIds;
	//regPairs.push_back(make_uint2(regSalInfos[0].id, regSalInfos[1].id));

	//std::cout << "SalGuidedRegMergion" << "\n";
	//for (size_t i = 0; i < regSalInfos.size(); i++)
	//	std::cout << "\tregion " << i << ":" << regSalInfos[i] << "\n";

	std::sort(regSalInfos.begin(), regSalInfos.end(), RegionSalDescCmp());
	if (debug)
	{
		std::cout << idx << "----------------------\n";
		for (size_t i = 0; i < regSalInfos.size(); i++)
		{
			std::cout << regSalInfos[i];
			std::cout << "region hist variance " << HistogramVariance(regions[regSalInfos[i].id].colorHist) << "\n\n";
		}

	}
	/*cv::Mat bmask;
	ShowRegionBorder(img, computer, newLabels, regions, regSalInfos, bmask);*/


	regPairs.push_back(make_uint2(regSalInfos[regSalInfos.size() - 1].id, regSalInfos[regSalInfos.size() - 2].id));
	/*if (regSalInfos.size() > 4)
	{
	sregPairs.push_back(make_uint2(regSalInfos[0].id, regSalInfos[1].id));
	}*/


	char name[200];
	char outpath[200];
	sprintf(outpath, "%s\\SGGrowing\\", path);
	if (debug)
	{
		CreateDir(outpath);
		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);

		sprintf(name, "%s%dBKMergeB.jpg", outpath, idx);
		cv::imwrite(name, mask);
	}

	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}




	if (debug)
	{
		float bkBorderRatio = regSalInfos[regSalInfos.size() - 1].borderRatio + regSalInfos[regSalInfos.size() - 2].borderRatio;
		cv::Mat rmask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
		sprintf(name, "%s%dBKMergeF_%d_%d.jpg", outpath, idx, regSalInfos.size() - 1, (int)(bkBorderRatio * 100));
		cv::imwrite(name, rmask);
	}
	idx++;




}
void AllRegionGrowing(const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug)
{
	static int idx = 0;
	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);

	std::vector<RegDist> RegDists;
	float avgColorDist(0);
	float avgHogDist(0);
	float avgSizeDist(0);
	int sum(0);
	int spSize = computer.GetSuperpixelSize();
	//std::cout << idx << ":\n";
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions.size(); j++)
		{
			int n = j;
			if (regions[i].size > 0 && regions[j].size > 0 && i < n)
			{
				float ad2cI = sqrt(sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y));
				float ad2cJ = sqrt(sqr(regions[n].ad2c.x) + sqr(regions[n].ad2c.y));
				float ad2cDis = abs(ad2cI - ad2cJ);
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				float minBorder = abs(regions[i].edgeSpNum - regions[n].edgeSpNum) / (computer.GetSPHeight() + computer.GetSPWidth()) / 2;
				rd.sizeDist = abs(regions[i].size - regions[n].size)*1.0 / spSize;
				double colorDist = cv::compareHist(regions[i].colorHist, regions[n].colorHist, CV_COMP_BHATTACHARYYA);
				double hogDist = cv::compareHist(regions[i].hog, regions[n].hog, CV_COMP_BHATTACHARYYA);
				double lbpDist = cv::compareHist(regions[i].lbpHist, regions[n].lbpHist, CV_COMP_BHATTACHARYYA);
				avgColorDist += colorDist;
				avgHogDist += hogDist;
				float avgcolorDist = L1Distance(regions[i].color, regions[n].color);
				std::vector<int>::iterator itr = std::find(regions[i].neighbors.begin(), regions[i].neighbors.end(), n);
				if (itr != regions[i].neighbors.end())
				{
					float borderLen = regions[i].borderPixelNum[itr - regions[i].neighbors.begin()];
					float borderLenI = regions[i].regCircum;
					float borderLenN = regions[n].regCircum;
					double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
					rd.sizeDist = shapeDist;
				}
				else
				{
					rd.sizeDist = 1.0;
				}
				rd.sizeDist = ad2cDis;
				rd.edgeness = minBorder;
				rd.colorDist = colorDist;
				rd.hogDist = hogDist;
				rd.lbpDist = lbpDist;
				RegDists.push_back(rd);
				//std::cout << rd;
			}
		}
	}
	int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int RegSize = regions.size() - ZeroReg;


	char name[200];
	char path[200];
	sprintf(path, "%s\\AllReggrowing\\", outPath);


	avgColorDist /= sum;
	avgHogDist /= sum;
	avgSizeDist /= sum;
	double cw, hw, shw;
	double avgDistSum = avgColorDist + avgHogDist + avgSizeDist;
	cw = 0.4;
	hw = 0.6;
	shw = 0;
	//std::cout << idx << ": avgColorDist= " << avgColorDist << ",avgHogDist= " << avgHogDist << ",avgSizeDist= " << avgSizeDist << "\n";
	//std::cout << cw << "," << hw << "," << sw << "\n";
	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(cw, hw, shw, 0));




	std::vector<uint2> regPairs;
	std::vector<int> sIds;
	for (int i = 0; i < 1; i++)
	{
		if (regions[RegDists[i].bRid].size > 0 && regions[RegDists[i].sRid].size > 0)
		{
			if (std::find(sIds.begin(), sIds.end(), RegDists[i].sRid) == sIds.end() &&
				std::find(sIds.begin(), sIds.end(), RegDists[i].bRid) == sIds.end())
			{
				sIds.push_back(RegDists[i].sRid);
				sIds.push_back(RegDists[i].bRid);
				regPairs.push_back(make_uint2(RegDists[i].bRid, RegDists[i].sRid));
			}

		}

	}

	if (debug)
	{
		CreateDir(path);
		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);

		sprintf(name, "%s\\%dARegMergeB.jpg", path, idx);
		cv::imwrite(name, mask);
	}

	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}
	RegSize--;

	if (debug)
	{
		cv::Mat rmask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
		sprintf(name, "%s%dARegMergeF_%d.jpg", path, idx, RegSize);
		cv::imwrite(name, rmask);
	}

	idx++;

}
float RegionDistance(std::vector<SPRegion>& regions, int i, int j)
{
	float dx = abs(regions[i].cX - regions[j].cX);
	float dy = abs(regions[i].cY - regions[j].cY);
	return sqrt(dx*dx + dy*dy);
}
void RegionSaliency(int width, int height, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, RegionPartition& regionPartition, std::vector<RegionSalInfo>& regInfos)
{
	float edgeSPNum = (computer->GetSPWidth() + computer->GetSPHeight()) * 2 - 4;
	regInfos.clear();
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{

			//contrast /= regions[i].neighbors.size();
			float ad2c = sqrt(sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y));
			float relSize = regions[i].size*1.0 / computer->GetSuperpixelSize();
			float borderRatio = regions[i].edgeSpNum / edgeSPNum;
			std::vector<cv::Point> borders;
			for (size_t j = 0; j < regions[i].borderPixels.size(); j++)
			{
				for (size_t k = 0; k < regions[i].borderPixels[j].size(); k++)
				{
					borders.push_back(cv::Point(regions[i].borderPixels[j][k].x, regions[i].borderPixels[j][k].y));
				}
			}
			for (size_t j = 0; j < regions[i].borderEdgePixels.size(); j++)
			{
				borders.push_back(regions[i].borderEdgePixels[j]);
			}
			if (borders.size() == 0)
			{
				regions[i].size = 0;
				continue;
			}

			cv::vector<cv::Point> hull;
			cv::convexHull(cv::Mat(borders), hull, false);
			cv::vector<cv::vector<cv::Point>> convexContour;  // Convex hull contour points   
			convexContour.push_back(hull);
			float area = cv::contourArea(convexContour[0]);
			float fill = regions[i].pixels / area;

			RegionSalInfo si;
			si.ad2c = ad2c;

			si.relSize = relSize;
			si.borderRatio = borderRatio;
			si.contrast = 0;
			si.compactness = exp(-sqr(regions[i].compactness - compactnessMean) / compactnessTheta);
			si.id = i;
			si.fillness = fill > 0.4 ? 1 : 0;
			//si.compactness = regions[i].compactness > 0.4 ? 1 : 0;
			//std::cout << si << "\n";
			regInfos.push_back(si);
		}
	}

	std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());


	for (size_t i = 0; i < regInfos.size() - 1; i++)
	{
		SPRegion& region = regions[regInfos[i].id];
		float totalWeight(0);
		float bkgContrast(0);
		for (size_t j = 0; j < regionPartition.bkgRegIds.size(); j++)
		{
			int bkgId = regionPartition.bkgRegIds[j];
			float contrast = cv::compareHist(regionPartition.regions[bkgId].colorHist, regions[regInfos[i].id].colorHist, CV_COMP_BHATTACHARYYA);
			int rid;
			for (size_t k = 0; k < regionPartition.regions.size(); k++)
			{
				if (regionPartition.regions[k].id == regInfos[i].id)
				{
					rid = k;
					break;
				}
			}
			/*float distance = regionPartition.minDistances[rid][bkgId];
			distance = 1.0 / distance;*/
			float distance = 1;
			totalWeight += distance;
			bkgContrast += contrast*distance;

		}
		//std::cout << "region " << regInfos[i].id<<" contrast min id " << minId << "\n";
		regInfos[i].contrast = bkgContrast / totalWeight;
	}


	cv::Mat dbgMap = cv::Mat::zeros(height, width, CV_8UC3);
	char text[20];
	for (int i = 0; i < regInfos.size(); i++)
	{
		bool bkgFlag(false);
		int id = regInfos[i].id;
		for (size_t j = 0; j < regionPartition.bkgRegIds.size(); j++)
		{
			SPRegion& region = regionPartition.regions[regionPartition.bkgRegIds[j]];
			if (region.id == id)
			{
				bkgFlag = true;
				break;
			}


		}

		for (int j = 0; j < regions[id].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[id].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[id].spIndices[j]][k].x;
				int r = spPoses[regions[id].spIndices[j]][k].y;
				if (i == 0)
				{
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = 255;
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = 255;
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = 0;


				}
				else if (bkgFlag)
				{
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = 0;
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = 255;
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = 255;
				}
				else
				{
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = (uchar)(regions[id].color.x);
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = (uchar)(regions[id].color.y);
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = (uchar)(regions[id].color.z);

				}
			}
		}
		sprintf(text, "%d", i);
		int x = regions[id].cX * 16;
		int y = regions[id].cY * 16;
		x = x >= width ? width - 1 : x;
		y = y >= height ? height - 1 : y;
		cv::putText(dbgMap, text, cv::Point(x, y), CV_FONT_ITALIC, 0.8, CV_RGB(255, 215, 0));
	}
	char fileName[200];
	sprintf(fileName, "%s\\regionSal_%d.jpg", outputPath, regInfos.size());
	cv::imwrite(fileName, dbgMap);


}
bool RegionSaliency(int width, int height, HISTOGRAMS& colorHists, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, bool debug)
{
	float edgeSPNum = (computer->GetSPWidth() + computer->GetSPHeight()) * 2 - 4;
	regInfos.clear();
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);
	SLICClusterCenter* centers;
	int* labels;
	int spSize;
	computer->GetSuperpixelResult(spSize, labels, centers);
	bool compactFlag(true);
	float avgAd2c(0), avgContrast(0), avgFill(0);
	float minContrast(0), maxContrast(0);


	float minBorderRatio(1), maxBorderRatio(0);


	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{
			float localContrast = RegionBoxLocalContrast(*computer, nLabels, regions, i, colorHists);
			float ad2c = sqrt(sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y));

			float relSize = regions[i].size*1.0 / computer->GetSuperpixelSize();
			float borderRatio = regions[i].edgeSpNum / edgeSPNum;
			if (borderRatio > maxBorderRatio)
				maxBorderRatio = borderRatio;
			if (borderRatio < minBorderRatio)
				minBorderRatio = borderRatio;
			avgAd2c += (ad2c + borderRatio);


			RegionSalInfo si;
			si.ad2c = ad2c;
			si.relSize = relSize;
			si.borderRatio = borderRatio;
			si.localContrast = localContrast;
			si.id = i;
			if (!regions[i].regFlag)
			{
				std::vector<cv::Point> borders;
				GetRegionBorder(regions[i], borders);
				cv::vector<cv::Point> hull;
				cv::convexHull(cv::Mat(borders), hull, false);
				cv::vector<cv::vector<cv::Point>> convexContour;  // Convex hull contour points   
				convexContour.push_back(hull);
				float area = cv::contourArea(convexContour[0]);
				/*cv::Rect box = cv::boundingRect(borders);
				float area =  cv::boundingRect(borders).area()*/
				regions[i].filleness = regions[i].pixels *1.0 / area;

			}

			if (regions[i].compactness < 0.15)
			{
				compactFlag = false;
				si.compactness = 0;
			}
			else
				si.compactness = 1;

			si.fillness = regions[i].filleness > 0.4 ? 1 : 0;

			avgFill += (si.compactness + si.fillness);
			si.neighbors = regions[i].neighbors;
			si.contrast = 0;
			regInfos.push_back(si);
		}

	}

	std::sort(regInfos.begin(), regInfos.end(), RegionSalBorderCmp());
	int bkgId = regInfos[regInfos.size() - 1].id;

	std::vector<float>& bkgHist = regions[bkgId].colorHist;
	float4 bkgColor = regions[bkgId].color;
	for (size_t i = 0; i < regInfos.size(); i++)
	{
		if (i < regInfos.size() - 1)
		{
			regInfos[i].contrast = RegionColorDist(regions[regInfos[i].id], regions[bkgId]);
			if (regInfos[i].contrast > maxContrast)
			{
				maxContrast = regInfos[i].contrast;
			}

			avgContrast += regInfos[i].contrast;
		}

		regInfos[i].neighRatio = regions[regInfos[i].id].neighbors.size()*1.0 / (regInfos.size() - 1);
	}
	avgContrast /= regInfos.size();
	avgAd2c /= regInfos.size();
	avgFill /= regInfos.size();

	//normalize
	/*for (int i = 0; i < regInfos.size(); i++)
	{
	regInfos[i].contrast = (regInfos[i].contrast - minContrast) / (maxContrast - minContrast);
	regInfos[i].borderRatio = (regInfos[i].borderRatio - minBorderRatio) / (maxBorderRatio - minBorderRatio);
	}*/


	std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());

	if (debug)
	{
		cv::Mat salMap = cv::Mat::zeros(height, width, CV_8U);
		for (int i = 0; i < regInfos.size() - 1; i++)
		{
			int id = regInfos[i].id;
			uchar salV = regInfos[i].contrast * exp(-9.0*regInfos[i].ad2c) * 255;
			for (int j = 0; j < regions[id].spIndices.size(); j++)
			{
				for (int k = 0; k < spPoses[regions[id].spIndices[j]].size(); k++)
				{
					int c = spPoses[regions[id].spIndices[j]][k].x;
					int r = spPoses[regions[id].spIndices[j]][k].y;
					*(char*)(salMap.data + (r*width + c)) = salV;

				}
			}
		}
		cv::normalize(salMap, salMap, 255, 0, CV_MINMAX);
		char fileName[200];
		sprintf(fileName, "%s\\regionSal_%d.jpg", outputPath, regInfos.size());
		cv::imwrite(fileName, salMap);
	}

	//bool borderFlag = (regInfos[regInfos.size() - 1].borderRatio > 0.8 && regInfos.size() < 8) ||
	//	regInfos.size() < 5;
	//return compactFlag && borderFlag;
	return true;
}
void RegionSaliencyL(int width, int height, HISTOGRAMS& colorHists, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, cv::Mat& salMap, bool debug)
{
	float edgeSPNum = (computer->GetSPWidth() + computer->GetSPHeight()) * 2;
	regInfos.clear();
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{
			float localContrast = RegionBoxLocalContrast(*computer, nLabels, regions, i, colorHists);


			float ad2c = sqrt(sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y));
			float relSize = regions[i].size*1.0 / computer->GetSuperpixelSize();
			float borderRatio = regions[i].edgeSpNum / edgeSPNum;



			RegionSalInfo si;
			si.ad2c = ad2c;
			si.relSize = relSize;
			si.borderRatio = borderRatio;
			si.localContrast = localContrast;
			//si.compactness = exp(-sqr(regions[i].compactness - compactnessMean) / compactnessTheta);

			si.compactness = regions[i].compactness;

			si.id = i;
			if (!regions[i].regFlag)
			{
				std::vector<cv::Point> borders;
				GetRegionBorder(regions[i], borders);
				cv::vector<cv::Point> hull;
				cv::convexHull(cv::Mat(borders), hull, false);
				cv::vector<cv::vector<cv::Point>> convexContour;  // Convex hull contour points   
				convexContour.push_back(hull);
				float area = cv::contourArea(convexContour[0]);
				/*cv::Rect box = cv::boundingRect(borders);
				float area =  cv::boundingRect(borders).area()*/
				regions[i].filleness = regions[i].pixels *1.0 / area;

			}

			si.compactness = regions[i].compactness > 0.15 ? 1 : 0;
			si.fillness = regions[i].filleness > 0.4 ? 1 : 0;
			//si.localContrast = localContrast;

			//si.compactness = regions[i].compactness > 0.4 ? 1 : 0;
			//std::cout << si << "\n";

			si.neighbors = regions[i].neighbors;
			si.contrast = 0;
			regInfos.push_back(si);
		}

	}

	std::sort(regInfos.begin(), regInfos.end(), RegionSalBorderCmp());


	std::vector<float>& bkgHist = regions[regInfos[regInfos.size() - 1].id].colorHist;
	float4 bkgColor = regions[regInfos[regInfos.size() - 1].id].color;
	for (size_t i = 0; i < regInfos.size(); i++)
	{
		if (i < regInfos.size() - 1)
		{

			float colorDist = L1Distance(regions[regions[i].id].color, bkgColor) / 255;
			regInfos[i].contrast = cv::compareHist(regions[regInfos[i].id].colorHist, bkgHist, CV_COMP_BHATTACHARYYA);

		}

		regInfos[i].neighRatio = regions[regInfos[i].id].neighbors.size()*1.0 / (regInfos.size() - 1);
	}

	std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());
	salMap = cv::Mat::zeros(height, width, CV_8U);
	for (int i = 0; i < regInfos.size() - 1; i++)
	{
		int id = regInfos[i].id;


		for (int j = 0; j < regions[id].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[id].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[id].spIndices[j]][k].x;
				int r = spPoses[regions[id].spIndices[j]][k].y;
				*(char*)(salMap.data + (r*width + c)) = 0xff;

			}
		}
	}


}


void RegionSaliency(int width, int height, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, cv::Mat& salMap, bool debug)
{
	float edgeSPNum = (computer->GetSPWidth() + computer->GetSPHeight()) * 2;
	regInfos.clear();
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{
			float localContrast(1);
			for (size_t j = 0; j < regions[i].neighbors.size(); j++)
			{
				int nid = regions[i].neighbors[j];
				float c = cv::compareHist(regions[i].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);
				if (c < localContrast)
					localContrast = c;
			}

			//contrast /= regions[i].neighbors.size();
			float ad2c = sqrt(sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y));
			float relSize = regions[i].size*1.0 / computer->GetSuperpixelSize();
			float borderRatio = regions[i].edgeSpNum / edgeSPNum;
			std::vector<cv::Point> borders;
			for (size_t j = 0; j < regions[i].borderPixels.size(); j++)
			{
				for (size_t k = 0; k < regions[i].borderPixels[j].size(); k++)
				{
					borders.push_back(cv::Point(regions[i].borderPixels[j][k].x, regions[i].borderPixels[j][k].y));
				}
			}
			for (size_t j = 0; j < regions[i].borderEdgePixels.size(); j++)
			{
				borders.push_back(regions[i].borderEdgePixels[j]);
			}
			if (borders.size() == 0)
			{
				regions[i].size = 0;
				continue;
			}

			cv::vector<cv::Point> hull;
			cv::convexHull(cv::Mat(borders), hull, false);
			cv::vector<cv::vector<cv::Point>> convexContour;  // Convex hull contour points   
			convexContour.push_back(hull);
			float area = cv::contourArea(convexContour[0]);
			float fill = regions[i].pixels / area;

			RegionSalInfo si;
			si.ad2c = ad2c;
			si.relSize = relSize;
			si.borderRatio = borderRatio;
			si.contrast = 0;
			//si.compactness = exp(-sqr(regions[i].compactness - compactnessMean) / compactnessTheta);
			si.compactness = regions[i].compactness > 0.15 ? 1 : 0;
			si.id = i;
			si.localContrast = localContrast;
			si.fillness = fill > 0.4 ? 1 : 0;
			//si.compactness = regions[i].compactness > 0.4 ? 1 : 0;
			//std::cout << si << "\n";

			si.neighbors = regions[i].neighbors;
			regInfos.push_back(si);
		}

	}
	std::sort(regInfos.begin(), regInfos.end(), RegionSalBorderCmp());
	std::vector<float>& bkgHist = regions[regInfos[regInfos.size() - 1].id].colorHist;
	float4 bkgColor = regions[regInfos[regInfos.size() - 1].id].color;
	for (size_t i = 0; i < regInfos.size(); i++)
	{
		if (i < regInfos.size() - 1)
		{

			float colorDist = L1Distance(regions[regions[i].id].color, bkgColor) / 255;
			regInfos[i].contrast = cv::compareHist(regions[regInfos[i].id].colorHist, bkgHist, CV_COMP_BHATTACHARYYA);
		}

		regInfos[i].neighRatio = regions[regInfos[i].id].neighbors.size()*1.0 / (regInfos.size() - 1);
	}

	std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());
	salMap = cv::Mat::zeros(height, width, CV_8U);
	for (int i = 0; i < regInfos.size() - 1; i++)
	{
		int id = regInfos[i].id;


		for (int j = 0; j < regions[id].spIndices.size(); j++)
		{
			for (int k = 0; k < spPoses[regions[id].spIndices[j]].size(); k++)
			{
				int c = spPoses[regions[id].spIndices[j]][k].x;
				int r = spPoses[regions[id].spIndices[j]][k].y;
				*(char*)(salMap.data + (r*width + c)) = 0xff;

			}
		}
	}
	if (debug)
	{
		cv::Mat dbgMap = cv::Mat::zeros(height, width, CV_8UC3);
		char text[20];
		for (int i = 0; i < regInfos.size(); i++)
		{
			int id = regInfos[i].id;


			for (int j = 0; j < regions[id].spIndices.size(); j++)
			{
				for (int k = 0; k < spPoses[regions[id].spIndices[j]].size(); k++)
				{
					int c = spPoses[regions[id].spIndices[j]][k].x;
					int r = spPoses[regions[id].spIndices[j]][k].y;
					if (i == 0)
					{
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = 255;
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = 255;
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = 0;


					}
					/*else if (i == 1)
					{
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = 0;
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = 255;
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = 255;
					}*/
					else
					{
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = (uchar)(regions[id].color.x);
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = (uchar)(regions[id].color.y);
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = (uchar)(regions[id].color.z);

					}
				}
			}
			sprintf(text, "%d", i);
			int x = regions[id].cX * 16;
			int y = regions[id].cY * 16;
			x = x >= width ? width - 1 : x;
			y = y >= height ? height - 1 : y;
			cv::putText(dbgMap, text, cv::Point(x, y), CV_FONT_ITALIC, 0.8, CV_RGB(255, 215, 0));
		}
		char fileName[200];
		sprintf(fileName, "%s\\regionSal_%d.jpg", outputPath, regInfos.size());
		cv::imwrite(fileName, dbgMap);

	}


	//std::sort(regInfos.begin(), regInfos.end(), RegionSalBorderCmp());
}

void RegionSaliency(int width, int height, const char* outputPath, SuperpixelComputer* computer, std::vector<int>&nLabels, std::vector<SPRegion>& regions, std::vector<RegionSalInfo>& regInfos, bool debug)
{
	float edgeSPNum = (computer->GetSPWidth() + computer->GetSPHeight()) * 2;
	regInfos.clear();
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{
			float localContrast(1);
			for (size_t j = 0; j < regions[i].neighbors.size(); j++)
			{
				int nid = regions[i].neighbors[j];
				float c = cv::compareHist(regions[i].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);
				if (c < localContrast)
					localContrast = c;
			}

			float ad2c = sqrt(sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y));
			float relSize = regions[i].size*1.0 / computer->GetSuperpixelSize();
			float borderRatio = regions[i].edgeSpNum / edgeSPNum;
			std::vector<cv::Point> borders;
			for (size_t j = 0; j < regions[i].borderPixels.size(); j++)
			{
				for (size_t k = 0; k < regions[i].borderPixels[j].size(); k++)
				{
					borders.push_back(cv::Point(regions[i].borderPixels[j][k].x, regions[i].borderPixels[j][k].y));
				}
			}
			for (size_t j = 0; j < regions[i].borderEdgePixels.size(); j++)
			{
				borders.push_back(regions[i].borderEdgePixels[j]);
			}
			if (borders.size() == 0)
			{
				regions[i].size = 0;
				continue;
			}

			cv::vector<cv::Point> hull;
			cv::convexHull(cv::Mat(borders), hull, false);
			cv::vector<cv::vector<cv::Point>> convexContour;  // Convex hull contour points   
			convexContour.push_back(hull);
			float area = cv::contourArea(convexContour[0]);
			float fill = regions[i].pixels / area;

			RegionSalInfo si;
			si.ad2c = ad2c;
			si.relSize = relSize;
			si.borderRatio = borderRatio;
			si.contrast = 0;
			//si.compactness = exp(-sqr(regions[i].compactness - compactnessMean) / compactnessTheta);

			si.id = i;
			si.localContrast = localContrast;

			//si.compactness = regions[i].compactness > 0.4 ? 1 : 0;
			//std::cout << si << "\n";
			if (regions[i].regFlag)
			{
				si.compactness = 1;
				si.fillness = 1;
			}
			else
			{
				si.fillness = fill > 0.4 ? 1 : 0;
				si.compactness = regions[i].compactness > 0.15 ? 1 : 0;
			}
			regInfos.push_back(si);
		}

	}
	std::sort(regInfos.begin(), regInfos.end(), RegionSalBorderCmp());
	std::vector<float>& bkgHist = regions[regInfos[regInfos.size() - 1].id].colorHist;
	float4 bkgColor = regions[regInfos[regInfos.size() - 1].id].color;
	for (size_t i = 0; i < regInfos.size(); i++)
	{
		if (i < regInfos.size() - 1)
		{

			float colorDist = L1Distance(regions[regions[i].id].color, bkgColor) / 255;
			regInfos[i].contrast = cv::compareHist(regions[regInfos[i].id].colorHist, bkgHist, CV_COMP_BHATTACHARYYA);
		}

		regInfos[i].neighRatio = regions[regInfos[i].id].neighbors.size()*1.0 / (regInfos.size() - 1);
	}

	std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());

	if (debug)
	{
		cv::Mat dbgMap = cv::Mat::zeros(height, width, CV_8UC3);
		char text[20];
		for (int i = 0; i < regInfos.size(); i++)
		{
			int id = regInfos[i].id;


			for (int j = 0; j < regions[id].spIndices.size(); j++)
			{
				for (int k = 0; k < spPoses[regions[id].spIndices[j]].size(); k++)
				{
					int c = spPoses[regions[id].spIndices[j]][k].x;
					int r = spPoses[regions[id].spIndices[j]][k].y;
					if (i == 0)
					{
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = 255;
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = 255;
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = 0;


					}
					/*else if (i == 1)
					{
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = 0;
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = 255;
					((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = 255;
					}*/
					else
					{
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 0] = (uchar)(regions[id].color.x);
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 1] = (uchar)(regions[id].color.y);
						((uchar *)(dbgMap.data + r*dbgMap.step.p[0]))[c*dbgMap.step.p[1] + 2] = (uchar)(regions[id].color.z);

					}
				}
			}
			sprintf(text, "%d", i);
			int x = regions[id].cX * 16;
			int y = regions[id].cY * 16;
			x = x >= width ? width - 1 : x;
			y = y >= height ? height - 1 : y;
			cv::putText(dbgMap, text, cv::Point(x, y), CV_FONT_ITALIC, 0.8, CV_RGB(255, 215, 0));
		}
		char fileName[200];
		sprintf(fileName, "%s\\regionSal_%d.jpg", outputPath, regInfos.size());
		cv::imwrite(fileName, dbgMap);

	}


	//std::sort(regInfos.begin(), regInfos.end(), RegionSalBorderCmp());
}