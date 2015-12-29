﻿#include "RegionMerging.h"
#include <fstream>
#include <time.h>       /* time */
#include <numeric>
#include "DistanceUtils.h"
#include "Dijkstra.h"
#include "LBP.h"
#include "Common.h"


const float compactnessTheta = 0.4;
const float compactnessMean = 0.7;
double sqr(double a)
{
	return a*a;
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
	//��ǰ�±�ǩ
	int curLabel(0);
	int imgSize = spWidth*spHeight;
	char* visited = new char[imgSize];
	memset(visited, 0, imgSize);
	memset(segmented, 0, sizeof(int)*width*height);
	std::vector<cv::Point2i> neighbors;
	float4 regMean;
	std::vector<int> singleLabels;
	//region growing �����label
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
	//	//�Ե��������أ��������Χ�ǻ��е���������
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
	//��ǰ�±�ǩ
	int curLabel(0);
	int imgSize = spWidth*spHeight;
	char* visited = new char[imgSize];
	memset(visited, 0, imgSize);
	memset(segmented, 0, sizeof(int)*width*height);
	std::vector<cv::Point2i> neighbors;
	float4 regMean;
	std::vector<int> singleLabels;
	//region growing �����label
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
	//	//�Ե��������أ��������Χ�ǻ��е���������
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
	//����ƽ�����ڳ����ؾ���֮��ľ���
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
	//��ǰ�±�ǩ
	int curLabel(0);
	memset(_visited, 0, _spSize);
	memset(segmented, 0, sizeof(int)*_spSize);
	std::vector<int> singleLabels;
	//region growing �����label



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
		//����Ƿ���Ժϲ���������������ȥ
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
		//			//�ϲ�����������
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
		//			//������
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
		//������
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
	//����ƽ�����ڳ����ؾ���֮��ľ���
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
	//��ǰ�±�ǩ
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
			std::vector<uint2>& pixels = regions[i].borderPixels[j];
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
void GetRegionPixelBorder(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int* segment)
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
								uint2 pixel = spPoses[spIdx][pix];
								for (size_t pn = 0; pn < 4; pn++)
								{
									int y = pixel.y + dy4[pn];
									if (y<0 || y >= height)
										continue;
									int x = pixel.x + dx4[pn];
									if (x >= 0 && x < width)
									{
										if (segment[y*width + x] == labelJ)
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
		regions[i].compactness = std::min(width, height) / std::max(width, height);
		int step = computer->GetSuperpixelStep();
		minX = (minX - 0.5)*step;
		maxX = (maxX + 1)*step;
		maxY = (maxY + 1)*step;
		minY = (minY - 0.5)*step;
		minX = std::max(0.f, minX);
		minY = std::max(0.f, minY);
		maxX = std::min(width*1.f, maxX);
		maxY = std::min(height*1.f, maxY);

		float width = maxX - minX;
		float height = maxY - minY;


		regions[i].Bbox = cv::Rect(minX, minY, width, height);

	}
}
void UpdateRegionInfo(int width, int height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int* segment)
{
	GetRegionSegment(width, height, computer, nLabels, segment);
	//GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	GetRegionPixelBorder(width, height, computer, nLabels, regions, segment);


}
void UpdateRegionInfo(int width, int height, SuperpixelComputer* computer, const cv::Mat& gradMap, const cv::Mat& scaleMap, const cv::Mat& edgemap, std::vector<int>& nLabels, std::vector<SPRegion>& regions, int * segment)
{
	GetRegionSegment(width, height, computer, nLabels, segment);
	//GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	GetRegionPixelBorder(width, height, computer, nLabels, regions, segment);

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
void GetRegionMap(int _width, int _height, SuperpixelComputer* computer, std::vector<int>& nLabels, std::vector<SPRegion>& regions, std::vector<uint2>& regParis, cv::Mat& mask)
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
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = 0xff;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = 0;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = 0;
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
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 0] = 0xff;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 1] = 0;
					((uchar *)(mask.data + r*mask.step.p[0]))[c*mask.step.p[1] + 2] = 0;
				}

			}
		}
	}
	delete[] pixSeg;
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
	for (int b = 0; b< regions[nRegId].colorHist.size(); b++)
	{
		regions[nRegId].colorHist[b] += regions[i].colorHist[b];
	}
	for (int b = 0; b< regions[nRegId].hog.size(); b++)
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
		//��i�������ھ�n������n���ھ���ɾ��i
		std::vector<int>::iterator itr = std::find(regNeighbors[Idx].begin(),
			regNeighbors[Idx].end(), i);
		if (itr != regNeighbors[Idx].end())
			regNeighbors[Idx].erase(itr);
		//��n���ھ��м���regId
		if (reg.id != nRegId && std::find(regNeighbors[Idx].begin(), regNeighbors[Idx].end(), nRegId) == regNeighbors[Idx].end())
			regNeighbors[Idx].push_back(nRegId);
		//�ںϲ��������nRegId���ھ��м���n
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
	for (int b = 0; b< regions[nRegId].colorHist.size(); b++)
	{
		regions[nRegId].colorHist[b] += regions[i].colorHist[b];
	}
	for (int b = 0; b< regions[nRegId].hog.size(); b++)
	{
		regions[nRegId].hog[b] += regions[i].hog[b];
	}
	regions[i].size = 0;
}


void MergeRegions(int i, int j,
	std::vector<int>& newLabels,
	std::vector<std::vector<uint2>>& spPoses,
	std::vector<SPRegion>& regions)
{
	//std::cout << "merge " << i << " to " << j << "\n";

	for (size_t n = 0; n < regions[i].neighbors.size(); n++)
	{
		SPRegion& reg = regions[regions[i].neighbors[n]];
		int Idx = regions[i].neighbors[n];
		//��i�������ھ�n�����ھ���ɾ��i
		std::vector<int>::iterator itr = std::find(reg.neighbors.begin(),
			reg.neighbors.end(), i);
		if (itr != reg.neighbors.end())
			reg.neighbors.erase(itr);
		//���ھ��м���j
		if (reg.id != j && std::find(reg.neighbors.begin(), reg.neighbors.end(), j) == reg.neighbors.end())
			reg.neighbors.push_back(j);
		//�ںϲ��������nRegId���ھ��м���n
		if (reg.id != j && std::find(regions[j].neighbors.begin(), regions[j].neighbors.end(), reg.id) == regions[j].neighbors.end())
			regions[j].neighbors.push_back(reg.id);
	}
	regions[i].neighbors.clear();
	int size0 = regions[j].size;
	int size1 = regions[i].size;
	regions[j].color = (regions[j].color * size0 + regions[i].color * size1)*(1.0 / (size0 + size1));
	regions[j].cX = (regions[j].cX * size0 + regions[i].cX * size1)*(1.0 / (size0 + size1));
	regions[j].cY = (regions[j].cY * size0 + regions[i].cY * size1)*(1.0 / (size0 + size1));
	for (size_t s = 0; s < regions[i].spIndices.size(); s++)
	{
		regions[j].spIndices.push_back(regions[i].spIndices[s]);
		newLabels[regions[i].spIndices[s]] = j;
	}
	for (int b = 0; b< regions[j].colorHist.size(); b++)
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
	cv::normalize(regions[j].lbpHist, regions[j].lbpHist, 1, 0, cv::NORM_L1);
	regions[j].size = size0 + size1;
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
		//��i�������ھ�n������n���ھ���ɾ��i
		std::vector<int>::iterator itr = std::find(regNeighbors[idx].begin(),
			regNeighbors[idx].end(), i);
		if (itr != regNeighbors[idx].end())
			regNeighbors[idx].erase(itr);
		//��n���ھ��м���regId
		if (reg.id != nRegId && std::find(regNeighbors[idx].begin(), regNeighbors[idx].end(), nRegId) == regNeighbors[idx].end())
			regNeighbors[idx].push_back(nRegId);
		//�ںϲ��������nRegId���ھ��м���n
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
	//����ն����򣬽���ϲ�����ӽ����ھ���
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
	std::cout << "merge regions " << i << " " << j << std::endl;
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
	//���ϲ�����ھ��ǲ����ھӱ�����
	int minN = 2;
	for (size_t n = 0; n < INeighbors.size(); n++)
	{

		int nSize = regions[INeighbors[n]].neighbors.size();
		std::cout << INeighbors[n] << " neighbors " << nSize << std::endl;
		if (regions[INeighbors[n]].size>0 && regions[INeighbors[n]].size<HoleSize && nSize > 0 && nSize < minN)
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
				float borderLenN = regions[n].regCircum;
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

void SmartHoleHandling(int width, int height, const char* outPath, SuperpixelComputer* computer, std::vector<SPRegion>& regions, std::vector<int>& newLabels, int holeNThreshold, int holeSizeThreshold, bool debug = false)
{
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
			stack.push_back(i);
			//merge all the hole regions that connected to i to a new region
			while (!stack.empty())
			{
				int id = *stack.begin();
				stack.pop_front();
				for (size_t j = 0; j < regions[id].neighbors.size(); j++)
				{
					int nid = regions[i].neighbors[j];
					if (regions[nid].size > 0 && regions[nid].size < holeNThreshold)
					{
						stack.push_back(nid);
						MergeRegions(nid, id, newLabels, spPoses, regions);
					}
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

					double colorDist = cv::compareHist(regions[i].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);

					//double dist = RegionDist(regions[i], regions[n]);
					float borderLen = regions[i].borderPixelNum[n];
					/*float borderLenI = std::accumulate(regions[i].borderPixelNum.begin(), regions[i].borderPixelNum.end(), 0);
					float borderLenN = std::accumulate(regions[n].borderPixelNum.begin(), regions[n].borderPixelNum.end(), 0);*/
					float borderLenI = regions[i].regCircum;
					float borderLenN = regions[n].regCircum;
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
					char name[300];
					cv::Mat mask;
					//std::cout << "merge regions " << i << " " << j << std::endl;
					std::vector<uint2> spParis;
					spParis.push_back(make_uint2(i, j));

					GetRegionMap(width, height, computer, newLabels, regions, spParis, mask);

					sprintf(name, "%s%dregMergeH%d_%d.jpg", outPath, idx, i, j);
					cv::imwrite(name, mask);
				}

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
	//����ն��������С������ϲ�����ӽ����ھ���
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
	//���ϲ�����ھ��ǲ����ھӱ�����
	int minN = 2;
	for (size_t n = 0; n < INeighbors.size(); n++)
	{

		int nSize = regions[INeighbors[n]].neighbors.size();
		//std::cout << INeighbors[n] << " neighbors " << nSize << std::endl;
		if ((regions[INeighbors[n]].size>0 && nSize > 0 && nSize <= minN && regions[INeighbors[n]].size<HoleSize))
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
	//����ն����򣬽���ϲ�����ӽ����ھ���
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
	//���ϲ�����ھ��ǲ����ھӱ�����
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
	//����ն����򣬽���ϲ�����ӽ����ھ���
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
	//���ϲ�����ھ��ǲ����ھӱ�����
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
	//���������ھ���С��minSize�����򣬽���������С����һ��ϲ�����������
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
			//����ն����򣬽���ϲ�����ӽ����ھ���
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

	//��������ͼ����ھ�
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
	//���������ھ���С��minSize�����򣬽���������С����һ��ϲ�����������
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
			//����ն����򣬽���ϲ�����ӽ����ھ���
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

		//Ѱ�Ҿ�������ı������򣨾��뱳�������Ե��
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

//@idxImg ÿ���������������ɫid
//@colorNum ���������ɫ��
void BuildQHistorgram(const cv::Mat& idxImg, int colorNum, SuperpixelComputer* computer, HISTOGRAMS& colorHists)
{
	colorHists.clear();
	int _width = idxImg.cols;
	int _height = idxImg.rows;

	//����ÿ������������Χ�����صĲ��
	int spHeight = computer->GetSPHeight();
	int spWidth = computer->GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int _spSize(0);
	computer->GetSuperpixelResult(_spSize, labels, centers);
	colorHists.resize(_spSize);

	//ÿ���������а����������Լ�λ��
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

	//����ÿ������������Χ�����صĲ��
	int spHeight = computer->GetSPHeight();
	int spWidth = computer->GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int _spSize(0);
	computer->GetSuperpixelResult(_spSize, labels, centers);
	_colorHists.resize(_spSize);
	_HOGs.resize(_spSize);
	//ÿ���������а����������Լ�λ��
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

	//����ÿ������������Χ�����صĲ��
	int spHeight = computer->GetSPHeight();
	int spWidth = computer->GetSPWidth();
	int* labels;
	SLICClusterCenter* centers = NULL;
	int _spSize(0);
	computer->GetSuperpixelResult(_spSize, labels, centers);
	_colorHists.resize(_spSize);
	_HOGs.resize(_spSize);
	lbpHists.resize(_spSize);

	//ÿ���������а����������Լ�λ��
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

	for (size_t i = 0; i <regions.size(); i++)
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
	//ѡ���������������������Ҳ�������Ե������
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
	//������������ѡ��ratio�����������򣨰�����������߽����صĶ��ٽ�������
	std::vector<SPRegion> tmpRegions(regions);
	std::sort(tmpRegions.begin(), tmpRegions.end(), RegionWSizeDescCmp());
	std::vector<std::vector<uint2>> spPoses;
	computer->GetSuperpixelPoses(spPoses);

	//GetRegionMap(width, height, computer, nLabels, regions, salMap);
	salMap.create(height, width, CV_8U);
	salMap = cv::Scalar(0);
	int n = (int)(regSize*ratio + 0.5);
	int salReg(0);
	for (size_t i = 0; i < tmpRegions.size() && salReg<n; i++)
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
	float maxColorDist(0);
	float maxAd2cDist(0);
	float avgColorDist(0);
	int avgColorDistSize(0);
	float avgAd2cDis(0);
	float minDist(0);
	float threshold(0.4);
	int idx(0);
	for (size_t i = 0; i < regInfos.size() - 1; i++)
	{
		int id = regInfos[i].id;
		for (size_t j = i + 1; j < regInfos.size() - 1; j++)
		{
			int nid = regInfos[j].id;

			float colorDist = cv::compareHist(regions[id].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);
			avgColorDist += colorDist;
			avgColorDistSize++;
			float ad2cI = sqrt(sqr(regions[id].ad2c.x) + sqr(regions[id].ad2c.y));
			float ad2cJ = sqrt(sqr(regions[nid].ad2c.x) + sqr(regions[nid].ad2c.y));
			float ad2cDis = abs(ad2cI - ad2cJ);
			avgAd2cDis += ad2cDis;
		}
	}
	avgColorDist /= avgColorDistSize;
	avgAd2cDis /= avgColorDistSize;
	float wc = 1;
	float wa = 0.5;
	threshold = 0.65;
	if (debug)
		std::cout << "avgColorDist = " << avgColorDist << "\n";
	threshold = avgColorDist*0.8;
	std::cout << "threshold=" << threshold << "\n";

	std::vector<std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);

	char name[200];
	cv::Mat rmask;
	//�����ڵ�
	while (regInfos.size() > 2 && minDist < threshold)
	{

		std::vector<RegDist> RegDists;
		for (size_t i = 0; i < regInfos.size(); i++)
		{
			int id = regInfos[i].id;
			for (size_t j = i + 1; j < std::min(regInfos.size(), i + regInfos.size() / 2); j++)
			{
				int nid = regInfos[j].id;

				RegDist rd;
				rd.sRid = id;
				rd.bRid = nid;
				if ((regions[id].edgeSpNum < 1 && regions[nid].edgeSpNum < 1) ||
					(regions[id].edgeSpNum > 1 && regions[nid].edgeSpNum > 1))
				{
					rd.colorDist = cv::compareHist(regions[id].colorHist, regions[nid].colorHist, CV_COMP_BHATTACHARYYA);
					//rd.edgeness = regions[id].borders[j] * 1.0 / std::min(regions[id].bor);
					float ad2cI = sqrt(sqr(regions[id].ad2c.x) + sqr(regions[id].ad2c.y));
					float ad2cJ = sqrt(sqr(regions[nid].ad2c.x) + sqr(regions[nid].ad2c.y));
					float ad2cDis = abs(ad2cI - ad2cJ);
					rd.edgeness = ad2cDis;
					RegDists.push_back(rd);
					if (rd.colorDist > maxColorDist)
						maxColorDist = rd.colorDist;
					if (rd.edgeness > maxAd2cDist)
						maxAd2cDist = rd.edgeness;
				}

			}
		}
		if (RegDists.size() == 0)
			return;
		std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer(wc, wa, 0, 0));
		minDist = RegDists[0].colorDist*wc + RegDists[0].edgeness*wa;
		//std::cout << "minDist color " << RegDists[0].colorDist << " ad2c " << RegDists[0].edgeness << " minDist " << minDist << "\n";
		float ad2cI = sqrt(sqr(regions[RegDists[0].sRid].ad2c.x) + sqr(regions[RegDists[0].sRid].ad2c.y));
		float ad2cJ = sqrt(sqr(regions[RegDists[0].bRid].ad2c.x) + sqr(regions[RegDists[0].bRid].ad2c.y));
		//std::cout << "ad2c_" << RegDists[0].sRid << " " << ad2cI << " ad2c_" << RegDists[0].bRid << " " << ad2cJ << "\n";
		std::vector<uint2> pair;
		pair.push_back(make_uint2(RegDists[0].sRid, RegDists[0].bRid));


		if (debug)
		{
			GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, pair, rmask);
			sprintf(name, "%s%dHOccMerging_%d_%d_Region_%2d_%d.jpg", outPath, idx, RegDists[0].sRid, RegDists[0].bRid, (int)(minDist * 100), regInfos.size());

			cv::imwrite(name, rmask);
		}

		if (minDist > threshold)
			break;
		MergeRegions(RegDists[0].sRid, RegDists[0].bRid, newLabels, spPoses, regions);

		if (debug)
		{
			GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, 1);
			sprintf(name, "%s%dHOccMerged_%d_%d_Region_%2d_%d.jpg", outPath, idx, RegDists[0].sRid, RegDists[0].bRid, (int)(minDist * 100), regInfos.size());
			cv::imwrite(name, rmask);

		}
		idx++;

		UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, regions, segment);
		RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, debug);
	}
	GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, 1);
	sprintf(name, "%sOccHandled_%d.jpg", outPath, regInfos.size());
	cv::imwrite(name, rmask);
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

	//�����Ե
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
	//		//����ն����򣬽���ϲ�����ӽ����ھ���
	//		HandleHole(i, newLabels, spPoses, regions);
	//		//HandleHoleDemo(width, height, i, &computer, spPoses, newLabels, regions);
	//	}
	//}

	int iter(0);
	while (RegSize > regThreshold)
	{
		UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, regions, segment);
		GetRegionEdgeness(edgeMap, regions);

		//UpdateRegionInfo(img.cols, img.rows, &computer, gradMap, scaleMap, edgeMap, newLabels, regions, segment);
		int needToMerge = (RegSize - regThreshold) / 2;
		needToMerge = std::max(1, needToMerge);
		RegionGrowing(iter++, img, outPath, edgeMap, computer, newLabels, regions, needToMerge, debug);


		//for (size_t i = 0; i < regions.size(); i++)
		//{
		//	if (regions[i].size > 0 && regions[i].neighbors.size() <= HoleNeighborsNum && regions[i].size < HoleSize)
		//	{
		//		int regId = regions[i].id;
		//		//����ն����򣬽���ϲ�����ӽ����ھ���
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
	//		//����ն����򣬽���ϲ�����ӽ����ھ���
	//		holeRegNum += HandleHole(i, newLabels, spPoses, regions);
	//		//HandleHoleDemo(width, height, i, &computer, spPoses, newLabels, regions);
	//	}
	//}
	ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	RegSize = regions.size() - ZeroReg;
	//RegSize -= holeRegNum;

	PickSaliencyRegion(img.cols, img.rows, &computer, newLabels, regions, sal1, 0.6);
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
	UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, regions, segment);
	RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, debug);

	HandleOcculusion(img, computer, outPath, newLabels, regInfos, regions, segment, debug);
	UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, regions, segment);
	//UpdateRegionInfo(img.cols, img.rows, &computer, gradMap, scaleMap, edgeMap, newLabels, regions, segment);
	if (debug)
	{
		RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, debug);
		std::cout << "After Occulusion handling" << regInfos.size() << " regions remains\n";
		std::sort(regInfos.begin(), regInfos.end(), RegionSalDescCmp());
		for (size_t i = 0; i < regInfos.size(); i++)
			std::cout << regInfos[i] << "\n";
	}
	//������ʱ��״̬
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
	while (regInfos.size()>2)
	{
		cv::Mat salMap;
		SalGuidedRegMergion(img, (char*)outPath, regInfos, computer, newLabels, regions, debug);
		UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, regions, segment);
		RegionSaliency(img.cols, img.rows, outPath, &computer, newLabels, regions, regInfos, salMap, debug);
		if (regInfos.size() < 8 || borderRatio > 0.75)
			salMaps.push_back(salMap.clone());
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


		UpdateRegionInfo(img.cols, img.rows, &computer, newLabels, regions, segment);
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
	GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	//GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	GetRegionPixelBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
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
				//�����ز�������ϲ��Ĵ��������������ز���������࣬�����Գ����ز����������
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
	GetRegionSegment(img.cols, img.rows, &computer, newLabels, segment);
	//GetRegionBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
	GetRegionPixelBorder(img.cols, img.rows, &computer, newLabels, regions, segment);
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


void RegionGrowing(int idx, const cv::Mat& img, const char* outPath, const cv::Mat& edgeMap, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, float thresholdF, bool debug)
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
				double shapeDist = 1 - (borderLen) / std::min(borderLenI, borderLenN);
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

	//int N = std::max(1, (int)(thresholdF*RegDists.size()));
	//N = std::min(RegSize - 5, N);
	int N = thresholdF;

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
	char name[200];
	if (debug)
	{
		CreateDir((char*)outPath);
		cv::Mat mask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, regPairs, mask);

		sprintf(name, "%s%dregMergeB.jpg", outPath, idx);
		cv::imwrite(name, mask);
	}

	for (int i = 0; i < regPairs.size(); i++)
	{
		MergeRegions(regPairs[i].x, regPairs[i].y, newLabels, spPoses, regions);
	}
	ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	RegSize = regions.size() - ZeroReg;



	if (debug)
	{
		cv::Mat rmask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask, false, false);
		sprintf(name, "%s%dregMergeF_%d.jpg", outPath, idx, RegSize);
		cv::imwrite(name, rmask);
	}
	HandleHoles(idx, img.cols, img.rows, (const char*)outPath, &computer, regions, newLabels, HoleNeighborsNum, HoleSize, true);
	idx++;

}
bool isNeighbor(std::vector<SPRegion>& regions, int i, int j)
{
	return std::find(regions[i].neighbors.begin(), regions[i].neighbors.end(), j) != regions[i].neighbors.end();
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

	if (debug && sregPairs.size()>0)
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
		cv::Mat rmask;
		GetRegionMap(img.cols, img.rows, &computer, newLabels, regions, rmask);
		sprintf(name, "%s%dBKMergeF_%d.jpg", outpath, idx, regSalInfos.size() - 1);
		cv::imwrite(name, rmask);
	}
	idx++;

	if (debug && sregPairs.size()>0)
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
	float edgeSPNum = (computer->GetSPWidth() + computer->GetSPHeight()) * 2;
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