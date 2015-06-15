#include "RegionMerging.h"
#include <fstream>
#include <time.h>       /* time */
#include <numeric>
#include "DistanceUtils.h"
#include "Dijkstra.h"

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
void GetRegionMap(int widht, int height, SuperpixelComputer* computer, int* segmented, std::vector<SPRegion>& regions, cv::Mat& mask, int flag)
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
	std::cout << "merge " << i << " to " << j << "\n";
		
	for (size_t n = 0; n < regions[i].neighbors.size(); n++)
	{
		SPRegion& reg = regions[regions[i].neighbors[n]];
		int Idx = regions[i].neighbors[n];
		//对i的所有邻居n，在邻居中删除i
		std::vector<int>::iterator itr = std::find(reg.neighbors.begin(),
			reg.neighbors.end(), i);
		if (itr != reg.neighbors.end())
			reg.neighbors.erase(itr);
		//在邻居中加入j
		if (reg.id != j && std::find(reg.neighbors.begin(), reg.neighbors.end(), j) == reg.neighbors.end())
			reg.neighbors.push_back(j);
		//在合并后的区域nRegId的邻居中加入n
		if (reg.id != j && std::find(regions[j].neighbors.begin(), regions[j].neighbors.end(), reg.id) == regions[j].neighbors.end())
			regions[j].neighbors.push_back(reg.id);
	}
	regions[i].neighbors.clear();
	int size0 = regions[j].size;
	int size1 = regions[i].size;
	regions[j].color = (regions[j].color * size0 + regions[i].color * size1)*(1.0 / (size0 + size1));
	regions[j].size = size0 + size1;
	for (size_t s = 0; s < regions[i].spIndices.size(); s++)
	{
		regions[j].spIndices.push_back(regions[i].spIndices[s]);
		newLabels[regions[i].spIndices[s]] = j;
	}
	for (int b = 0; b< regions[j].colorHist.size(); b++)
	{
		regions[j].colorHist[b] += regions[i].colorHist[b];
	}
	cv::normalize(regions[j].colorHist, regions[j].colorHist, 1, 0, cv::NORM_L1);
	for (int b = 0; b< regions[j].hog.size(); b++)
	{
		regions[j].hog[b] += regions[i].hog[b];
	}
	cv::normalize(regions[j].hog, regions[j].hog, 1, 0, cv::NORM_L1);
	regions[i].size = 0;
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
			ret += HandleHole(INeighbors[n], newLabels, spPoses, regions,regNeighbors);
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
	int nsize = std::find_if(regions.begin(), regions.end(), RegionSizeZero())-regions.begin();
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
	std::swap(nRegSizes,regSizes);
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
		for (int m =  y- K*step; m <= y + K*step; m += step)
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
		
		momentSum+=regions[i].moment;
	}
	mask.create(height, width, CV_8U);
	for (int i = 0; i < regions.size(); i++)
	{
		float sizeSal = (1 - 1.0*regions[i].size / regions[0].size);
		float momentSal = 1-regions[i].moment / momentSum;
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
		if (bgSize > 0.45*totalSize)
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
		

		for (size_t j = 0; j < regions[i].spIndices.size(); j++)
		{
			for (size_t s = 0; s < spPoses[regions[i].spIndices[j]].size(); s++)
			{
				uint2 xy = spPoses[regions[i].spIndices[j]][s];

				int idx = xy.x + xy.y*width;
				//mask.at<cv::Vec3b>(xy.y, xy.x) = color;
				mask.at<float>(xy.y, xy.x) = nBgColorDist;
				//*(float*)(mask.data + idx * 4) = minDist;
				//mask.at<float>(xy.y, xy.x) = (regions[minId].color.x + regions[minId].color.y + regions[minId].color.z) / 3 / 255;
			}
		}
	}
	normalize(mask, mask, 0, 1, cv::NORM_MINMAX, CV_32F);
	/*double min, max;
	cv::minMaxLoc(mask, &min, &max);*/
	//cv::threshold(mask, mask, 1.5, 255, CV_THRESH_BINARY);
	mask.convertTo(mask, CV_8U,255);

}



void BuildHistogram(const cv::Mat& img, SuperpixelComputer* computer, HISTOGRAMS& _colorHists, HISTOGRAMS& _HOGs)
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
	int _hogBins(36);
	int _totalColorBins = _colorBins*_colorBins*_colorBins;
	int _hogStep = 360.0 / _hogBins;

	float _colorSteps[3], _colorMins[3];
	//rgb color space
	_colorSteps[0] = _colorSteps[1] = _colorSteps[2] = 255.0 / _colorBins;
	_colorMins[0] = _colorMins[1] = _colorMins[2] = 0;

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
			//cv::Vec3f* labPtr = _labImg.ptr<cv::Vec3f>(m);
			cv::Vec3b* rgbPtr = (cv::Vec3b*)img.ptr<cv::Vec3b>(m);
			int bin = std::min<float>(floor(angPtr[n] / _hogStep), _hogBins - 1);
			_HOGs[i][bin] += magPtr[n];
			bin = 0;
			int s = 1;
			for (int c = 0; c < 3; c++)
			{
				//bin += s*std::min<float>(floor((labPtr[n][c]-_colorMins[c]) /_colorSteps[c]),_colorBins-1);
				bin += s*std::min<float>(floor((rgbPtr[n][c] - _colorMins[c]) / _colorSteps[c]), _colorBins - 1);
				s *= _colorBins;
			}
			_colorHists[i][bin] ++;
		}
		cv::normalize(_colorHists[i], _colorHists[i], 1, 0, cv::NORM_L1);
		cv::normalize(_HOGs[i], _HOGs[i], 1, 0, cv::NORM_L1);
	}
}

//iterative region growing
void IterativeRegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF)
{
	int width = img.cols, height = img.rows;
	int spWidth = computer.GetSPWidth(), spHeight = computer.GetSPHeight();
	int spSize(spWidth*spHeight);
	//build historgram
	HISTOGRAMS colorHist, gradHist;
	BuildHistogram(img, &computer, colorHist, gradHist);
	
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
		region.neighbors = computer.GetNeighbors(i);
		region.spIndices.push_back(i);
		regions.push_back(region);
		//regNeighbors.push_back(computer.GetNeighbors(i));
	}
	int ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	int RegSize = regions.size() - ZeroReg;
	int regThreshold = 30;
	while (RegSize > regThreshold)
	{
		RegionGrowing(img,computer,newLabels,regions,regNeighbors,thresholdF);
		ZeroReg = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
		RegSize = regions.size() - ZeroReg;
	}

	std::sort(regions.begin(), regions.end(), RegionSizeCmp());
	int size = std::find_if(regions.begin(), regions.end(), RegionSizeZero()) - regions.begin();
	regions.resize(size);
}
struct RegDist
{
	int sRid;
	int bRid;
	double dist;
};
struct RegDistDescComparer
{
	bool operator()(const RegDist& rd1, const RegDist& rd2)
	{
		return rd1.dist < rd2.dist;
	}
};
void RegionGrowing(const cv::Mat& img, SuperpixelComputer& computer, std::vector<int>& newLabels, std::vector<SPRegion>& regions, std::vector<std::vector<int>>& regNeighbors, float thresholdF)
{
	std::vector<RegDist> RegDists;
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].neighbors.size(); j++)
		{
			int n = regions[i].neighbors[j];
			if (i < n)
			{
				double dist = cv::compareHist(regions[i].colorHist, regions[n].colorHist,CV_COMP_BHATTACHARYYA);
				//double gdist = cv::compareHist(regions[i].colorHist, regions[j].colorHist, CV_COMP_BHATTACHARYYA);
				RegDist rd;
				rd.sRid = i;
				rd.bRid = n;
				rd.dist = dist;
				RegDists.push_back(rd);
			}
		}
	}

	std::sort(RegDists.begin(), RegDists.end(), RegDistDescComparer());
	int N = thresholdF*RegDists.size();

	std::vector < std::vector<uint2>> spPoses;
	computer.GetSuperpixelPoses(spPoses);

	for (int i = 0; i < N; i++)
	{
		if (regions[RegDists[i].bRid].size> 0 && regions[RegDists[i].sRid].size > 0)
			MergeRegions(RegDists[i].bRid, RegDists[i].sRid, newLabels, spPoses, regions);
	}

	/*std::sort(regions.begin(), regions.end(), RegionSizeCmp());*/
	//int size = std::count_if(regions.begin(), regions.end(), RegionSizeZero());
	
	

}