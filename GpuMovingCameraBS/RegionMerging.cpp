#include "RegionMerging.h"
#include <fstream>
#include <time.h>       /* time */


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


		while (pixDist < regMaxDist && regSize<imgSize)
		{
			file << "iy:" << iy << " ix:" << ix << "\n";

			for (int d = 0; d<4; d++)
			{
				int x = ix + dx4[d];
				int y = iy + dy4[d];
				if (x >= 0 && x<spWidth && y >= 0 && y<spHeight && !visited[x + y*spWidth])
				{
					neighbors.push_back(cv::Point2i(x, y));
					visited[x + y*spWidth] = true;

				}
			}
			file << "	neighbors: ";
			for (int i = 0; i<neighbors.size(); i++)
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

			for (int j = 0; j<neighbors.size(); j++)
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
				for (int i = 0; i<histograms[label].size(); i++)
				{
					histograms[label][i] += histograms[minIdx][i];

				}
				cv::normalize(histograms[label], histograms[label], 1, 0, cv::NORM_L1);
				for (int i = 0; i<lhistograms[label].size(); i++)
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
		for (int i = 0; i<labelGroup.size(); i++)
		{
			newLabels[labelGroup[i]] = curLabel;
		}
		regColor.x /= regSize;
		regColor.y /= regSize;
		regColor.z /= regSize;
		regAvgColors.push_back(regColor);
		regSizes.push_back(regSize);
		curLabel++;
		for (int i = 0; i<neighbors.size(); i++)
		{
			int label = neighbors[i].x + neighbors[i].y*spWidth;
			visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);

		}
		if (regSize <2)
			singleLabels.push_back(label);
	}

	for (int i = 0; i<newLabels.size(); i++)
	{
		int x = centers[i].xy.x;
		int y = centers[i].xy.y;
		for (int dx = -step; dx <= step; dx++)
		{
			int sx = x + dx;
			if (sx<0 || sx >= width)
				continue;
			for (int dy = -step; dy <= step; dy++)
			{

				int sy = y + dy;
				if (sy >= 0 && sy<height)
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

void SuperPixelRegionMerging(int width, int height, int step,const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented, 
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors,float confidence)
{
	std::ofstream file("mergeOut.txt");
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
	//boundarySet.insert(95);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;
	
	while(!boundarySet.empty())
	{
		//std::cout<<boundarySet.size()<<std::endl;
		labelGroup.clear();
		std::set<int>::iterator itr = boundarySet.begin();
		int label = *itr;
		file<<"seed: "<<label<<"\n";
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
			file<<"iy:"<<iy<<" ix:"<<ix<<"\n";
			
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
			file<<"	neighbors: ";
			for (int i=0; i<neighbors.size(); i++)
			{
				int x = neighbors[i].x;
				int y = neighbors[i].y;
				file<<x+y*spWidth<<"("<<y<<","<<x<<"),";
			}
			file<<"\n";
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
				file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"\n";
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
				//cv::normalize(histograms[label], histograms[label], 1, 0, cv::NORM_L1);
				for(int i=0; i<lhistograms[label].size(); i++)
				{
					lhistograms[label][i] += lhistograms[minIdx][i];
				}
				//cv::normalize(lhistograms[label], lhistograms[label], 1, 0, cv::NORM_L1);
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
				file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"overpass threshold "<<regMaxDist<<"\n";
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

void SuperPixelRegionMergingFast(int width, int height, int step,const int*  labels, const SLICClusterCenter* centers,
	std::vector<std::vector<uint2>>& pos,
	std::vector<std::vector<float>>& histograms,
	std::vector<std::vector<float>>& lhistograms,
	std::vector<std::vector<uint2>>& newPos,
	std::vector<std::vector<float>>& newHistograms,
	float threshold, int*& segmented, 
	std::vector<int>& regSizes, std::vector<float4>& regAvgColors,float confidence)
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

				cv::normalize(histograms[label],histograms[label],1,0,cv::NORM_L1 );
				
				for(int i=0; i<lhistograms[label].size(); i++)
				{
					lhistograms[label][i] += lhistograms[minIdx][i];
				}
				cv::normalize(lhistograms[label],lhistograms[label],1,0,cv::NORM_L1 );
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
		std::vector<RegInfo> *vtor = (std::vector<RegInfo> *)&neighbors;
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
		{
				newLabels[label] = ulabel[0];
				regSizes[ulabel[0]]++;
		}
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
	delete[] visited;
	//SaveSegment(width,height,segmented,"region.png");
	//delete[] segmented;
	//file.close();
}

void SuperPixelRegionMergingFast(int width, int height, int step, const int*  labels, const SLICClusterCenter* centers,
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
	//std::ofstream file("mergeOut.txt");
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
	//std::vector<cv::Point2i> neighbors;

	float4 regMean;
	std::vector<int> singleLabels;
	//region growing 后的新label
	std::vector<int> newLabels;

	newLabels.resize(imgSize);
	//nih::Timer timer;
	//timer.start();
	std::set<int> boundarySet;
	boundarySet.insert(rand() % imgSize);
	//boundarySet.insert(3);
	//boundarySet.insert(190);
	std::vector<int> labelGroup;

	while (!boundarySet.empty())
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
		int iy = label / spWidth;
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


		while (pixDist < regMaxDist && regSize<imgSize)
		{
			//file<<"iy:"<<iy<<"ix:"<<ix<<"\n";

			for (int d = 0; d<4; d++)
			{
				int x = ix + dx4[d];
				int y = iy + dy4[d];
				size_t idx = x + y*spWidth;
				if (x >= 0 && x<spWidth && y >= 0 && y<spHeight && !visited[idx])
				{

					visited[idx] = true;
					float rd = histComp1->Distance(histograms[idx], histograms[label]);
					float hd = histComp2->Distance(lhistograms[idx], lhistograms[label]);
					float dist = confidence*rd + hd*(1 - confidence);
					neighbors.push(RegInfo(idx, x, y, dist));
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
				regColor.x = (regColor.x*regSize + centers[minIdx].rgb.x) / (regSize + 1);
				regColor.y = (regColor.y*regSize + centers[minIdx].rgb.y) / (regSize + 1);
				regColor.z = (regColor.z*regSize + centers[minIdx].rgb.z) / (regSize + 1);
				float t = 2.0;
				float dx = abs(tmpx - regColor.x);
				float dy = abs(tmpy - regColor.y);
				float dz = abs(tmpz - regColor.z);

				/*regColor.x += centers[minIdx].rgb.x;
				regColor.y += centers[minIdx].rgb.y;
				regColor.z += centers[minIdx].rgb.z;*/
				regSize++;
				labelGroup.push_back(minIdx);

				for (int i = 0; i<histograms[label].size(); i++)
				{
					histograms[label][i] += histograms[minIdx][i];

				}

				cv::normalize(histograms[label], histograms[label], 1, 0, cv::NORM_L1);

				for (int i = 0; i<lhistograms[label].size(); i++)
				{
					lhistograms[label][i] += lhistograms[minIdx][i];
				}
				cv::normalize(lhistograms[label], lhistograms[label], 1, 0, cv::NORM_L1);
				visited[minIdx] = true;
				if (sqrt(dx*dx + dy*dy + dz*dz) > t)
				{
					while (!tneighbors.empty())
						tneighbors.pop();
					while (!neighbors.empty())
					{
						RegInfo sp = neighbors.top();
						neighbors.pop();
						float rd = histComp1->Distance(histograms[sp.label], histograms[label]);
						float hd = histComp2->Distance(lhistograms[sp.label], lhistograms[label]);
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
			else
			{
				//file<<"nearst neighbor "<<minIdx<<"("<<iy<<","<<ix<<") with distance:"<<pixDist<<"overpass threshold "<<regMaxDist<<"\n";
			}
		}
		newHistograms.push_back(histograms[label]);
		for (int i = 0; i<labelGroup.size(); i++)
		{
			newLabels[labelGroup[i]] = curLabel;
		}
		/*	regColor.x/=regSize;
		regColor.y/=regSize;
		regColor.z/=regSize;*/
		regAvgColors.push_back(regColor);
		regSizes.push_back(regSize);
		curLabel++;
		std::vector<RegInfo> *vtor = (std::vector<RegInfo> *)&neighbors;
		for (int i = 0; i<vtor->size(); i++)
		{
			int label = ((RegInfo)vtor->operator [](i)).label;
			visited[label] = false;
			if (boundarySet.find(label) == boundarySet.end())
				boundarySet.insert(label);

		}
		if (regSize <2)
			singleLabels.push_back(label);
	}


	//对单个超像素，检查其是否在大区域之中（周边三个以上label一样）
	for (int i = 0; i<singleLabels.size(); i++)
	{
		int label = singleLabels[i];
		int ix = label%spWidth;
		int iy = label / spWidth;
		std::vector<int> ulabel;

		for (int d = 0; d<4; d++)
		{
			int x = ix + dx4[d];
			int y = iy + dy4[d];
			if (x >= 0 && x<spWidth && y >= 0 && y<spHeight)
			{
				int nlabel = x + y*spWidth;
				if (std::find(ulabel.begin(), ulabel.end(), newLabels[nlabel]) == ulabel.end())
					ulabel.push_back(newLabels[nlabel]);
			}

		}
		if (ulabel.size() <= 2)
		{
			newLabels[label] = ulabel[0];
			regSizes[ulabel[0]]++;
		}
	}
	for (int i = 0; i<newLabels.size(); i++)
	{
		int x = centers[i].xy.x;
		int y = centers[i].xy.y;
		for (int dx = -step; dx <= step; dx++)
		{
			int sx = x + dx;
			if (sx<0 || sx >= width)
				continue;
			for (int dy = -step; dy <= step; dy++)
			{

				int sy = y + dy;
				if (sy >= 0 && sy<height)
				{
					int idx = sx + sy*width;
					if (labels[idx] == i)
						segmented[idx] = newLabels[i];
				}
			}
		}

	}
	delete[] visited;
	//SaveSegment(width,height,segmented,"region.png");
	//delete[] segmented;
	//file.close();
}