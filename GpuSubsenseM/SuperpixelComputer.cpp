#include "SuperpixelComputer.h"
#include "Common.h"
#include "timer.h"
void SuperpixelComputer::init()
{
	_spHeight = (_height+_step-1)/_step;
	_spWidth = (_width + _step -1)/_step;
	_nPixels = _spHeight*_spWidth;
	_imgSize = _width*_height;
	_labels = new int[_imgSize];
	_centers = new SLICClusterCenter[_nPixels];
	_preLabels = NULL;
	_preCenters = NULL;
	_gs = new GpuSuperpixel(_width,_height,_step,_alpha);
	_neighbors.resize(_nPixels);
	_bgLabels = new int[_nPixels];
	_visited = new char[_nPixels];
	_segmented = new char[_nPixels];
	for(int i=0; i<_nPixels; i++)
	{

		if (i-1>=0)
		{
			_neighbors[i].push_back(i-1);
			if(i+_spWidth-1<_nPixels)
				_neighbors[i].push_back(i+_spWidth-1);
			if (i-_spWidth-1>=0)
				_neighbors[i].push_back(i-_spWidth-1);
		}
		if(i+1<_nPixels)
		{
			_neighbors[i].push_back(i+1);
			if(i+_spWidth+1<_nPixels)
				_neighbors[i].push_back(i+_spWidth+1);
			if (i-_spWidth+1>=0)
				_neighbors[i].push_back(i-_spWidth+1);
		}

		if (i-_spWidth>=0)
			_neighbors[i].push_back(i-_spWidth);		

		if(i+_spWidth<_nPixels)
			_neighbors[i].push_back(i+_spWidth);

	}
}

void SuperpixelComputer::release()
{
	delete _gs;
	if (_labels != NULL)
	{
		delete[] _labels;
		_labels = NULL;
	}
	if (_centers != NULL)
	{
		delete[] _labels;
		_centers = NULL;
	}
	if (_preLabels != NULL)
	{
		delete[] _labels;
		_preLabels = NULL;
	}
	if (_preCenters != NULL)
	{
		delete[] _labels;
		_preCenters = NULL;
	}
	safe_delete_array(_segmented);
	safe_delete_array(_visited);
	safe_delete_array(_bgLabels);

}

void SuperpixelComputer::ComputeSuperpixel(const cv::Mat& img, int& num, int*& labels, SLICClusterCenter*& centers)
{
	cv::Mat rgbaImg;
	cv::cvtColor(img,rgbaImg,CV_BGR2BGRA);
	_gs->Superpixel(rgbaImg,num,_labels,_centers);
	labels = _labels;
	centers = _centers;
	if (_preLabels == NULL)
	{
		_preLabels = new int[_imgSize];
		_preCenters = new SLICClusterCenter[_nPixels];
		memcpy(_preLabels,_labels,sizeof(int)*_imgSize);
		memcpy(_preCenters,_centers,sizeof(SLICClusterCenter)*_nPixels);
	}
}

void SuperpixelComputer::RegionGrowing(const std::vector<int>& seedLabels, float threshold,int*& resultLabel)
{
	const int dx4[] = {-1,0,1,0};
	const int dy4[] = {0,-1,0,1};
	//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	
	float pixDist(0);
	float regMaxDist = threshold;
	int regSize(0);
	
	memset(_bgLabels,0,sizeof(int)*_nPixels);
	memset(_segmented,0,_nPixels);
	std::vector<cv::Point2i> neighbors;
	float4 regMean;
	
	nih::Timer timer;
	timer.start();
	for(int i=0; i<seedLabels.size(); i++)
	{
		memset(_visited ,0,_nPixels);
		int seedLabel = seedLabels[i];
		_bgLabels[seedLabel] = 1;
		SLICClusterCenter cc = _centers[seedLabel];
		int k = cc.xy.x;
		int j = cc.xy.y;		
		
		int ix = seedLabel%_spWidth;
		int iy = seedLabel/_spWidth;
		pixDist = 0;
		regSize = 1;
		
		neighbors.clear();
	
		regMean = cc.rgb;
		while(pixDist < regMaxDist && regSize<_nPixels)
		{
			for(int d=0; d<4; d++)
			{
				int x = ix+dx4[d];
				int y = iy + dy4[d];
				if (x>=0 && x<_spWidth && y>=0 && y<_spHeight && !_visited[x+y*_spWidth] && !_segmented[x+y*_spWidth])
				{
					neighbors.push_back(cv::Point2i(x,y));
					_visited[x+y*_spWidth] = true;
				}
			}
			int idxMin = 0;
			pixDist = 255;
			if (neighbors.size() == 0)
				break;
			for(int j=0; j<neighbors.size(); j++)
			{
				size_t idx = neighbors[j].x+neighbors[j].y*_spWidth;
				float4 rgb = _centers[idx].rgb;
				float dx = rgb.x - regMean.x;
				float dy = rgb.y -regMean.y;
				float dz = rgb.z - regMean.z;
				//float dist = (abs(dx) + abs(dy)+ abs(dz))/3;
				float dist = sqrt(dx*dx + dy*dy + dz*dz);
				if (dist < pixDist)
				{
					pixDist = dist;
					idxMin = j;
				}				
			}
			
			ix = neighbors[idxMin].x;
			iy = neighbors[idxMin].y;
			int minIdx =ix + iy*_spWidth;
			float4 rgb = _centers[minIdx].rgb;
			//std::cout<<ix<<" "<<iy<<" added ,regMean = "<< regMean<<" pixDist "<<pixDist<<std::endl;
			regMean.x = (rgb.x + regMean.x*regSize )/(regSize+1);
			regMean.y = (rgb.y + regMean.y*regSize )/(regSize+1);
			regMean.z = (rgb.z + regMean.z*regSize )/(regSize+1);
			regSize++;
			int label = minIdx;
			_bgLabels[label] = 1;
			_segmented[minIdx] = 1;
			//result.data[minIdx] = 0xff;
			//smask.data[minIdx] = 0xff;
			neighbors[idxMin] = neighbors[neighbors.size()-1];
			neighbors.pop_back();
		}
	}
	resultLabel = _bgLabels;
	timer.stop();
	std::cout<<"\t superpixel region growing "<<timer.seconds()*1000<<std::endl;
}

void SuperpixelComputer::GetRegionGrowingImg(cv::Mat& rstImg)
{
	if (rstImg.empty())
	{
		rstImg.create(_height,_width,CV_8U);
		
	}
	rstImg = cv::Scalar(0);
	std::set<int> resLabels;
	for(int i=0; i<_nPixels; i++)
	{
		if (_bgLabels[i] == 1)
			resLabels.insert(i);
	}

	for(int i=0; i<_width; i++)
	{
		for(int j=0; j<_height; j++)
		{
			int idx = i+j*_width;
			if (resLabels.find(_labels[idx]) != resLabels.end())
				rstImg.data[idx] = 0xff;
		}
	}
}