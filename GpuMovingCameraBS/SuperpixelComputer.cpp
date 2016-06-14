#include "SuperpixelComputer.h"
#include "SLIC.h"
#include "PictureHandler.h"
#include "Common.h"
#include "timer.h"
#include <queue>
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
	_neighbors4.resize(_nPixels);
	_neighbors8.resize(_nPixels);
	_bgLabels = new int[_nPixels];
	_visited = new char[_nPixels];
	_segmented = new char[_nPixels];
	for (size_t j = 1; j < _spWidth - 1; j++)
	{
		int i = 0;
		int idx = j + i*_spWidth;
		_neighbors8[idx].push_back(idx - 1);
		_neighbors8[idx].push_back(idx + 1);
		_neighbors8[idx].push_back(idx + _spWidth);
		_neighbors8[idx].push_back(idx - 1 + _spWidth);
		_neighbors8[idx].push_back(idx + _spWidth + 1);
		_neighbors4[idx].push_back(idx - 1);
		_neighbors4[idx].push_back(idx + 1);
		_neighbors4[idx].push_back(idx + _spWidth);
		i = _spHeight - 1;
		idx = j + i*_spWidth;
		_neighbors8[idx].push_back(idx - 1);
		_neighbors8[idx].push_back(idx + 1);
		_neighbors8[idx].push_back(idx - 1 - _spWidth);
		_neighbors8[idx].push_back(idx - _spWidth + 1);
		_neighbors8[idx].push_back(idx - _spWidth);
		_neighbors4[idx].push_back(idx - 1);
		_neighbors4[idx].push_back(idx + 1);
		_neighbors4[idx].push_back(idx - _spWidth);
	}
	for (size_t i = 1; i < _spHeight - 1; i++)
	{
		int j = 0;
		int idx = j + i*_spWidth;		
		_neighbors8[idx].push_back(idx + 1);
		_neighbors8[idx].push_back(idx + _spWidth);
		_neighbors8[idx].push_back(idx + 1 + _spWidth);
		_neighbors8[idx].push_back(idx - _spWidth + 1);
		_neighbors8[idx].push_back(idx - _spWidth);
		_neighbors4[idx].push_back(idx - _spWidth);
		_neighbors4[idx].push_back(idx + 1);
		_neighbors4[idx].push_back(idx + _spWidth);
		j = _spWidth - 1;
		idx = j + i*_spWidth;
		_neighbors8[idx].push_back(idx - 1);
		_neighbors8[idx].push_back(idx + _spWidth);
		_neighbors8[idx].push_back(idx - 1 + _spWidth);
		_neighbors8[idx].push_back(idx - _spWidth - 1);
		_neighbors8[idx].push_back(idx - _spWidth);
		_neighbors4[idx].push_back(idx - _spWidth);
		_neighbors4[idx].push_back(idx - 1);
		_neighbors4[idx].push_back(idx + _spWidth);
	}
	for (size_t i = 1; i < _spHeight - 1; i++)
	{
		for (size_t j = 1; j < _spWidth - 1; j++)
		{
			int idx = j + i*_spWidth;
			_neighbors8[idx].push_back(idx - 1);
			_neighbors8[idx].push_back(idx + 1);
			_neighbors8[idx].push_back(idx + _spWidth);
			_neighbors8[idx].push_back(idx - 1 + _spWidth);
			_neighbors8[idx].push_back(idx + 1 + _spWidth);
			_neighbors8[idx].push_back(idx - _spWidth - 1);
			_neighbors8[idx].push_back(idx - _spWidth);
			_neighbors8[idx].push_back(idx - _spWidth + 1);
			_neighbors4[idx].push_back(idx - _spWidth);
			_neighbors4[idx].push_back(idx + 1);
			_neighbors4[idx].push_back(idx + _spWidth);
			_neighbors4[idx].push_back(idx - 1);
		}
	}
	_neighbors8[0].push_back(1);
	_neighbors8[0].push_back(_spWidth);
	_neighbors8[0].push_back(_spWidth + 1);

	_neighbors4[0].push_back(1);
	_neighbors4[0].push_back(_spWidth);

	_neighbors8[_spWidth - 1].push_back(_spWidth - 2);
	_neighbors8[_spWidth - 1].push_back(2 * _spWidth - 1);
	_neighbors8[_spWidth - 1].push_back(2 * _spWidth - 2);
	_neighbors4[_spWidth - 1].push_back(_spWidth - 2);
	_neighbors4[_spWidth - 1].push_back(2 * _spWidth - 1);

	int idx = _spWidth*_spHeight-1;
	_neighbors8[idx].push_back(idx - _spWidth);
	_neighbors8[idx].push_back(idx - 1);
	_neighbors8[idx].push_back(idx - _spWidth - 1);
	_neighbors4[idx].push_back(idx - _spWidth);
	_neighbors4[idx].push_back(idx - 1);

	idx -= (_spWidth - 1);
	_neighbors8[idx].push_back(idx - _spWidth);
	_neighbors8[idx].push_back(idx + 1);
	_neighbors8[idx].push_back(idx - _spWidth + 1);
	_neighbors4[idx].push_back(idx - _spWidth);
	_neighbors4[idx].push_back(idx + 1);

	_spPoints.clear();
	_spPoses.clear();
}

void SuperpixelComputer::release()
{
	safe_delete(_gs);
	safe_delete(_labels);
	safe_delete(_centers);
	safe_delete(_preLabels);
	safe_delete(_preCenters);
	
	safe_delete_array(_segmented);
	safe_delete_array(_visited);
	safe_delete_array(_bgLabels);

}
void SuperpixelComputer::ComputeSuperpixel(uchar4* d_rgbaBuffer, int& num, int*& labels, SLICClusterCenter*& centers)
{
	_gs->DSuperpixel(d_rgbaBuffer,num,_labels,_centers);
	labels = _labels;
	centers = _centers;
	if (_preLabels == NULL)
	{
		_preLabels = new int[_imgSize];
		_preCenters = new SLICClusterCenter[_nPixels];
		memcpy(_preLabels,_labels,sizeof(int)*_imgSize);
		memcpy(_preCenters,_centers,sizeof(SLICClusterCenter)*_nPixels);
	}
	_spPoints.clear();
	_spPoses.clear();
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
	_spPoints.clear();
	_spPoses.clear();
}
void SuperpixelComputer::ComputeBigSuperpixel(uchar4* d_rgbaBuffer)
{
	int num(0);
	_gs->DSuperpixelB(d_rgbaBuffer,num,_labels,_centers);
	if (_preLabels == NULL)
	{
		_preLabels = new int[_imgSize];
		_preCenters = new SLICClusterCenter[_nPixels];
		memcpy(_preLabels,_labels,sizeof(int)*_imgSize);
		memcpy(_preCenters,_centers,sizeof(SLICClusterCenter)*_nPixels);
	}
	_spPoints.clear();
	_spPoses.clear();
}
void SuperpixelComputer::ComputeBigSuperpixel(const cv::Mat& img)
{
	cv::Mat rgbaImg;
	cv::cvtColor(img,rgbaImg,CV_BGR2BGRA);
	int num(0);
	_gs->SuperpixelB(rgbaImg,num,_labels,_centers);
	if (_preLabels == NULL)
	{
		_preLabels = new int[_imgSize];
		_preCenters = new SLICClusterCenter[_nPixels];
		memcpy(_preLabels,_labels,sizeof(int)*_imgSize);
		memcpy(_preCenters,_centers,sizeof(SLICClusterCenter)*_nPixels);
	}
	_spPoints.clear();
	_spPoses.clear();
}
void SuperpixelComputer::ComputeSuperpixel(const cv::Mat& img)
{
	cv::Mat rgbaImg;
	cv::cvtColor(img,rgbaImg,CV_BGR2BGRA);
	int num(0);
	_gs->Superpixel(rgbaImg,num,_labels,_centers);
	if (_preLabels == NULL)
	{
		_preLabels = new int[_imgSize];
		_preCenters = new SLICClusterCenter[_nPixels];
		memcpy(_preLabels,_labels,sizeof(int)*_imgSize);
		memcpy(_preCenters,_centers,sizeof(SLICClusterCenter)*_nPixels);
	}
	_spPoints.clear();
	_spPoses.clear();
}
void SuperpixelComputer::GetSuperpixelPoints(std::vector<std::vector<cv::Point>>& poses)
{
	poses = _spPoints;
	/*if (_labels == NULL)
		return;
	if (_spPoints.size() > 0)
	{
		poses = _spPoints;
		return;
	}
		
	int _spSize = _spWidth*_spHeight;
	_spPoints.clear();
	_spPoints.resize(_spSize);
	for (int i = 0; i < _spSize; i++)
	{
		_spPoints[i].clear();
		int x = int(_centers[i].xy.x + 0.5);
		int y = int(_centers[i].xy.y + 0.5);
		for (int m = -_step + y; m <= _step + y; m++)
		{
			if (m < 0 || m >= _height)
				continue;
			for (int n = -_step + x; n <= _step + x; n++)
			{
				if (n < 0 || n >= _width)
					continue;
				int id = m*_width + n;
				if (_labels[id] == i)
				{
					_spPoints[i].push_back(cv::Point(n, m));
				}
			}
		}
	}
	poses = _spPoints;*/
}

void SuperpixelComputer::GetSuperpixelPointsNeighbors(std::vector<std::vector<cv::Point>>& points, std::vector<std::vector<int>>& neighbors, int numOfNeighbors)
{
	static int ndx[] = { 1, 0, -1, 0, 1, -1, -1, 1 };
	static int ndy[] = { 0, -1, 0, 1, -1, -1, 1, 1 };
	if (_labels == NULL)
		return;
	if (_spPoints.size() > 0)
	{
		points = _spPoints;
		neighbors = _neighbors4;
		return;
	}
	int _spSize = _spWidth*_spHeight;
	_spPoints.clear();
	_spPoints.resize(_spSize);
	_neighbors4.resize(_spSize);

	for (int i = 0; i < _spSize; i++)
	{
		_spPoints[i].clear();
		int x = int(_centers[i].xy.x + 0.5);
		int y = int(_centers[i].xy.y + 0.5);
		for (int m = -_step + y; m <= _step + y; m++)
		{
			if (m < 0 || m >= _height)
				continue;
			for (int n = -_step + x; n <= _step + x; n++)
			{
				if (n < 0 || n >= _width)
					continue;
				int id = m*_width + n;
				if (_labels[id] == i)
				{
					_spPoints[i].push_back(cv::Point(n, m));
					for (size_t ni = 0; ni < numOfNeighbors; ni++)
					{
						int px = n + ndx[ni];
						int py = m + ndy[ni];
						if (px >= 0 && px < _width && py >= 0 && py < _height)
						{
							int idx = px + py*_width;
							if (_labels[idx] != i)
							{
								if (std::find(_neighbors4[i].begin(), _neighbors4[i].end(), idx) == _neighbors4[i].end())
									_neighbors4[i].push_back(_labels[idx]);
							}
						}

					}
				}
			}
		}
	}
	points = _spPoints;
}
void SuperpixelComputer::GetSuperpixelPosesNeighbors(std::vector<std::vector<uint2>>& poses, std::vector<std::vector<int>>& neighbors, int numOfNeighbors)
{
	static int ndx[] = { 1, 0, -1, 0, 1, -1, -1, 1 };
	static int ndy[] = { 0, -1, 0, 1, -1, -1, 1, 1 };
	if (_labels == NULL)
		return;
	if (_spPoses.size() > 0)
	{
		poses = _spPoses;
		neighbors = _neighbors4;
		return;
	}
	
	_spPoses.clear();
	_spPoses.resize(_nPixels);
	_neighbors4.clear();
	_neighbors4.resize(_nPixels);
	

	for (int y = 0; y < _height; y++)
	{
		int* labelPtr = &_labels[y*_width];
		
		for (int x = 0; x < _width; x++)
		{
			
			int label = labelPtr[x];
			_spPoses[label].push_back( make_uint2(x, y));
			for (int n = 0; n < numOfNeighbors; n++)
			{
				int dy = y + ndy[n];
				if (dy <0 || dy >= _height)
					continue;
				int dx = x + ndx[n];
				if (dx < 0 || dx >= _width)
					continue;
				int nlabel = _labels[dx + dy*_width];
				if (nlabel != label)
				{
					if (std::find(_neighbors4[label].begin(), _neighbors4[label].end(), nlabel) == _neighbors4[label].end())
						_neighbors4[label].push_back(nlabel);

				}
			}

		}
	}
	poses = _spPoses;
	neighbors = _neighbors4;
}
void SuperpixelComputer::GetSuperpixelPoses(std::vector<std::vector<uint2>>& spPoses)
{
	if (_labels == NULL)
		return;
	if (_spPoses.size() > 0)
	{
		spPoses = _spPoses;
		return;
	}
	int _spSize = _spWidth*_spHeight;
	_spPoses.clear();
	_spPoses.resize(_spSize);
	for (int i = 0; i < _spSize; i++)
	{
		_spPoses[i].clear();
		int x = int(_centers[i].xy.x + 0.5);
		int y = int(_centers[i].xy.y + 0.5);
		for (int m = -_step + y; m <= _step + y; m++)
		{
			if (m < 0 || m >= _height)
				continue;		
			for (int n = -_step + x; n <= _step + x; n++)
			{
				if (n < 0 || n >= _width)
					continue;
				int id = m*_width + n;
				if (_labels[id] == i)
				{			
					_spPoses[i].push_back(make_uint2(n, m));
				}
			}
		}	
	}
	spPoses = _spPoses;
}
void SuperpixelComputer::ComputeSLICSuperpixel(const cv::Mat& img)
{
	cv::Mat rgbaImg;
	cv::cvtColor(img, rgbaImg, CV_BGR2BGRA);
	SLIC slic;
	
	double m(0);
	slic.PerformSLICO_ForGivenStepSize((unsigned int*)rgbaImg.data, img.cols, img.rows, _labels, _centers, _nPixels, _step, m);
	//slic.SaveSuperpixelLabels(_labels, img.cols, img.rows, "labels.txt", "./");
	
	GetSuperpixelPosesNeighbors(_spPoses, _neighbors4);
	_centers = new SLICClusterCenter[_nPixels];
	//#pragma omp parallel for
	for (int i = 0; i < _nPixels; i++)
	{
		double sr(0), sg(0), sb(0), sy(0), sx(0); 
		for (int j = 0; j < _spPoses[i].size(); j++)
		{
			int x = _spPoses[i][j].x;
			int y = _spPoses[i][j].y;
			sx += x;
			sy += y;
			uchar* ptr = (uchar*)(img.data + (y*_width + x) * 3);
			sr += ptr[0];
			sg += ptr[1];
			sb += ptr[2];
		}
		sx /= _spPoses[i].size();
		sy /= _spPoses[i].size();
		sr /= _spPoses[i].size();
		sg /= _spPoses[i].size();
		sb /= _spPoses[i].size();
		_centers[i].nPoints = _spPoses[i].size();
		_centers[i].rgb = make_float4(sr, sg, sb, 0);
		_centers[i].xy = make_float2(sx, sy);
	}
	
}
struct SPInfo
{
	SPInfo(){}
	SPInfo(int l,float d):dist(d),label(l){} 
	float dist;
	int label;
};
//结构体的比较方法 改写operator()  
struct SPInfoCmp  
{  
    bool operator()(const SPInfo &na, const SPInfo &nb)  
    {  
		return na.dist > nb.dist;
    }  
};
typedef std::priority_queue<SPInfo,std::vector<SPInfo>,SPInfoCmp> SPInfos;
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
	
	/*nih::Timer timer;
	timer.start();*/
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
				int idx = x+y*_spWidth;
				if (x>=0 && x<_spWidth && y>=0 && y<_spHeight && !_visited[idx] && !_segmented[idx])
				{
					neighbors.push_back(cv::Point2i(x,y));
					_visited[idx] = true;
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
	/*timer.stop();
	std::cout<<"\t superpixel region growing "<<timer.seconds()*1000<<std::endl;*/
}
void SuperpixelComputer::RegionGrowingFast(const std::vector<int>& seedLabels, float threshold,int*& resultLabel)
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
	
	
	float4 regMean;
	
	/*nih::Timer timer;
	timer.start();*/
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
		
		
		SPInfos sps,tsps;
	
		regMean = cc.rgb;
		while(pixDist < regMaxDist && regSize<_nPixels)
		{
			for(int d=0; d<4; d++)
			{
				int x = ix+dx4[d];
				int y = iy + dy4[d];
				size_t idx = x+y*_spWidth;
				if (x>=0 && x<_spWidth && y>=0 && y<_spHeight && !_visited[idx] && !_segmented[idx])
				{
					//neighbors.push_back(cv::Point2i(x,y));
					float4 rgb = _centers[idx].rgb;
					float dx = rgb.x - regMean.x;
					float dy = rgb.y -regMean.y;
					float dz = rgb.z - regMean.z;
					//float dist = (abs(dx) + abs(dy)+ abs(dz))/3;
					float dist = sqrt(dx*dx + dy*dy + dz*dz);
					SPInfo sp(idx,dist);
					sps.push(sp);
					_visited[idx] = true;
				}
			}
			//std::cout<<i<<": "<<sps.size()<<std::endl;
			if (sps.empty())
				break;
			
			SPInfo sp = sps.top();
			pixDist = sp.dist;
			sps.pop();
			int minIdx = sp.label;
			float4 rgb = _centers[minIdx].rgb;
			//std::cout<<ix<<" "<<iy<<" added ,minIdx = "<< minIdx<<" pixDist "<<pixDist<<std::endl;
			float tmpx = regMean.x;
			float tmpy = regMean.y;
			float tmpz = regMean.z;
			regMean.x = (rgb.x + regMean.x*regSize )/(regSize+1);
			regMean.y = (rgb.y + regMean.y*regSize )/(regSize+1);
			regMean.z = (rgb.z + regMean.z*regSize )/(regSize+1);
			float t = 5;
			float dx = abs(tmpx - regMean.x);
			float dy = abs(tmpy - regMean.y);
			float dz = abs(tmpz - regMean.z);
			if (sqrt(dx*dx +dy*dy +dz*dz) > t)
			{
				while(!tsps.empty())
					tsps.pop();
				while(!sps.empty())
				{
					SPInfo sp = sps.top();
					sps.pop();
					float4 rgb = _centers[sp.label].rgb;
					float dx = rgb.x - regMean.x;
					float dy = rgb.y -regMean.y;
					float dz = rgb.z - regMean.z;				
					sp.dist =  sqrt(dx*dx + dy*dy + dz*dz);
					tsps.push(sp);
				}
				std::swap(sps,tsps);
				//std::cout<<"update neighbors"<<std::endl;
			}
			regSize++;
			
			_bgLabels[minIdx] = 1;
			_segmented[minIdx] = 1;
			ix = minIdx%_spWidth;
			iy = minIdx/_spWidth;
			
		}
	}
	resultLabel = _bgLabels;
	/*timer.stop();
	std::cout<<"\t superpixel region growing "<<timer.seconds()*1000<<std::endl;*/
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
void SuperpixelComputer::GetRegionGrowingSeedImg(const std::vector<int>& seeds, cv::Mat& rstImg)
{
	if (rstImg.empty())
	{
		rstImg.create(_height,_width,CV_8U);
		
	}
	rstImg = cv::Scalar(0);
	std::set<int> resLabels;
	for(int i=0; i<seeds.size(); i++)
	{
		resLabels.insert(seeds[i]);
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
 void SuperpixelComputer::GetSuperpixelDownSampleImg(cv::Mat& rstImg)
 {
	 rstImg = cv::Mat::zeros(_spHeight,_spWidth,CV_8UC3);
	 for(int i=0; i< _spHeight; i++)
	 {
		 cv::Vec3b* ptr = rstImg.ptr<cv::Vec3b>(i);
		 for(int j=0; j<_spWidth; j++)
		 {
			 int idx = _spWidth*i+j;
			 if (_centers[idx].nPoints > 0)
			 {
				 ptr[j][0] = _centers[idx].rgb.x;
				 ptr[j][1] = _centers[idx].rgb.y;
				 ptr[j][2] = _centers[idx].rgb.z;
			 }
		 }
	 }
 }
 void SuperpixelComputer:: GetSuperpixelDownSampleGrayImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& src, cv::Mat &dstImg)
 {
	 dstImg = cv::Mat::zeros(_spHeight,_spWidth,CV_8U);
	 for(int i=0; i<_nPixels; i++)
	  {
		  int k = (int)(centers[i].xy.x+0.5);
		  int j = (int)(centers[i].xy.y + 0.5);
		  if (src.data[k+j*_width] == 0xff)
			  dstImg.data[i] = 0xff;		 
	  }
 }
 void SuperpixelComputer::GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, cv::Mat& dstColorImg)
 {
	 dstColorImg = cv::Mat::zeros(_spHeight,_spWidth,CV_8UC3);

	  int rgb[3];
	  for(int i=0; i<_nPixels; i++)
	  {
		  int k = (int)(_centers[i].xy.x+0.5);
		  int j = (int)(_centers[i].xy.y + 0.5);
		  memset(rgb,0,sizeof(int)*3);
		  
		  int count(0);
		  for(int y=j-_step; y<=j+_step; y++)
		  {
			  if (y>=0 && y<_height)
			  {
				  const cv::Vec3b* ptr = srcColorImg.ptr<cv::Vec3b>(y);
				  for(int x = k-_step; x<= k+_step; x++)
				  {
					  if (x>=0 && x<_width && labels[y*_width+x] == i)
					  {
						  for(int c=0; c<3; c++)
						  {
							  rgb[c]+=(int)ptr[x][c];
							  
						  }
						  
						  count++;
					  }
				  }
			  }
		  }
		  uchar* dstPtr = (uchar*)(dstColorImg.data + i*3);
		  for(int c=0; c<3; c++)
			  dstPtr[c] = (uchar)(rgb[c]*1.0/count);
		 
	  }
 }
  void SuperpixelComputer::GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, const cv::Mat& srcMapXImg, const cv::Mat& srcMapYImg, const cv::Mat& srcInvMapXImg, const cv::Mat& srcInvMapYImg, 
		  cv::Mat& dstColorImg, cv::Mat& dstMapXImg, cv::Mat& dstMapYImg, cv::Mat& dstInvMapXImg,  cv::Mat& dstInvMapYImg )
  {
	  dstColorImg = cv::Mat::zeros(_spHeight,_spWidth,CV_8UC3);
	  dstMapXImg = cv::Mat::zeros(_spHeight,_spWidth,CV_32F);
	  dstMapYImg = cv::Mat::zeros(_spHeight,_spWidth,CV_32F);
	  dstInvMapXImg = dstMapXImg.clone();
	  dstInvMapYImg = dstMapYImg.clone();
	  int rgb[3];
	  float map[2];
	  float imap[2]; 
	  int count(0);
	  for(int i=0; i<_nPixels; i++)
	  {
		  int k = (int)(centers[i].xy.x+0.5);
		  int j = (int)(centers[i].xy.y + 0.5);
		  memset(rgb,0,sizeof(int)*3);
		  memset(map,0,sizeof(float)*2);
		  memset(imap,0,sizeof(float)*2);
		 count = 0;
		  for(int y=j-_step; y<=j+_step; y++)
		  {
			  if (y>=0 && y<_height)
			  {
				  const cv::Vec3b* ptr = srcColorImg.ptr<cv::Vec3b>(y);
				  const float* mxPtr = srcMapXImg.ptr<float>(y);
				  const float* myPtr = srcMapYImg.ptr<float>(y);
				  const float* imxPtr = srcInvMapXImg.ptr<float>(y);
				  const float* imyPtr = srcInvMapYImg.ptr<float>(y);
		
				  for(int x = k-_step; x<= k+_step; x++)
				  {
					  if (x>=0 && x<_width)
					  {
						  if (_labels[x+y*_width] == i)
						  {
							  for(int c=0; c<3; c++)
							  {
								  rgb[c]+=(int)ptr[x][c];

							  }
							  map[0] += (mxPtr[x] -x);
							  map[1] += (myPtr[x] - y);
							  imap[0]+= (imxPtr[x] - x);
							  imap[1] += (imyPtr[x] -y);
							  count++;
						  }
					  }
				  }
			  }
		  }
		  uchar* dstPtr = (uchar*)(dstColorImg.data + i*3);
		  for(int c=0; c<3; c++)
			  dstPtr[c] = (uchar)(rgb[c]*1.0/count);
		  float* dstXPtr = (float*)(dstMapXImg.data+i*4);
		  float* dstYPtr = (float*)(dstMapYImg.data+i*4);
		  float* dstIxPtr = (float*)(dstInvMapXImg.data+i*4);
		  float* dstIyPtr = (float*)(dstInvMapYImg.data+i*4);
		  int imx = (int)(map[0]/count + centers[i].xy.x+0.5);
		  int imy = (int)(map[1]/count + centers[i].xy.y+0.5);
		  if (imx<0 || imx > _width)
			  *dstXPtr = -1;
		  else if(imy<0 || imy>_height)
			  *dstYPtr = -1;
		  else
		  {
			  int label =  labels[imx + imy*_width];
			  *dstXPtr = label%_spWidth;
			  *dstYPtr =  label/_spWidth;
		  }
		  imx = (int)(imap[0]/count + centers[i].xy.x+0.5);
		  imy = (int)(imap[1]/count + centers[i].xy.y+0.5);
		  if (imx<0 || imx > _width)
			  *dstIxPtr = -1;
		  else if(imy<0 || imy>_height)
			  *dstIyPtr = -1;
		  else
		  {
			  int label =  labels[imx + imy*_width];
			  *dstIxPtr = label%_spWidth;
			  *dstIyPtr =  label/_spWidth;
		  }
		
		 
	  }
  }
 void SuperpixelComputer::GetSuperpixelDownSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& srcColorImg, const cv::Mat& srcMapXImg, const cv::Mat& srcMapYImg, cv::Mat& dstColorImg, cv::Mat& dstMapXImg, cv::Mat& dstMapYImg)
 {
	 dstColorImg = cv::Mat::zeros(_spHeight,_spWidth,CV_8UC3);
	  dstMapXImg = cv::Mat::zeros(_spHeight,_spWidth,CV_32F);
	  dstMapYImg = cv::Mat::zeros(_spHeight,_spWidth,CV_32F);

	  int rgb[3];
	  float map[2];
	  for(int i=0; i<_nPixels; i++)
	  {
		  int k = (int)(centers[i].xy.x+0.5);
		  int j = (int)(centers[i].xy.y + 0.5);
		  memset(rgb,0,sizeof(int)*3);
		  memset(map,0,sizeof(float)*2);
		  int count(0);
		  for(int y=j-_step; y<=j+_step; y++)
		  {
			  if (y>=0 && y<_height)
			  {
				  const cv::Vec3b* ptr = srcColorImg.ptr<cv::Vec3b>(y);
				  const float* mxPtr = srcMapXImg.ptr<float>(y);
				  const float* myPtr = srcMapYImg.ptr<float>(y);
		
				  for(int x = k-_step; x<= k+_step; x++)
				  {
					  if (x>=0 && x<_width && labels[y*_width+x] == i)
					  {
						  for(int c=0; c<3; c++)
						  {
							  rgb[c]+=(int)ptr[x][c];
							  
						  }
						  map[0] += mxPtr[x];
						  map[1] += myPtr[x];
						  count++;
					  }
				  }
			  }
		  }
		  uchar* dstPtr = (uchar*)(dstColorImg.data + i*3);
		  for(int c=0; c<3; c++)
			  dstPtr[c] = (uchar)(rgb[c]*1.0/count);
		  float* dstXPtr = (float*)(dstMapXImg.data+i*4);
		  float* dstYPtr = (float*)(dstMapYImg.data+i*4);
		  *dstXPtr = map[0]/count;
		  *dstYPtr = map[1]/count;
	  }
 }

 void SuperpixelComputer::GetSuperpixelUpSampleImg(const int* labels, const SLICClusterCenter* centers, const cv::Mat& src, cv::Mat& dstImg)
 {
	 dstImg = cv::Mat::zeros(_height,_width,src.type());
	 for(int i=0; i< _nPixels; i++)
	 {
		 uchar value = src.data[i];
		 int k = (int)(centers[i].xy.x+0.5);
		  int j = (int)(centers[i].xy.y + 0.5);
		  
		  for(int y=j-_step; y<=j+_step; y++)
		  {
			  if (y>=0 && y<_height)
			  {			  
				  
				  uchar* ptr = dstImg.ptr<uchar>(y);
		
				  for(int x = k-_step; x<= k+_step; x++)
				  {
					  if (x>=0 && x<_width && labels[y*_width+x] == i)
					  {
						  ptr[x] = value;
					  }
				  }
			  }
		  }
	 }
 }

 void SuperpixelComputer::GetVisualResult(const cv::Mat& img, cv::Mat& rstMat)
 {
	cv::Mat rgbaImg;
	cv::cvtColor(img,rgbaImg,CV_BGR2BGRA);
	SLIC aslic;	
	PictureHandler handler;
	unsigned int* idata = (unsigned int*) rgbaImg.data;
	aslic.DrawContoursAroundSegments(idata, _labels, _width,_height,0x00ff00);
	cv::cvtColor(rgbaImg,rstMat,CV_BGRA2BGR);
 }