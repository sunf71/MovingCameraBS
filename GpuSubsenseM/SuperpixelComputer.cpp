#include "SuperpixelComputer.h"
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