#include "SuperpixelComputer.h"

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