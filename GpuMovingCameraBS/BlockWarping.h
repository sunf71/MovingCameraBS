#pragma once
#include "ImageWarping.h"
#include <opencv\cv.h>
#include <vector>
#undef min
#undef max
#include <opencv2\gpu\gpu.hpp>
#include "cuda.h"
#include "cuda_runtime.h"
typedef std::vector<cv::Point2f> Points;
struct Cell
{
	Points points0,points1;
	int idx;
};
class GlobalWarping : public ImageWarping
{
public:
	GlobalWarping(int width, int height, float threshold = 0.1) :_width(width), _height(height), _threshold(threshold)
	{
		_map.create(height, width, CV_32FC2);
		_invMap.create(height, width, CV_32FC2);
	}
	void SetFeaturePoints(const Points& p1, const Points& p2);
	virtual void Warp(const cv::Mat& img, cv::Mat& warpedImg);
	virtual void getFlow(cv::Mat& flow);
	virtual void GpuWarp(const cv::gpu::GpuMat& dimg, cv::gpu::GpuMat& dwimg);
private:
	int _width;
	int _height;
	float _threshold;
	cv::Mat _homo, _invHomo;
	std::vector<uchar> _inliers;
};
class BlockWarping:public ImageWarping
{
public:
	BlockWarping(int width, int height, int quadStep):_width(width),_height(height),_quadStep(quadStep)
	{
		_srcPtMat.create(height,width,CV_32FC2);
		_map.create(height, width, CV_32FC2);
		_invMap.create(height, width, CV_32FC2);
		for(int i=0; i<height; i++)
		{
			float* ptr = _srcPtMat.ptr<float>(i);
			for(int j=0; j<width; j++)
			{
				ptr[2*j] = j;
				ptr[2*j+1] = i;
			}
		}
		

		_blkWidth = _width/_quadStep;
		_blkHeight = _height/_quadStep;
		_blkSize = _quadStep*_quadStep;
		//_blkPoints1.resize(_blkSize);
		//_blkPoints0.resize(_blkSize);
		
		_blkErrors.resize(_blkSize);
		for(int i=0; i<_blkSize; i++)
		{
			Cell cell;
			cell.idx = i;
			_cells.push_back(cell);
			_blkErrors[i] = -1;
			_emptyBlkIdx0.push_back(i);
		}
		_minNumForHomo = 10;
		_blkHomoVec.resize(_blkSize*8);
		_blkInvHomoVec.resize(_blkSize*8);
		cudaMalloc(&_dBlkHomoVec,sizeof(double)*_blkSize*8);
		cudaMalloc(&_dBlkInvHomoVec,sizeof(double)*_blkSize*8);
		_dMapXY[0].create(height,width,CV_32F);
		_dMapXY[1].create(height,width,CV_32F);
		_dIMapXY[0].create(height,width,CV_32F);
		_dIMapXY[1].create(height,width,CV_32F);
		InitMesh();
	};
	~BlockWarping()
	{
		cudaFree(_dBlkHomoVec);
		cudaFree(_dBlkInvHomoVec);
	}
	void Reset()
	{
		for(int i=0; i<_blkSize; i++)
		{
			_cells[i].points0.clear();
			_cells[i].points1.clear();
			_cells[i].idx = i;
			_blkErrors[i] = -1;
		}
		
		
		for(int i=0; i<_blkSize; i++)
			_emptyBlkIdx0.push_back(i);
	}
	void InitMesh()
	{
		std::vector<int> quadX, quadY;
	
		int x = 0;
		int halfWidth = _blkWidth/2;
		while( _width-1 - x > halfWidth)
		{
			quadX.push_back(x);
			x += _blkWidth;
		}
		quadX.push_back(_width-1);
		int halfHeight = _blkHeight/2;
		int y = 0;
		while(_height - y -1> halfHeight)
		{
			quadY.push_back(y);
			y += _blkHeight;
		}
		quadY.push_back(_height-1);

		

		_mesh.create(quadY.size(),quadX.size(),CV_32SC2);
		

		for(int i=0; i< _mesh.rows; i++)
		{
			int * ptr = _mesh.ptr<int>(i);
			for(int j=0; j<_mesh.cols; j++)
			{
				ptr[2*j] = quadX[j];
				ptr[2*j+1] = quadY[i];
			}
		}
	}
	void CalcBlkHomography();
	float MappingError(const cv::Mat& homo, const Points& p1, const Points& p2);
	float MappingError(const double* ptr, const Points& p1, const Points& p2);
	float MappingError(const std::vector<double>& homoVec, const Points& p1, const Points& p2);
	void SetFeaturePoints(const Points& p1, const Points& p2);
	virtual void Warp(const cv::Mat& img, cv::Mat& warpedImg);
	void GpuWarp(const cv::Mat& img, cv::Mat& warpedImg);
	virtual void GpuWarp(const cv::gpu::GpuMat& dimg, cv::gpu::GpuMat& dwimg);
	//·Ö¿éId
	int blockId(cv::Point2f pt)
	{

		int idx = (int)(pt.x+0.5) / _blkWidth;
		int idy = (int)(pt.y+0.5) / _blkHeight;
		return idx + idy*_quadStep;

	}
	virtual void getFlow(cv::Mat& flow);
	
	
private:
	int _width;
	int _height;
	int _quadStep;
	int _blkWidth;
	int _blkHeight;
	int _blkSize;
	//std::vector<Points> _blkPoints0,_blkPoints1;
	std::vector<Cell> _cells;
	//std::vector<cv::Mat> _blkHomos, _blkInvHomos;
	std::vector<double> _blkHomoVec,_blkInvHomoVec;
	std::vector<float> _blkErrors;
	std::vector<int> _emptyBlkIdx1, _emptyBlkIdx0;	
	cv::Mat _mesh;
	cv::Mat _srcPtMat;
	int _minNumForHomo;
	
	double* _dBlkHomoVec,*_dBlkInvHomoVec;

};

