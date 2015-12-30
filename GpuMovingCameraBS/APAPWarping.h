#include "BlockWarping.h"

class APAPWarping :public ImageWarping
{
public:

	APAPWarping();
	APAPWarping(int width, int height, int quadStep) :_width(width), _height(height), _quadStep(quadStep)
	{
		_srcPtMat.create(height, width, CV_32FC2);
		
		for (int i = 0; i<height; i++)
		{
			float* ptr = _srcPtMat.ptr<float>(i);
			for (int j = 0; j<width; j++)
			{
				ptr[2 * j] = j;
				ptr[2 * j + 1] = i;
			}
		}
		_map.create(height, width, CV_32FC2);
		_invMap.create(height, width, CV_32FC2);
		


		_blkWidth = _width / _quadStep;
		_blkHeight = _height / _quadStep;
		_blkSize = _quadStep*_quadStep;
		InitMesh();
	
		_gamma = 0.01;
		_sigma = 12.f;
		_sigmaSquared = _sigma*_sigma;

		
	}
	virtual void Reset()
	{
		
	}
	
	virtual void Solve();
	virtual void getFlow(cv::Mat& flow);
	virtual void Warp(const cv::Mat& img, cv::Mat& warpedImg);
	virtual void GpuWarp(const cv::gpu::GpuMat& dimg, cv::gpu::GpuMat& dwimg);
	virtual void WarpPt(const cv::Point2f& input, cv::Point2f& output);
	virtual void SetFeaturePoints(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2);
protected:
	void InitMesh()
	{
		_blkCenters.resize(_blkSize);
		for (size_t i = 0; i < _blkSize; i++)
		{
			int y = i / _quadStep;
			int x = i%_quadStep;
			_blkCenters[i].x = x*_blkWidth + _blkWidth*0.5;
			_blkCenters[i].y = y*_blkHeight + _blkHeight*0.5;
		}

		std::vector<int> quadX, quadY;

		int x = 0;
		int halfWidth = _blkWidth / 2;
		while (_width - 1 - x > halfWidth)
		{
			quadX.push_back(x);
			x += _blkWidth;
		}
		quadX.push_back(_width - 1);
		int halfHeight = _blkHeight / 2;
		int y = 0;
		while (_height - y - 1 > halfHeight)
		{
			quadY.push_back(y);
			y += _blkHeight;
		}
		quadY.push_back(_height - 1);



		_mesh.create(quadY.size(), quadX.size(), CV_32SC2);


		for (int i = 0; i < _mesh.rows; i++)
		{
			int * ptr = _mesh.ptr<int>(i);
			for (int j = 0; j < _mesh.cols; j++)
			{
				ptr[2 * j] = quadX[j];
				ptr[2 * j + 1] = quadY[i];
			}
		}

	}
	int _width;
	int _height;
	int _quadStep;	
	int _blkWidth;
	int _blkHeight;
	int _blkSize;
	//std::vector<Points> _blkPoints0,_blkPoints1;
	std::vector<Cell> _cells;
	//std::vector<cv::Mat> _blkHomos, _blkInvHomos;
	std::vector<double> _blkHomoVec, _blkInvHomoVec;
	std::vector<cv::Point2f> _blkCenters;
	cv::Mat _mesh;
	cv::Mat _T1, _T2;//normalize matrix;
	cv::Mat _A;//DLT Matrix;
	double* _dBlkHomoVec, *_dBlkInvHomoVec;
	double _gamma;
	double _sigma;
	double _sigmaSquared;
	std::vector<cv::Point2f> *_x1, *_x2;
	cv::Mat _srcPtMat;
};