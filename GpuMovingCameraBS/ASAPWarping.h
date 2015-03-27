#pragma once
#include "ImageWarping.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\gpu\gpumat.hpp>
#include "cuda.h"
#include "cuda_runtime.h"
using namespace std;
bool isPointInTriangular(const cv::Point2f& pt, const cv::Point2f& V0, const cv::Point2f& V1, const cv::Point2f& V2);




class Quad
{
public:
	Quad(const cv::Point2f& v00, const cv::Point2f& v01, const cv::Point2f& v10, const cv::Point2f& v11):_V00(v00),_V01(v01),_V10(v10),_V11(v11){};
	bool isPointIn(const cv::Point2f& pt)
	{
		bool in1 = isPointInTriangular(pt,_V00,_V01,_V11);
		bool in2 = isPointInTriangular(pt,_V00,_V10,_V11);
		return in1 || in2;
	}
	bool getBilinearCoordinates(const cv::Point2f& pt, std::vector<float>& coefficients)
	{
		coefficients.clear();
		float a_x = _V00.x - _V01.x - _V10.x + _V11.x;
		float  b_x = -_V00.x + _V01.x;
		float c_x = -_V00.x + _V10.x;
		float d_x = _V00.x - pt.x;

		float  a_y = _V00.y - _V01.y - _V10.y + _V11.y;
		float b_y = -_V00.y + _V01.y;
		float c_y = -_V00.y + _V10.y;
		float d_y = _V00.y - pt.y;

		float bigA = -a_y*b_x + b_y*a_x;
		float bigB = -a_y*d_x - c_y*b_x + d_y*a_x +b_y*c_x;
		float  bigC = -c_y*d_x + d_y*c_x;

		float tmp1 = -1;
		float tmp2 = -1;
		float tmp3 = -1;
		float tmp4 = -1;
		float k1(2),k2(2);
		if (bigB*bigB - 4*bigA*bigC >= 0.0)
		{
			if( abs(bigA) >= 0.000001)
			{
				tmp1 = ( -bigB + sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
				tmp2 = ( -bigB - sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
			}
			else                   
			{
				tmp1 = -bigC/bigB;
			}


			if ( tmp1 >= -0.999999 && tmp1 <= 1.000001)
			{   
				tmp3 = -(b_y*tmp1 + d_y) / (a_y*tmp1 + c_y);
				tmp4 = -(b_x*tmp1 + d_x) / (a_x*tmp1 + c_x);
				if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
				{   
					k1 = tmp1;

					k2 = tmp3;
				}
				else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
				{
					k1 = tmp1;
					k2 = tmp4;
				}
			}
			if ( tmp2 >= -0.999999 && tmp2 <= 1.000001)
			{   
				if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
				{
					k1 = tmp2;
					k2 = tmp3;
				}
				else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
				{
					k1 = tmp2;
					k2 = tmp4;
				}
			}
		}
		


		if (k1>=-0.999999 && k1<=1.000001 && k2>=-0.999999 && k2<=1.000001)
		{

			coefficients.push_back((1.0-k1)*(1.0-k2));
			coefficients.push_back(k1*(1.0-k2));
			coefficients.push_back((1.0-k1)*k2);
			coefficients.push_back(k1*k2);



			return true;
		}
		else
		{
			return false;
		}

	}
	float getMinX()
	{

		float  minx = min(_V00.x,_V01.x);
		minx = min(minx, _V10.x);
		minx = min(minx,_V11.x);
		return minx;
	}
	float getMaxX()
	{
		float  maxx = max(_V00.x,_V01.x);
		maxx = max(maxx, _V10.x);
		maxx = max(maxx,_V11.x);
		return maxx;
	}

	float  getMinY()
	{     
		float miny = min(_V00.y,_V01.y);
		miny = min(miny,_V10.y);
		miny = min(miny,_V11.y);
		return miny;
	}
	float getMaxY()
	{
		float maxy = max(_V00.y,_V01.y);
		maxy = max(maxy,_V10.y);
		maxy = max(maxy,_V11.y);
		return maxy;
	}
	cv::Point2f getV00()
	{
		return _V00;
	}
	cv::Point2f getV01()
	{
		return _V01;
	}
	cv::Point2f getV10()
	{
		return _V10;
	}
	cv::Point2f getV11()
	{
		return _V11;
	}
private:

	cv::Point2f _V00,_V01,_V10,_V11;
};

class Mesh
{
public:
	Mesh(int rows, int cols, int quadWidth, int quadHeight):_imgWidth(cols), _imgHeight(rows),_quadWidth(quadWidth), _quadHeight(quadHeight)
	{
		std::vector<int> quadX, quadY;
	
		int x = 0;
		int halfWidth = _quadWidth/2;
		while( _imgWidth-1 - x > halfWidth)
		{
			quadX.push_back(x);
			x += _quadWidth;
		}
		quadX.push_back(_imgWidth-1);
		int halfHeight = _quadHeight/2;
		int y = 0;
		while(_imgHeight - y -1> halfHeight)
		{
			quadY.push_back(y);
			y += _quadHeight;
		}
		quadY.push_back(_imgHeight-1);

		_meshWidth = quadX.size();
		_meshHeight = quadY.size();

		_xMat.create(_meshHeight,_meshWidth,CV_32F);
		_yMat.create(_meshHeight,_meshWidth,CV_32F);

		for(int i=0; i< _meshHeight; i++)
		{
			float* xPtr = _xMat.ptr<float>(i);
			float* yPtr = _yMat.ptr<float>(i);
			for(int j=0; j<_meshWidth; j++)
			{
				xPtr[j] = quadX[j];
				yPtr[j] = quadY[i];
			}
		}
	};
public:
	cv::Point2f getVertex(int i, int j)
	{
		if ( i<_meshWidth && j < _meshHeight)
			return cv::Point2f(_xMat.at<float>(i,j), _yMat.at<float>(i,j));
		else
			return cv::Point2f();
	}
	void setVertex(int i, int j, cv::Point2f& pt)
	{
		_xMat.at<float>(i,j) = pt.x;
		_yMat.at<float>(i,j) = pt.y;
	}

	Quad getQuad(int i, int j)
	{
		cv::Point2f v00 = getVertex(i-1,j-1);
		cv::Point2f v01 = getVertex(i-1,j);
		cv::Point2f v10 = getVertex(i,j-1);
		cv::Point2f v11 = getVertex(i,j);
		return Quad(v00,v01,v10,v11);
	}
	void drawMesh(const cv::Mat& img, int gap, cv::Mat& nimg)
	{
		nimg.create(img.rows+2*gap,img.cols+2*gap,img.type());		
		img.copyTo(nimg(cv::Rect(gap,gap,img.cols,img.rows)));
		cv::Scalar color(255,0,0);
		cv::Scalar color2(0,255,0);
		for(int i=0; i<_meshHeight; i++)
		{
			for(int j=0; j<_meshWidth; j++)
			{
				cv::Point2f point;
				point.x = _xMat.at<float>(i,j)+gap;
				point.y = _yMat.at<float>(i,j) +gap;
				cv::circle(nimg, point,3,color);
			}
		}
		for(int i=0; i<_meshHeight; i++)
		{
			cv::Point2f start,end;
			start.x = _xMat.at<float>(i,0)+gap;
			start.y = _yMat.at<float>(i,0)+gap;
			end.x = _xMat.at<float>(i,_meshWidth-1) + gap;
			end.y = _yMat.at<float>(i,_meshWidth-1) + gap;
			cv::line(nimg,start,end,color2);

		}
		for(int i=0; i<_meshWidth; i++)
		{
			cv::Point2f start,end;
			start.x = _xMat.at<float>(0,i)+gap;
			start.y = _yMat.at<float>(0,i)+gap;
			end.x = _xMat.at<float>(_meshHeight-1,i) + gap;
			end.y = _yMat.at<float>(_meshHeight-1,i) + gap;
			cv::line(nimg,start,end,color2);
		}
	}


	int _imgWidth, _imgHeight, _meshWidth, _meshHeight, _quadWidth, _quadHeight;

	cv::Mat _xMat, _yMat;
};

class ASAPWarping : public ImageWarping
{
public:
	ASAPWarping(int width, int height, int quadStep, float weight):_quadStep(quadStep),_quadWidth(width/quadStep),_quadHeight(height/quadStep),_weight(weight)
	{
		_source = new Mesh(height,width,_quadWidth,_quadHeight);
		_destin = new Mesh(height,width,_quadWidth,_quadHeight);
		_height = _source->_meshHeight;
		_width = _source->_meshWidth;

		_x_index.resize(_height*_width);
		_y_index.resize(_height*_width);
		for(int i=0; i<_height*_width; i++)
		{
			_x_index[i] = i;
			_y_index[i] = _height*_width+i;
		}
		_num_smooth_cons = (_height-2)*(_width-2)*16 + (2*(_width+_height)-8)*8+4*4;

		_columns = _width*_height*2;
		_homographies.resize((_width-1)*(_height-1));
		_invHomographies.resize((_width-1)*(_height-1));
		_SmoothConstraints = cv::Mat::zeros(_num_smooth_cons*5,3,CV_32F);
		_SCc = 0;

		//CreateSmoothCons(weight);
		_mapXY[0].create(height,width,CV_32F);
		_mapXY[1].create(height, width, CV_32F);
		_outMask = cv::Mat::zeros(height,width,CV_32F);
		_invMapXY[0] = _mapXY[0].clone();
		_invMapXY[1] = _mapXY[1].clone();
		//std::cout<<_SmoothConstraints;
		_blkSize = _quadStep*_quadStep;
		cudaMalloc(&_dBlkHomoVec,sizeof(double)*_blkSize*8);
		cudaMalloc(&_dBlkInvHomoVec,sizeof(double)*_blkSize*8);
		_dMapXY[0].create(height,width,CV_32F);
		_dMapXY[1].create(height,width,CV_32F);
		_dIMapXY[0].create(height,width,CV_32F);
		_dIMapXY[1].create(height,width,CV_32F);
		_blkHomoVec.resize(_blkSize*8);
		_blkInvHomoVec.resize(_blkSize*8);
	};
	void CreateSmoothCons(float weight);
	void CreateSmoothCons(std::vector<float> weights);
	~ASAPWarping()
	{
		delete _source;
		delete _destin;
		cudaFree(_dBlkHomoVec);
		cudaFree(_dBlkInvHomoVec);
	}
	void SetControlPts(std::vector<cv::Point2f>& inputsPts, std::vector<cv::Point2f>& outputsPts);
	void Reset()
	{
		//_rowCount = _sRowCount;
		_rowCount = 0;
		_sRowCount = 0;
		_SCc = 0;
	}
	void AddDataCons(int i, int j, double* hPtr, cv::Mat& b);
	//添加双线性差值dataterm
	void CreateMyDataConsB(int num, std::vector<cv::Mat>& homographies, cv::Mat& b);
	void CreateMyDataCons(int num, std::vector<cv::Mat>& homographies, cv::Mat& b);
	void CreateDataCons(cv::Mat& b);
	void MySolve(cv::Mat& b);
	void Solve();
	virtual void Warp(const cv::Mat& img1, cv::Mat& warpImg);
	virtual void GpuWarp(const cv::gpu::GpuMat& img1, cv::gpu::GpuMat& warpImg);
	void InvWarp(const cv::Mat& img1, cv::Mat& warpImg, int gap = 0);
	std::vector<cv::Mat>& getHomographies()
	{
		return _homographies;
	}
	std::vector<cv::Mat>& getInvHomographies()
	{
		return _invHomographies;
	}
	
	virtual void getFlow(cv::Mat& flow);
protected:
	void quadWarp(const cv::Mat& img, int row, int col, Quad& qd1, Quad& qd2);
	void calcQuadHomography(int row, int col, Quad& qd1, Quad& qd2);
	void getSmoothWeight(const cv::Point2f& V1, const cv::Point2f& V2, const cv::Point2f& V3, float& u, float& v)
	{
		float d1 = sqrt((V1.x - V2.x)*(V1.x - V2.x) + (V1.y - V2.y)*(V1.y - V2.y));
		float   d3 = sqrt((V2.x - V3.x)*(V2.x - V3.x) + (V2.y - V3.y)*(V2.y - V3.y));

		cv::Point2f v21 = cv::Point2f(V1.x-V2.x,V1.y-V2.y);
		cv::Point2f v23 = cv::Point2f(V3.x-V2.x,V3.y-V2.y);

		float cosin = v21.x*v23.x + v21.y*v23.y;
		cosin = cosin/(d1*d3);

		float u_dis = cosin*d1;
		u = u_dis/d3;

		float v_dis = sqrt(d1*d1 - u_dis*u_dis);
		v = v_dis/d3;
	}

	void addCoefficient_1(int i, int j, float weight)
	{
		//V3(i-1,j-1)
		//|
		//|____
		//V2   V1(i,j)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i,j-1);
		cv::Point2f  V3 = _source->getVertex(i-1,j-1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  i*_width+j-1;
		int coordv3 = (i-1)*_width+j-1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_2(int i, int j, float weight)
	{
		//     V3   V2
		//       _____
		//       |
		//       |
		//      V1(i,j)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i-1,j);
		cv::Point2f  V3 = _source->getVertex(i-1,j-1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  (i-1)*_width+j;
		int coordv3 = (i-1)*_width+j-1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_3(int i, int j, float weight)
	{
		//V3(i-1,j-1)
		//|
		//|____
		//V2   V1(i,j)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i-1,j);
		cv::Point2f  V3 = _source->getVertex(i-1,j+1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  (i-1)*_width+j;
		int coordv3 = (i-1)*_width+j+1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_4(int i, int j, float weight)
	{
		//          V3(i-1,j+1)
		//            |
		//    ______|
		//   V1     V2
		//   
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i,j+1);
		cv::Point2f  V3 = _source->getVertex(i-1,j+1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  i*_width+j+1;
		int coordv3 = (i-1)*_width+j+1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_5(int i, int j, float weight)
	{
		//             V1   V2
		//             _____
		//             |
		//             |
		//             V3
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i,j+1);
		cv::Point2f  V3 = _source->getVertex(i+1,j+1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  i*_width+j+1;
		int coordv3 = (i+1)*_width+j+1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_6(int i, int j, float weight)
	{
		//             V1(i,j)
		//             |
		//             |____
		//             V2   V3(i+1,j+1)
		//             (i+1,j)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i+1,j);
		cv::Point2f  V3 = _source->getVertex(i+1,j+1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  (i+1)*_width+j;
		int coordv3 = (i+1)*_width+j+1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_7(int i, int j, float weight)
	{
		//             V1(i,j)
		//             |
		//             |
		//             _______|
		//             V3      V2
		//             (i+1,j-1)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i+1,j);
		cv::Point2f  V3 = _source->getVertex(i+1,j-1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  (i+1)*_width+j;
		int coordv3 = (i+1)*_width+j-1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	void addCoefficient_8(int i, int j, float weight)
	{
		//             V2        V1(i,j)
		//             _________
		//             |
		//             |
		//             |
		//             V3(i+1,j-1)
		cv::Point2f V1 = _source->getVertex(i,j);
		cv::Point2f  V2 = _source->getVertex(i,j-1);
		cv::Point2f  V3 = _source->getVertex(i+1,j-1);

		float u,v;
		getSmoothWeight(V1,V2,V3,u,v);

		int coordv1 =  i*_width+j;
		int coordv2 =  i*_width+j-1;
		int coordv3 = (i+1)*_width+j-1;            


		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),v*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv1),(-1.0)*weight);

		//V1.x =  V2.x + u * (V3.x - V2.x) + v * (V2.y - V3.y);
		float* ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 


		//            _SmoothCons.add_row(rowCount,_y_index(coordv2),(1.0-u)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv3),u*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv3),v*weight);
		//            _SmoothCons.add_row(rowCount,_x_index(coordv2),(-1.0*v)*weight);
		//            _SmoothCons.add_row(rowCount,_y_index(coordv1),(-1.0)*weight);
		//            rowCount = rowCount+1;

		//V1.y = V2.y + u * (V3.y - V2.y) + v * (V3.x - V2.x);
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv2]; ptr[2] = (1.0-u)*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv3]; ptr[2] = u*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv2]; ptr[2] = v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _x_index[coordv3]; ptr[2] = -1.0*v*weight;  _SCc = _SCc + 1;
		ptr = _SmoothConstraints.ptr<float>(_SCc);
		ptr[0] = _rowCount; ptr[1] = _y_index[coordv1]; ptr[2] = -1.0*weight;  _SCc = _SCc + 1;

		_rowCount = _rowCount+1; 



	}
	
	
private:
	int  _height;
	int _width;
	int _quadWidth;
	int _quadHeight;
	int _quadStep;
	int _blkSize;
	float _weight;
	std::vector<float> _V00, _V01, _V10, _V11;
	std::vector<cv::Point2f> _orgPts, _destPts;
	std::vector<int> _x_index, _y_index;
	Mesh* _source, *_destin;
	int _columns;
	
	//smooth constraints
	int _num_smooth_cons;	
	cv::Mat _SmoothConstraints;
	int _SCc;
	std::vector<cv::Mat> _homographies;
	std::vector<cv::Mat> _invHomographies;
	
	cv::Mat _outMask;
	
	//data constraints
	std::vector<float>    _dataterm_element_i;
	std::vector<float>    _dataterm_element_j;
	std::vector<cv::Point2f>   _dataterm_element_orgPt;
	std::vector<cv::Point2f>    _dataterm_element_desPt;
	std::vector<float>   _dataterm_element_V00;
	std::vector<float>   _dataterm_element_V01;
	std::vector<float>  _dataterm_element_V10;
	std::vector<float>   _dataterm_element_V11;
	cv::Mat _DataConstraints;
	int _DCc;
	int _num_data_cons;
	cv::Mat _warpImg;
	int _rowCount;
	int _sRowCount;
	double* _dBlkHomoVec,*_dBlkInvHomoVec;
	std::vector<double> _blkHomoVec,_blkInvHomoVec;
	
};
