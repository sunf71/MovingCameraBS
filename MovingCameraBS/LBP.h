#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>


double dxdy[] = 
{
	2,-0,
	1.84776,-0.765367,
	1.41421,-1.41421,
	0.765367,-1.84776,
	1.22465e-016,-2,
	-0.765367,-1.84776,
	-1.41421,-1.41421,
	-1.84776,-0.765367,
	-2,-2.44929e-016,
	-1.84776,0.765367,
	-1.41421,1.41421,
	-0.765367,1.84776,
	-3.67394e-016,2,
	0.765367,1.84776,
	1.41421,1.41421,
	1.84776,0.765367
};

//locol binary pattern
class LBP
{
protected:

const static size_t PATCH_SIZE = 5;
const static size_t DESC_SIZE = 2;
const static size_t NEIGHBOUR_SIZE = 16;

public:

	
	template<typename T>
	static void BilinearInterpolation(int width, int height, const T* data, float x, float y,T* out, int step = 3,int channel = 3)
	{

		if ( x >=0 && x <width && y>=0&& y< height)
		{
			int sx = (int)x;
			int sy = (int)y;
			int bx = sx +1;
			int by = sy +1;
			float tx = x - sx;
			float ty = y - sy;
			size_t idx_rgb_lu = (sx+sy*width)*step;
			size_t idx_rgb_ru = (bx+sy*width)*step;
			size_t idx_rgb_ld =(sx+by*width)*step;
			size_t idx_rgb_rd = (bx+by*width)*step;
			for(int c=0; c<channel; c++)
			{
				out[c] =(1- ty)*((1-tx)*data[idx_rgb_lu+c]+tx*data[idx_rgb_ru+c]) + ty*((1-tx)*data[idx_rgb_ld+c] + tx*data[idx_rgb_rd+c]);
			}

		}


	}
	template<typename T>
	static void BilinearInterpolation(int width, int height,const T* data, const int x,const int y,double* dxdy,T* out, int step = 3, int channel = 3)
	{
		double fx = x+dxdy[0];
		double fy = y+dxdy[1];
		if ( fx >=0 && fx <width && fy>=0&& fy< height)
		{
			int sx = (int)fx;
			int sy = (int)fy;
			int bx = sx +1;
			int by = sy +1;
			float tx = fx - sx;
			float ty = fy - sy;
			size_t idx_rgb_lu = (sx+sy*width)*step;
			size_t idx_rgb_ru = (bx+sy*width)*step;
			size_t idx_rgb_ld =(sx+by*width)*step;
			size_t idx_rgb_rd = (bx+by*width)*step;
			for(int c=0; c<channel; c++)
			{
				out[c] =(1- ty)*((1-tx)*data[idx_rgb_lu+c]+tx*data[idx_rgb_ru+c]) + ty*((1-tx)*data[idx_rgb_ld+c] + tx*data[idx_rgb_rd+c]);
			}

		}


	}
	static void LBPcomputeGrayscaleDescriptor(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _t, ushort& _res)
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC1);
		CV_DbgAssert(LBP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBP::PATCH_SIZE/2 && _y>=(int)LBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		int width = oInputImg.cols;
		int height = oInputImg.rows;
		uchar value[NEIGHBOUR_SIZE];
		_res = 0;
		for( int i=0; i<NEIGHBOUR_SIZE; i++)
			BilinearInterpolation(width,height,_data,_x,_y,&dxdy[2*i],value+i,1,1);

		/*for(int i=0; i<16; i++)
		{
			std::cout<<(int)value[i]<<" ";
		}
		std::cout<<std::endl;*/
		for(int i=0; i<NEIGHBOUR_SIZE/2; i++)
		{
			_res |= (value[i]-value[i+NEIGHBOUR_SIZE/2]>=0) << NEIGHBOUR_SIZE-1-i;
		}
		for(int i=0; i<NEIGHBOUR_SIZE; i+=2)
		{
			_res |= (value[i]-value[i+1]>=0) << NEIGHBOUR_SIZE/2-1-i;
		}
	
	

	}
	static void LBPcomputeRGBDescriptor(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _t, ushort* _res)
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(LBP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBP::PATCH_SIZE/2 && _y>=(int)LBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		int width = oInputImg.cols;
		int height = oInputImg.rows;
		uchar value[NEIGHBOUR_SIZE*3];
		
		for( int i=0; i<NEIGHBOUR_SIZE; i++)
			BilinearInterpolation(width,height,_data,_x,_y,&dxdy[2*i],value+3*i);
		for(int c=0; c<3; c++)
		{
			_res[c] = 0;
			for(int i=0; i<NEIGHBOUR_SIZE/2; i++)
			{
				_res[c] |= (value[i*3+c]-value[(i+NEIGHBOUR_SIZE/2)*3+c]>=0) << NEIGHBOUR_SIZE-1-i;
			}
			for(int i=0; i<NEIGHBOUR_SIZE; i+=2)
			{
				_res[c] |= (value[3*i+c]-value[3*(i+1)+c]>=0) << NEIGHBOUR_SIZE/2-1-i;
			}
		}

	

	}
	
};


//volum locol binary pattern
class VLBP
{
protected:

const static size_t PATCH_SIZE = 3;
const static size_t DESC_SIZE = 2;
const static size_t NEIGHBOUR_SIZE = 4;
const static size_t r = 1;
public:	
	static void computeVLBPGrayscaleDescriptor(const cv::Mat& img1, const int _x, const int _y,const cv::Mat& img2,  const int wx, const int wy, ushort& _res)
	{
		CV_DbgAssert(!img1.empty() && !img2.empty());
		CV_DbgAssert(img1.type()==CV_8UC1);
		
		CV_DbgAssert(_x>=(int)VLBP::PATCH_SIZE/2 && _y>=(int)VLBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<img1.cols-(int)VLBP::PATCH_SIZE/2 && _y<img1.rows-(int)VLBP::PATCH_SIZE/2);
		const size_t _step_row = img1.step.p[0];
		const uchar* const _data1= img1.data;
		const uchar* const _data2 = img2.data;
		int width = img1.cols;
		int height = img1.rows;
		uchar value[NEIGHBOUR_SIZE*2+1];
		_res = 0;
		value[0] = *(_data1 + (_x-1) + _y*_step_row);
		value[1] = *(_data1 + (_x) + (_y+1)*_step_row);
		value[2] = *(_data1 + (_x+1) + _y*_step_row);
		value[3] = *(_data1 + (_x) +( _y-1)*_step_row);
		value[4] = *(_data2 + (wx) +( wy)*_step_row);
		value[5] = *(_data2 + (wx-1) +( wy)*_step_row);
		value[5] = *(_data2 + (wx) +( wy+1)*_step_row);
		value[7] = *(_data2 + (wx+1) +( wy)*_step_row);
		value[8] = *(_data2 + (wx) +( wy-1)*_step_row);
		/*for(int i=0; i<16; i++)
		{
			std::cout<<(int)value[i]<<" ";
		}
		std::cout<<std::endl;*/
		uchar cvalue = *(_data1+_x+_y*_step_row);
		size_t size = NEIGHBOUR_SIZE*2+1;
		for(int i=0; i<size; i++)
		{
			_res |= (value[i]-cvalue>=0) << size-1-i;
		}
		
	
	

	}
	static void computeLBPGrayscaleDescriptor(const cv::Mat& img1, const int _x, const int _y, ushort& _res)
	{
		CV_DbgAssert(!img1.empty());
		CV_DbgAssert(img1.type()==CV_8UC1);
		
		CV_DbgAssert(_x>=(int)VLBP::PATCH_SIZE/2 && _y>=(int)VLBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<img1.cols-(int)VLBP::PATCH_SIZE/2 && _y<img1.rows-(int)VLBP::PATCH_SIZE/2);
		const size_t _step_row = img1.step.p[0];
		const uchar* const _data1= img1.data;
		
		int width = img1.cols;
		int height = img1.rows;
		uchar value[NEIGHBOUR_SIZE];
		
		_res = 0;
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			double dx = r*cos(2*M_PI*i/NEIGHBOUR_SIZE);
			std::cout<< -1*r*sin(2*M_PI*i/4)<<std::endl;
			double dy = -1*r*sin(2*M_PI*i/4);
			LBP::BilinearInterpolation(width,height,_data1,_x+dx,_y+dy,value+i,1,1);
		}
		
		uchar cvalue = *(_data1+_x+_y*_step_row);
		
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			_res |= (value[i]-cvalue>=0) << NEIGHBOUR_SIZE-1-i;
		}
		
	
	

	}
	
	
};


