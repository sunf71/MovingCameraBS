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


