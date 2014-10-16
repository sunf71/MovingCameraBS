#pragma once

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "DistanceUtils.h"
#define PI 3.1415926
//local structural binary similarity  pattern
class LSBSP
{
public:

	const static size_t PATCH_SIZE = 5;
	const static size_t DESC_SIZE = 2;
	static void SobelOperator(const cv::Mat& oInputImg, const int x, const int y, float &gx, float &gy)
	{
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		CV_DbgAssert(x>=1 && y>=1);
		CV_DbgAssert(x<oInputImg.cols-1 && y<oInputImg.rows-1);
		size_t lu_idx = x - 1 + (y-1)*_step_row;
		size_t l_idx = (x-1) + y*_step_row;
		size_t ld_idx = (x+1) + (y+1)*_step_row;
		size_t ru_idx = x + 1 + (y-1)*_step_row;
		size_t r_idx = (x+1) + y*_step_row;
		size_t rd_idx = (x+1) + (y+1)*_step_row;
		size_t u_idx = x + (y-1)*_step_row;
		size_t d_idx = x + (y+1)*_step_row;
		gx = abs((_data[lu_idx] + _data[l_idx]*2 + _data[ld_idx]) - (_data[ru_idx] + _data[r_idx]*2 + _data[rd_idx]));
		gy = abs((_data[lu_idx] + _data[u_idx]*2 + _data[ru_idx]) - (_data[ld_idx] + _data[d_idx]*2 + _data[rd_idx]));
	}
	static void SobelOperatorRGB(const cv::Mat& oInputImg, const int x, const int y, float* gx, float* gy)
	{
		const size_t _step_row = oInputImg.cols;
		const uchar* const _data = oInputImg.data;
		CV_DbgAssert(x>=1 && y>=1);
		CV_DbgAssert(x<oInputImg.cols-1 && y<oInputImg.rows-1);
		size_t lu_idx = (x - 3 + (y-1)*_step_row);
		size_t l_idx = ((x-3) + y*_step_row);
		size_t ld_idx = ((x-3) + (y+1)*_step_row);
		size_t ru_idx = (x + 3 + (y-1)*_step_row);
		size_t r_idx = ((x+3) + y*_step_row);
		size_t rd_idx = ((x+3) + (y+1)*_step_row);
		size_t u_idx = (x + (y-1)*_step_row);
		size_t d_idx = (x + (y+1)*_step_row);
		for(int c=0; c<3; c++)
		{
			gx[c] = abs((_data[lu_idx+c] + _data[l_idx+c]*2 + _data[ld_idx+c]) - (_data[ru_idx+c] + _data[r_idx+c]*2 + _data[rd_idx+c]));
			gy[c] = abs((_data[lu_idx+c] + _data[u_idx+c]*2 + _data[ru_idx+c]) - (_data[ld_idx+c] + _data[d_idx+c]*2 + _data[rd_idx+c]));
		}
	}
	static void LSBSPcomputeGrayscaleDescriptor(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _t, ushort& _res)
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC1);
		CV_DbgAssert(LSBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LSBSP::PATCH_SIZE/2 && _y>=(int)LSBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LSBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LSBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		const size_t binCap = DESC_SIZE*8;
		float bins[binCap];
		
		float bin_size = 180.0/DESC_SIZE*8;
		memset(bins,0,sizeof(float)*16);
		//进行梯度直方图计算
		for(int i=-2; i<2; i++)
		{
			for(int j=-2; j<2; j++)
			{
				float gx,gy;
				SobelOperator(oInputImg,_x+i,_y+j,gx,gy);
				float ang = atan(gy/(gx+0.000001))/PI*180 + 90;
				bins[(int)ang/16] += (gx+gy);
			}
		}
		
		_res = 0;
		for(int i=0; i<binCap; i++)
		{
			_res +=(bins[i] > _t) << (binCap-1-i);
		}
	

	}
	static void LSBSPcomputeRGBDescriptor(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _t, ushort* _res)
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(LSBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LSBSP::PATCH_SIZE/2 && _y>=(int)LSBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LSBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LSBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		const size_t binCap = DESC_SIZE*8;
		float bins[3][binCap];
		float bin_size = 180.0/DESC_SIZE*8;
		memset(bins,0,sizeof(float)*16*3);
		//进行梯度直方图计算
		for(int i=-2; i<2; i++)
		{
			for(int j=-2; j<2; j++)
			{
				float gx[3],gy[3];
				SobelOperatorRGB(oInputImg,_x+i,_y+j,gx,gy);
				for(int c=0; c<3; c++)
				{
					float ang = atan(gy[c]/(gx[c]+0.000001))/PI*180 + 90;
					bins[c][(int)ang/16] += (gx[c]+gy[c]);
				}
			}
		}
		
		for(int c=0; c<3; c++)
		{
			_res[c] = 0;
			for(int i=0; i<binCap; i++)
			{
				_res[c] +=(bins[c][i] > _t) << (binCap-1-i);
			}
		}
	

	}
};