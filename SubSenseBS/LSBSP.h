#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "DistanceUtils.h"

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
		size_t ld_idx = (x-1) + (y+1)*_step_row;
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
				float ang = atan(gy/(gx+0.000001))/M_PI*180 + 90;
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
					float ang = atan(gy[c]/(gx[c]+0.000001))/M_PI*180 + 90;
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
	static void LSBSPcomputeGrayscaleHistogram(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _t, std::vector<float>& hist)
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC1);
		CV_DbgAssert(LSBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LSBSP::PATCH_SIZE/2 && _y>=(int)LSBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LSBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LSBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		const size_t binCap = DESC_SIZE*8;
		hist.resize(binCap);
	
		
		float bin_size = 180.0/DESC_SIZE*8;
		memset(&hist[0],0,sizeof(float)*binCap);
		//进行梯度直方图计算
		for(int i=-2; i<2; i++)
		{
			for(int j=-2; j<2; j++)
			{
				float gx,gy;
				SobelOperator(oInputImg,_x+i,_y+j,gx,gy);
				float ang = atan(gy/(gx+0.000001))/M_PI*180 + 90;
				hist[(int)ang/16] += (gx+gy);
			}
		}

	}
	static void LSBSPcomputeHistogram(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _t, std::vector<std::vector<float>>& hist)
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(LSBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LSBSP::PATCH_SIZE/2 && _y>=(int)LSBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LSBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LSBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		const size_t binCap = DESC_SIZE*8;
		hist.resize(3);
		for(int c=0; c<3; c++)
		{
			hist[c].resize(binCap);
			memset(&hist[c][0],0,sizeof(float)*binCap);
		}
		float bin_size = 180.0/binCap;
	
		//进行梯度直方图计算
		for(int i=-2; i<2; i++)
		{
			for(int j=-2; j<2; j++)
			{
				float gx[3],gy[3];
				SobelOperatorRGB(oInputImg,_x+i,_y+j,gx,gy);
				for(int c=0; c<3; c++)
				{
					float ang = atan(gy[c]/(gx[c]+0.000001))/M_PI*180 + 90;
					hist[c][(int)ang/16] += (gx[c]+gy[c]);
				}
			}
		}
		

	}
};

// Local Structrue Tensor Binary Pattern
class LSTBP
{
public:
	const static size_t binSize = 16 ;
	const static size_t patchSize = 5;
	const static int halfPSize = patchSize/2;
	inline static void computeRGBDescriptor(const cv::Mat& dxImg, const cv::Mat& dyImg, const int _x, const int _y,ushort* binPattern)
	{
		std::vector<std::vector<float>>histogram;
		histogram.resize(3);			
		int width = dxImg.cols;
		int height = dxImg.rows;
		CV_DbgAssert(!dxImg.empty() && !dyImg.empty());
		CV_DbgAssert(dxImg.type()==CV_16SC3 && dyImg.type()==CV_16SC3);
		if ( _x - halfPSize< 0 || _x+halfPSize > width-1 || _y-halfPSize <0 || _y+halfPSize > height-1)
			return;
		const int _step0 = dxImg.step.p[0];
		const int _step1 = dxImg.step.p[1];
		const uchar* const _dxData = dxImg.data;
		const uchar* const _dyData = dyImg.data;
		float bin_step = (180+binSize-1)/binSize;
		for(int c=0; c<3; c++)
		{
			histogram[c].resize(binSize);
			memset(&histogram[c][0],0,sizeof(float)*binSize);
			for(int i=-halfPSize; i<=halfPSize; i++)
			{
				for(int j=-halfPSize; j<= halfPSize; j++)
				{
					int x = i+_x;
					int y = j+_y;
					int idx = y*_step0 +x*_step1;
					short dx = *((short*)(_dxData +idx)+c);
					short dy = *((short*)(_dyData +idx)+c);
					double j11 = dx*dx;
					double j12 = dx*dy;
					double j22 = dy*dy;
					double tmp = sqrt(1.0*(j22-j11)*(j22-j11)+4*j12*j12);
					double lmdMax = 0.5*(j11+j22+tmp);
					double lmdMin =  0.5*(j11+j22-tmp);
					double orientation(0);		
					if (abs((j22-j11+tmp)< 1e-6))
					{
						orientation = 90.0;
					}
					else
					{
						orientation = atan(2*j12/(j22-j11+tmp))/M_PI*180.0;

					}
					if (orientation < 0)
						orientation += 180;
					/*float ang = atan(dy/(dx+1e-6))/M_PI*180;
					if (ang<0)
					ang+=180;
					histogram[ang/bin_step] += (abs(dx)+abs(dy));*/
					histogram[c][(int)(orientation/bin_step)] += lmdMax;

				}
			}
			double max = histogram[c][0];
			int idx(0);
			for(int i=1; i<binSize; i++)
			{
				if (histogram[c][i] > max)
				{
					max = histogram[c][i];
					idx = i;
				}
			}
			binPattern[c] = 0;
			for(int i=0; i<binSize; i++)
			{
				if (histogram[c][i] > max*0.5)
				{
					binPattern[c] |= 1 << (binSize-i-1);
				}
			}
		}
	}
	inline static void computeGrayscaleDescriptor(const cv::Mat& dxImg, const cv::Mat& dyImg,const int _x, const int _y, ushort& binPattern)
	{	
		std::vector<float> histogram;
		histogram.resize(binSize);
		memset(&histogram[0],0,sizeof(float)*binSize);	
		int width = dxImg.cols;
		int height = dxImg.rows;
		CV_DbgAssert(!dxImg.empty());
		CV_DbgAssert(dxImg.type()==CV_16SC1);
		if ( _x - halfPSize< 0 || _x+halfPSize > width-1 || _y-halfPSize <0 || _y+halfPSize > height-1)
			return;
		const int _step0 = dxImg.step.p[0];
		const int _step1 = dxImg.step.p[1];
		const uchar* const _dxData = dxImg.data;
		const uchar* const _dyData = dyImg.data;
		float bin_step = (180+binSize-1)/binSize;
		for(int i=-halfPSize; i<=halfPSize; i++)
		{
			for(int j=-halfPSize; j<= halfPSize; j++)
			{
				int x = i+_x;
				int y = j+_y;
				int idx = y*_step0 +x*_step1;
				short dx = *((short*)(_dxData +idx ));
				short dy = *((short*)(_dyData +idx));
				double j11 = dx*dx;
				double j12 = dx*dy;
				double j22 = dy*dy;
				double tmp = sqrt(1.0*(j22-j11)*(j22-j11)+4*j12*j12);
				double lmdMax = 0.5*(j11+j22+tmp);
				double lmdMin =  0.5*(j11+j22-tmp);
				double orientation(0);		
				if (abs((j22-j11+tmp)< 1e-6))
				{
					orientation = 90.0;
				}
				else
				{
					orientation = atan(2*j12/(j22-j11+tmp))/M_PI*180.0;

				}
				if (orientation < 0)
					orientation += 180;
				/*float ang = atan(dy/(dx+1e-6))/M_PI*180;
				if (ang<0)
				ang+=180;
				histogram[ang/bin_step] += (abs(dx)+abs(dy));*/
				histogram[(int)(orientation/bin_step)] += lmdMax;

			}
		}
		double max = histogram[0];
		int idx(0);
		for(int i=1; i<binSize; i++)
		{
			if (histogram[i] > max)
			{
				max = histogram[i];
				idx = i;
			}
		}
		binPattern = 0;
		for(int i=0; i<binSize; i++)
		{
			if (histogram[i] > max*0.5)
			{
				binPattern |= 1 << (binSize-i-1);
			}
		}

	}
};
