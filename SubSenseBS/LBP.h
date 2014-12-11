#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

//double dxdy[] = 
//{
//	2,-0,
//	1.84776,-0.765367,
//	1.41421,-1.41421,
//	0.765367,-1.84776,
//	1.22465e-016,-2,
//	-0.765367,-1.84776,
//	-1.41421,-1.41421,
//	-1.84776,-0.765367,
//	-2,-2.44929e-016,
//	-1.84776,0.765367,
//	-1.41421,1.41421,
//	-0.765367,1.84776,
//	-3.67394e-016,2,
//	0.765367,1.84776,
//	1.41421,1.41421,
//	1.84776,0.765367
//};
double dxdy[] = 
{
	2,-0,
	2,-1,
	2,-2,
	1,-2,
	0,-2,
	-1,-2,
	-2,-2,
	-2,-1,
	-2,0,
	-2,1,
	-2,2,
	-1,2,
	0,2,
	1,2,
	2,2,
	2,1
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
			int bx = min(sx +1,width-1);
			int by = min(sy +1,height-1);
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

const static size_t PATCH_SIZE = 5;
const static size_t DESC_SIZE = 2;
const static size_t NEIGHBOUR_SIZE = 16;
const static size_t r = 2;
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
		double dx[NEIGHBOUR_SIZE];
		double dy[NEIGHBOUR_SIZE];
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			dx[i] = r*cos(2*M_PI*i/NEIGHBOUR_SIZE);			
			dy[i] = -1.0*r*sin(2*M_PI*i/NEIGHBOUR_SIZE);
			
		}
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{			
			LBP::BilinearInterpolation(width,height,_data1,_x+dx[i],_y+dy[i],value+i,1,1);
			LBP::BilinearInterpolation(width,height,_data2,wx+dx[i],wy+dy[i],value+i+NEIGHBOUR_SIZE,1,1);
		}
		value[NEIGHBOUR_SIZE*2] = *(_data2+wx+wy*_step_row);
	
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
	static void computeVLBPRGBDescriptor(const cv::Mat& img1, const int _x, const int _y,const cv::Mat& img2,  const int wx, const int wy, const size_t* t, ushort* _res)
	{
		
		CV_DbgAssert(!img1.empty() && !img2.empty());
		CV_DbgAssert(img1.type()==CV_8UC3 && img2.type()== CV_8UC3);
		
		CV_DbgAssert(_x>=(int)VLBP::PATCH_SIZE/2 && _y>=(int)VLBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<img1.cols-(int)VLBP::PATCH_SIZE/2 && _y<img1.rows-(int)VLBP::PATCH_SIZE/2);
		memset(_res,0,3*sizeof(ushort));
		const size_t _step_row = img1.step.p[0];
		const size_t _step_col = img1.step.p[1];
		const uchar* const _data1= img1.data;
		const uchar* const _data2 = img2.data;
		int width = img1.cols;
		int height = img1.rows;
		const size_t size = NEIGHBOUR_SIZE*2+1;
		uchar value[size*3];
		
		double dx[NEIGHBOUR_SIZE];
		double dy[NEIGHBOUR_SIZE];
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			dx[i] = r*cos(2*M_PI*i/NEIGHBOUR_SIZE);			
			dy[i] = -1.0*r*sin(2*M_PI*i/NEIGHBOUR_SIZE);
			
		}
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{			
			LBP::BilinearInterpolation(width,height,_data1,_x+dx[i],_y+dy[i],value+i*3,3,3);
			LBP::BilinearInterpolation(width,height,_data2,wx+dx[i],wy+dy[i],value+(i+NEIGHBOUR_SIZE)*3,3,3);
		}
		
		for(int c=0; c<3; c++)
		{
			value[(NEIGHBOUR_SIZE*2)*3 +c] = *(_data2+wx*_step_col+wy*_step_row+c);
		
			uchar cvalue = *(_data1+_x*_step_col+_y*_step_row+c);
			
			for(int i=0; i<size; i++)
			{
				_res[c] |= (value[i*3+c]-cvalue>=t[c]) << size-1-i;
			}
		}
	}
	static void computeLBPGrayscaleDescriptor(const cv::Mat& img1, const int _x, const int _y, const size_t t, ushort& _res)
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
			
			double dy = -1.0*r*sin(2*M_PI*i/4);
			LBP::BilinearInterpolation(width,height,_data1,_x+dx,_y+dy,value+i,1,1);
		}
		
		uchar cvalue = *(_data1+_x+_y*_step_row);
		
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			_res |= (value[i]-cvalue>=t) << NEIGHBOUR_SIZE-1-i;
		}

	}

	static void computeLBPRGBDescriptor(const cv::Mat& img1, const int _x, const int _y, const size_t* t, ushort* _res)
	{
		CV_DbgAssert(!img1.empty());
		CV_DbgAssert(img1.type()==CV_8UC3);
		
		CV_DbgAssert(_x>=(int)VLBP::PATCH_SIZE/2 && _y>=(int)VLBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<img1.cols-(int)VLBP::PATCH_SIZE/2 && _y<img1.rows-(int)VLBP::PATCH_SIZE/2);
		const size_t _step_row = img1.step.p[0];
		const size_t _step_col = img1.step.p[1];
		const uchar* const _data1= img1.data;
		
		int width = img1.cols;
		int height = img1.rows;
		uchar value[NEIGHBOUR_SIZE];
		
		memset(_res,0,sizeof(ushort)*3);
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			double dx = r*cos(2*M_PI*i/NEIGHBOUR_SIZE);			
			double dy = -1.0*r*sin(2*M_PI*i/NEIGHBOUR_SIZE);
			LBP::BilinearInterpolation(width,height,_data1,_x+dx,_y+dy,value+3*i,3,3);
		}
		
		const uchar* cvalue =(_data1+_x*_step_col+_y*_step_row);
		
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			for(int c=0; c=3; c++)
				_res[c] |= (value[i*3+c]-cvalue[c] >=t[c]) << NEIGHBOUR_SIZE-1-i;
		}

	}
	
	
};

//multi block LBP
class MBLBP
{
public:

const static size_t PATCH_SIZE = 9;
const static size_t DESC_SIZE = 2;


public:	
	template<typename T>
	inline static T imgData(const T* data, const int step_row, const int channels, const int x, const int y, const int c=0)
	{
		return data[y*step_row+x*channels + c];

	}
	template<typename T>
	inline static void AVGBlock3x3(int width, int height,const T* data, const int x,const int y,T* out, int step_row, int channels = 3)
	{
		
		CV_DbgAssert(MBLBP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(x>=1 && y>=1);
		
		//中心3*3的平局值
		for(int c=0; c<channels; c++)
		{
			out[c] = (imgData(data,step_row,channels,x-1,y,c) +  imgData(data,step_row,channels,x-1,y-1,c) + imgData(data,step_row,channels,x-1,y+1,c)+
				 imgData(data,step_row,channels,x,y,c) +  imgData(data,step_row,channels,x,y-1,c)+  imgData(data,step_row,channels,x,y+1,c)+
				  imgData(data,step_row,channels,x+1,y,c) +  imgData(data,step_row,channels,x+1,y-1,c) +  imgData(data,step_row,channels,x+1,y+1,c))/9;
			
		}


	}
	                                                            
	inline static void computeGrayscaleDescriptor(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _t, ushort& _res)
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC1);
		CV_DbgAssert(MBLBP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)MBLBP::PATCH_SIZE/2 && _y>=(int)MBLBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)MBLBP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)MBLBP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		int width = oInputImg.cols;
		int height = oInputImg.rows;
		uchar values[8];
		AVGBlock3x3(width,height,_data,_x+3,_y,&values[0],_step_row,1);
		AVGBlock3x3(width,height,_data,_x+3,_y-3,&values[1],_step_row,1);
		AVGBlock3x3(width,height,_data,_x,_y-3,&values[2],_step_row,1);
		AVGBlock3x3(width,height,_data,_x-3,_y-3,&values[3],_step_row,1);
		AVGBlock3x3(width,height,_data,_x-3,_y,&values[4],_step_row,1);
		AVGBlock3x3(width,height,_data,_x-3,_y+3,&values[5],_step_row,1);
		AVGBlock3x3(width,height,_data,_x,_y+3,&values[6],_step_row,1);
		AVGBlock3x3(width,height,_data,_x+3,_y+3,&values[7],_step_row,1);
		uchar center;
		AVGBlock3x3(width,height,_data,_x,_y,&center,_step_row,1);
		_res = 0;
		for(int i=0; i<8; i++)
			_res |= ((values[i] - center) >= _t) << 7-i;
	

	}
	inline static void computeRGBDescriptor(const cv::Mat& oInputImg, const int _x, const int _y, const size_t* _t, ushort* _res)
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(MBLBP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)MBLBP::PATCH_SIZE/2 && _y>=(int)MBLBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)MBLBP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)MBLBP::PATCH_SIZE/2);
		const int _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		int width = oInputImg.cols;
		int height = oInputImg.rows;
		uchar values[24];
		AVGBlock3x3(width,height,_data,_x+3,_y,&values[0],_step_row);
		AVGBlock3x3(width,height,_data,_x+3,_y-3,&values[3],_step_row);
		AVGBlock3x3(width,height,_data,_x,_y-3,&values[6],_step_row);
		AVGBlock3x3(width,height,_data,_x-3,_y-3,&values[9],_step_row);
		AVGBlock3x3(width,height,_data,_x-3,_y,&values[12],_step_row);
		AVGBlock3x3(width,height,_data,_x-3,_y+3,&values[15],_step_row);
		AVGBlock3x3(width,height,_data,_x,_y+3,&values[18],_step_row);
		AVGBlock3x3(width,height,_data,_x+3,_y+3,&values[21],_step_row);
		uchar center[3];
		AVGBlock3x3(width,height,_data,_x,_y,center,_step_row);
		
		for (int c=0; c<3; c++)
		{
			_res[c] = 0;
			for(int i=0; i<8; i++)
				_res[c] |= ((values[3*i+c] - center[c]) >= _t[center[c]]) << 7-i;
		}

	

	}
	inline static void computeSingleRGBDescriptor(const cv::Mat& oInputImg, const uchar _ref, const int _x, const int _y, const size_t _c, const size_t _t, ushort& _res) 
	{

	}
	static void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) {
	cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImgSize,PATCH_SIZE/2);
	}
};

//locol gradient binary pattern
class LGBP
{
protected:

const static size_t PATCH_SIZE = 5;
const static size_t DESC_SIZE = 2;
const static size_t NEIGHBOUR_SIZE = 16;

public:
	                                                            
	static void computeGrayscaleDescriptor(const cv::Mat& dxImg, const cv::Mat& dyImg,const int _x, const int _y, const size_t _t, ushort* _res)
	{
		CV_DbgAssert(!dxImg.empty());
		CV_DbgAssert(dxImg.type()==CV_16SC1);
		CV_DbgAssert(LGBP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LGBP::PATCH_SIZE/2 && _y>=(int)LGBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<dxImg.cols-(int)LGBP::PATCH_SIZE/2 && _y<dxImg.rows-(int)LGBP::PATCH_SIZE/2);
		const size_t _step_row_x = dyImg.step.p[0];
		const size_t _step_row_y = dyImg.step.p[0];
		const short* const _dxData = (short*)dxImg.data;
		const short* const _dyData = (short*)dyImg.data;
		int width = dxImg.cols;
		int height = dxImg.rows;
		short dxValue[NEIGHBOUR_SIZE];
		short dyValue[NEIGHBOUR_SIZE];
		_res = 0;
		for( int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			LBP::BilinearInterpolation(width,height,_dxData,_x,_y,&dxdy[2*i],dxValue+i,2,1);
			LBP::BilinearInterpolation(width,height,_dyData,_x,_y,&dxdy[2*i],dyValue+i,2,1);
		}
		double rad[NEIGHBOUR_SIZE];
		double ang[NEIGHBOUR_SIZE];
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			rad[i] = sqrt((double)(dxValue[i]*dxValue[i] + dyValue[i]*dyValue[i]));
			ang[i] = atan(dyValue[i]*1.f/dxValue[i]);
		}
		/*for(int i=0; i<16; i++)
		{
			std::cout<<(int)value[i]<<" ";
		}
		std::cout<<std::endl;*/
		for(int i=0; i<NEIGHBOUR_SIZE/2; i++)
		{
			_res[0] |= (abs(rad[i]-rad[i+NEIGHBOUR_SIZE/2])>=0 )<< NEIGHBOUR_SIZE-1-i;
		}
		for(int i=0; i<NEIGHBOUR_SIZE; i+=2)
		{
			_res[0] |= (abs(ang[i]-ang[i+1])>=0 )<< NEIGHBOUR_SIZE/2-1-i;
		}
	
	

	}
	static void computeRGBDescriptor(const cv::Mat& dxImg, const cv::Mat& dyImg,const int _x, const int _y, const size_t* _t, ushort* _res)
	{
		CV_DbgAssert(!dxImg.empty());
		CV_DbgAssert(dxImg.type()==CV_16SC3);
		CV_DbgAssert(LGBP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LGBP::PATCH_SIZE/2 && _y>=(int)LGBP::PATCH_SIZE/2);
		CV_DbgAssert(_x<dxImg.cols-(int)LGBP::PATCH_SIZE/2 && _y<dxImg.rows-(int)LGBP::PATCH_SIZE/2);
		const size_t _step_row = dxImg.step.p[0];
		const short* const _dxData = (short*)dxImg.data;
		const short* const _dyData = (short*)dyImg.data;
		int width = dxImg.cols;
		int height = dxImg.rows;
		
		short dxValue[NEIGHBOUR_SIZE*3];
		short dyValue[NEIGHBOUR_SIZE*3];
		//short dxCenter = _dxData+ (_x+_y*width)*3*2
		for( int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			LBP::BilinearInterpolation(width,height,_dxData,_x,_y,&dxdy[2*i],dxValue+i*3,2,3);
			LBP::BilinearInterpolation(width,height,_dyData,_x,_y,&dxdy[2*i],dyValue+i*3,2,3);
		}
		double rad[NEIGHBOUR_SIZE*3];
		double ang[NEIGHBOUR_SIZE*3];
		for(int i=0; i<NEIGHBOUR_SIZE; i++)
		{
			for(int c=0; c<3; c++)
			{
				rad[i*3+c] = sqrt((double)(dxValue[i]*dxValue[i] + dyValue[i]*dyValue[i]));
				ang[3*i+c] = atan(dyValue[i]*1.f/dxValue[i]);
			}
		}
		
		for(int c=0; c<3; c++)
		{
			_res[c] = 0;
			/*for(int i=0; i<NEIGHBOUR_SIZE; i++)
			{
				_res[c] |= (abs(value[i*3+c]-avg[c]) >=_t[c])<< NEIGHBOUR_SIZE-1-i;
			}*/
			for(int i=0; i<NEIGHBOUR_SIZE/2; i++)
			{
				_res[c] |= ((rad[i*3+c]-rad[(i+NEIGHBOUR_SIZE/2)*3+c]) >= 0)<< NEIGHBOUR_SIZE-1-i;
				//_res[c] |= ((ang[(i+NEIGHBOUR_SIZE/2)*3+c]-ang[i*3+c]) >= 0)<< NEIGHBOUR_SIZE/2-1-i;
			}
			/*for(int i=0; i<NEIGHBOUR_SIZE; i+=2)
			{
				_res[c] |= (abs(value[3*i+c]-value[3*(i+1)+c])>=_t[c]) << NEIGHBOUR_SIZE/2-1-i;
			}*/
		}

	

	}
	
};