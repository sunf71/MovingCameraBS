#include "MotionEstimate.h"
#include "GpuSuperpixel.h"
void MotionEstimate::EstimateMotion(const Mat& curImg, const Mat& prevImg, Mat& transM)
{
	//super pixel
	
	int num;
	cv::Mat img;
	cv::cvtColor(curImg,img,CV_BGR2BGRA);
	_imgData = new uchar4[_width*_height];
	
	for(int i=0; i< img.cols; i++)
	{		
		for(int j=0; j<img.rows; j++)
		{
			int idx = img.step[0]*j + img.step[1]*i;
			_imgData[i + j*img.cols].x = img.data[idx];
			_imgData[i + j*img.cols].y = img.data[idx+ img.elemSize1()];
			_imgData[i + j*img.cols].z = img.data[idx+2*img.elemSize1()];
			_imgData[i + j*img.cols].w = img.data[idx+3*img.elemSize1()];			
		}
	}
	_gs->Superpixel(_imgData,num,_labels0);
	//find most match superpixel
}