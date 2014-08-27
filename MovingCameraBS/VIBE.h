#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define NUM_SAMPLES 20		//ÿ�����ص����������
#define MIN_MATCHES 2		//#minָ��
#define RADIUS 20		//Sqthere�뾶
#define SUBSAMPLE_FACTOR 16	//�Ӳ�������


class ViBe_BGS
{
public:
	ViBe_BGS(void);
	~ViBe_BGS(void);

	void init(const Mat& _image);   //��ʼ��
	void processFirstFrame(const Mat& _image);
	void processFirstFrame(const Mat& _image, const Mat& mask);
	void testAndUpdate(const Mat& _image, const Mat& mask);  //����
	void testAndUpdate(const Mat& _image);
	Mat getMask(void){return m_mask;};
	
	void WarpPerspective(Mat& homography)
	{
		cv::warpPerspective(m_mask, // input image
			m_mask,			// output image
			homography,		// homography
			cv::Size(m_mask.cols,m_mask.rows)); // size of output image

		cv::warpPerspective(m_foregroundMatchCount, // input image
			m_foregroundMatchCount,			// output image
			homography,		// homography
			cv::Size(m_foregroundMatchCount.cols,m_foregroundMatchCount.rows)); // size of output image

		for( int i=0; i<NUM_SAMPLES; i++)
		{
			cv::warpPerspective(m_samples[i], // input image
			m_samples[i],			// output image
			homography,		// homography
			cv::Size(m_samples[i].cols,m_samples[i].rows)); // size of output image
		}
	}
	
private:
	Mat m_samples[NUM_SAMPLES];
	Mat m_foregroundMatchCount;
	Mat m_mask;
};