#include <opencv\cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
inline void postProcessa(const Mat& img, Mat& mask)
{
	cv::Mat m_oFGMask_PreFlood(img.size(),CV_8U);
	cv::Mat m_oFGMask_FloodedHoles(img.size(),CV_8U);
	cv::morphologyEx(mask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_or(mask,m_oFGMask_FloodedHoles,mask);
	cv::bitwise_or(mask,m_oFGMask_PreFlood,mask);
	cv::medianBlur(mask,mask,3);
	
}
inline void postProcessSegments(const Mat& img, Mat& mask)
{
	int niters = 1;

	vector<vector<cv::Point> > contours,imgContours;
	vector<Vec4i> hierarchy,imgHierarchy;
	
	Mat temp;


	dilate(mask, temp, Mat(), cv::Point(-1, -1), niters);//ÅòÕÍ£¬3*3µÄelement£¬µü´ú´ÎÊýÎªniters
	erode(temp, temp, Mat(), cv::Point(-1, -1), niters * 2);//¸¯Ê´
	dilate(temp, temp, Mat(), cv::Point(-1, -1), niters);
	
	findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//ÕÒÂÖÀª
	
	

	cv::Mat cimg(mask.size(),CV_8UC3);
	cimg = cv::Scalar(0);
	double minArea = 10*10;
	Scalar color( 255, 255, 255 );
	for( int i = 0; i< contours.size(); i++ )
	{
		const vector<cv::Point>& c = contours[i];
		double area = fabs(contourArea(Mat(c)));
		if( area > minArea )
		{
			drawContours(cimg, contours, i, color, CV_FILLED, 8, hierarchy, 0, cv::Point());
			
		}
		
	}
	cv::cvtColor(cimg,mask,CV_BGR2GRAY);
}

inline void drawImageContour(const Mat& img, Mat& contourImg, vector<vector<cv::Point> >& contours)
{
	cv::Mat gray;
	if (img.channels() == 3)
	{
		cv::cvtColor(img, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = img.clone();
	}
	cv::GaussianBlur(gray, gray, cv::Size(3, 3),0);
	cv::Mat edge;
	cv::Canny(gray, edge, 100, 300);
	
	contourImg.create(img.size(), CV_8UC3);
	contourImg = cv::Scalar(0);
	contours.clear();
	vector<Vec4i> hierarchy;
	findContours(edge, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);//ÕÒÂÖÀª
	Scalar color(255, 255, 255);
	for (int i = 0; i< contours.size(); i++)
	{
		//const vector<cv::Point>& c = contours[i];
		/*double area = fabs(contourArea(Mat(c)));
		if (area > 100)*/
		{
			drawContours(contourImg, contours, i, color, 1, 8, hierarchy, 0, cv::Point());

		}

	}
	cv::cvtColor(contourImg, contourImg, CV_BGR2GRAY);
}