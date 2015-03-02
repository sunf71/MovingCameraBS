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

	vector<vector<Point> > contours,imgContours;
	vector<Vec4i> hierarchy,imgHierarchy;
	
	Mat temp;


	dilate(mask, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
	erode(temp, temp, Mat(), Point(-1,-1), niters*2);//腐蚀
	dilate(temp, temp, Mat(), Point(-1,-1), niters);
	
	findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//找轮廓
	
	

	cv::Mat cimg(mask.size(),CV_8UC3);
	cimg = cv::Scalar(0);
	double minArea = 10*10;
	Scalar color( 255, 255, 255 );
	for( int i = 0; i< contours.size(); i++ )
	{
		const vector<Point>& c = contours[i];
		double area = fabs(contourArea(Mat(c)));
		if( area > minArea )
		{
			drawContours( cimg, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );
			
		}
		
	}
	cv::cvtColor(cimg,mask,CV_BGR2GRAY);
}