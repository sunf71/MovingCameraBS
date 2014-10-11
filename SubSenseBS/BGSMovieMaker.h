#pragma once
#include <string>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
class BGSMovieMaker
{
public:
	

	static void MakeMovie(const char* maskPath,const char* imgPath,cv::Size size, int from, int to, const char* outFileName)
	{
		char fileName[200];
		cv::VideoWriter writer;
		writer.open(std::string(outFileName),0,25,size);
		
		for(int i=from; i<=to; i++)
		{
			sprintf(fileName,"%s\\bin%06d.png",maskPath,i);			
			cv::Mat mask = cv::imread(fileName);
			cv::cvtColor(mask,mask,CV_BGR2GRAY);
			bool conit = mask.isContinuous();
			
			sprintf(fileName,"%s\\in%06d.jpg",imgPath,i);
			cv::Mat img = cv::imread(fileName);
			conit = img.isContinuous();
			for(int i=0; i<img.rows; i++)
			{
				for(int j=0; j<img.cols; j++)
				{
					int idx_uchar = i*img.cols + j;
					int idx_uchar_rgb = idx_uchar*3;
					uchar* mPtr = mask.data+idx_uchar;
					uchar* imgPtr = img.data+idx_uchar_rgb;
					if (*mPtr == 0xff)
					{
						imgPtr[2] = 200;
					}
				}
			}
			char text[20];
			sprintf(text,"%d",i);
			cv::putText(img,text,cv::Point(10,20),CV_FONT_ITALIC,1,CV_RGB(250,250,250));
			writer<<img;
			/*sprintf(fileName,"%06d.jpg",i);
			cv::imwrite(fileName,img);*/
		}
		writer.release();
	}
};