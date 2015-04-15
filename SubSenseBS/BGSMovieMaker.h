#pragma once
#include <string>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
class BGSMovieMaker
{
public:
	
	
	static void MakeMovie(const char* maskPath,const char* imgPath,cv::Size size, int from, int to, const char* outFileName, 
		int saveImg = 0, int type = 0, int fps = 25)
	{
		char fileName[200];
		cv::VideoWriter writer;
		writer.open(std::string(outFileName),CV_FOURCC('X', 'V', 'I', 'D'),fps,size);
		
		for(int i=from; i<=to; i++)
		{
			sprintf(fileName,"%s\\bin%06d.png",maskPath,i);			
			cv::Mat mask = cv::imread(fileName);
			cv::cvtColor(mask,mask,CV_BGR2GRAY);
			bool conit = mask.isContinuous();
			if (mask.size() != size)
			{
				cv::resize(mask, mask, size);
			}
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
					if (type == 0)
					{
						if (*mPtr != 0xff)
						{

							int gray(0);
							for (int c = 0; c < 3; c++)
								gray += imgPtr[c];
							gray /= 10;
							for (int c = 0; c < 3; c++)
								imgPtr[c] = gray;
						}
					}
					else
					{
						if (*mPtr == 0xff)
						{							
							for (int c = 2; c < 3; c++)
								imgPtr[c] = 0xff;
						}
					}
				}
			}
			if (saveImg)
			{
				sprintf(fileName, "%s\\%06d.jpg", maskPath, i);
				cv::imwrite(fileName,img);
			}
			char text[20];
			sprintf(text,"%d",i);
			cv::putText(img,text,cv::Point(10,20),CV_FONT_ITALIC,1,CV_RGB(255,215,0));
			writer<<img;
			
			
		}
		writer.release();
	}
};