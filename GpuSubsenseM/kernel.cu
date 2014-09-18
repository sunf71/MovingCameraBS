#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "GpuSuperpixel.h"
#include "SLIC.h"
#include "PictureHandler.h"
void testCudaGpu()
{
	try

	{

		cv::Mat src_host = cv::imread("in0000001.jpg");

		cv::gpu::GpuMat dst, src;

		src.upload(src_host);

		cv::gpu::cvtColor(src,src,CV_BGR2GRAY);

		cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);



		//cv::Mat result_host = dst;

		cv::Mat result_host;

		dst.download(result_host);

		cv::imshow("Result", result_host);

		cv::waitKey();

	}

	catch(const cv::Exception& ex)

	{

		std::cout << "Error: " << ex.what() << std::endl;

	}
}

int main (int argc, char* argv[])
{
	using namespace cv;
	Mat img = imread("in000001.jpg");
	cv::resize(img,img,cv::Size(16,16));
	float4* imgData = new float4[img.rows*img.cols];
	unsigned int* idata = new unsigned int[img.rows*img.cols];
	for(int i=0; i< img.cols; i++)
	{
		
		for(int j=0; j<img.rows; j++)
		{
			int idx = img.step[0]*j + img.step[1]*i;
			imgData[i + j*img.cols].x = img.data[idx];
			imgData[i + j*img.cols].y = img.data[idx+ img.elemSize1()];
			imgData[i + j*img.cols].z = img.data[idx+2*img.elemSize1()];
			imgData[i + j*img.cols].w = img.data[idx+3*img.elemSize1()];
			unsigned char tmp[4];
			for(int k=0; k<4; k++)
				tmp[k] = img.data[img.step[0]*j + img.step[1]*i + img.elemSize1()*k];
			idata[i + j*img.cols] = tmp[3]<<24 | tmp[2]<<16| tmp[1]<<8 | tmp[0];
		}
	}
	GpuSuperpixel gs;
	int num(0);
	int* labels = new int[img.rows*img.cols];
	gs.Superixel(imgData,img.cols,img.rows,3,0.9,num,labels);

	SLIC aslic;
	
	PictureHandler handler;
	Mat fimg(img.rows,img.cols,CV_32FC4,imgData);
	imwrite("border.jpg",fimg);
	aslic.DrawContoursAroundSegments(idata, labels, img.cols,img.rows,0x00ff00);
	handler.SavePicture(idata,img.cols,img.rows,std::string("mysuper.jpg"),std::string(".\\"));
	delete[] labels;

	return 0;

}
