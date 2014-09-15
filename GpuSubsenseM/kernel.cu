#include <iostream>

#include <opencv2/opencv.hpp>

#include <opencv2/gpu/gpu.hpp>



int main (int argc, char* argv[])

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

    return 0;

}
