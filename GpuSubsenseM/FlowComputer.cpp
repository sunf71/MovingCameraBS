#include "FlowComputer.h"
#include "flowIO.h"
using namespace cv;
template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}
void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}
void GpuDenseOptialFlow::DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow)
{
	cv::Mat gray1, gray2;
	if (curImg.channels() == 3)
		cv::cvtColor(curImg,gray1,CV_BGR2GRAY);
	else
		gray1 = curImg.clone();
	if (prevImg.channels() ==3)
		cv::cvtColor(prevImg,gray2,CV_BGR2GRAY);
	else
		gray2 = prevImg.clone();
	cv::gpu::GpuMat dCurImg(gray1);
	cv::gpu::GpuMat dPrevImg(gray2);
	
	cv::gpu::GpuMat du,dv;
	cv::gpu::PyrLKOpticalFlow dPyrLK;

	dPyrLK.dense(dCurImg,dPrevImg,du,dv);
	
	std::vector<cv::Mat> flows(2);
	du.download(flows[0]);
	dv.download(flows[1]);
	/*cv::Mat flowField;
	getFlowField(flows[0],flows[1],flowField);
	cv::imwrite("flow.jpg",flowField);
	*/
	cv::merge(flows,flow);
}
void SFDenseOptialFlow::DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow)
{
	cv::Mat img0,img1;
	cv::cvtColor(curImg,img0,CV_BGR2GRAY);
	cv::cvtColor(prevImg,img1,CV_BGR2GRAY);
	cv::calcOpticalFlowSF(img0,img1,flow,3,2,4);
	cv::Mat flowField;
	cv::Mat flows[2];
	cv::split(flow,flows);
	getFlowField(flows[0],flows[1],flowField);
	cv::imwrite("flow_SF.jpg",flowField);
}
void FarnebackDenseOptialFlow::DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow)
{
	cv::Mat img0,img1;
	if (curImg.channels() == 3)
	{
		cv::cvtColor(curImg,img0,CV_BGR2GRAY);
		cv::cvtColor(prevImg,img1,CV_BGR2GRAY);
	}
	else
	{
		img0 = curImg;
		img1 = prevImg;
	}
	//cv::calcOpticalFlowSF(img0,img1,flow,3,2,55);
	cv::calcOpticalFlowFarneback(img0,img1,flow,0.5, 5, 15, 5, 5, 1.2, 0);
	/*cv::Mat flowField;
	cv::Mat flows[2];
	cv::split(flow,flows);
	getFlowField(flows[0],flows[1],flowField);
	cv::imwrite("flow_Farnback.jpg",flowField);*/
}
void EPPMDenseOptialFlow::DenseOpticalFlow(const cv::Mat& curImg, const cv::Mat& prevImg, cv::Mat& flow)
{
	cv::Mat img0,img1;
	if (curImg.channels() == 3)
	{
		cv::cvtColor(curImg,img0,CV_BGR2GRAY);
		cv::cvtColor(prevImg,img1,CV_BGR2GRAY);
	}
	else
	{
		img0 = curImg;
		img1 = prevImg;
	}
	cv::imwrite("img0.png",img0);
	cv::imwrite("img1.png",img1);
	std::system(" EPPM_flow.exe  img0.png img1.png flow.flo flow.png ");
	ReadFlowFile(flow,"flow.flo");
}