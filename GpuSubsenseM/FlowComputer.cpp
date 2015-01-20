#include "FlowComputer.h"
#include <opencv2/gpu/gpu.hpp>
#include "CudaSuperpixel.h"
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

//features0是超像素中心以及特征点的位置（前面nF个特征点，后面spSize个中心点）
//features1，status是KLT计算结果
//输出时去掉计算失败的点，返回KLT计算成功的特征点对应点坐标
void SuperpixelFlow(int spWidth, int spHeight, int spSize, const SLICClusterCenter* centers, int& nF,
	std::vector<cv::Point2f>& features0, std::vector<cv::Point2f>& features1, std::vector<uchar>& status,cv::Mat& flow)
{
	flow.create(spHeight,spWidth,CV_32FC2);
	flow = cv::Scalar(0);
	
	int k=0; 
	for(int i=0; i<nF; i++)
	{
		if (status[i] == 1)
		{
			features0[k] = features0[i];
			features1[k] = features1[i];
			k++;
		}
	}
	int tmp = k;
	for(int i=0; i<spHeight; i++)
	{
		float2 * ptr = flow.ptr<float2>(i);
		for(int j=0; j<spWidth; j++)
		{
			int idx = j + i*spWidth + nF;
			if (status[idx] == 1)
			{
				ptr[j].x = features1[idx].x - features0[idx].x;
				ptr[j].y = features1[idx].y - features0[idx].y;
				features0[k] = features0[idx];
				features1[k] = features1[idx];
				k++;
			}
			else
			{
				//KLT失败
				ptr[j].x = UNKNOWN_FLOW;
				ptr[j].y = UNKNOWN_FLOW;
			}
		}
	}
	//std::cout<<"tracking succeeded "<<k<<" total "<<spSize<<std::endl;
	nF = tmp;
	features0.resize(k);
	features1.resize(k);
}
void SuperpixelFlow(const cv::Mat& sgray, const cv::Mat& tgray,int step, int spSize, const SLICClusterCenter* centers, 
	std::vector<cv::Point2f>& features0, std::vector<cv::Point2f>& features1, cv::Mat& flow)
{
	features0.clear();
	features1.clear();
	std::vector<uchar> status;
	std::vector<float> err;
	int spWidth = (sgray.cols+step-1)/step;
	int spHeight = (sgray.rows+step-1)/step;
	flow.create(spHeight,spWidth,CV_32FC2);
	flow = cv::Scalar(0);
	for(int i=0; i<spSize; i++)
	{
		features0.push_back(cv::Point2f(centers[i].xy.x,centers[i].xy.y));
	}
	cv::calcOpticalFlowPyrLK(sgray,tgray,features0,features1,status,err);
	
	int k=0; 
	for(int i=0; i<spHeight; i++)
	{
		float2 * ptr = flow.ptr<float2>(i);
		for(int j=0; j<spWidth; j++)
		{
			int idx = j + i*spWidth;
			if (status[idx] == 1)
			{
				ptr[j].x = features1[idx].x - features0[idx].x;
				ptr[j].y = features1[idx].y - features0[idx].y;
				k++;
			}
			else
			{
				//KLT失败
				ptr[j].x = UNKNOWN_FLOW;
				ptr[j].y = UNKNOWN_FLOW;
			}
		}
	}
	
	
	
	std::cout<<"tracking succeeded "<<k<<" total "<<spSize<<std::endl;
}
float L1Dist(const SLICClusterCenter& center, const float2& pos, const uchar* rgb)
{
	float dx =center.xy.x - pos.x;
	float dy = center.xy.y - pos.y;
	float4 crgb = center.rgb;
	float dr = rgb[0] - crgb.x;
	float dg = rgb[1] - crgb.y;
	float db = rgb[2] - crgb.z;
	float d_rgb = sqrt(dr*dr + dg*dg + db*db);
	float d_xy = (dx*dx + dy*dy);
	float alpha(0.9);
	return (1-alpha)*d_rgb + alpha*d_xy/(5.f);
}
//用稀疏光流进行superpixel matching
void SuperpixelMatching(const int* labels0, const SLICClusterCenter* centers0, const cv::Mat& img0, int* labels1, const SLICClusterCenter* centers1, const cv::Mat& img1, int spSize, int spStep, int width, int height,
	const cv::Mat& spFlow, std::vector<int>& matchedId)
{
	int spWidth = spFlow.cols;
	int spHeight = spFlow.rows;
	matchedId.resize(spSize);
	memset(&matchedId[0],-1,sizeof(int)*spSize);
	for(int i=0; i<spSize; i++)
	{		

		int k = (int)(centers0[i].xy.x+0.5);
		int j = (int)(centers0[i].xy.y+0.5);

		if (centers0[i].nPoints >0)			
		{
			float avgX(0),avgY(0);
			int n(0);
			
			float2 flow = *(float2*)(spFlow.data+i*8);
			avgX = flow.x;
			avgY = flow.y;
			const float2 point = make_float2(avgX+k,avgY+j);
			int iavgX = (int)(avgX +0.5+k);
			int iavgY = (int)(avgY + 0.5+j);
			if (iavgX <0 || iavgX > width-1 || iavgY <0 || iavgY > height-1)
				continue;
			int idx = iavgX + iavgY*width;
			const uchar* color1 = (uchar*)(img1.data + idx*3);
			int label = labels1[idx];
			//在8邻域内查找中心点与avgX和avgY最接近的超像素
			//float disMin = L1Dist(centers1[label].xy,point);
			float disMin = L1Dist(centers1[label],point,color1);
			int minLabel = label;
			//left
			if (label-1 >=0)
			{
				//float dist =L1Dist(centers1[label-1].xy,point);
				float dist = L1Dist(centers1[label-1],point,color1);
				if (dist<disMin)
				{
					minLabel = label-1;
					disMin = dist;
				}
				if (label-1-spWidth >=0)
				{
					//dist =L1Dist(centers1[label-1-spWidth-1].xy,point);
					dist = L1Dist(centers1[label-1-spWidth-1],point,color1);
					if (dist<disMin)
					{
						minLabel = label-1-spWidth;
						disMin = dist;
					}
				}
				if(label-1+spWidth < spSize)
				{
					//dist =L1Dist(centers1[label-1+spWidth].xy,point);
					dist = L1Dist(centers1[label-1+spWidth],point,color1);
					if (dist<disMin)
					{
						minLabel = label-1+spWidth;
						disMin = dist;
					}
				}
			}
			if (label+1 <spSize)
			{
				//float dist =L1Dist(centers1[label+1].xy,point);
				float dist = L1Dist(centers1[label+1],point,color1);
				if (dist<disMin)
				{
					minLabel = label+1;
					disMin = dist;
				}
				if (label+1-spWidth >=0)
				{
					//dist =L1Dist(centers1[label+1-spWidth-1].xy,point);
					dist = L1Dist(centers1[label+1-spWidth ],point,color1);
					if (dist<disMin)
					{
						minLabel = label+1-spWidth;
						disMin = dist;
					}
				}
				if(label+1+spWidth < spSize)
				{
					//dist =L1Dist(centers1[label+1+spWidth].xy,point);
					dist =L1Dist(centers1[label+1+spWidth],point,color1);
					if (dist<disMin)
					{
						minLabel = label+1+spWidth;
						disMin = dist;
					}
				}
			}
			if (label+spWidth <spSize)
			{
				//dist =L1Dist(centers1[label+spWidth].xy,point);
				float dist =L1Dist(centers1[label+spWidth],point,color1);
				if (dist<disMin)
				{
					minLabel = label+spWidth;
					disMin = dist;
				}
			}
			if (label-spWidth >=0)
			{
				//dist =L1Dist(centers1[label-spWidth].xy,point);
				float dist =L1Dist(centers1[label-spWidth],point,color1);
				if (dist<disMin)
				{
					minLabel = label-spWidth;
					disMin = dist;
				}
			}
			matchedId[i] = (minLabel);
		}
		
	}
}