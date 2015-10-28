#include "ImageFocusness.h"
#include <opencv\highgui.h>
#define _USE_MATH_DEFINES
#include <math.h>



void CalScale(const cv::Mat& gray, cv::Mat& scaleMap)
{
	int width = gray.cols;
	int height = gray.rows;

	std::vector<float> scale;
	for (size_t i = 1; i < 16; i++)
	{
		for (size_t j = 0; j < 4; j++)
			scale.push_back(i + 0.25*j);
	}
	scale.push_back(16);
	scaleMap = cv::Mat::zeros(gray.size(), CV_32F);
	cv::Mat gradp = cv::Mat::zeros(gray.size(), CV_32F);
	cv::Mat graddp = cv::Mat::zeros(gray.size(), CV_32F);

	for (size_t i = 0; i < scale.size(); i++)
	{
		float w = 4 * scale[i];
		std::vector<float> x;
		for (float f = -w; f <= w; f++)
			x.push_back(f);

		float slsq = scale[i] * scale[i];
		cv::Mat_<float> smoothFilter(1, x.size());
		cv::Mat_<float> differFileter(1, x.size());

	
	
		for (size_t f = 0; f < x.size(); f++)
		{
			float s = M_SQRT1_2*M_2_SQRTPI / 2 / scale[i] * exp(-x[f] * x[f] / 2 / slsq);
			*((float*)(smoothFilter.data + f * 4)) = s;
			float d = -x[f] * M_SQRT1_2*M_2_SQRTPI / 2 / scale[i] * exp(-x[f] * x[f] / 2 / slsq);
			*((float*)(differFileter.data + f * 4)) = d;
			
		}
		
		//std::cout << "differFileter" << differFileter << "\n";
		//std::cout << "smoothFilter" << smoothFilter << "\n\n";

		differFileter /= cv::sum(cv::abs(differFileter))[0];
		smoothFilter /= cv::sum(smoothFilter)[0];

		//std::cout << "differFileter" << differFileter << "\n";
		//std::cout << "smoothFilter" << smoothFilter << "\n\n";
		
		
		cv::Mat smoothIx, gradIy, smoothIy, gradIx, tmp;

		cv::filter2D(gray, smoothIx, CV_32F, smoothFilter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);		
		smoothIx(cv::Rect(w, 0, smoothIx.cols - 2 * w, smoothIx.rows)).copyTo(tmp);
		cv::copyMakeBorder(tmp, smoothIx, 0, 0, w, w, cv::BORDER_REPLICATE);
		
		cv::filter2D(smoothIx, gradIy, CV_32F, differFileter.t(), cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	    gradIy(cv::Rect(0, w, smoothIx.cols, smoothIx.rows - 2 * w)).copyTo(tmp);
		cv::copyMakeBorder(tmp, gradIy, w, w, 0, 0, cv::BORDER_REPLICATE);
		

		cv::filter2D(gray, smoothIy, CV_32F, smoothFilter.t(), cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);		
		smoothIy(cv::Rect(0, w, smoothIx.cols, smoothIx.rows - 2 * w)).copyTo(tmp);
		cv::copyMakeBorder(tmp, smoothIy, w, w, 0, 0, cv::BORDER_REPLICATE);
		
		cv::filter2D(smoothIy, gradIx, CV_32F, differFileter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
		gradIx(cv::Rect(w, 0, smoothIx.cols - 2 * w, smoothIx.rows)).copyTo(tmp);
		cv::copyMakeBorder(tmp, gradIx, 0, 0, w, w, cv::BORDER_REPLICATE);

		
		
		cv::pow(gradIx, 2, gradIx);
		cv::pow(gradIy, 2, gradIy);
		cv::Mat_<float> gradI;
		cv::add(gradIx, gradIy, gradI);
		cv::pow(gradI, 0.5, gradI);


		
		cv::Mat gradd = gradI - gradp;
		cv::threshold(gradd, gradd, 0, 255, cv::THRESH_TOZERO);
		/*std::cout << "--------\n";
		tmp = gradd(cv::Rect(100, 100, 5, 5));
		std::cout << tmp << "\n";*/

		gradp = gradI.clone();

		if (i > 1)
		{
			cv::Mat diff = gradd - graddp;
			cv::threshold(diff, diff, 0, 255, cv::THRESH_BINARY);
			diff.convertTo(diff, CV_8U);			
			scaleMap.setTo(scale[i], diff);
			
		}

		graddp = gradd.clone();
	}

	

	for (size_t i = 0; i < scaleMap.rows; i++)
	{
		float* ptr = scaleMap.ptr<float>(i);
		for (size_t j = 0; j < scaleMap.cols; j++)
		{
			if (i == 0 || i == scaleMap.rows - 1 || j == 0 || j == scaleMap.cols - 1)
				ptr[j] = -0.2;
			else if (abs(ptr[j]) < 1e-10)
			{
				ptr[j] = 0.5*sqrt(2)*scale[scale.size() - 1];

			}
			else
			{
				ptr[j] = 0.5*sqrt(2)*ptr[j];
			}
		}
	}
	cv::Mat gradScale;
	cv::normalize(scaleMap, gradScale, 255, 0, CV_MINMAX, CV_8U);
	cv::imshow("scale", gradScale);
	cv::waitKey();
}



void DogGradMap(const cv::Mat& grayImg, cv::Mat& grad)
{
	
	//float sigmaBig = 70.0 / 10.0f;
	//float sigmaSmall = 60.0 / 100.0f;
	int ksize = 7;
	float sigmaSmall = 0.3*((ksize - 1)*0.5 - 1) + 0.8;
	float sigmaBig = sigmaSmall * 70 / 6;
	//int ksize = ceilf((sigmaBig - 0.8f) / 0.3f) * 2 + 3;
	
	cv::Mat gauBig = cv::getGaussianKernel(ksize, sigmaBig, CV_32F);
	cv::Mat gauSmall = cv::getGaussianKernel(ksize, sigmaSmall, CV_32F);

	cv::Mat DoG = gauSmall - gauBig;
	cv::filter2D(grayImg, grad, CV_32F, DoG.t());
	grad = cv::abs(grad);
	cv::Mat gradGray;
	cv::normalize(grad, gradGray, 255, 0, CV_MINMAX, CV_8U);
	cv::imshow("dog", gradGray);
	cv::waitKey();

}
void CalRegionFocusness(const cv::Mat& gray, const cv::Mat& edgeMap, std::vector<std::vector<uint2>>& spPoses, std::vector<SPRegion>& regions, cv::Mat& rst)
{
	cv::Mat img = gray.clone();
	cv::Mat scaleMap, grad;
	CalScale(gray, scaleMap);
	DogGradMap(gray, grad);
	float threshold = 128;
	
	rst = cv::Mat::zeros(gray.size(), CV_32F);


	for (size_t i = 0; i < regions.size(); i++)
	{
		if (regions[i].size > 0)
		{
			int mi(0);
			float gradSum(0);
			float scaleMapSum(0);
			for (size_t j = 0; j < regions[i].borderPixels.size(); j++)
			{
				for (size_t k = 0; k < regions[i].borderPixels[j].size(); k++)
				{
					cv::Point borderPixel(regions[i].borderPixels[j][k].x, regions[i].borderPixels[j][k].y);
					gradSum += grad.at<float>(borderPixel);
					scaleMapSum += scaleMap.at<float>(borderPixel);
					cv::circle(img, borderPixel, 2, cv::Scalar(regions[i].id%255));
					mi++; 
				}

			}
			for (size_t j = 0; j < regions[i].borderEdgePixels.size(); j++)
			{
				cv::Point borderPixel = regions[i].borderEdgePixels[j];
				gradSum += grad.at<float>(borderPixel);
				scaleMapSum += scaleMap.at<float>(borderPixel);
				cv::circle(img, borderPixel, 2, cv::Scalar(regions[i].id % 255));
				mi++;
				

			}
			int ni(0);
			
			for (size_t j = 0; j < regions[i].spIndices.size(); j++)
			{
				for (size_t m = 0; m < spPoses[regions[i].spIndices[j]].size(); m++)
				{
					uint2 pos = spPoses[regions[i].spIndices[j]][m];
					cv::Point point(pos.x, pos.y);
					if (edgeMap.at<float>(point) > threshold)
					{
						ni++;
						scaleMapSum += scaleMap.at<float>(point);
					}
				}
			}
			
			float regFocusness = gradSum / mi*exp(1.0 / (mi + ni) /scaleMapSum);
			regions[i].focusness = regFocusness;
			for (size_t j = 0; j < regions[i].spIndices.size(); j++)
			{
				for (size_t m = 0; m < spPoses[regions[i].spIndices[j]].size(); m++)
				{
					uint2 pos = spPoses[regions[i].spIndices[j]][m];
					cv::Point point(pos.x, pos.y);
					rst.at<float>(point) = regFocusness;
				}
			}
		}
		else
			regions[i].focusness = 0;
	}
	cv::normalize(rst, rst, 255, 0, CV_MINMAX, CV_8U);
	cv::imshow("border", img);
	cv::waitKey();
}
