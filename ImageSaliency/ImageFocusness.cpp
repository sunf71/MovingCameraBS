#include "ImageFocusness.h"
#define _USE_MATH_DEFINES
#include <math.h>

void CalScale(cv::Mat& gray, cv::Mat& scaleMap)
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
		cv::Mat pad(gray.size(), CV_32F);
		cv::filter2D(gray, smoothIx, CV_32F, smoothFilter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);		
		smoothIx(cv::Rect(w, 0, smoothIx.cols - 2 * w, smoothIx.rows)).copyTo(tmp);
		cv::copyMakeBorder(tmp, smoothIx, 0, 0, w, w, cv::BORDER_REPLICATE);

		cv::Mat fuck = smoothIx(cv::Rect(0, 0, 10, 10));
		std::cout << fuck<< "\n";;


		cv::filter2D(smoothIx, gradIy, CV_32F, differFileter.t(), cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	    gradIy(cv::Rect(0, w, smoothIx.cols, smoothIx.rows - 2 * w)).copyTo(tmp);
		cv::copyMakeBorder(tmp, gradIy, w, w, 0, 0, cv::BORDER_REPLICATE);


		cv::filter2D(gray, smoothIy, CV_32F, smoothFilter.t(), cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);		
		smoothIy(cv::Rect(w, 0, smoothIx.cols - 2 * w, smoothIx.rows)).copyTo(tmp);
		cv::copyMakeBorder(tmp, smoothIy, 0, 0, w, w, cv::BORDER_REPLICATE);

		cv::filter2D(smoothIy, gradIx, CV_32F, differFileter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
		gradIx(cv::Rect(0, w, smoothIx.cols, smoothIx.rows - 2 * w)).copyTo(tmp);
		cv::copyMakeBorder(tmp, gradIy, w, w, 0, 0, cv::BORDER_REPLICATE);
		//std::cout << "gradIx"<<gradIx << "\n";
		//std::cout << "gradIy" << gradIy << "\n";
		
		cv::pow(gradIx, 2, gradIx);
		cv::pow(gradIy, 2, gradIy);
		cv::Mat_<float> gradI;
		cv::add(gradIx, gradIy, gradI);
		cv::pow(gradI, 0.5, gradI);

		
		
		cv::Mat gradd = gradI - gradp;
		cv::threshold(gradd, gradd, 0, 255, cv::THRESH_TOZERO);
		cv::Mat p = gradd(cv::Rect(0, 0, 5, 5));
		std::cout << p;
		gradp = gradI;

		if (i > 1)
		{
			cv::Mat diff = gradd - graddp;
			cv::threshold(diff, diff, 0, 255, cv::THRESH_BINARY);
			diff.convertTo(diff, CV_8U);
			scaleMap.setTo(scale[i], diff);
			
		}

		graddp = gradd;
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

}