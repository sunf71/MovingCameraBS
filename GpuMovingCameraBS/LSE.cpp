#include "LSE.h"
#include "timer.h"
#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
template<typename T>
void MGradient(cv::Mat& mat, cv::Mat& dx, cv::Mat& dy)
{
	Mat kernelx = (Mat_<T>(1, 3) << -0.5, 0, 0.5);
	Mat kernely = (Mat_<T>(3, 1) << -0.5, 0, 0.5);
	size_t rows(mat.rows), cols(mat.cols);

	filter2D(mat, dx, -1, kernelx);
	filter2D(mat, dy, -1, kernely);
	for (size_t i = 0; i < mat.rows; i++)
	{
		T* dxPtr = dx.ptr<T>(i);
		T* ptr = mat.ptr<T>(i);
		dxPtr[0] = ptr[1] - ptr[0];
		dxPtr[cols - 1] = ptr[cols - 1] - ptr[cols - 2];
	}
	for (size_t i = 0; i < cols; i++)
	{
		dy.at<T>(0, i) = mat.at<T>(1, i) - mat.at<T>(0, i);
		dy.at<T>(rows - 1, i) = mat.at<T>(rows - 1, i) - mat.at<T>(rows - 2, i);
	}
}
template<typename T>
void NeumannBoundCond(cv::Mat& u)
{
	int rows = u.rows;
	int cols = u.cols;
	u.at<T>(0, 0) = u.at<T>(2, 2);
	u.at<T>(0, cols - 1) = u.at<T>(2, cols - 3);
	u.at<T>(rows - 1, 0) = u.at<T>(rows - 3, 2);
	u.at<T>(rows - 1, cols - 1) = u.at<T>(rows - 3, cols - 3);
	u(cv::Rect(1, 2, cols - 2, 1)).copyTo(u(cv::Rect(1, 0, cols - 2, 1)));
	u(cv::Rect(1, rows - 3, cols - 2, 1)).copyTo(u(cv::Rect(1, rows - 1, cols - 2, 1)));
	u(cv::Rect(2, 1, 1, rows - 2)).copyTo(u(cv::Rect(0, 1, 1, rows - 2)));
	u(cv::Rect(cols - 3, 1, 1, rows - 2)).copyTo(u(cv::Rect(cols - 1, 1, 1, rows - 2)));

}
template<typename T>
void CurvatureCenter(cv::Mat& u, cv::Mat& k)
{
	cv::Mat ux, uy;
	MGradient<T>(u, ux, uy);
	cv::Mat Nx, Ny;
	size_t rows(u.rows), cols(u.cols);
	Nx.create(u.size(), u.type());
	Ny.create(u.size(), u.type());
	for (size_t i = 0; i < rows; i++)
	{
		T* uxptr = ux.ptr<T>(i);
		T* uyptr = ux.ptr<T>(i);
		T* nxPtr = Nx.ptr<T>(i);
		T* nyPtr = Ny.ptr<T>(i);
		for (size_t j = 0; j < cols; j++)
		{
			T normDu = sqrt(uxPtr[j] * uxPtr[j] + uyPtr[j] * uyPtr[j] + 1e-10);
			nxPtr[j] = uxptr[j] / normDu;
			nyPtr[j] = uyptr[j] / normDu;
		}
	}
	cv::Mat nxx, nyy, junk;
	MGradient(Nx, nxx, junk);
	MGradient(Ny, junk, nyy);
	cv::add(nxx, nyy, k);
}
template<typename T>
void Dirac(cv::Mat& x, float sigma, cv::Mat& f)
{
	f.create(x.size(), x.type());
	double theta = M_PI / sigma;
	double a = 0.5 / sigma;
	for (size_t i = 0; i < x.rows; i++)
	{
		T* ptr = x.ptr<T>(i);
		T* fptr = f.ptr<T>(i);
		for (size_t j = 0; j < x.cols; j++)
		{
			fptr[j] = (ptr[j] <= sigma) && (ptr[j] >= -sigma) ? a*(cos(ptr[j] * theta) + 1) : 0;
		}
	}
}
template<typename T>
void Evolution(cv::Mat& u, cv::Mat& g, float lmbda, float mu, float alf, float epsilon, float delt, int N)
{
	cv::Mat vx, vy;
	MGradient<T>(g, vx, vy);
	for (size_t i = 0; i < N; i++)
	{
		NeumannBoundCond<float>(u);
		//std::cout << "u = \n" << u(cv::Rect(0, 0, 10, 10)) << "\n";
		cv::Mat k, diracU;
		Dirac<T>(u, epsilon, diracU);
		cv::Mat ux, uy;
		MGradient<T>(u, ux, uy);
		cv::Mat Nx, Ny;
		size_t rows(u.rows), cols(u.cols);
		Nx.create(u.size(), u.type());
		Ny.create(u.size(), u.type());
		for (size_t i = 0; i < rows; i++)
		{
			T* uxPtr = ux.ptr<T>(i);
			T* uyPtr = uy.ptr<T>(i);
			T* nxPtr = Nx.ptr<T>(i);
			T* nyPtr = Ny.ptr<T>(i);
			for (size_t j = 0; j < cols; j++)
			{
				T normDu = sqrt(uxPtr[j] * uxPtr[j] + uyPtr[j] * uyPtr[j] + 1e-10);
				nxPtr[j] = uxPtr[j] / normDu;
				nyPtr[j] = uyPtr[j] / normDu;
			}
		}

		cv::Mat nxx, nyy, junk;
		MGradient<T>(Nx, nxx, junk);
		MGradient<T>(Ny, junk, nyy);
		cv::add(nxx, nyy, k);

		//std::cout << "k = \n" << k(cv::Rect(10, 10, 10, 10)) << "\n";
		cv::Mat lapU;
		cv::Laplacian(u, lapU, CV_32F);
		//std::cout << "lapU = \n" << lapU(cv::Rect(10, 10, 10, 10)) << "\n";
		for (size_t i = 0; i < rows; i++)
		{
			T* ptr = u.ptr<T>(i);
			T* vxPtr = vx.ptr<T>(i);
			T* vyPtr = vy.ptr <T>(i);
			T* nxPtr = Nx.ptr<T>(i);
			T* nyPtr = Ny.ptr<T>(i);
			T* duPtr = diracU.ptr<T>(i);
			T* gPtr = g.ptr<T>(i);
			T* uPtr = u.ptr<T>(i);
			T* kPtr = k.ptr<T>(i);
			float* lapUPtr = lapU.ptr<float>(i);
			for (size_t j = 0; j < cols; j++)
			{
				T WLTerm = lmbda*duPtr[j] * (vxPtr[j] * nxPtr[j] + vyPtr[j] * nyPtr[j] + gPtr[j] * kPtr[j]);
				T pTerm = mu*(lapUPtr[j] - kPtr[j]);
				T WATerm = alf*duPtr[j] * gPtr[j];
				uPtr[j] += delt*(WLTerm + pTerm + WATerm);
			}
		}

	}
}
void LSEWR(const cv::Mat& img, float sigma, float epsilon, float mu, float lambda, float alf, float c0, int N, const cv::Mat& mask, cv::Mat& u)
{
	int rows = img.rows;
	int cols = img.cols;
	cv::Mat gray, fimg;
	if (img.channels() == 3)
		cv::cvtColor(img, gray, CV_BGR2GRAY);
	else
		gray = img.clone();
	//cv::imshow("before filtering", gray);
	gray.convertTo(fimg, CV_32F);

	cv::Mat dst;
	cv::GaussianBlur(fimg, fimg, cv::Size(15, 15), sigma, 0, cv::BORDER_CONSTANT);


	cv::Mat gImg, GImg, dx, dy;
	GImg.create(img.size(), CV_32F);
	u.create(img.size(), CV_32F);
	MGradient<float>(fimg, dx, dy);

	for (size_t i = 0; i < rows; i++)
	{
		float* ptrDx = dx.ptr<float>(i);
		float* ptrDy = dy.ptr<float>(i);
		float* gptr = GImg.ptr<float>(i);
		float* uptr = u.ptr<float>(i);
		const uchar* mptr = mask.ptr<uchar>(i);
		for (size_t j = 0; j < cols; j++)
		{
			gptr[j] = 1 / (1 + ptrDx[j] * ptrDx[j] + ptrDy[j] * ptrDy[j]);
			if (mptr[j] == 0)
				uptr[j] = c0;
			else
				uptr[j] = -c0;
		}
	}

	float timeStep = 0.2 / mu;
	nih::Timer timer;
	timer.start();
	for (size_t i = 0; i < N; i++)
	{
		//std::cout << "u = \n" << u(cv::Rect(0, 0, 10, 10)) << "\n";
		Evolution<float>(u, GImg, lambda, mu, alf, epsilon, timeStep, 1);
	}
	timer.stop();
	std::cout << N << " iterations " << timer.seconds() * 1000 << " ms\n";
	//std::cout << "u = \n" << u(cv::Rect(30,10,10,10)) << "\n";
	cv::threshold(u, u, 0, 255, CV_THRESH_BINARY);
	u.convertTo(u, CV_8U);
	//cv::GaussianBlur(u, u, cv::Size(3, 3), 0);
	//cv::Mat edge;
	//cv::Canny(u, edge, 100, 300);
	//vector<Vec4i> hierarchy;
	//vector<vector<cv::Point>> contours;
	//findContours(edge, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);//ÕÒÂÖÀª
	//Scalar color(255, 0, 0);
	//for (int i = 0; i< contours.size(); i++)
	//{
	//	//const vector<cv::Point>& c = contours[i];
	//	/*double area = fabs(contourArea(Mat(c)));
	//	if (area > 100)*/
	//	{
	//		drawContours(img, contours, i, color, -1, 8, hierarchy, 0, cv::Point());

	//	}

	//}
	/*cv::imshow("result", u);
	cv::waitKey();*/
}