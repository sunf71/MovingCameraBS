#pragma once
#include <opencv\cv.h>
#include "SparseSolver.h"
inline void MatrixTimesPoint(const double* ptr, cv::Point2f& point)
{
	float x = ptr[0]*point.x + ptr[1]*point.y + ptr[2];
	float y = ptr[3]*point.x + ptr[4]*point.y + ptr[5];
	float z = ptr[6]*point.x + ptr[7]*point.y + 1 + 1e-6;
	point.x = x/z;
	point.y = y/z;
}
inline void MatrixTimesPoint(const cv::Mat& mat, const cv::Point2f& point, cv::Point2f& out)
{
	const double * ptr = mat.ptr<double>(0);
	float x = ptr[0] * point.x + ptr[1] * point.y + ptr[2];
	float y = ptr[3] * point.x + ptr[4] * point.y + ptr[5];
	float z = ptr[6] * point.x + ptr[7] * point.y + ptr[8] + 1e-6;
	out.x = x / z;
	out.y = y / z;
}
inline void MatrixTimesPoint(const cv::Mat& mat, cv::Point2f& point)
{
	const double * ptr = mat.ptr<double>(0);
	float x = ptr[0]*point.x + ptr[1]*point.y + ptr[2];
	float y = ptr[3]*point.x + ptr[4]*point.y + ptr[5];
	float z = ptr[6]*point.x + ptr[7]*point.y + ptr[8] + 1e-6;
	point.x = x/z;
	point.y = y/z;
}
//inPts, float2 矩阵
inline void MatrixTimesMatPoints(const cv::Mat& homo, const cv::Mat& inPts, cv::Mat& outPts)
{
	const double * hptr = homo.ptr<double>(0);
	for(int i=0; i<inPts.rows; i++)
	{
		const float* ptr = inPts.ptr<float>(i);
		float* dstPtr = outPts.ptr<float>(i);
		for(int j=0; j<inPts.cols; j++)
		{
			float x = hptr[0]*ptr[2*j] + hptr[1]*ptr[2*j+1] + hptr[2];
			float y = hptr[3]*ptr[2*j] + hptr[4]*ptr[2*j+1] + hptr[5];
			float z = hptr[6]*ptr[2*j] + hptr[7]*ptr[2*j+1] + hptr[8] + 1e-6;
			dstPtr[2*j] = x/z;
			dstPtr[2*j+1] = y/z;			
		}
	}
}
inline void MatrixTimesPoints(const cv::Mat& mat, std::vector<cv::Point2f>& points)
{
	for(int i=0; i<points.size(); i++)
	{
		MatrixTimesPoint(mat,points[i]);
	}
}
inline void MatrixTimesPoints(const double* ptr, std::vector<cv::Point2f>& points)
{
	for(int i=0; i<points.size(); i++)
	{
		
		MatrixTimesPoint(ptr,points[i]);
	}
}
inline void MatrixTimesPoints(const std::vector<double>& ptr, std::vector<cv::Point2f>& points)
{
	for(int i=0; i<points.size(); i++)
	{
		cv::Point2f point = points[i];
		float x = ptr[0]*point.x + ptr[1]*point.y + ptr[2];
		float y = ptr[3]*point.x + ptr[4]*point.y + ptr[5];
		float z = ptr[6]*point.x + ptr[7]*point.y +  1;
		point.x = x/z;
		point.y = y/z;
	}
}
//normalize points to the center of the points
inline void NormalizePoints(const std::vector<cv::Point2f >& points, cv::Mat& trans, std::vector<cv::Point2f >& normalizedPoints)
{
	trans.create(3,3,CV_64F);
	normalizedPoints.resize(points.size());
	float avgX(0),avgY(0);
	for(int i=0; i<points.size(); i++)
	{
		avgX += points[i].x;
		avgY += points[i].y;
	}
	avgX /= points.size();
	avgY /= points.size();
	//Transform taking x's centroid to the origin
	for(int i=0; i<points.size(); i++)
	{
		normalizedPoints[i].x =  points[i].x - avgX;
		normalizedPoints[i].y = points[i].y - avgY;
	}

	double dist(0);
	//Calculate appropriate scaling factor
	for(int i=0; i<points.size(); i++)
	{
		dist += sqrt(normalizedPoints[i].x*normalizedPoints[i].x + normalizedPoints[i].y*normalizedPoints[i].y);
	}
	dist/=points.size();
	//Transform scaling x to an average length of sqrt(2)
	double scale = sqrt(2.f) / dist;
	double* ptr = trans.ptr<double>(0);
	// Compose the transforms
	ptr[0] = scale ; ptr[1] =  0; ptr[2] =    -scale*avgX;
	ptr[3] = 0; ptr[4] = scale; ptr[5] = -scale*avgY;
	ptr[6] = 0; ptr[7] = 0; ptr[8] = 1;

	normalizedPoints = points;
	MatrixTimesPoints(trans,normalizedPoints);


}

inline void findHomographyDLT(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,cv::Mat& homography)
{
	homography.create(3,3,CV_64F);
	double* homoData = (double*)homography.data;
	cv::Mat dataM(2*f1.size(),9,CV_64F);
	double* ptr = (double*)dataM.data;
	int rowStep = dataM.step.p[0];
	for(int i=0; i<f1.size(); i++)
	{
		ptr[0] = ptr[1] = ptr[2] =0;
		ptr[3] = -1*f1[i].x;
		ptr[4] = -1*f1[i].y;
		ptr[5] = -1;
		ptr[6] = f2[i].y*f1[i].x;
		ptr[7] = f2[i].y*f1[i].y;
		ptr[8] = f2[i].y;
		ptr += 9;
		ptr[0] = f1[i].x;
		ptr[1] = f1[i].y;
		ptr[2] = 1;
		ptr[3] = ptr[4] = ptr[5] = 0;
		ptr[6] = -f2[i].x * f1[i].x;
		ptr[7] = -f2[i].x * f1[i].y;
		ptr[8] = -f2[i].x;
		ptr += 9;
	}
	cv::Mat w,u,vt;
	//std::cout<<"A = "<<dataM<<std::endl;
	cv::SVDecomp(dataM,w,u,vt);
	//std::cout<<"vt = "<<vt<<std::endl;
	ptr = (double*)(vt.data + (vt.rows-1)*vt.step.p[0]);
	for(int i=0; i<9; i++)
		homoData[i] = ptr[i]/ptr[8];


}


inline void findHomographyNormalizedDLT(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, cv::Mat& homography)
{
	// Normalise each set of points 
	cv::Mat T1, T2;
	std::vector<cv::Point2f> x1, x2;
	NormalizePoints(f1, T1, x1);
	NormalizePoints(f2, T2, x2);
	//std::cout<<"\nT1\n";
	//std::cout<<T1; 
	//std::cout<<"\nT2\n";
	//std::cout<<T2;



	findHomographyDLT(x1, x2, homography);
	double* homoData = (double*)homography.data;
	cv::Mat invT2 = T2.inv();
	homography = invT2*homography*T1;


	for (int i = 0; i<9; i++)
		homoData[i] = homoData[i] / homoData[8];
}

inline void findHomographySVD(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,cv::Mat& homography)
{
	homography.create(3,3,CV_64F);
	double* homoData = (double*)homography.data;

	if (f1.size() != f2.size())
		return;
	int n = f1.size();
	if (n<4)
		return;
	cv::Mat h(9,2*n,CV_64F);
	cv::Mat rowsXY = cv::Mat::ones(3,n,CV_64F);
	double* ptr = rowsXY.ptr<double>(0);
	for(int i=0; i<n; i++)
		ptr[i] = -1*f1[i].x;
	ptr = rowsXY.ptr<double>(1);
	for(int i=0; i<n; i++)
		ptr[i] = -1*f1[i].y;
	ptr = rowsXY.ptr<double>(2);
	for(int i=0; i<n; i++)
		ptr[i] = -1;
	cv::Mat rows0 = cv::Mat::zeros(3,n,CV_64F);
	/*std::cout<<"rowsXY\n"<<rowsXY<<std::endl;*/
	rowsXY.copyTo(h(cv::Rect(0,0,rowsXY.cols,rowsXY.rows)));
	rows0.copyTo(h(cv::Rect(0,rowsXY.rows,n,3)));
	ptr = h.ptr<double>(6);
	for(int i=0; i<n; i++)
		ptr[i] = f1[i].x*f2[i].x;

	ptr = h.ptr<double>(7);
	for(int i=0; i<n; i++)
		ptr[i] = f1[i].y*f2[i].x;
	ptr = h.ptr<double>(8);
	for(int i=0; i<n; i++)
		ptr[i] = f2[i].x;

	rows0.copyTo(h(cv::Rect(n,0,n,3)));
	rowsXY.copyTo(h(cv::Rect(n,rows0.rows,rowsXY.cols,rowsXY.rows)));
	ptr = h.ptr<double>(6);
	for(int i=0; i<n; i++)
		ptr[n+i] = f1[i].x*f2[i].y;
	ptr = h.ptr<double>(7);
	for(int i=0; i<n; i++)
		ptr[n+i] = f1[i].y*f2[i].y;
	ptr = h.ptr<double>(8);
	for(int i=0; i<n; i++)
		ptr[i+n] = f2[i].y;
	/*std::cout<<"h\n"<<h<<std::endl;*/
	cv::Mat w,u,v;
	cv::SVD::compute(h,w,u,v,cv::SVD::FULL_UV);
	/*std::cout<<"w\n"<<w<<std::endl;
	std::cout<<"u\n"<<u<<std::endl;
	std::cout<<"v\n"<<v<<std::endl;*/
	cv::Mat ut;
	cv::transpose(u,ut);
	/*std::cout<<"ut\n"<<ut<<std::endl;*/
	ptr = (double*)(ut.data + (ut.rows-1)*ut.step.p[0]);
	for(int i=0; i<9; i++)
		homoData[i] = ptr[i]/ptr[8];
}

//直接通过解方程求解
inline void findHomographyEqa(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2,cv::Mat& homography)
{
	cv::Mat Amat = cv::Mat::zeros(3*f1.size(),8,CV_32F);
	cv::Mat bmat = cv::Mat(3*f1.size(),1,CV_32F);
	for(int i=0; i<f1.size(); i++)
	{
		float* ptr = Amat.ptr<float>(3*i);
		ptr[0] = f1[i].x;
		ptr[1] = f1[i].y;
		ptr[2] = 1;
		ptr = bmat.ptr<float>(3*i);
		ptr[0] = f2[i].x;

		ptr = Amat.ptr<float>(i*3+1);
		ptr[3] = f1[i].x;
		ptr[4] = f1[i].y;
		ptr[5] = 1;
		ptr = bmat.ptr<float>(3*i+1);
		ptr[0] = f2[i].y;

		ptr = Amat.ptr<float>(i*3+2);
		ptr[6] = f1[i].x;
		ptr[7] = f1[i].y;
		ptr = bmat.ptr<float>(3*i+2);
		ptr[0] = 0;

	}
	//std::cout<<"A \n"<<Amat<<std::endl;
	//std::cout<<"bmat "<<bmat<<std::endl;
	//std::vector<float> result;
	cv::Mat result(8,1,CV_32F);
	cv::solve(Amat,bmat,result,cv::DECOMP_SVD);
	homography.create(3,3,CV_64F);
	for(int i=0; i<result.rows; i++)
	{
		((double*)homography.data)[i] = *result.ptr<float>(i);
	}
	((double*)homography.data)[8] = 1;
}

//直接通过解方程求解
inline void findHomographyEqa(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, std::vector<double>& homoVec)
{
	cv::Mat Amat = cv::Mat::zeros(3*f1.size(),8,CV_32F);
	cv::Mat bmat = cv::Mat(3*f1.size(),1,CV_32F);
	for(int i=0; i<f1.size(); i++)
	{
		float* ptr = Amat.ptr<float>(3*i);
		ptr[0] = f1[i].x;
		ptr[1] = f1[i].y;
		ptr[2] = 1;
		ptr = bmat.ptr<float>(3*i);
		ptr[0] = f2[i].x;

		ptr = Amat.ptr<float>(i*3+1);
		ptr[3] = f1[i].x;
		ptr[4] = f1[i].y;
		ptr[5] = 1;
		ptr = bmat.ptr<float>(3*i+1);
		ptr[0] = f2[i].y;

		ptr = Amat.ptr<float>(i*3+2);
		ptr[6] = f1[i].x;
		ptr[7] = f1[i].y;
		ptr = bmat.ptr<float>(3*i+2);
		ptr[0] = 0;

	}
	//std::cout<<"A \n"<<Amat<<std::endl;
	//std::cout<<"bmat "<<bmat<<std::endl;
	//std::vector<float> result;
	/*cv::Mat result(8,1,CV_32F);*/
	cv::solve(Amat,bmat,homoVec,cv::DECOMP_SVD);
	/*homography.create(3,3,CV_64F);
	for(int i=0; i<result.rows; i++)
	{
		((double*)homography.data)[i] = *result.ptr<float>(i);
	}
	((double*)homography.data)[8] = 1;*/
}