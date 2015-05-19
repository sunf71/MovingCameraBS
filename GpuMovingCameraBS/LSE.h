#pragma once
#include <opencv2\opencv.hpp>


//Level Set Evolution Method for Image Segmentation
template<typename T>
void MGradient(cv::Mat& mat, cv::Mat& dx, cv::Mat& dy);
template<typename T>
void NeumannBoundCond(cv::Mat& u);
template<typename T>
void CurvatureCenter(cv::Mat& u, cv::Mat& k);
template<typename T>
void Dirac(cv::Mat& x, float sigma, cv::Mat& f);
template<typename T>
void Evolution(cv::Mat& u, cv::Mat& g, float lmbda, float mu, float alf, float epsilon, float delt, int N);
void LSEWR(const cv::Mat& img, float sigma, float epsilon, float mu, float lambda, float alf, float c0, int N, const cv::Mat& mask, cv::Mat& u);