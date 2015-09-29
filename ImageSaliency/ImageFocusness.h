#include <opencv\cv.h>
#include "../GpuMovingCameraBS/RegionMerging.h"
#include <vector>

void CalScale(cv::Mat& gray, cv::Mat& scaleMap);


void CalRegionFocusness(const cv::Mat& grayImg, const cv::Mat& scaleMap, const cv::Mat& edgeMap, std::vector<std::vector<uint2>>& spPoses, std::vector<SPRegion>& regions, cv::Mat& rst);


void DogGradMap(cv::Mat& grayImg, cv::Mat& gradMap);