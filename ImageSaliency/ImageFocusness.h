#include <opencv\cv.h>
#include "../GpuMovingCameraBS/RegionMerging.h"
#include <vector>

void CalScale(cv::Mat& gray, cv::Mat& scaleMap);


void CalRegionFocusness(const cv::Mat& img, const cv::Mat& scaleMap, const cv::Mat& edgeMap, std::vector<SPRegion>& regions, cv::Mat& rst);