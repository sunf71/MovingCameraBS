#include <opencv\cv.h>
#include "../GpuMovingCameraBS/RegionMerging.h"
#include <vector>

float RegionObjectness(std::vector<SPRegion>& regions, int i, SuperpixelComputer* computer, HISTOGRAMS& colorHist, cv::Mat& edgeMap);

void RegionOutBorder(int i,  std::vector<SPRegion>& regions);


void GetRegionOutBorder(std::vector<SPRegion>& regions);