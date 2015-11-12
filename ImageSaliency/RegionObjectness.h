#include <opencv\cv.h>
#include "../GpuMovingCameraBS/RegionMerging.h"
#include <vector>

const float meanRelSize = 0.3;
const float meanFillness = 0.558;
const float thetaSize = 0.25;
const float thetaFill = 0.11;


struct PropScore
{
	int id;
	float score;
};
struct PropScoreCmp
{
	bool operator() (const PropScore& a, const PropScore& b)
	{
		return a.score > b.score;
	}
};


float RegionObjectness(std::vector<SPRegion>& regions, int i, SuperpixelComputer* computer, HISTOGRAMS& colorHist, cv::Mat& edgeMap);

void RegionOutBorder(int i,  std::vector<SPRegion>& regions);


void GetRegionOutBorder(std::vector<SPRegion>& regions);