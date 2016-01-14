#include <opencv\cv.h>
const float MAXD 1e10;
using namespace cv;


void FastMBD(const Mat& img, Mat& U, Mat& L, int k, Mat& seeds, Mat& mbdMap);

void RasterScan(const Mat& img, Mat& mbdMap, Mat& U, Mat& L, bool order);

