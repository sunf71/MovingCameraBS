#include "Test.h"

int main (int argc, char* argv[])
{
	//TestSuperpixelDownSample();
	//TestSuperpielxComputer();
	//TestFlowHistogram();
	//TestColorHistogram();
	//TestSuperpixelMatching();
	//TestSuperpixelFlow();
	//TCMRFOptimization();
	//TestRegionGrowing();
	//TestRegioinGrowingSegment();
	//TestMotionEstimate();
	//TestRandom();
	TestGpuSubsense(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),argv[4],argv[5]);
	//MRFOptimization();
	//TestSuperpixel();
	//testCudaGpu();
	//TestDescDiff();
	return 0;

}
