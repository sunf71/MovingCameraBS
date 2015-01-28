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
	if (argc == 6)
		TestGpuSubsense(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),argv[4],argv[5]);
	else
		TestGpuSubsense(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),argv[4],argv[5],atof(argv[6]),atof(argv[7]),atof(argv[8]),atof(argv[9]));
	//MRFOptimization();
	//TestSuperpixel();
	//testCudaGpu();
	//TestDescDiff();
	return 0;

}
