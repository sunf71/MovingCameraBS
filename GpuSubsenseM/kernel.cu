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
	printf("gpu 1 cpu 0 %s\n",argv[1]);
	printf("from %s\n",argv[2]);
	printf("to %s\n",argv[3]);
	printf("input %s\n",argv[4]);
	printf("output %s\n",argv[5]);
	printf("region growing threshold %s\n",argv[6]);
	printf("region growing seed threshold %s\n",argv[7]);
	printf("model confidence %s\n",argv[8]);
	printf("tc confidence %s\n",argv[9]);
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
