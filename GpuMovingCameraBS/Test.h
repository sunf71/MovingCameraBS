#undef min
#undef max
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "GpuSuperpixel.h"
#include "SLIC.h"
#include "PictureHandler.h"
#include "ComSuperpixel.h"
#include "GpuTimer.h"
#include "timer.h"
#include "MRFOptimize.h"
#include "GpuBackgroundSubtractor.h"
#include "SubSenseBSProcessor.h"
#include "videoprocessor.h"
#include "CudaBSOperator.h"
#include "RandUtils.h"
#include "MotionEstimate.h"

void testCudaGpu();
void CpuSuperpixel();
void TestSuperpixel(int argc, char* argv[]);
void MRFOptimization();
void TCMRFOptimization();
void TestRandom();
void TestGpuSubsense(int procId, int start, int end, const char* input, const char* output, int warpId = 1, float rggThre = 1.0, float rggSeedThres = 0.4, float mdlConfidence = 0.75, float tcConfidence = 0.15, float scConfidence = 0.35);
void TestMotionEstimate();
void TestRegionGrowing();
void TestFeaturesRefine(int argc, char* argv[]);
void TestSuperpixelFlow();
void TestSuperpixelMatching();
void TestFlowHistogram();
void TestColorHistogram();

void TestSuperpielxComputer();

void TestDescDiff();
void TestSuperpixelDownSample();

void TestFeaturesRefineHistogram(int argc, char* argv[]);
void TestBlockHomography();
void TestBlockWarping();
void TestSaliency(int argc, char* argv[]);
void GpuSubsenseMain(int argc, char* argv[]);
void TestRTBS(int argc, char** argv);
void TestQuantize();

void TestLBP();
void TestGpuKLT();
void TestWarpError(int argc, char* argv[]);