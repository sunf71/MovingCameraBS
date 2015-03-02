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
void TestSuperpixel();
void MRFOptimization();
void TCMRFOptimization();
void TestRandom();
void TestGpuSubsense(int procId, int start, int end, const char* input, const char* output, float rggThre = 2.0, float rggSeedThres = 0.8, float mdlConfidence = 0.8, float tcConfidence = 0.25, float scConfidence = 0.35);
void TestMotionEstimate();
void TestRegionGrowing();

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
void GpuSubsenseMain(int argc, char* argv[]);