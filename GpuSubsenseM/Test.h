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
void TestGpuSubsense();
void TestMotionEstimate();
void TestRegionGrowing();

void TestSuperpixelFlow();
void TestSuperpixelMatching();
void TestFlowHistogram();
void TestColorHistogram();

void TestSuperpielxComputer();
