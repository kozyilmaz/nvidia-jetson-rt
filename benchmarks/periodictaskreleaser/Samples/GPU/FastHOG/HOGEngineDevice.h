#ifndef __CUDA_HOG__
#define __CUDA_HOG__


#include "HOGDefines.h"

void InitHOG(int width, int height);

void CloseHOG();

void BeginHOGProcessing(unsigned char* hostImage, float minScale,
    float maxScale);

float* EndHOGProcessing();

void GetHOGParameters();

void InitCUDAHOG(int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
    int windowSizeX, int windowSizeY, int noOfHistogramBins, float wtscale,
    float svmBias, float* svmWeights, int svmWeightsCount, bool useGrayscale);

void CloseCUDAHOG();

void DeviceAllocHOGEngineDeviceMemory(void);
void HostAllocHOGEngineDeviceMemory(void);
void CopyInHOGEngineDevice(void);
void DeviceFreeHOGEngineDeviceMemory(void);
void HostFreeHOGEngineDeviceMemory(void);

#endif
