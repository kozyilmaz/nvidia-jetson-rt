#ifndef __HOG_HISTOGRAM__
#define __HOG_HISTOGRAM__
#include "HOGDefines.h"

void InitHistograms(int cellSizeX, int cellSizeY, int blockSizeX,
    int blockSizeY, int noHistogramBins, float wtscale);
void CloseHistogram();
void ComputeBlockHistogramsWithGauss(float2* inputImage,
    float1* blockHistograms, int noHistogramBins, int cellSizeX, int cellSizeY,
    int blockSizeX, int blockSizeY, int windowSizeX, int windowSizeY,
    int width, int height);
void NormalizeBlockHistograms(float1* blockHistograms, int noHistogramBins,
    int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY, int width,
    int height);
__global__ void computeBlockHistogramsWithGauss(float2* inputImage,
    float1* blockHistograms, int noHistogramBins, int cellSizeX, int cellSizeY,
    int blockSizeX, int blockSizeY, int leftoverX, int leftoverY, int width,
    int height);
__global__ void normalizeBlockHistograms(float1 *blockHistograms,
    int noHistogramBins, int rNoOfHOGBlocksX, int rNoOfHOGBlocksY,
    int blockSizeX, int blockSizeY, int alignedBlockDimX, int alignedBlockDimY,
    int alignedBlockDimZ, int width, int height);

void HostAllocHOGHistogramMemory(void);
void DeviceAllocHOGHistogramMemory(void);
void CopyInHOGHistogram(void);
void HostFreeHOGHistogramMemory(void);
void DeviceFreeHOGHistogramMemory(void);

#endif
