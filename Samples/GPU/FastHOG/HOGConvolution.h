#ifndef __HOG_CONVOLUTION__
#define __HOG_CONVOLUTION__

#include "HOGDefines.h"

void InitConvolution(int width, int height, bool useGrayscale);
void SetConvolutionSize(int width, int height);
void CloseConvolution();
void ComputeColorGradients1to2(float1* inputImage, float2* outputImage);
void ComputeColorGradients4to2(float4* inputImage, float2* outputImage);
void DeviceAllocHOGConvolutionMemory(void);
void CopyInHOGConvolution(void);
void DeviceFreeHOGConvolutionMemory(void);

__global__ void convolutionRowGPU1(float1 *d_Result, float1 *d_Data, int dataW, int dataH);
__global__ void convolutionRowGPU4(float4 *d_Result, float4 *d_Data, int dataW, int dataH);
__global__ void convolutionColumnGPU1to2(float1 *d_Result, float1 *d_Data, float1 *d_DataRow,
    int dataW, int dataH, int smemStride, int gmemStride);
__global__ void convolutionColumnGPU4to2(float2 *d_Result, float4 *d_Data, float4 *d_DataRow,
    int dataW, int dataH, int smemStride, int gmemStride);

#endif
