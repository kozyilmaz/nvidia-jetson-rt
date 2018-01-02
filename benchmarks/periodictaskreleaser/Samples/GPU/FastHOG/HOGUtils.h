#ifndef __HOG_UTILS__
#define __HOG_UTILS__

#include "HOGDefines.h"

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

int iDivUpF(int a, float b);
int iClosestPowerOfTwo(int x);

void Float4ToUchar4(float4 *inputImage, uchar4 *outputImage, int width, int height);
void Float2ToUchar4(float2 *inputImage, uchar4 *outputImage, int width, int height, int index);
void Float2ToUchar1(float2 *inputImage, uchar1 *outputImage, int width, int height, int index);
void Float1ToUchar4(float1 *inputImage, uchar4 *outputImage, int width, int height);
void Float1ToUchar1(float1 *inputImage, uchar1 *outputImage, int width, int height);
void Uchar4ToFloat4(uchar4 *inputImage, float4 *outputImage, int width, int height);

__global__ void float4toUchar4(float4 *inputImage, uchar4 *outputImage, int width, int height);
__global__ void float2toUchar4(float2 *inputImage, uchar4 *outputImage, int width, int height, int index);
__global__ void float2toUchar1(float2 *inputImage, uchar1 *outputImage, int width, int height, int index);
__global__ void float1toUchar4(float1 *inputImage, uchar4 *outputImage, int width, int height);
__global__ void float1toUchar1(float1 *inputImage, uchar1 *outputImage, int width, int height);
__global__ void uchar4tofloat4(uchar4 *inputImage, float4 *outputImage, int width, int height);

#endif
