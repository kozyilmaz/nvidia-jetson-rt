#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "HOGEngine.h"
#include "HOGUtils.h"

// Short function to synchronize with our global user-defined stream.
static void sync(void) {
  checkCudaErrors(cudaStreamSynchronize(stream));
}

//Round a / b to nearest higher integer value
int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b) {
  return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b) {
  return (a % b != 0) ? (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b) {
  return a - a % b;
}

//Round a / b to nearest higher integer value
int iDivUpF(int a, float b) {
  return (a % int(b) != 0) ? int(a / b + 1) : int(a / b);
}

int iClosestPowerOfTwo(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

void Uchar4ToFloat4(uchar4 *inputImage, float4 *outputImage, int width,
    int height) {
  dim3 threads_in_block(16,16);
  dim3 blocks(iDivUp(width,16), iDivUp(height,16));
  uchar4tofloat4<<<blocks, threads_in_block, 0, stream>>>(inputImage,
    outputImage, width, height);
  sync();
}
void Float4ToUchar4(float4 *inputImage, uchar4 *outputImage, int width,
    int height) {
  dim3 threads_in_block(16,16);
  dim3 blocks(iDivUp(width,16), iDivUp(height,16));
  float4toUchar4<<<blocks, threads_in_block, 0, stream>>>(inputImage,
    outputImage, width, height);
  sync();
}
void Float2ToUchar4(float2 *inputImage, uchar4 *outputImage, int width,
    int height, int index) {
  dim3 threads_in_block(16,16);
  dim3 blocks(iDivUp(width,16), iDivUp(height,16));
  float2toUchar4<<<blocks, threads_in_block, 0, stream>>>(inputImage,
    outputImage, width, height, index);
  sync();
}
void Float2ToUchar1(float2 *inputImage, uchar1 *outputImage, int width,
    int height, int index) {
  dim3 threads_in_block(16,16);
  dim3 blocks(iDivUp(width,16), iDivUp(height,16));
  float2toUchar1<<<blocks, threads_in_block, 0, stream>>>(inputImage,
    outputImage, width, height, index);
  sync();
}
void Float1ToUchar4(float1 *inputImage, uchar4 *outputImage, int width,
    int height) {
  dim3 threads_in_block(16,16);
  dim3 blocks(iDivUp(width,16), iDivUp(height,16));
  float1toUchar4<<<blocks, threads_in_block, 0, stream>>>(inputImage,
    outputImage, width, height);
  sync();
}
void Float1ToUchar1(float1 *inputImage, uchar1 *outputImage, int width,
    int height) {
  dim3 threads_in_block(16,16);
  dim3 blocks(iDivUp(width,16), iDivUp(height,16));
  float1toUchar1<<<blocks, threads_in_block, 0, stream>>>(inputImage,
    outputImage, width, height);
  sync();
}
__global__ void float4toUchar4(float4 *inputImage, uchar4 *outputImage,
    int width, int height) {
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
  float4 pixelf = inputImage[offset];
  uchar4 pixel;
  pixel.x = (unsigned char) pixelf.x;
  pixel.y = (unsigned char) pixelf.y;
  pixel.z = (unsigned char) pixelf.z;
  pixel.w = (unsigned char) pixelf.w;
  outputImage[offset] = pixel;
}

__global__ void float2toUchar4(float2 *inputImage, uchar4 *outputImage,
    int width, int height, int index) {
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
  float2 pixelf = inputImage[offset];
  float pixelfIndexed = (index == 0) ? pixelf.x : pixelf.y;
  uchar4 pixel;
  pixel.x = (unsigned char) abs(pixelfIndexed);
  pixel.y = (unsigned char) abs(pixelfIndexed);
  pixel.z = (unsigned char) abs(pixelfIndexed);
  pixel.w = (unsigned char) abs(pixelfIndexed);
  outputImage[offset] = pixel;
}

__global__ void float2toUchar1(float2 *inputImage, uchar1 *outputImage,
    int width, int height, int index) {
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
  float2 pixelf = inputImage[offset];
  float pixelfIndexed = (index == 0) ? pixelf.x : pixelf.y;
  uchar1 pixel;
  pixel.x = (unsigned char) pixelfIndexed;
  outputImage[offset] = pixel;
}

__global__ void float1toUchar4(float1 *inputImage, uchar4 *outputImage,
    int width, int height) {
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
  float1 pixelf = inputImage[offset];
  uchar4 pixel;
  pixel.x = (unsigned char) pixelf.x;
  pixel.y = (unsigned char) pixelf.x;
  pixel.z = (unsigned char) pixelf.x;
  pixel.w = (unsigned char) pixelf.x;
  outputImage[offset] = pixel;
}

__global__ void float1toUchar1(float1 *inputImage, uchar1 *outputImage,
    int width, int height) {
  int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
  int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
  float1 pixelf = inputImage[offset];
  uchar1 pixel;
  pixel.x = (unsigned char) pixelf.x;
  outputImage[offset] = pixel;
}

__global__ void uchar4tofloat4(uchar4 *inputImage, float4 *outputImage,
    int width, int height) {
  int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetY = blockIdx.y * blockDim.y + threadIdx.y;
  if (offsetX < width && offsetY < height) {
    int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y *
      width;
    int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
    uchar4 pixel = inputImage[offset];
    float4 pixelf;
    pixelf.x = pixel.x;
    pixelf.y = pixel.y;
    pixelf.z = pixel.z;
    pixelf.w = pixel.w;
    outputImage[offset] = pixelf;
  }
}
