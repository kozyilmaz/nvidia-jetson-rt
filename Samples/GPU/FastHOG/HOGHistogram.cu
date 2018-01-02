#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "HOGEngine.h"
#include "HOGUtils.h"
#include "HOGHistogram.h"

__device__ __constant__ float cenBound[3], halfBin[3], bandWidth[3],
    oneHalf = 0.5f;
__device__ __constant__ int tvbin[3];

texture<float, 1, cudaReadModeElementType> texGauss;
cudaArray* gaussArray;
cudaChannelFormatDesc channelDescGauss;
float *hostWeights;
extern __shared__ float allShared[];
extern int rNoHistogramBins, rNoOfCellsX, rNoOfCellsY, rNoOfBlocksX,
    rNoOfBlocksY, rNumberOfWindowsX, rNumberOfWindowsY;

// Stuff set during the InitHistograms function, but needed during allocation
static struct {
  int cellSizeX, cellSizeY, blockSizeX, blockSizeY, noHistogramBins;
  float wtscale;
  float var2x, var2y;
  float centerX, centerY;
  int h_tvbin[3];
  float h_cenBound[3], h_halfBin[3], h_bandWidth[3];
} initVars;

void HostAllocHOGHistogramMemory(void) {
  int i, j;
  float tx, ty;
  int cellSizeX = initVars.cellSizeX;
  int cellSizeY = initVars.cellSizeY;
  int blockSizeX = initVars.blockSizeX;
  int blockSizeY = initVars.blockSizeY;
  checkCudaErrors(cudaMallocHost(&hostWeights, cellSizeX * blockSizeX *
    cellSizeY * blockSizeY * sizeof(float)));
  for (i = 0; i < cellSizeX * blockSizeX; i++) {
    for (j = 0; j < cellSizeY * blockSizeY; j++) {
      tx = i - initVars.centerX;
      ty = j - initVars.centerY;
      tx *= tx / initVars.var2x;
      ty *= ty / initVars.var2y;
      hostWeights[i + j * cellSizeX * blockSizeX] = exp(-(tx + ty));
    }
  }
}

void DeviceAllocHOGHistogramMemory(void) {
  checkCudaErrors(cudaMallocArray(&gaussArray, &channelDescGauss,
    initVars.cellSizeX * initVars.blockSizeX * initVars.cellSizeY *
    initVars.blockSizeY, 1));
}

void CopyInHOGHistogram(void) {
  checkCudaErrors(cudaMemcpyToArrayAsync(gaussArray, 0, 0, hostWeights,
    sizeof(float) * initVars.cellSizeX * initVars.blockSizeX *
    initVars.cellSizeY * initVars.blockSizeY, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyToSymbolAsync(cenBound, initVars.h_cenBound, 3 *
    sizeof(float), 0, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyToSymbolAsync(halfBin, initVars.h_halfBin, 3 *
    sizeof(float), 0, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyToSymbolAsync(bandWidth, initVars.h_bandWidth, 3 *
    sizeof(float), 0, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyToSymbolAsync(tvbin, initVars.h_tvbin, 3 *
    sizeof(int), 0, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
}

void HostFreeHOGHistogramMemory(void) {
  checkCudaErrors(cudaFreeHost(hostWeights));
  hostWeights = NULL;
}

void DeviceFreeHOGHistogramMemory(void) {
  checkCudaErrors(cudaFreeArray(gaussArray));
  gaussArray = NULL;
}

// wt scale == scale for weighting function span
void InitHistograms(int cellSizeX, int cellSizeY, int blockSizeX,
    int blockSizeY, int noHistogramBins, float wtscale) {
  initVars.cellSizeX = cellSizeX;
  initVars.cellSizeY = cellSizeY;
  initVars.blockSizeX = blockSizeX;
  initVars.blockSizeY = blockSizeY;
  initVars.var2x = cellSizeX * blockSizeX / (2 * wtscale);
  initVars.var2y = cellSizeY * blockSizeY / (2 * wtscale);
  initVars.var2x *= initVars.var2x * 2;
  initVars.var2y *= initVars.var2y * 2;
  initVars.centerX = cellSizeX * blockSizeX / 2.0f;
  initVars.centerY = cellSizeY * blockSizeY / 2.0f;
  channelDescGauss = cudaCreateChannelDesc<float>();
  initVars.h_cenBound[0] = cellSizeX * blockSizeX / 2.0f;
  initVars.h_cenBound[1] = cellSizeY * blockSizeY / 2.0f;
  // TODO: Can be 360
  initVars.h_cenBound[2] = 180 / 2.0f;
  initVars.h_halfBin[0] = blockSizeX / 2.0f;
  initVars.h_halfBin[1] = blockSizeY / 2.0f;
  initVars.h_halfBin[2] = noHistogramBins / 2.0f;
  initVars.h_bandWidth[0] = (float) cellSizeX;
  initVars.h_bandWidth[0] = 1.0f / initVars.h_bandWidth[0];
  initVars.h_bandWidth[1] = (float) cellSizeY;
  initVars.h_bandWidth[1] = 1.0f / initVars.h_bandWidth[1];
  // TODO: Can be 360
  initVars.h_bandWidth[2] = 180.0f / (float) noHistogramBins;
  initVars.h_bandWidth[2] = 1.0f / initVars.h_bandWidth[2];
  initVars.h_tvbin[0] = blockSizeX;
  initVars.h_tvbin[1] = blockSizeY;
  initVars.h_tvbin[2] = noHistogramBins;
}

void CloseHistogram() {}

__global__ void computeBlockHistogramsWithGauss(float2* inputImage,
    float1* blockHistograms, int noHistogramBins, int cellSizeX, int cellSizeY,
    int blockSizeX, int blockSizeY, int leftoverX, int leftoverY, int width,
    int height) {
  int i;
  float2 localValue;
  float* shLocalHistograms = (float*) allShared;
  int cellIdx = threadIdx.y;
  int cellIdy = threadIdx.z;
  int columnId = threadIdx.x;
  int smemReadPos = __mul24(cellIdx, noHistogramBins) + __mul24(cellIdy,
    blockSizeX) * noHistogramBins;
  int gmemWritePos = __mul24(threadIdx.y, noHistogramBins) +
    __mul24(threadIdx.z, gridDim.x) * __mul24(blockDim.y, noHistogramBins) +
    __mul24(blockIdx.x, noHistogramBins) * blockDim.y + __mul24(blockIdx.y,
    gridDim.x) * __mul24(blockDim.y, noHistogramBins) * blockDim.z;
  int gmemReadStride = width;
  int gmemReadPos = leftoverX + __mul24(leftoverY, gmemReadStride) +
    (__mul24(blockIdx.x, cellSizeX) + __mul24(blockIdx.y, cellSizeY) *
    gmemReadStride) + (columnId + __mul24(cellIdx, cellSizeX) +
    __mul24(cellIdy, cellSizeY) * gmemReadStride);
  int histogramSize = __mul24(noHistogramBins, blockSizeX) * blockSizeY;
  int smemLocalHistogramPos = (columnId + __mul24(cellIdx, cellSizeX)) *
    histogramSize + __mul24(cellIdy, histogramSize) * __mul24(blockSizeX,
    cellSizeX);
  int cmemReadPos = columnId + __mul24(cellIdx, cellSizeX) + __mul24(cellIdy,
    cellSizeY) * __mul24(cellSizeX, blockSizeX);
  float atx, aty;
  float pIx, pIy, pIz;
  int fIx, fIy, fIz;
  int cIx, cIy, cIz;
  float dx, dy, dz;
  float cx, cy, cz;
  bool lowervalidx, lowervalidy;
  bool uppervalidx, uppervalidy;
  bool canWrite;
  int offset;
  for (i = 0; i < histogramSize; i++) {
    shLocalHistograms[smemLocalHistogramPos + i] = 0;
  }

#ifdef UNROLL_LOOPS
  int halfSizeYm1 = cellSizeY / 2 - 1;
#endif

  //if (blockIdx.x == 5 && blockIdx.y == 4)
  //{
  //  int asasa;
  //  asasa = 0;
  //  asasa++;
  //}

  for (i = 0; i < cellSizeY; i++) {
    localValue = inputImage[gmemReadPos + i * gmemReadStride];
    localValue.x *= tex1D(texGauss, cmemReadPos + i * cellSizeX * blockSizeX);
    atx = cellIdx * cellSizeX + columnId + 0.5;
    aty = cellIdy * cellSizeY + i + 0.5;
    pIx = halfBin[0] - oneHalf + (atx - cenBound[0]) * bandWidth[0];
    pIy = halfBin[1] - oneHalf + (aty - cenBound[1]) * bandWidth[1];
    pIz = halfBin[2] - oneHalf + (localValue.y - cenBound[2]) * bandWidth[2];
    fIx = floorf(pIx);
    fIy = floorf(pIy);
    fIz = floorf(pIz);
    cIx = fIx + 1;
    cIy = fIy + 1;
    cIz = fIz + 1; //eq ceilf(pI.)
    dx = pIx - fIx;
    dy = pIy - fIy;
    dz = pIz - fIz;
    cx = 1 - dx;
    cy = 1 - dy;
    cz = 1 - dz;
    cIz %= tvbin[2];
    fIz %= tvbin[2];
    if (fIz < 0) fIz += tvbin[2];
    if (cIz < 0) cIz += tvbin[2];

#ifdef UNROLL_LOOPS
    if ((i & halfSizeYm1) == 0)
#endif
    {
      uppervalidx = !(cIx >= tvbin[0] - oneHalf || cIx < -oneHalf);
      uppervalidy = !(cIy >= tvbin[1] - oneHalf || cIy < -oneHalf);
      lowervalidx = !(fIx < -oneHalf || fIx >= tvbin[0] - oneHalf);
      lowervalidy = !(fIy < -oneHalf || fIy >= tvbin[1] - oneHalf);
    }
    canWrite = lowervalidx && lowervalidy;
    if (canWrite) {
      offset = smemLocalHistogramPos + (fIx + fIy * blockSizeY) *
        noHistogramBins;
      shLocalHistograms[offset + fIz] += localValue.x * cx * cy * cz;
      shLocalHistograms[offset + cIz] += localValue.x * cx * cy * dz;
    }
    canWrite = lowervalidx && uppervalidy;
    if (canWrite) {
      offset = smemLocalHistogramPos + (fIx + cIy * blockSizeY) *
        noHistogramBins;
      shLocalHistograms[offset + fIz] += localValue.x * cx * dy * cz;
      shLocalHistograms[offset + cIz] += localValue.x * cx * dy * dz;
    }
    canWrite = uppervalidx && lowervalidy;
    if (canWrite) {
      offset = smemLocalHistogramPos + (cIx + fIy * blockSizeY) *
        noHistogramBins;
      shLocalHistograms[offset + fIz] += localValue.x * dx * cy * cz;
      shLocalHistograms[offset + cIz] += localValue.x * dx * cy * dz;
    }
    canWrite = (uppervalidx) && (uppervalidy);
    if (canWrite) {
      offset = smemLocalHistogramPos + (cIx + cIy * blockSizeY) *
        noHistogramBins;
      shLocalHistograms[offset + fIz] += localValue.x * dx * dy * cz;
      shLocalHistograms[offset + cIz] += localValue.x * dx * dy * dz;
    }
  }
  __syncthreads();
  //TODO -> aligned block size * cell size
  int smemTargetHistogramPos;
  for (unsigned int s = blockSizeY >> 1; s > 0; s >>= 1) {
    if (cellIdy < s && (cellIdy + s) < blockSizeY) {
      smemTargetHistogramPos = (columnId + __mul24(cellIdx, cellSizeX)) *
        histogramSize + __mul24((cellIdy + s), histogramSize) *
        __mul24(blockSizeX, cellSizeX);

#ifdef UNROLL_LOOPS
      shLocalHistograms[smemLocalHistogramPos + 0] += shLocalHistograms[smemTargetHistogramPos + 0];
      shLocalHistograms[smemLocalHistogramPos + 1] += shLocalHistograms[smemTargetHistogramPos + 1];
      shLocalHistograms[smemLocalHistogramPos + 2] += shLocalHistograms[smemTargetHistogramPos + 2];
      shLocalHistograms[smemLocalHistogramPos + 3] += shLocalHistograms[smemTargetHistogramPos + 3];
      shLocalHistograms[smemLocalHistogramPos + 4] += shLocalHistograms[smemTargetHistogramPos + 4];
      shLocalHistograms[smemLocalHistogramPos + 5] += shLocalHistograms[smemTargetHistogramPos + 5];
      shLocalHistograms[smemLocalHistogramPos + 6] += shLocalHistograms[smemTargetHistogramPos + 6];
      shLocalHistograms[smemLocalHistogramPos + 7] += shLocalHistograms[smemTargetHistogramPos + 7];
      shLocalHistograms[smemLocalHistogramPos + 8] += shLocalHistograms[smemTargetHistogramPos + 8];
      shLocalHistograms[smemLocalHistogramPos + 9] += shLocalHistograms[smemTargetHistogramPos + 9];
      shLocalHistograms[smemLocalHistogramPos + 10] += shLocalHistograms[smemTargetHistogramPos + 10];
      shLocalHistograms[smemLocalHistogramPos + 11] += shLocalHistograms[smemTargetHistogramPos + 11];
      shLocalHistograms[smemLocalHistogramPos + 12] += shLocalHistograms[smemTargetHistogramPos + 12];
      shLocalHistograms[smemLocalHistogramPos + 13] += shLocalHistograms[smemTargetHistogramPos + 13];
      shLocalHistograms[smemLocalHistogramPos + 14] += shLocalHistograms[smemTargetHistogramPos + 14];
      shLocalHistograms[smemLocalHistogramPos + 15] += shLocalHistograms[smemTargetHistogramPos + 15];
      shLocalHistograms[smemLocalHistogramPos + 16] += shLocalHistograms[smemTargetHistogramPos + 16];
      shLocalHistograms[smemLocalHistogramPos + 17] += shLocalHistograms[smemTargetHistogramPos + 17];
      shLocalHistograms[smemLocalHistogramPos + 18] += shLocalHistograms[smemTargetHistogramPos + 18];
      shLocalHistograms[smemLocalHistogramPos + 19] += shLocalHistograms[smemTargetHistogramPos + 19];
      shLocalHistograms[smemLocalHistogramPos + 20] += shLocalHistograms[smemTargetHistogramPos + 20];
      shLocalHistograms[smemLocalHistogramPos + 21] += shLocalHistograms[smemTargetHistogramPos + 21];
      shLocalHistograms[smemLocalHistogramPos + 22] += shLocalHistograms[smemTargetHistogramPos + 22];
      shLocalHistograms[smemLocalHistogramPos + 23] += shLocalHistograms[smemTargetHistogramPos + 23];
      shLocalHistograms[smemLocalHistogramPos + 24] += shLocalHistograms[smemTargetHistogramPos + 24];
      shLocalHistograms[smemLocalHistogramPos + 25] += shLocalHistograms[smemTargetHistogramPos + 25];
      shLocalHistograms[smemLocalHistogramPos + 26] += shLocalHistograms[smemTargetHistogramPos + 26];
      shLocalHistograms[smemLocalHistogramPos + 27] += shLocalHistograms[smemTargetHistogramPos + 27];
      shLocalHistograms[smemLocalHistogramPos + 28] += shLocalHistograms[smemTargetHistogramPos + 28];
      shLocalHistograms[smemLocalHistogramPos + 29] += shLocalHistograms[smemTargetHistogramPos + 29];
      shLocalHistograms[smemLocalHistogramPos + 30] += shLocalHistograms[smemTargetHistogramPos + 30];
      shLocalHistograms[smemLocalHistogramPos + 31] += shLocalHistograms[smemTargetHistogramPos + 31];
      shLocalHistograms[smemLocalHistogramPos + 32] += shLocalHistograms[smemTargetHistogramPos + 32];
      shLocalHistograms[smemLocalHistogramPos + 33] += shLocalHistograms[smemTargetHistogramPos + 33];
      shLocalHistograms[smemLocalHistogramPos + 34] += shLocalHistograms[smemTargetHistogramPos + 34];
      shLocalHistograms[smemLocalHistogramPos + 35] += shLocalHistograms[smemTargetHistogramPos + 35];
#else
      for (i = 0; i < histogramSize; i++) {
        shLocalHistograms[smemLocalHistogramPos + i] +=
          shLocalHistograms[smemTargetHistogramPos + i];
      }
#endif
    }
    __syncthreads();
  }
  for (unsigned int s = blockSizeX >> 1; s > 0; s >>= 1) {
    if (cellIdx < s && (cellIdx + s) < blockSizeX) {
      smemTargetHistogramPos = (columnId + __mul24((cellIdx + s), cellSizeX)) *
        histogramSize + __mul24(cellIdy, histogramSize) * __mul24(blockSizeX,
        cellSizeX);
#ifdef UNROLL_LOOPS
      shLocalHistograms[smemLocalHistogramPos + 0] += shLocalHistograms[smemTargetHistogramPos + 0];
      shLocalHistograms[smemLocalHistogramPos + 1] += shLocalHistograms[smemTargetHistogramPos + 1];
      shLocalHistograms[smemLocalHistogramPos + 2] += shLocalHistograms[smemTargetHistogramPos + 2];
      shLocalHistograms[smemLocalHistogramPos + 3] += shLocalHistograms[smemTargetHistogramPos + 3];
      shLocalHistograms[smemLocalHistogramPos + 4] += shLocalHistograms[smemTargetHistogramPos + 4];
      shLocalHistograms[smemLocalHistogramPos + 5] += shLocalHistograms[smemTargetHistogramPos + 5];
      shLocalHistograms[smemLocalHistogramPos + 6] += shLocalHistograms[smemTargetHistogramPos + 6];
      shLocalHistograms[smemLocalHistogramPos + 7] += shLocalHistograms[smemTargetHistogramPos + 7];
      shLocalHistograms[smemLocalHistogramPos + 8] += shLocalHistograms[smemTargetHistogramPos + 8];
      shLocalHistograms[smemLocalHistogramPos + 9] += shLocalHistograms[smemTargetHistogramPos + 9];
      shLocalHistograms[smemLocalHistogramPos + 10] += shLocalHistograms[smemTargetHistogramPos + 10];
      shLocalHistograms[smemLocalHistogramPos + 11] += shLocalHistograms[smemTargetHistogramPos + 11];
      shLocalHistograms[smemLocalHistogramPos + 12] += shLocalHistograms[smemTargetHistogramPos + 12];
      shLocalHistograms[smemLocalHistogramPos + 13] += shLocalHistograms[smemTargetHistogramPos + 13];
      shLocalHistograms[smemLocalHistogramPos + 14] += shLocalHistograms[smemTargetHistogramPos + 14];
      shLocalHistograms[smemLocalHistogramPos + 15] += shLocalHistograms[smemTargetHistogramPos + 15];
      shLocalHistograms[smemLocalHistogramPos + 16] += shLocalHistograms[smemTargetHistogramPos + 16];
      shLocalHistograms[smemLocalHistogramPos + 17] += shLocalHistograms[smemTargetHistogramPos + 17];
      shLocalHistograms[smemLocalHistogramPos + 18] += shLocalHistograms[smemTargetHistogramPos + 18];
      shLocalHistograms[smemLocalHistogramPos + 19] += shLocalHistograms[smemTargetHistogramPos + 19];
      shLocalHistograms[smemLocalHistogramPos + 20] += shLocalHistograms[smemTargetHistogramPos + 20];
      shLocalHistograms[smemLocalHistogramPos + 21] += shLocalHistograms[smemTargetHistogramPos + 21];
      shLocalHistograms[smemLocalHistogramPos + 22] += shLocalHistograms[smemTargetHistogramPos + 22];
      shLocalHistograms[smemLocalHistogramPos + 23] += shLocalHistograms[smemTargetHistogramPos + 23];
      shLocalHistograms[smemLocalHistogramPos + 24] += shLocalHistograms[smemTargetHistogramPos + 24];
      shLocalHistograms[smemLocalHistogramPos + 25] += shLocalHistograms[smemTargetHistogramPos + 25];
      shLocalHistograms[smemLocalHistogramPos + 26] += shLocalHistograms[smemTargetHistogramPos + 26];
      shLocalHistograms[smemLocalHistogramPos + 27] += shLocalHistograms[smemTargetHistogramPos + 27];
      shLocalHistograms[smemLocalHistogramPos + 28] += shLocalHistograms[smemTargetHistogramPos + 28];
      shLocalHistograms[smemLocalHistogramPos + 29] += shLocalHistograms[smemTargetHistogramPos + 29];
      shLocalHistograms[smemLocalHistogramPos + 30] += shLocalHistograms[smemTargetHistogramPos + 30];
      shLocalHistograms[smemLocalHistogramPos + 31] += shLocalHistograms[smemTargetHistogramPos + 31];
      shLocalHistograms[smemLocalHistogramPos + 32] += shLocalHistograms[smemTargetHistogramPos + 32];
      shLocalHistograms[smemLocalHistogramPos + 33] += shLocalHistograms[smemTargetHistogramPos + 33];
      shLocalHistograms[smemLocalHistogramPos + 34] += shLocalHistograms[smemTargetHistogramPos + 34];
      shLocalHistograms[smemLocalHistogramPos + 35] += shLocalHistograms[smemTargetHistogramPos + 35];
#else
      for (i = 0; i < histogramSize; i++) {
        shLocalHistograms[smemLocalHistogramPos + i] +=
          shLocalHistograms[smemTargetHistogramPos + i];
      }
#endif
    }
    __syncthreads();
  }

  for (unsigned int s = cellSizeX >> 1; s > 0; s >>= 1) {
    if (columnId < s && (columnId + s) < cellSizeX) {
      smemTargetHistogramPos = (columnId + s + __mul24(cellIdx, cellSizeX)) *
        histogramSize + __mul24(cellIdy, histogramSize) * __mul24(blockSizeX,
        cellSizeX);

#ifdef UNROLL_LOOPS
      shLocalHistograms[smemLocalHistogramPos + 0] += shLocalHistograms[smemTargetHistogramPos + 0];
      shLocalHistograms[smemLocalHistogramPos + 1] += shLocalHistograms[smemTargetHistogramPos + 1];
      shLocalHistograms[smemLocalHistogramPos + 2] += shLocalHistograms[smemTargetHistogramPos + 2];
      shLocalHistograms[smemLocalHistogramPos + 3] += shLocalHistograms[smemTargetHistogramPos + 3];
      shLocalHistograms[smemLocalHistogramPos + 4] += shLocalHistograms[smemTargetHistogramPos + 4];
      shLocalHistograms[smemLocalHistogramPos + 5] += shLocalHistograms[smemTargetHistogramPos + 5];
      shLocalHistograms[smemLocalHistogramPos + 6] += shLocalHistograms[smemTargetHistogramPos + 6];
      shLocalHistograms[smemLocalHistogramPos + 7] += shLocalHistograms[smemTargetHistogramPos + 7];
      shLocalHistograms[smemLocalHistogramPos + 8] += shLocalHistograms[smemTargetHistogramPos + 8];
      shLocalHistograms[smemLocalHistogramPos + 9] += shLocalHistograms[smemTargetHistogramPos + 9];
      shLocalHistograms[smemLocalHistogramPos + 10] += shLocalHistograms[smemTargetHistogramPos + 10];
      shLocalHistograms[smemLocalHistogramPos + 11] += shLocalHistograms[smemTargetHistogramPos + 11];
      shLocalHistograms[smemLocalHistogramPos + 12] += shLocalHistograms[smemTargetHistogramPos + 12];
      shLocalHistograms[smemLocalHistogramPos + 13] += shLocalHistograms[smemTargetHistogramPos + 13];
      shLocalHistograms[smemLocalHistogramPos + 14] += shLocalHistograms[smemTargetHistogramPos + 14];
      shLocalHistograms[smemLocalHistogramPos + 15] += shLocalHistograms[smemTargetHistogramPos + 15];
      shLocalHistograms[smemLocalHistogramPos + 16] += shLocalHistograms[smemTargetHistogramPos + 16];
      shLocalHistograms[smemLocalHistogramPos + 17] += shLocalHistograms[smemTargetHistogramPos + 17];
      shLocalHistograms[smemLocalHistogramPos + 18] += shLocalHistograms[smemTargetHistogramPos + 18];
      shLocalHistograms[smemLocalHistogramPos + 19] += shLocalHistograms[smemTargetHistogramPos + 19];
      shLocalHistograms[smemLocalHistogramPos + 20] += shLocalHistograms[smemTargetHistogramPos + 20];
      shLocalHistograms[smemLocalHistogramPos + 21] += shLocalHistograms[smemTargetHistogramPos + 21];
      shLocalHistograms[smemLocalHistogramPos + 22] += shLocalHistograms[smemTargetHistogramPos + 22];
      shLocalHistograms[smemLocalHistogramPos + 23] += shLocalHistograms[smemTargetHistogramPos + 23];
      shLocalHistograms[smemLocalHistogramPos + 24] += shLocalHistograms[smemTargetHistogramPos + 24];
      shLocalHistograms[smemLocalHistogramPos + 25] += shLocalHistograms[smemTargetHistogramPos + 25];
      shLocalHistograms[smemLocalHistogramPos + 26] += shLocalHistograms[smemTargetHistogramPos + 26];
      shLocalHistograms[smemLocalHistogramPos + 27] += shLocalHistograms[smemTargetHistogramPos + 27];
      shLocalHistograms[smemLocalHistogramPos + 28] += shLocalHistograms[smemTargetHistogramPos + 28];
      shLocalHistograms[smemLocalHistogramPos + 29] += shLocalHistograms[smemTargetHistogramPos + 29];
      shLocalHistograms[smemLocalHistogramPos + 30] += shLocalHistograms[smemTargetHistogramPos + 30];
      shLocalHistograms[smemLocalHistogramPos + 31] += shLocalHistograms[smemTargetHistogramPos + 31];
      shLocalHistograms[smemLocalHistogramPos + 32] += shLocalHistograms[smemTargetHistogramPos + 32];
      shLocalHistograms[smemLocalHistogramPos + 33] += shLocalHistograms[smemTargetHistogramPos + 33];
      shLocalHistograms[smemLocalHistogramPos + 34] += shLocalHistograms[smemTargetHistogramPos + 34];
      shLocalHistograms[smemLocalHistogramPos + 35] += shLocalHistograms[smemTargetHistogramPos + 35];
#else
      for (i = 0; i < histogramSize; i++) {
        shLocalHistograms[smemLocalHistogramPos + i] +=
          shLocalHistograms[smemTargetHistogramPos + i];
      }
#endif
    }
    __syncthreads();
  }

  if (columnId == 0) {
    //write result to gmem
#ifdef UNROLL_LOOPS
    blockHistograms[gmemWritePos + 0].x = shLocalHistograms[smemReadPos + 0];
    blockHistograms[gmemWritePos + 1].x = shLocalHistograms[smemReadPos + 1];
    blockHistograms[gmemWritePos + 2].x = shLocalHistograms[smemReadPos + 2];
    blockHistograms[gmemWritePos + 3].x = shLocalHistograms[smemReadPos + 3];
    blockHistograms[gmemWritePos + 4].x = shLocalHistograms[smemReadPos + 4];
    blockHistograms[gmemWritePos + 5].x = shLocalHistograms[smemReadPos + 5];
    blockHistograms[gmemWritePos + 6].x = shLocalHistograms[smemReadPos + 6];
    blockHistograms[gmemWritePos + 7].x = shLocalHistograms[smemReadPos + 7];
    blockHistograms[gmemWritePos + 8].x = shLocalHistograms[smemReadPos + 8];
#else
    for (i=0; i<noHistogramBins; i++) {
      blockHistograms[gmemWritePos + i].x = shLocalHistograms[smemReadPos + i];
    }
#endif
  }

  if (blockIdx.x == 10 && blockIdx.y == 8) {
    int asasa;
    asasa = 0;
    asasa++;
  }
}

void ComputeBlockHistogramsWithGauss(float2* inputImage,
    float1* blockHistograms, int noHistogramBins, int cellSizeX, int cellSizeY,
    int blockSizeX, int blockSizeY, int windowSizeX, int windowSizeY,
    int width, int height) {
  int leftoverX;
  int leftoverY;
  dim3 hThreadSize, hBlockSize;
  rNoOfCellsX = width / cellSizeX;
  rNoOfCellsY = height / cellSizeY;
  rNoOfBlocksX = rNoOfCellsX - blockSizeX + 1;
  rNoOfBlocksY = rNoOfCellsY - blockSizeY + 1;
  rNumberOfWindowsX = (width-windowSizeX)/cellSizeX + 1;
  rNumberOfWindowsY = (height-windowSizeY)/cellSizeY + 1;
  leftoverX = (width - windowSizeX - cellSizeX * (rNumberOfWindowsX - 1)) / 2;
  leftoverY = (height - windowSizeY - cellSizeY * (rNumberOfWindowsY - 1)) / 2;
  hThreadSize = dim3(cellSizeX, blockSizeX, blockSizeY);
  hBlockSize = dim3(rNoOfBlocksX, rNoOfBlocksY);
  checkCudaErrors(cudaBindTextureToArray(texGauss, gaussArray, channelDescGauss));
  computeBlockHistogramsWithGauss<<<hBlockSize, hThreadSize, noHistogramBins *
    blockSizeX * blockSizeY * cellSizeX * blockSizeY * blockSizeX *
    sizeof(float), stream>>>(inputImage, blockHistograms, noHistogramBins,
    cellSizeX, cellSizeY, blockSizeX, blockSizeY, leftoverX, leftoverY, width,
    height);
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaUnbindTexture(texGauss));
}

void NormalizeBlockHistograms(float1* blockHistograms, int noHistogramBins,
    int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY, int width,
    int height) {
  dim3 hThreadSize, hBlockSize;
  rNoOfCellsX = width / cellSizeX;
  rNoOfCellsY = height / cellSizeY;
  rNoOfBlocksX = rNoOfCellsX - blockSizeX + 1;
  rNoOfBlocksY = rNoOfCellsY - blockSizeY + 1;
  hThreadSize = dim3(noHistogramBins, blockSizeX, blockSizeY);
  hBlockSize = dim3(rNoOfBlocksX, rNoOfBlocksY);
  int alignedBlockDimX = iClosestPowerOfTwo(noHistogramBins);
  int alignedBlockDimY = iClosestPowerOfTwo(blockSizeX);
  int alignedBlockDimZ = iClosestPowerOfTwo(blockSizeY);
  normalizeBlockHistograms<<<hBlockSize, hThreadSize, noHistogramBins *
    blockSizeX * blockSizeY * sizeof(float), stream>>>(blockHistograms,
    noHistogramBins, rNoOfBlocksX, rNoOfBlocksY, blockSizeX, blockSizeY,
    alignedBlockDimX, alignedBlockDimY, alignedBlockDimZ, noHistogramBins *
    rNoOfCellsX, rNoOfCellsY);
  checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void normalizeBlockHistograms(float1 *blockHistograms,
    int noHistogramBins, int rNoOfHOGBlocksX, int rNoOfHOGBlocksY,
    int blockSizeX, int blockSizeY, int alignedBlockDimX, int alignedBlockDimY,
    int alignedBlockDimZ, int width, int height) {
  int smemLocalHistogramPos, smemTargetHistogramPos, gmemPosBlock,
    gmemWritePosBlock;
  float* shLocalHistogram = (float*) allShared;
  float localValue, norm1, norm2;
  float eps2 = 0.01f;
  smemLocalHistogramPos = __mul24(threadIdx.y, noHistogramBins) +
    __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;
  gmemPosBlock = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z,
    gridDim.x) * __mul24(blockDim.y, blockDim.x) + threadIdx.x +
    __mul24(blockIdx.x, noHistogramBins) * blockDim.y + __mul24(blockIdx.y,
    gridDim.x) * __mul24(blockDim.y, blockDim.x) * blockDim.z;
  gmemWritePosBlock = __mul24(threadIdx.z, noHistogramBins) +
    __mul24(threadIdx.y, gridDim.x) * __mul24(blockDim.y, blockDim.x) +
    threadIdx.x + __mul24(blockIdx.x, noHistogramBins) * blockDim.y +
    __mul24(blockIdx.y, gridDim.x) * __mul24(blockDim.y, blockDim.x) *
    blockDim.z;
  localValue = blockHistograms[gmemPosBlock].x;
  shLocalHistogram[smemLocalHistogramPos] = localValue * localValue;
  if (blockIdx.x == 10 && blockIdx.y == 8) {
    int asasa;
    asasa = 0;
    asasa++;
  }
  __syncthreads();
  for(unsigned int s = alignedBlockDimZ >> 1; s > 0; s >>= 1) {
    if (threadIdx.z < s && (threadIdx.z + s) < blockDim.z) {
      smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) +
        __mul24((threadIdx.z + s), blockDim.x) * blockDim.y + threadIdx.x;
      shLocalHistogram[smemLocalHistogramPos] +=
        shLocalHistogram[smemTargetHistogramPos];
    }
    __syncthreads();
  }
  for (unsigned int s = alignedBlockDimY >> 1; s > 0; s >>= 1) {
    if (threadIdx.y < s && (threadIdx.y + s) < blockDim.y) {
      smemTargetHistogramPos = __mul24((threadIdx.y + s), noHistogramBins) +
        __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;
      shLocalHistogram[smemLocalHistogramPos] +=
        shLocalHistogram[smemTargetHistogramPos];
    }
    __syncthreads();
  }
  for(unsigned int s = alignedBlockDimX >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
      smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) +
        __mul24(threadIdx.z, blockDim.x) * blockDim.y + (threadIdx.x + s);
      shLocalHistogram[smemLocalHistogramPos] +=
        shLocalHistogram[smemTargetHistogramPos];
    }
    __syncthreads();
  }
  norm1 = sqrtf(shLocalHistogram[0]) + __mul24(noHistogramBins, blockSizeX) *
    blockSizeY;
  localValue /= norm1;

  localValue = fminf(0.2f, localValue); //why 0.2 ??
  __syncthreads();
  shLocalHistogram[smemLocalHistogramPos] = localValue * localValue;
  __syncthreads();
  for(unsigned int s = alignedBlockDimZ >> 1; s > 0; s >>= 1) {
    if (threadIdx.z < s && (threadIdx.z + s) < blockDim.z) {
      smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) +
        __mul24((threadIdx.z + s), blockDim.x) * blockDim.y + threadIdx.x;
      shLocalHistogram[smemLocalHistogramPos] +=
        shLocalHistogram[smemTargetHistogramPos];
    }
    __syncthreads();
  }
  for (unsigned int s = alignedBlockDimY >> 1; s > 0; s >>= 1) {
    if (threadIdx.y < s && (threadIdx.y + s) < blockDim.y) {
      smemTargetHistogramPos = __mul24((threadIdx.y + s), noHistogramBins) +
        __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;
      shLocalHistogram[smemLocalHistogramPos] +=
        shLocalHistogram[smemTargetHistogramPos];
    }
    __syncthreads();
  }
  for(unsigned int s = alignedBlockDimX >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
      smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) +
        __mul24(threadIdx.z, blockDim.x) * blockDim.y + (threadIdx.x + s);
      shLocalHistogram[smemLocalHistogramPos] +=
        shLocalHistogram[smemTargetHistogramPos];
    }
    __syncthreads();
  }
  norm2 = sqrtf(shLocalHistogram[0]) + eps2;
  localValue /= norm2;
  blockHistograms[gmemWritePosBlock].x = localValue;
  if (blockIdx.x == 10 && blockIdx.y == 8) {
    int asasa;
    asasa = 0;
    asasa++;
  }
}
