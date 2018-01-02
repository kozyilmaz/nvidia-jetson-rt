#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "HOGEngine.h"
#include "HOGUtils.h"
#include "HOGSVMSlider.h"

texture<float, 1, cudaReadModeElementType> texSVM;
cudaArray *svmArray = 0;
cudaChannelFormatDesc channelDescSVM;
extern int scaleCount;
extern int hNumberOfWindowsX, hNumberOfWindowsY;
extern int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;
extern int rNumberOfWindowsX, rNumberOfWindowsY;
extern __shared__ float1 allSharedF1[];
float svmBias;

void DeviceAllocHOGSVMMemory(void) {
  checkCudaErrors(cudaMallocArray(&svmArray, &channelDescSVM,
    HOG.svmWeightsCount, 1));
}

void CopyInHOGSVM(void) {
  checkCudaErrors(cudaMemcpyToArrayAsync(svmArray, 0, 0, HOG.svmWeights,
    HOG.svmWeightsCount * sizeof(float), cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
}

void DeviceFreeHOGSVMMemory(void) {
  checkCudaErrors(cudaFreeArray(svmArray));
  svmArray = NULL;
}

void InitSVM() {
  channelDescSVM = cudaCreateChannelDesc<float>();
  svmBias = HOG.svmBias;
}

void CloseSVM() {}

__global__ void linearSVMEvaluation(float1* svmScores, float svmBias,
    float1* blockHistograms, int noHistogramBins, int windowSizeX,
    int windowSizeY, int hogBlockCountX, int hogBlockCountY, int cellSizeX,
    int cellSizeY, int numberOfBlockPerWindowX, int numberOfBlockPerWindowY,
    int blockSizeX, int blockSizeY, int alignedBlockDimX, int scaleId,
    int scaleCount, int hNumberOfWindowsX, int hNumberOfWindowsY, int width,
    int height) {
  int i;
  int texPos;
  float1 localValue;
  float texValue;
  float1* smem = (float1*) allSharedF1;
  int gmemPosWindow, gmemPosInWindow, gmemPosInWindowDown, smemLocalPos,
    smemTargetPos;
  int gmemStride = hogBlockCountX * noHistogramBins * blockSizeX;
  gmemPosWindow = blockIdx.x * noHistogramBins * blockSizeX + blockIdx.y *
    blockSizeY * gmemStride;
  gmemPosInWindow = gmemPosWindow + threadIdx.x;
  smemLocalPos = threadIdx.x;
  int val1 = (blockSizeY * blockSizeX * noHistogramBins) *
    numberOfBlockPerWindowY;
  int val2 = blockSizeX * noHistogramBins;
  localValue.x = 0;
  if (blockIdx.x == 10 && blockIdx.y == 8) {
    int asasasa;
    asasasa = 0;
    asasasa++;
  }
  for (i = 0; i < blockSizeY * numberOfBlockPerWindowY; i++) {
    gmemPosInWindowDown = gmemPosInWindow + i * gmemStride;
    texPos = threadIdx.x % val2 + i * val2 + threadIdx.x / val2 * val1;
    texValue = tex1D(texSVM, texPos);
    localValue.x += blockHistograms[gmemPosInWindowDown].x * texValue;
  }
  smem[smemLocalPos] = localValue;
  __syncthreads();
  for(unsigned int s = alignedBlockDimX >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
      smemTargetPos = threadIdx.x + s;
      smem[smemLocalPos].x += smem[smemTargetPos].x;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    smem[smemLocalPos].x -= svmBias;
    svmScores[blockIdx.x + blockIdx.y * hNumberOfWindowsX + scaleId *
      hNumberOfWindowsX * hNumberOfWindowsY] = smem[smemLocalPos];
  }
  if (blockIdx.x == 10 && blockIdx.y == 8) {
    int asasasa;
    asasasa = 0;
    asasasa++;
  }
}

void ResetSVMScores(float1* svmScores) {
  checkCudaErrors(cudaMemsetAsync(svmScores, 0, sizeof(float) * scaleCount *
    hNumberOfWindowsX * hNumberOfWindowsY));
  checkCudaErrors(cudaStreamSynchronize(stream));
}

void LinearSVMEvaluation(float1* svmScores, float1* blockHistograms,
    int noHistogramBins, int windowSizeX, int windowSizeY, int cellSizeX,
    int cellSizeY, int blockSizeX, int blockSizeY, int hogBlockCountX,
    int hogBlockCountY, int scaleId, int width, int height) {
  rNumberOfWindowsX = (width - windowSizeX) / cellSizeX + 1;
  rNumberOfWindowsY = (height - windowSizeY) / cellSizeY + 1;
  dim3 threadCount = dim3(noHistogramBins * blockSizeX *
    hNumberOfBlockPerWindowX);
  dim3 blockCount = dim3(rNumberOfWindowsX, rNumberOfWindowsY);
  int alignedBlockDimX = iClosestPowerOfTwo(noHistogramBins * blockSizeX *
    hNumberOfBlockPerWindowX);
  checkCudaErrors(cudaBindTextureToArray(texSVM, svmArray, channelDescSVM));
  linearSVMEvaluation<<<blockCount, threadCount, noHistogramBins * blockSizeX *
    hNumberOfBlockPerWindowX * sizeof(float1), stream>>>(svmScores, svmBias,
    blockHistograms, noHistogramBins, windowSizeX, windowSizeY, hogBlockCountX,
    hogBlockCountY, cellSizeX, cellSizeY, hNumberOfBlockPerWindowX,
    hNumberOfBlockPerWindowY, blockSizeX, blockSizeY, alignedBlockDimX,
    scaleId, scaleCount, hNumberOfWindowsX, hNumberOfWindowsY, width, height);
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaUnbindTexture(texSVM));
}
