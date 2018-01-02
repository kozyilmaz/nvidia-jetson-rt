#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "HOGConvolution.h"
#include "HOGEngine.h"
#include "HOGHistogram.h"
#include "HOGPadding.h"
#include "HOGScale.h"
#include "HOGSVMSlider.h"
#include "HOGUtils.h"
#include "HOGEngineDevice.h"

int hWidth, hHeight;
int hWidthROI, hHeightROI;
int hPaddedWidth, hPaddedHeight;
int rPaddedWidth, rPaddedHeight;

int minX, minY, maxX, maxY;

int hNoHistogramBins, rNoHistogramBins;

int hPaddingSizeX, hPaddingSizeY;
int hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, hWindowSizeX, hWindowSizeY;
int hNoOfCellsX, hNoOfCellsY, hNoOfBlocksX, hNoOfBlocksY;
int rNoOfCellsX, rNoOfCellsY, rNoOfBlocksX, rNoOfBlocksY;

int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;
int hNumberOfWindowsX, hNumberOfWindowsY;
int rNumberOfWindowsX, rNumberOfWindowsY;

float4 *paddedRegisteredImage;

float1 *resizedPaddedImageF1;
float4 *resizedPaddedImageF4;

float2 *colorGradientsF2;

float1 *blockHistograms;
float1 *cellHistograms;

float1 *svmScores;

bool hUseGrayscale;

uchar1* outputTest1;
uchar4* outputTest4;

float* hResult;

float scaleRatio;
float startScale;
float endScale;
int scaleCount;

int avSizeX, avSizeY, marginX, marginY;

extern uchar4* paddedRegisteredImageU4;

void DeviceAllocHOGEngineDeviceMemory(void) {
  DeviceAllocHOGConvolutionMemory();
  DeviceAllocHOGHistogramMemory();
  DeviceAllocHOGSVMMemory();
  DeviceAllocHOGPaddingMemory();
  DeviceAllocHOGScaleMemory();
  checkCudaErrors(cudaMalloc(&paddedRegisteredImage, sizeof(float4) *
    hPaddedWidth * hPaddedHeight));
  if (hUseGrayscale) {
    checkCudaErrors(cudaMalloc(&resizedPaddedImageF1, sizeof(float1) *
      hPaddedWidth * hPaddedHeight));
  } else {
    checkCudaErrors(cudaMalloc(&resizedPaddedImageF4, sizeof(float4) *
      hPaddedWidth * hPaddedHeight));
  }
  checkCudaErrors(cudaMalloc(&colorGradientsF2, sizeof(float2) * hPaddedWidth *
    hPaddedHeight));
  checkCudaErrors(cudaMalloc(&blockHistograms, sizeof(float1) * hNoOfBlocksX *
    hNoOfBlocksY * hCellSizeX * hCellSizeY * hNoHistogramBins));
  checkCudaErrors(cudaMalloc(&cellHistograms, sizeof(float1) * hNoOfCellsX *
    hNoOfCellsY * hNoHistogramBins));
  checkCudaErrors(cudaMalloc(&svmScores, sizeof(float1) * hNumberOfWindowsX *
    hNumberOfWindowsY * scaleCount));
  if (hUseGrayscale) {
    checkCudaErrors(cudaMalloc(&outputTest1, sizeof(uchar1) * hPaddedWidth *
      hPaddedHeight));
  } else {
    checkCudaErrors(cudaMalloc(&outputTest4, sizeof(uchar4) * hPaddedWidth *
      hPaddedHeight));
  }
}

void HostAllocHOGEngineDeviceMemory(void) {
  HostAllocHOGHistogramMemory();
  checkCudaErrors(cudaMallocHost(&hResult, sizeof(float) * hNumberOfWindowsX *
    hNumberOfWindowsY * scaleCount));
}

void CopyInHOGEngineDevice(void) {
  CopyInHOGConvolution();
  CopyInHOGHistogram();
  CopyInHOGSVM();
}

void HostFreeHOGEngineDeviceMemory(void) {
  checkCudaErrors(cudaFreeHost(hResult));
  hResult = NULL;
  HostFreeHOGHistogramMemory();
}

void DeviceFreeHOGEngineDeviceMemory(void) {
  checkCudaErrors(cudaFree(paddedRegisteredImage));
  if (hUseGrayscale) {
    checkCudaErrors(cudaFree(resizedPaddedImageF1));
  } else {
    checkCudaErrors(cudaFree(resizedPaddedImageF4));
  }
  checkCudaErrors(cudaFree(colorGradientsF2));
  checkCudaErrors(cudaFree(blockHistograms));
  checkCudaErrors(cudaFree(cellHistograms));
  checkCudaErrors(cudaFree(svmScores));
  DeviceFreeHOGConvolutionMemory();
  DeviceFreeHOGHistogramMemory();
  DeviceFreeHOGSVMMemory();
  DeviceFreeHOGPaddingMemory();
  DeviceFreeHOGScaleMemory();
  if (hUseGrayscale) {
    checkCudaErrors(cudaFree(outputTest1));
  } else {
    checkCudaErrors(cudaFree(outputTest4));
  }
}

void InitHOG(int width, int height) {
  cudaSetDevice(0);
  int i;
  int toaddxx = 0, toaddxy = 0, toaddyx = 0, toaddyy = 0;
  hWidth = width;
  hHeight = height;
  avSizeX = HOG.avSizeX;
  avSizeY = HOG.avSizeY;
  marginX = HOG.marginX;
  marginY = HOG.marginY;
  if (avSizeX != 0) {
    toaddxx = hWidth * marginX / avSizeX;
    toaddxy = hHeight * marginY / avSizeX;
  }
  if (avSizeY != 0) {
    toaddyx = hWidth * marginX / avSizeY;
    toaddyy = hHeight * marginY / avSizeY;
  }
  hPaddingSizeX = max(toaddxx, toaddyx);
  hPaddingSizeY = max(toaddxy, toaddyy);
  hPaddedWidth = hWidth + hPaddingSizeX * 2;
  hPaddedHeight = hHeight + hPaddingSizeY * 2;
  hUseGrayscale = HOG.useGrayscale;
  hNoHistogramBins = HOG.hNoOfHistogramBins;
  hCellSizeX = HOG.hCellSizeX;
  hCellSizeY = HOG.hCellSizeY;
  hBlockSizeX = HOG.hBlockSizeX;
  hBlockSizeY = HOG.hBlockSizeY;
  hWindowSizeX = HOG.hWindowSizeX;
  hWindowSizeY = HOG.hWindowSizeY;
  hNoOfCellsX = hPaddedWidth / hCellSizeX;
  hNoOfCellsY = hPaddedHeight / hCellSizeY;
  hNoOfBlocksX = hNoOfCellsX - hBlockSizeX + 1;
  hNoOfBlocksY = hNoOfCellsY - hBlockSizeY + 1;
  hNumberOfBlockPerWindowX = (hWindowSizeX - hCellSizeX * hBlockSizeX) /
    hCellSizeX + 1;
  hNumberOfBlockPerWindowY = (hWindowSizeY - hCellSizeY * hBlockSizeY) /
    hCellSizeY + 1;
  hNumberOfWindowsX = 0;
  for (i = 0; i < hNumberOfBlockPerWindowX; i++) {
    hNumberOfWindowsX += (hNoOfBlocksX - i) / hNumberOfBlockPerWindowX;
  }
  hNumberOfWindowsY = 0;
  for (i = 0; i < hNumberOfBlockPerWindowY; i++) {
    hNumberOfWindowsY += (hNoOfBlocksY - i) / hNumberOfBlockPerWindowY;
  }
  scaleRatio = 1.05f;
  startScale = 1.0f;
  endScale = min(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight /
    (float) hWindowSizeY);
  scaleCount = (int)floor(logf(endScale / startScale) / logf(scaleRatio)) + 1;
  InitConvolution(hPaddedWidth, hPaddedHeight, hUseGrayscale);
  InitHistograms(hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY,
    hNoHistogramBins, HOG.wtScale);
  InitSVM();
  InitScale();
  InitPadding();
  rPaddedWidth = hPaddedWidth;
  rPaddedHeight = hPaddedHeight;
}

void CloseHOG() {
  CloseConvolution();
  CloseHistogram();
  CloseSVM();
  CloseScale();
  ClosePadding();
}

void BeginHOGProcessing(unsigned char* hostImage, float minScale,
    float maxScale) {
  int i;
  minX = HOG.minX;
  minY = HOG.minY;
  maxX = HOG.maxX;
  maxY = HOG.maxY;
  PadHostImage((uchar4*)hostImage, paddedRegisteredImage, minX, minY, maxX,
    maxY);

  rPaddedWidth = hPaddedWidth; rPaddedHeight = hPaddedHeight;
  scaleRatio = 1.05f;
  startScale = (minScale < 0.0f) ? 1.0f : minScale;
  if (maxScale < 0.0f) {
    endScale = min(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight /
      (float) hWindowSizeY);
  } else {
    endScale = maxScale;
  }
  scaleCount = (int) floor(logf(endScale / startScale) / logf(scaleRatio)) + 1;
  float currentScale = startScale;
  ResetSVMScores(svmScores);
  for (i = 0; i < scaleCount; i++) {
    DownscaleImage(0, scaleCount, i, currentScale, hUseGrayscale,
      paddedRegisteredImage, resizedPaddedImageF1, resizedPaddedImageF4);
    SetConvolutionSize(rPaddedWidth, rPaddedHeight);
    if (hUseGrayscale) {
      ComputeColorGradients1to2(resizedPaddedImageF1, colorGradientsF2);
    } else {
      ComputeColorGradients4to2(resizedPaddedImageF4, colorGradientsF2);
    }
    ComputeBlockHistogramsWithGauss(colorGradientsF2, blockHistograms,
      hNoHistogramBins, hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY,
      hWindowSizeX, hWindowSizeY,  rPaddedWidth, rPaddedHeight);
    NormalizeBlockHistograms(blockHistograms, hNoHistogramBins, hCellSizeX,
      hCellSizeY, hBlockSizeX, hBlockSizeY, rPaddedWidth, rPaddedHeight);
    LinearSVMEvaluation(svmScores, blockHistograms, hNoHistogramBins,
      hWindowSizeX, hWindowSizeY, hCellSizeX, hCellSizeY, hBlockSizeX,
      hBlockSizeY, rNoOfBlocksX, rNoOfBlocksY, i, rPaddedWidth, rPaddedHeight);
    currentScale *= scaleRatio;
  }
}

float* EndHOGProcessing() {
  checkCudaErrors(cudaMemcpyAsync(hResult, svmScores, sizeof(float) *
    scaleCount * hNumberOfWindowsX * hNumberOfWindowsY,
    cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
  return hResult;
}

void GetHOGParameters() {
  HOG.startScale = startScale;
  HOG.endScale = endScale;
  HOG.scaleRatio = scaleRatio;
  HOG.scaleCount = scaleCount;
  HOG.hPaddingSizeX = hPaddingSizeX;
  HOG.hPaddingSizeY = hPaddingSizeY;
  HOG.hPaddedWidth = hPaddedWidth;
  HOG.hPaddedHeight = hPaddedHeight;
  HOG.hNoOfCellsX = hNoOfCellsX;
  HOG.hNoOfCellsY = hNoOfCellsY;
  HOG.hNoOfBlocksX = hNoOfBlocksX;
  HOG.hNoOfBlocksY = hNoOfBlocksY;
  HOG.hNumberOfWindowsX = hNumberOfWindowsX;
  HOG.hNumberOfWindowsY = hNumberOfWindowsY;
  HOG.hNumberOfBlockPerWindowX = hNumberOfBlockPerWindowX;
  HOG.hNumberOfBlockPerWindowY = hNumberOfBlockPerWindowY;
}

cudaArray *imageArray2 = 0;
texture<float4, 2, cudaReadModeElementType> tex2;
cudaChannelFormatDesc channelDescDownscale2;

__global__ void resizeFastBicubic3(float4 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale)
{
  int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int i = __umul24(y, width) + x;

  float u = x*scale;
  float v = y*scale;

  if (x < width && y < height)
  {
    float4 cF;

    if (scale == 1.0f)
      cF = paddedRegisteredImage[x + y * width];
    else
      cF = tex2D(tex2, u, v);

    outputFloat[i] = cF;
  }
}
