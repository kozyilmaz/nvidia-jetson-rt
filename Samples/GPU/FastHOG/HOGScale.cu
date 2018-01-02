#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "HOGEngine.h"
#include "HOGUtils.h"
#include "HOGScale.h"

extern int rPaddedHeight;
extern int rPaddedWidth;
extern int hPaddedHeight;
extern int hPaddedWidth;
cudaArray *imageArray = 0;
texture<float4, 2, cudaReadModeElementType> tex;
cudaChannelFormatDesc channelDescDownscale;

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__device__ float w0(float a) { return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f); }
__device__ float w1(float a) { return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f); }
__device__ float w2(float a) { return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f); }
__device__ float w3(float a) { return (1.0f/6.0f)*(a*a*a); }

// g0 and g1 are the two amplitude functions
__device__ float g0(float a) { return w0(a) + w1(a); }
__device__ float g1(float a) { return w2(a) + w3(a); }

// h0 and h1 are the two offset functions
__device__ float h0(float a) { return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f; }
__device__ float h1(float a) { return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f; }

void DeviceAllocHOGScaleMemory(void) {
  checkCudaErrors(cudaMallocArray(&imageArray, &channelDescDownscale,
    hPaddedWidth, hPaddedHeight));
}

void DeviceFreeHOGScaleMemory(void) {
  checkCudaErrors(cudaFreeArray(imageArray));
}

void InitScale() {
  channelDescDownscale = cudaCreateChannelDesc<float4>();
  tex.filterMode = cudaFilterModeLinear;
  tex.normalized = false;
}

void CloseScale() {}

void DownscaleImage(int startScaleId, int endScaleId, int scaleId, float scale,
    bool useGrayscale, float4* paddedRegisteredImage,
    float1* resizedPaddedImageF1, float4* resizedPaddedImageF4) {
  dim3 hThreadSize, hBlockSize;
  hThreadSize = dim3(THREAD_SIZE_W, THREAD_SIZE_H);
  rPaddedWidth = iDivUpF(hPaddedWidth, scale);
  rPaddedHeight = iDivUpF(hPaddedHeight, scale);
  hBlockSize = dim3(iDivUp(rPaddedWidth, hThreadSize.x), iDivUp(rPaddedHeight,
    hThreadSize.y));
  if (scaleId == startScaleId) {
    checkCudaErrors(cudaMemcpyToArrayAsync(imageArray, 0, 0,
      paddedRegisteredImage, sizeof(float4) * hPaddedWidth * hPaddedHeight,
      cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  checkCudaErrors(cudaBindTextureToArray(tex, imageArray, channelDescDownscale));

  if (useGrayscale) {
    checkCudaErrors(cudaMemsetAsync(resizedPaddedImageF1, 0, hPaddedWidth *
      hPaddedHeight * sizeof(float1), stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    resizeFastBicubic1<<<hBlockSize, hThreadSize, 0, stream>>>(
      resizedPaddedImageF1, paddedRegisteredImage, rPaddedWidth, rPaddedHeight,
      scale);
    checkCudaErrors(cudaStreamSynchronize(stream));
  } else {
    checkCudaErrors(cudaMemsetAsync(resizedPaddedImageF4, 0, hPaddedWidth *
      hPaddedHeight * sizeof(float4), stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    resizeFastBicubic4<<<hBlockSize, hThreadSize, 0, stream>>>(
      resizedPaddedImageF4, paddedRegisteredImage, rPaddedWidth, rPaddedHeight,
      scale);
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  checkCudaErrors(cudaUnbindTexture(tex));
}

__device__ float4 tex2DFastBicubic(const texture<float4, 2, cudaReadModeElementType> texref, float x, float y)
{
  float4 r;
  float4 val0, val1, val2, val3;

  x -= 0.5f;
  y -= 0.5f;
  float px = floor(x);
  float py = floor(y);
  float fx = x - px;
  float fy = y - py;

  float g0x = g0(fx);
  float g1x = g1(fx);
  float h0x = h0(fx);
  float h1x = h1(fx);
  float h0y = h0(fy);
  float h1y = h1(fy);

  val0 = tex2D(texref, px + h0x, py + h0y);
  val1 = tex2D(texref, px + h1x, py + h0y);
  val2 = tex2D(texref, px + h0x, py + h1y);
  val3 = tex2D(texref, px + h1x, py + h1y);

  r.x = (g0(fy) * (g0x * val0.x + g1x * val1.x) + g1(fy) * (g0x * val2.x + g1x * val3.x));
  r.y = (g0(fy) * (g0x * val0.y + g1x * val1.y) + g1(fy) * (g0x * val2.y + g1x * val3.y));
  r.z = (g0(fy) * (g0x * val0.z + g1x * val1.z) + g1(fy) * (g0x * val2.z + g1x * val3.z));
  r.w = (g0(fy) * (g0x * val0.w + g1x * val1.w) + g1(fy) * (g0x * val2.w + g1x * val3.w));

  return r;
}

__global__ void resizeFastBicubic4(float4 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale)
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
    {
      cF = paddedRegisteredImage[x + y * width];
      cF.w = 0;
    }
    else
    {
      cF = tex2D(tex, u, v);
      cF.w = 0;
    }

    cF.x = sqrtf(cF.x); cF.y = sqrtf(cF.y); cF.z = sqrtf(cF.z); cF.w = 0;
    outputFloat[i] = cF;
  }
}

__global__ void resizeFastBicubic1(float1 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale)
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
    {
      cF = paddedRegisteredImage[x + y * width];
      cF.w = 0;
    }
    else
    {
      cF = tex2D(tex, u, v);
      cF.w = 0;
    }

    outputFloat[i].x = sqrtf(0.2989f * cF.x + 0.5870f * cF.y + 0.1140f * cF.z);
  }
}
