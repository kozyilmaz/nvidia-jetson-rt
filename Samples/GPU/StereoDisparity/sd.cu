/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A CUDA program that demonstrates how to compute a stereo disparity map using
 *   SIMD SAD (Sum of Absolute Difference) intrinsics
 */

/*
 * The program's performance is dominated by 
 * the computation on the execution engine (EE) while memory copies 
 * between Host and Device using the copy engine (CE) are significantly
 * less time consuming.
 *
 * This version uses a user allocated stream and asynchronous memory
 * copy operations (cudaMemcpyAsync()).  Cuda kernel invocations on the
 * stream are also asynchronous.  cudaStreamSynchronize() is used to 
 * synchronize with both the copy and kernel executions.  Host pinned
 * memory is not used because the copy operations are not a significant 
 * element of performance.
 *
 * The program depends on two input files containing the image 
 * representations for the left and right stereo images 
 * (stereo.im0.640x533.ppm and stereo.im1.640x533.ppm)
 * which must be in the directory with the executable.
 *
 */

#include <errno.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
extern "C" {
#include "gpusync.h"
}
#include "sd_kernel.cuh"

// Relative path to images
static const char fname0[] = "../Samples/GPU/StereoDisparity/data/stereo.im0.640x533.ppm";
static const char fname1[] = "../Samples/GPU/StereoDisparity/data/stereo.im1.640x533.ppm";

// Holds per-thread state for this algorithm.
typedef struct {
  cudaStream_t stream;
  // Host Memory
  unsigned int *h_odata;
  unsigned char *h_img0;
  unsigned char *h_img1;
  uint64_t *h_block_times;
  // Device memory
  unsigned int *d_odata;
  unsigned int *d_img0;
  unsigned int *d_img1;
  uint64_t *d_block_times;
  // Kernel execution parameters
  unsigned int w, h;
  size_t offset;
  dim3 numThreads;
  dim3 numBlocks;
  size_t block_times_size;
  unsigned int numData;
  unsigned int memSize;
  cudaChannelFormatDesc ca_desc0;
  cudaChannelFormatDesc ca_desc1;
  // Search parameters
  int minDisp;
  int maxDisp;
  // Set to 1 if block times should be printed during CopyOut()
  int show_block_times;
} ThreadContext;

// Used for work-in-progress migration of this task to one that doesn't rely on
// global state.
ThreadContext *g;

int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// Override helper_image.h
inline bool loadPPM4ub(const char *file, unsigned char **data,
  unsigned int *w, unsigned int *h) {
  unsigned char *idata = 0;
  unsigned int channels;

  if (!__loadPPM(file, &idata, w, h, &channels)) {
    free(idata);
    return false;
  }
  // pad 4th component
  int size = *w * *h;
  // keep the original pointer
  unsigned char *idata_orig = idata;
  checkCudaErrors(cudaMallocHost(data, sizeof(unsigned char) * size * 4));
  unsigned char *ptr = *data;
  for (int i = 0; i < size; i++) {
    *ptr++ = *idata++;
    *ptr++ = *idata++;
    *ptr++ = *idata++;
    *ptr++ = 0;
  }
  free(idata_orig);
  return true;
}

// Converts a 64-bit count of nanoseconds to a floating-point number of
// seconds.
static double ConvertToSeconds(uint64_t nanoseconds) {
  return ((double) nanoseconds) / 1e9;
}

void* Initialize(GPUParameters *parameters) {
  g = (ThreadContext*) malloc(sizeof(ThreadContext));
  if (!g) {
    printf("Failed to allocate Thread Context.\n");
    exit(1);
  }
  g->minDisp = -16;
  g->maxDisp = 0;
  g->show_block_times = parameters->show_block_times;
  // Pin code
  if(!mlockall(MCL_CURRENT | MCL_FUTURE)) {
    fprintf(stderr, "Failed to lock code pages.\n");
    exit(EXIT_FAILURE);
  }
  if (parameters->cuda_device >= 0) {
    checkCudaErrors(cudaSetDevice(parameters->cuda_device));
  }
  checkCudaErrors(cudaStreamCreate(&(g->stream)));
  return NULL;
}


void MallocCPU(void *thread_data) {
  // Load image data
  // functions allocate memory for the images on host side
  // initialize pointers to NULL to request lib call to allocate as needed
  // PPM images are loaded into 4 byte/pixel memory (RGBX)
  g->h_img0 = NULL;
  g->h_img1 = NULL;
  if (!loadPPM4ub(fname0, &(g->h_img0), &(g->w), &(g->h))) {
    fprintf(stderr, "Failed to load <%s>\n", fname0);
    exit(-1);
  }
  if (!loadPPM4ub(fname1, &(g->h_img1), &(g->w), &(g->h))) {
    fprintf(stderr, "Failed to load <%s>\n", fname1);
    exit(-1);
  }
  // set up parameters used in the rest of program
  g->numThreads = dim3(blockSize_x, blockSize_y, 1);
  g->numBlocks = dim3(iDivUp(g->w, g->numThreads.x), iDivUp(g->h,
    g->numThreads.y));
  printf("Number of blocks: (%dx%d)\n", (int) g->numBlocks.x,
    (int) g->numBlocks.y);
  g->numData = g->w * g->h;
  g->memSize = sizeof(int) * g->numData;
  // We hold space for a start and end time of each block.
  g->block_times_size = g->numBlocks.x * g->numBlocks.y * 2 * sizeof(uint64_t);

  checkCudaErrors(cudaMallocHost(&(g->h_block_times), g->block_times_size));

  // allocate memory for the result on host side
  checkCudaErrors(cudaMallocHost(&(g->h_odata), g->memSize));

  // more setup for using the GPU
  g->offset = 0;
  g->ca_desc0 = cudaCreateChannelDesc<unsigned int>();
  g->ca_desc1 = cudaCreateChannelDesc<unsigned int>();

  tex2Dleft.addressMode[0] = cudaAddressModeClamp;
  tex2Dleft.addressMode[1] = cudaAddressModeClamp;
  tex2Dleft.filterMode     = cudaFilterModePoint;
  tex2Dleft.normalized     = false;
  tex2Dright.addressMode[0] = cudaAddressModeClamp;
  tex2Dright.addressMode[1] = cudaAddressModeClamp;
  tex2Dright.filterMode     = cudaFilterModePoint;
  tex2Dright.normalized     = false;
}


void MallocGPU(void *thread_data) {
  // allocate device memory for inputs and result
  checkCudaErrors(cudaMalloc(&(g->d_odata), g->memSize));
  checkCudaErrors(cudaMalloc(&(g->d_img0), g->memSize));
  checkCudaErrors(cudaMalloc(&(g->d_img1), g->memSize));
  checkCudaErrors(cudaMalloc(&(g->d_block_times), g->block_times_size));
  checkCudaErrors(cudaBindTexture2D(&(g->offset), tex2Dleft, g->d_img0,
    g->ca_desc0, g->w, g->h, g->w * 4));
  assert(g->offset == 0);
  checkCudaErrors(cudaBindTexture2D(&(g->offset), tex2Dright, g->d_img1,
    g->ca_desc1, g->w, g->h, g->w * 4));
  assert(g->offset == 0);
}

void CopyIn(void *thread_data) {
  // copy host memory with images to device
  checkCudaErrors(cudaMemcpyAsync(g->d_img0, g->h_img0, g->memSize,
    cudaMemcpyHostToDevice, g->stream));
  checkCudaErrors(cudaMemcpyAsync(g->d_img1, g->h_img1, g->memSize,
    cudaMemcpyHostToDevice, g->stream));
  // copy host memory that was set to zero to initialize device output
  checkCudaErrors(cudaMemcpyAsync(g->d_odata, g->h_odata, g->memSize,
    cudaMemcpyHostToDevice, g->stream));
  cudaStreamSynchronize(g->stream);
}

void Exec(void *thread_data) {
  stereoDisparityKernel<<<g->numBlocks, g->numThreads, 0, g->stream>>>(
    g->d_img0, g->d_img1, g->d_odata, g->w, g->h, g->minDisp, g->maxDisp,
    g->d_block_times);
  cudaStreamSynchronize(g->stream);
  getLastCudaError("Kernel execution failed");
}

void CopyOut(void *thread_data) {
  int total_blocks, i;
  double start, end;
  checkCudaErrors(cudaMemcpyAsync(g->h_odata, g->d_odata, g->memSize,
    cudaMemcpyDeviceToHost, g->stream));
  checkCudaErrors(cudaMemcpyAsync(g->h_block_times, g->d_block_times,
    g->block_times_size, cudaMemcpyDeviceToHost, g->stream));
  cudaStreamSynchronize(g->stream);

  if (g->show_block_times) {
    total_blocks = g->numBlocks.x * g->numBlocks.y;
    printf("Block times (s * 1e5): ");
    for (i = 0; i < total_blocks; i++) {
      start = ConvertToSeconds(g->h_block_times[i * 2]) * 1e5;
      end = ConvertToSeconds(g->h_block_times[i * 2 + 1]) * 1e5;
      printf("%.04f,%.04f ", start, end);
    }
    printf("\n");
  }
}

void FreeGPU(void *thread_data) {
  checkCudaErrors(cudaFree(g->d_odata));
  checkCudaErrors(cudaFree(g->d_img0));
  checkCudaErrors(cudaFree(g->d_img1));
  checkCudaErrors(cudaFree(g->d_block_times));
}

void FreeCPU(void *thread_data) {
  cudaFreeHost(g->h_odata);
  cudaFreeHost(g->h_img0);
  cudaFreeHost(g->h_img1);
  cudaFreeHost(g->h_block_times);
}

void Finish(void *thread_data) {
  cudaStreamSynchronize(g->stream);
  cudaStreamDestroy(g->stream);
  checkCudaErrors(cudaDeviceReset());
}
