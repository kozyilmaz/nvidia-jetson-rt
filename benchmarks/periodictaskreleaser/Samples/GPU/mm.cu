/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <sys/mman.h>

// CUDA runtime
#include <cuda_runtime.h>

extern "C" {
#include "../gpusync.h"
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
  template <int BLOCK_SIZE> __global__ void
      matrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

// Stream for the thread's GPU Operations
cudaStream_t stream;

float *hA, *hB, *hC;
float *dA, *dB, *dC;
unsigned int mem_size;
int matrix_size;

dim3 dimsA;
dim3 dimsB;
dim3 threads;
dim3 grid;

void* Initialize(GPUParameters *parameters) {
  /*
   * The sync_level parameter is an integer that indicates the desired level of
   * synchronization used by the GPU driver (values defined below).  The
   * specified level is used in cudaSetDeviceFlags() to set the level
   * prior to initialization.
   */
  switch (parameters->sync_level) {
    case 0:
      cudaSetDeviceFlags(cudaDeviceScheduleSpin);
      break;
    case 1:
      cudaSetDeviceFlags(cudaDeviceScheduleYield);
      break;
    case 2:
      cudaSetDeviceFlags(cudaDeviceBlockingSync);
      break;
    default:
      fprintf(stderr, "Unknown sync level: %d\n", parameters->sync_level);
      break;
  }
  matrix_size = sqrt(parameters->element_count) * sqrt(
    parameters->element_count);
  if (parameters->cuda_device >= 0) {
    if (cudaSetDevice(parameters->cuda_device) != cudaSuccess) {
      printf("Failed setting CUDA device.\n");
      exit(1);
    }
  }
  cudaStreamCreate(&stream);
  return NULL;
}

void MallocCPU(void *thread_data) {
  // Allocate host memory for matrices A and B
  mem_size = sizeof(float) * matrix_size;
  cudaError_t err = cudaMallocHost((void **) &hA, mem_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host memory A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMallocHost((void **) &hB, mem_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host memory B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // Allocate host matrix C
  err = cudaMallocHost((void **) &hC, mem_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host memory C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Initialize host memory
  constantInit(hA, matrix_size, 1.0f);
  constantInit(hB, matrix_size, 0.01f);

  // Setup execution parameters
  int block_size = 16;
  dimsA = dim3(5*2*block_size, 5*2*block_size, 1);
  dimsA.x = sqrt(matrix_size);
  dimsA.y = sqrt(matrix_size);
  dimsB = dim3(5*4*block_size, 5*2*block_size, 1);
  dimsB.x = sqrt(matrix_size);
  dimsB.y = sqrt(matrix_size);
  threads = dim3(block_size, block_size);
  grid = dim3(ceil(dimsB.x / (float) threads.x), ceil(dimsA.y / (float) threads.y));
}

void MallocGPU(void *thread_data) {
  // Allocate device memory
  cudaError_t err = cudaMalloc((void **) &dA, mem_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **) &dB, mem_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **) &dC, mem_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

void CopyIn(void *thread_data) {
  // copy the A and B blocks from Host to Device memory
  // these calls are asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(dA, hA, mem_size, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory A from host to device (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpyAsync(dB, hB, mem_size, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory B from host to device (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);
}

void Exec(void *thread_data) {
  cudaError_t err = cudaSuccess;
  matrixMulCUDA<16><<< grid, threads, 0, stream>>>(dC, dA, dB, dimsA.x, dimsB.x);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream after kernel execution
  // the wrapper for this function releases any lock held (EE here)
  cudaStreamSynchronize(stream);
}

void CopyOut(void *thread_data) {
  // copy the result memory from Device to Host memory
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(hC, dC, mem_size, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory C from device to host (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);
}

void FreeGPU(void *thread_data) {
  // Free device global memory for inputs A and B and result C
  cudaError_t err = cudaFree(dA);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  err = cudaFree(dB);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  err = cudaFree(dC);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

void FreeCPU(void *thread_data) {
  // Free host memory that was pinned
  cudaFreeHost(hA);
  cudaFreeHost(hB);
  cudaFreeHost(hC);
}

void Finish(void *thread_data) {
  // clean up the user allocated stream
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  // Reset the device and return
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application returns
  cudaError_t err = cudaDeviceReset();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
  }
}
