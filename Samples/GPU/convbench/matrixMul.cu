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

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>

#define TOTAL_SHARED 25600
#define BLOCK_SIZE_WIDTH 16
#define EXTRA_SHARED_MEM_USAGE ((TOTAL_SHARED - BLOCK_SIZE_WIDTH*BLOCK_SIZE_WIDTH*8) / 4)

// assuming all gpus are the same
float occupancyCalc(const void* func, int block_size) {
	struct cudaFuncAttributes attr;
	cudaError_t error;
	error = cudaFuncGetAttributes(&attr, func);
	if (error != cudaSuccess)
		printf("cudaFuncGetAttributes returned error code %d, line(%d)\n", error, __LINE__);

	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	error = cudaGetDeviceProperties(&deviceProp, 0);
	if (error != cudaSuccess)
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);

	/* prepare some configuration variables */
	int maxWarpsPerSM = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
	//printf("max warps per sm: %d\n", maxWarpsPerSM);
	/* variable starting by b means related to block size */
	//printf("block_size: %d\n", block_size);
	int b_warpsPerBlock = block_size / deviceProp.warpSize + ((block_size % deviceProp.warpSize > 0) ? 1 : 0);
	//printf("warps per block: %d, block_size/deviceProp.warpSize: %d\n", b_warpsPerBlock, block_size/deviceProp.warpSize);
	int b_blocksPerSM = maxWarpsPerSM / b_warpsPerBlock;

	/* variable starting by r means related to register file */
	int r_warpsPerBlock = b_warpsPerBlock;
	if (attr.numRegs > deviceProp.regsPerBlock) {
		printf("Register usage is larger than %d limit per block.", deviceProp.regsPerBlock);
		return 0;
	}

	int r_warpsPerSM = deviceProp.regsPerMultiprocessor / (deviceProp.warpSize * attr.numRegs);
	int r_blocksPerSM = r_warpsPerSM / r_warpsPerBlock;

	/* variable starting by s means related to shared memory */
	int s_blocksPerSM = deviceProp.sharedMemPerMultiprocessor / attr.sharedSizeBytes;

	int numBlocks = (b_blocksPerSM < r_blocksPerSM) ? b_blocksPerSM : r_blocksPerSM;
	numBlocks = (numBlocks < s_blocksPerSM) ? numBlocks : s_blocksPerSM;

	//printf("b: %d, r: %d, s: %d\n", b_blocksPerSM, r_blocksPerSM, s_blocksPerSM);
	int numWarps = numBlocks * b_warpsPerBlock;
	printf("%d,%d,%d,%d,", attr.numRegs, (int) (attr.sharedSizeBytes), numBlocks, numWarps);
	float occupancy = (float)numWarps / maxWarpsPerSM;

	return occupancy;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
#define BLOCK_SIZE 32
extern "C" __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
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
	for (int a = aBegin, b = bBegin;
	     a <= aEnd;
	     a += aStep, b += bStep)
	{

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

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
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
