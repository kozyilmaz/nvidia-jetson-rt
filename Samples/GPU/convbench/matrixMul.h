#ifndef __MATRIX_MUL_H__
#define __MATRIX_MUL_H__
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" __global__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB);
#endif
#endif
