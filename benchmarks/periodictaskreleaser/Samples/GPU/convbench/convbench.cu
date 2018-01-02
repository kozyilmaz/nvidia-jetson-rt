#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "im2col.h"
#include "matrixMul.h"
extern "C" {
#include "../../gpusync.h"
}

typedef struct {
  // Current number of channels in image and filter.
  int channels;
  // Height of image
  int height;
  // Width of image
  int width;
  // Height of filter
  int kernel_h;
  // Width of filter
  int kernel_w;
  // Padding size at top and bottom
  int pad_h;
  // Padding size at left and right, always equal to pad_h
  int pad_w;
  // Stride of moving filter at horizontal direction
  int stride_h;
  // at vertical direction
  int stride_w; // at vertical direction
  // Number of filters used, equal to the number of channels for the next layer
  int num_filters;
} im2col_info;

// 4 layers of convolution
im2col_info info[4] = {
  // 206,116 image size
  {4, 227, 227, 11, 11, 0, 0, 4, 4, 96},
  // 4,946,784
  {96, 27, 27, 5, 5, 2, 2, 1, 1, 256},
  // 43,264
  {256, 13, 13, 3, 3, 1, 1, 1, 1, 384},
  // 64,896
  {384, 13, 13, 3, 3, 1, 1, 1, 1, 128},
};
float *h_image, *h_filter, *h_result;
float *d_image, *d_filter, *d_image_col, *d_result;
int max_image_size, max_filter_size, max_image_col_size, max_result_size;
cudaStream_t stream;

/* generate random float numbers for the array of size @size. */
void random_float(float *ptr, int size) {
  srand((int) time(NULL));
  for (int i = 0; i < size; i++) {
    ptr[i] = (float)(rand() % 255);
  }
}

void* Initialize(GPUParameters *parameters) {
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
    printf("Unknown sync level: %d\n", parameters->sync_level);
    break;
  }
  if (parameters->cuda_device >= 0) {
    checkCudaErrors(cudaSetDevice(parameters->cuda_device));
  }
  checkCudaErrors(cudaStreamCreate(&stream));
  max_image_size = 0;
  // A(M*K)
  max_filter_size = 0;
  // B(K*N)
  max_image_col_size = 0;
  // C(M*N) = A*B
  max_result_size = 0;
  for (int i = 0; i < sizeof(info) / sizeof(im2col_info); i++) {
    im2col_info curr = info[i];
    int tmp = curr.channels * curr.height * curr.width;
    if (tmp > max_image_size) max_image_size = tmp;
    // A(M*K): filter matrix
    // M: each row represents one filter for one channel.
    // K: equals the size of filter matrix,
    // e.g., 11(width) * 11(height) * 4(channels,RGBD)
    //  -> filter matrix has size 484.
    // B(K*N): image matrix
    // K: still filter matrix size
    // N: Number of values produced. For 227*227*4 image, 0 padding, 4 stride,
    //    11*11*4 filter, (227-11)/4+1 = 55 rows(columns). So N = 55*55 = 3025.
    int M = curr.num_filters;
    int K = curr.kernel_h * curr.kernel_w * curr.channels;
    int N = (curr.height + 2 * curr.pad_h - curr.kernel_w) / curr.stride_h +1;
    N *= N;
    if (M * K > max_filter_size) max_filter_size = M * K;
    if (K * N > max_image_col_size) max_image_col_size = K * N;
    if (M * N > max_result_size) max_result_size = M * N;
  }
  return NULL;
}

void MallocCPU(void *thread_data) {
  checkCudaErrors(cudaMallocHost(&h_image, max_image_size * sizeof(float)));
  checkCudaErrors(cudaMallocHost(&h_filter, max_filter_size * sizeof(float)));
  checkCudaErrors(cudaMallocHost(&h_result, max_result_size * sizeof(float)));
  random_float(h_image, max_image_size);
  random_float(h_filter, max_filter_size);
}

void MallocGPU(void *thread_data) {
  checkCudaErrors(cudaMalloc(&d_image, max_image_size * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_filter, max_filter_size * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_image_col, max_image_col_size *
    sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_result, max_result_size * sizeof(float)));
}

void CopyIn(void *thread_data) {
  checkCudaErrors(cudaMemcpyAsync(d_image, h_image, max_image_size *
    sizeof(float), cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_filter, h_filter, max_filter_size *
    sizeof(float), cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
}

void Exec(void *thread_data) {
  for (int i = 0; i < sizeof(info) / sizeof(im2col_info); i++) {
    im2col_info curr = info[i];
    int M = curr.num_filters;
    int K = curr.kernel_h * curr.kernel_w * curr.channels;
    int N = (curr.height + 2 * curr.pad_h - curr.kernel_w) / curr.stride_h + 1;
    N *= N;
    im2col_gpu(d_image, curr.channels, curr.height, curr.width, curr.kernel_h,
      curr.kernel_w, curr.pad_h, curr.pad_w, curr.stride_h, curr.stride_w,
      d_image_col, stream);
    dim3 threads(32, 32);
    dim3 grid((int) ((N - 1) / threads.x) + 1,
      (int) ((M - 1) / threads.y) + 1);
    matrixMulCUDA<<<grid, threads, 0, stream>>>(d_result, d_filter,
      d_image_col, K, N); // can refer to sample code
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
}

void CopyOut(void *thread_data) {
  checkCudaErrors(cudaMemcpyAsync(h_result, d_result, max_result_size *
    sizeof(float), cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
}

void FreeGPU(void *thread_data) {
  cudaFree(d_image);
  d_image = NULL;
  cudaFree(d_image_col);
  d_image_col = NULL;
  cudaFree(d_filter);
  d_filter = NULL;
  cudaFree(d_result);
  d_result = NULL;
}

void FreeCPU(void *thread_data) {
  cudaFreeHost(h_image);
  h_image = NULL;
  cudaFreeHost(h_filter);
  h_filter = NULL;
  cudaFreeHost(h_result);
  h_result = NULL;
}

void Finish(void *thread_data) {
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  checkCudaErrors(cudaDeviceReset());
}
