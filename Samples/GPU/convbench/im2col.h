#ifndef __IM2COL_H__
#define __IM2COL_H__
// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
	/* Code block avoids redefinition of cudaError_t error */ \
	do { \
		cudaError_t error = condition; \
		if (error != cudaSuccess) \
			printf("%s\n", cudaGetErrorString(error)); \
	} while (0)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	     i < (n); \
	     i += blockDim.x * gridDim.x)

#ifdef __cplusplus
extern "C" void im2col_gpu(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, float* data_col, cudaStream_t stream);
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
#endif
