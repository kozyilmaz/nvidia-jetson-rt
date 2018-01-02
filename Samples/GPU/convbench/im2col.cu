#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "im2col.h"


__global__ void im2col_gpu_kernel(const int n, const float* data_im,
				  const int height, const int width, const int kernel_h, const int kernel_w,
				  const int pad_h, const int pad_w,
				  const int stride_h, const int stride_w,
				  const int height_col, const int width_col,
				  float* data_col) {
	CUDA_KERNEL_LOOP(index, n) {
		const int h_index = index / width_col;
		const int h_col = h_index % height_col;
		const int w_col = index % width_col;
		const int c_im = h_index / height_col;
		const int c_col = c_im * kernel_h * kernel_w;
		const int h_offset = h_col * stride_h - pad_h;
		const int w_offset = w_col * stride_w - pad_w;
		float* data_col_ptr = data_col;
		data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
		const float* data_im_ptr = data_im;
		data_im_ptr += (c_im * height + h_offset) * width + w_offset;
		for (int i = 0; i < kernel_h; ++i) {
			for (int j = 0; j < kernel_w; ++j) {
				int h_im = h_offset + i;
				int w_im = w_offset + j;
				*data_col_ptr =
					(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
					data_im_ptr[i * width + j] : 0;
				data_col_ptr += height_col * width_col;
			}
		}
	}
}

extern "C" void im2col_gpu(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		float* data_col, cudaStream_t stream) {
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int num_kernels = channels * height_col * width_col;
	// NOLINT_NEXT_LINE(whitespace/operators)
	im2col_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
					  num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
					  pad_w, stride_h, stride_w, height_col,
					  width_col, data_col);
	checkCudaErrors(cudaStreamSynchronize(stream));
	CUDA_POST_KERNEL_CHECK;
}
