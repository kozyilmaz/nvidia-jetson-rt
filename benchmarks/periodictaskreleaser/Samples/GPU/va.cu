#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>

extern "C" {
#include "../gpusync.h"
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
    vectorAdd(const float *A, const float *B, float *C, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements)
  {
    C[i] = A[i] + B[i];
  }
}

typedef struct {
  float *hA, *hB, *hC;
  float *dA, *dB, *dC;
  int element_count;
  size_t vector_bytes;
  int v_threadsPerBlock;
  int v_blocksPerGrid;
  cudaStream_t stream;
} ThreadContext;

void* Initialize(GPUParameters *parameters) {
  ThreadContext *g = NULL;
  cudaError_t e;
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
  e = cudaMallocHost(&g, sizeof(ThreadContext));
  if (e != cudaSuccess) {
    printf("Failed allocating thread context: %d\n", (int) e);
    exit(1);
  }
  if (parameters->cuda_device >= 0) {
    if (cudaSetDevice(0) != cudaSuccess) {
      printf("Failed setting CUDA device.\n");
      exit(1);
    }
  }
  cudaStreamCreate(&(g->stream));
  g->element_count = parameters->element_count;
  g->vector_bytes = g->element_count * sizeof(float);
  return g;
}

void MallocCPU(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;

  // Host allocations in pinned memory
  // Allocate the host input vector A
  cudaError_t err = cudaMallocHost(&(g->hA), g->vector_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the host input vector B
  err = cudaMallocHost(&(g->hB), g->vector_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the host output vector C
  err = cudaMallocHost(&(g->hC), g->vector_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Initialize the host input vectors
  for (int i = 0; i < g->element_count; i++) {
    g->hA[i] = rand() / (float) RAND_MAX;
    g->hB[i] = rand() / (float) RAND_MAX;
  }
  g->v_threadsPerBlock = 256;
  g->v_blocksPerGrid = (g->element_count + g->v_threadsPerBlock - 1) / g->v_threadsPerBlock;
}

void MallocGPU(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  // Allocate the device input vector A
  cudaError_t err = cudaMalloc(&(g->dA), g->vector_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the device input vector B
  err = cudaMalloc(&(g->dB), g->vector_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the device output vector C
  err = cudaMalloc(&(g->dC), g->vector_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

void CopyIn(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  // copy the A and B vectors from Host to Device memory
  // these calls are asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(g->dA, g->hA, g->vector_bytes, cudaMemcpyHostToDevice, g->stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpyAsync(g->dB, g->hB, g->vector_bytes, cudaMemcpyHostToDevice, g->stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(g->stream);
}

void Exec(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  cudaError_t err = cudaSuccess;

  // Launch the Vector Add CUDA Kernel
  // lock of EE is handled in wrapper for cudaLaunch()
  vectorAdd<<<g->v_blocksPerGrid, g->v_threadsPerBlock, 0, g->stream>>>(g->dA, g->dB, g->dC, g->element_count);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream after kernel execution
  // the wrapper for this function releases any lock held (EE here)
  cudaStreamSynchronize(g->stream);
}

void CopyOut(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  // Copy the result vector from Device to Host memory
  // This call is asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(g->hC, g->dC, g->vector_bytes, cudaMemcpyDeviceToHost, g->stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(g->stream);
}

void FreeGPU(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  // Free device global memory for inputs A and B and result C
  cudaError_t err = cudaFree(g->dA);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  err = cudaFree(g->dB);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  err = cudaFree(g->dC);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

void FreeCPU(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  // Free device global memory for inputs A and B and result C
  // Free host memory that was pinned
  cudaFreeHost(g->hA);
  cudaFreeHost(g->hB);
  cudaFreeHost(g->hC);
}

void Finish(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  cudaStreamSynchronize(g->stream);
  cudaStreamDestroy(g->stream);
  cudaFreeHost(g);
}
