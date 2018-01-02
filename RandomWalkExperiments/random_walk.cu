// This file defines a CUDA random walk microbenchmark, which traverses an
// array in random order. This will print the times for each *block* in each
// kernel invocation. Specify the -zc command-line argument to use zero-copy
// memory, and -mm for managed memory.
//
// Usage: ./random_walk [-zc|-mm]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

// The seed to use for shuffling the array.
#define RANDOM_SEED (1337)

// The length, in number of 32-bit integers, of the array to walk
#define ARRAY_LENGTH (1024 * 1024 * 64)

// The number of steps in the random walk will equal these two values
// multiplied together.
#define STEPS_MAJOR (1)
#define STEPS_MINOR (ARRAY_LENGTH / 262)

// The total number of kernel invocations (measurement iterations) to perform.
#define KERNEL_INVOCATIONS (10000)

// These microbenchmarks will use 2 blocks of 32 threads.
#define THREAD_COUNT (32)
#define BLOCK_COUNT (2)

// A macro which exits the program with an error message if the given value is
// not equal to cudaSuccess.
#define CheckError(val) CheckCUDAError( (val), #val, __FILE__, __LINE__ )

static void CheckCUDAError(cudaError_t value, const char *function,
  const char *filename, int line) {
  if (value == cudaSuccess) return;
  printf("Cuda error %d. File %s, line %d: %s\n", (int) value, filename, line,
    function);
  exit(1);
}

// Holds variables and pointers that are passed between the phases of the
// experiment. The times arrays hold the start and end time stamps for each
// block, in the order [block1_start, block1_end, block2_start, ...].
typedef struct {
  uint64_t *device_times;
  uint64_t *host_times;
  uint32_t *host_array;
  uint32_t *device_array;
  uint8_t *host_outputs;
  uint8_t *device_outputs;
  cudaStream_t stream;
  // This will be nonzero if we're using zero-copy memory.
  uint8_t zero_copy;
  // This will be nonzero if we're using managed memory.
  uint8_t managed_memory;
} WalkState;

// Converts a 64-bit count of nanoseconds to a floating-point number of
// seconds.
static double ConvertToSeconds(uint64_t nanoseconds) {
  return ((double) nanoseconds) / 1e9;
}

// Returns the value of CUDA's global nanosecond timer.
static __device__ __inline__ uint64_t GlobalTimer64(void) {
  uint64_t to_return;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(to_return));
  return to_return;
}

__global__ void DoRandomWalk(uint32_t *array, uint8_t *outputs,
  uint64_t *times) {
  int i, j, current, thread_number;
  uint32_t sum;
  if (threadIdx.x == 0) {
    times[blockIdx.x * 2] = GlobalTimer64();
  }
  __syncthreads();
  thread_number = (blockIdx.x * THREAD_COUNT) + threadIdx.x;
  current = thread_number;
  for (i = 0; i < STEPS_MAJOR; i++) {
    for (j = 0; j < STEPS_MINOR; j++) {
      sum += current;
      current = array[current];
    }
  }
  outputs[thread_number] = (uint8_t) sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    times[(blockIdx.x * 2) + 1] = GlobalTimer64();
  }
}

// Returns a random 31-bit integer in the range [0, limit).
static inline uint32_t Rand32(uint32_t limit) {
  return lrand48() % limit;
}

// Takes an array and randomly shuffles its contents. The length parameter is
// the number of elements in the array. Won't work properly for arrays
// containing over 2^32 elements.
static void ShuffleArray(uint32_t *array, size_t length) {
  size_t i, j;
  uint32_t tmp;
  if (length <= 1) return;
  for (i = 0; i < length; i++) {
    j = i + Rand32(length - i);
    tmp = array[j];
    array[j] = array[i];
    array[i] = tmp;
  }
}

// Selects and initializes the device to run the benchmarks on.
void Initialize(int sync_level) {
  cudaError_t error = cudaErrorInvalidValue;
  switch (sync_level) {
  case 0:
    error = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    break;
  case 1:
    error = cudaSetDeviceFlags(cudaDeviceScheduleYield);
    break;
  case 2:
    error = cudaSetDeviceFlags(cudaDeviceBlockingSync);
    break;
  default:
    printf("Invalid sync level: %d\n", sync_level);
    exit(1);
  }
  CheckError(error);
  CheckError(cudaSetDevice(0));
}

// Allocates and initializes all values (including the stream), in the state
// struct. Requires the name of a file containing the pre-generated random walk
// array.
void AllocateMemory(WalkState *state) {
  uint32_t i = 0;
  size_t array_size = ARRAY_LENGTH * sizeof(uint32_t);
  size_t output_array_size = THREAD_COUNT * BLOCK_COUNT * sizeof(uint8_t);
  size_t times_array_size = BLOCK_COUNT * 2 * sizeof(uint64_t);
  if (state->managed_memory) {
    CheckError(cudaMallocManaged(&state->host_times, times_array_size));
    CheckError(cudaMallocManaged(&state->host_array, array_size));
    CheckError(cudaMallocManaged(&state->host_outputs, output_array_size));
  } else {
    CheckError(cudaMallocHost(&state->host_times, times_array_size));
    CheckError(cudaMallocHost(&state->host_array, array_size));
    CheckError(cudaMallocHost(&state->host_outputs, output_array_size));
    if (!state->zero_copy) {
      // Should we use zero-copy for the times array? For now, we will.
      CheckError(cudaMalloc(&state->device_times, times_array_size));
      CheckError(cudaMalloc(&state->device_array, array_size));
      CheckError(cudaMalloc(&state->device_outputs, output_array_size));
    } else {
      CheckError(cudaHostGetDevicePointer(&state->device_times,
        state->host_times, 0));
      CheckError(cudaHostGetDevicePointer(&state->device_array,
        state->host_array, 0));
      CheckError(cudaHostGetDevicePointer(&state->device_outputs,
        state->host_outputs, 0));
    }
  }
  CheckError(cudaStreamCreate(&state->stream));
  //printf("Generating random walk array... ");
  //fflush(stdout);
  for (i = 0; i < ARRAY_LENGTH; i++) {
    state->host_array[i] = i;
  }
  ShuffleArray(state->host_array, ARRAY_LENGTH);
  //printf("done!\n");
}

// Copies input arrays to the device. For this particular benchmark, this only
// needs to be called once, since the input array is only read, and the output
// array is always completely overwritten.
void CopyIn(WalkState *state) {
  if (state->zero_copy || state->managed_memory) return;
  size_t array_size = ARRAY_LENGTH * sizeof(uint32_t);
  CheckError(cudaMemcpyAsync(state->device_array, state->host_array,
    array_size, cudaMemcpyHostToDevice, state->stream));
  CheckError(cudaStreamSynchronize(state->stream));
}

// Copies the output array from the device. Should be called after every
// iteration, so that times can be recorded.
void CopyOut(WalkState *state) {
  if (state->zero_copy || state->managed_memory) return;
  size_t output_array_size = THREAD_COUNT * BLOCK_COUNT * sizeof(uint8_t);
  size_t times_array_size = BLOCK_COUNT * 2 * sizeof(uint64_t);
  CheckError(cudaMemcpyAsync(state->host_outputs, state->device_outputs,
    output_array_size, cudaMemcpyDeviceToHost, state->stream));
  CheckError(cudaMemcpyAsync(state->host_times, state->device_times,
    times_array_size, cudaMemcpyDeviceToHost, state->stream));
  CheckError(cudaStreamSynchronize(state->stream));
}

// Frees memory and closes the device stream. This will also reset the
// zero_copy field to 0.
void FreeMemory(WalkState *state) {
  CheckError(cudaStreamSynchronize(state->stream));
  CheckError(cudaStreamDestroy(state->stream));
  if (!state->zero_copy && !state->managed_memory) {
    CheckError(cudaFree(state->device_array));
    CheckError(cudaFree(state->device_outputs));
    CheckError(cudaFree(state->device_times));
  }
  if (state->managed_memory) {
    CheckError(cudaFree(state->host_array));
    CheckError(cudaFree(state->host_outputs));
    CheckError(cudaFree(state->host_times));
  } else {
    CheckError(cudaFreeHost(state->host_array));
    CheckError(cudaFreeHost(state->host_outputs));
    CheckError(cudaFreeHost(state->host_times));
  }
  memset(state, 0, sizeof(*state));
}

// Checks command-line arguments and sets members of the state struct if any
// are affected. May exit the program if any arguments are invalid.
static void ParseArgs(int argc, char **argv, WalkState *state) {
  int i;
  state->zero_copy = 0;
  if (argc == 1) return;
  if (argc != 2) {
    printf("Usage: %s [-zc]\n"
      "  Specify -zc to use zero-copy memory.\n", argv[0]);
    exit(1);
  }
  for (i = 1; i < argc; i++) {
    if (strncmp(argv[i], "-zc", 3) == 0) {
      state->zero_copy = 1;
      continue;
    }
    if (strncmp(argv[i], "-mm", 3) == 0) {
      state->managed_memory = 1;
      continue;
    }
    printf("Unknown argument: %s\n", argv[i]);
    exit(1);
  }
}

int main(int argc, char **argv) {
  int i, j;
  double block_start, block_end;
  WalkState state;
  srand48(RANDOM_SEED);
  ParseArgs(argc, argv, &state);
  // Initialize and allocate memory, then lock pages.
  Initialize(2);
  AllocateMemory(&state);
  if (!mlockall(MCL_CURRENT | MCL_FUTURE)) {
    printf("Error: failed locking pages in memory\n");
    return 1;
  }
  dim3 threads_per_block(THREAD_COUNT);
  dim3 block_count(BLOCK_COUNT);
  // We only need to copy in one time; the input array doesn't change.
  CopyIn(&state);
  for (i = 0; i < KERNEL_INVOCATIONS; i++) {
    if (state.managed_memory) {
      DoRandomWalk<<<block_count, threads_per_block, 0, state.stream>>>(
        state.host_array, state.host_outputs, state.host_times);
      CheckError(cudaDeviceSynchronize());
    } else {
      DoRandomWalk<<<block_count, threads_per_block, 0, state.stream>>>(
        state.device_array, state.device_outputs, state.device_times);
    }
    CheckError(cudaStreamSynchronize(state.stream));
    CopyOut(&state);
    for (j = 0; j < BLOCK_COUNT; j++) {
      block_start = ConvertToSeconds(state.host_times[j * 2]);
      block_end = ConvertToSeconds(state.host_times[(j * 2) + 1]);
      printf("Block %d: start: %f end: %f elapsed: %f\n", j, block_start, block_end, block_end - block_start);
    }
  }
  FreeMemory(&state);
  CheckError(cudaDeviceReset());
  return 0;
}
