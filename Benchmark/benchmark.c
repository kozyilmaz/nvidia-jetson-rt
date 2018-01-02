#define _GNU_SOURCE
#include <argp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "gpusync.h"

// 30 minutes
#define DEFAULT_EXPERIMENT_DURATION (1800)
#define DEFAULT_ITERATION_COUNT (0x7fffffff)
#define DEFAULT_DATA_SIZE 1024
#define DEFAULT_SYNC (2)
#define DEFAULT_RAND_SLEEP (1)
#define FIFTEEN_MS_IN_NS (15000000)
#define DEFAULT_CUDA_DEVICE (-1)

const char *argp_program_version = "v1";
const char *argp_program_bug_address = "<otternes@cs.unc.edu>";
static char doc[] = "GPU Sample Program Benchmarking.";
static char args_doc[] = "";
static struct argp_option options[] = {
  {0, 0, 0, 0, "Experiment configuration parameters:"},
  {"size", 's', "data_size", 0, "Specifies the size of input data to the task. Some tasks may disregard this value."},
  {"sync", 'y', "{0|1}", 0, "Specifies how the CPU should synchronize with the GPU kernel. {0: spin, 1: yield, default: block}."},
  {"randsleep", 'r', 0, OPTION_ARG_OPTIONAL, "Specifies that the program should sleep for a random amount of time between 0-15ms after each iteration."},
  {0, 0, 0, 0, "Experiment duration specifiers. If both are used, whichever limit is reached first will terminate the experiment."},
  {"iterations", 'n', "iteration_count", 0, "Specifies the maximum number of iterations of the benchmark program. Defaults to infinity."},
  {"duration", 'd', "experiment_duration", 0, "Specifies the duration the experiment should run in seconds. Defaults to 30 minutes."},
  {"show_blocks", 'b', 0, OPTION_ARG_OPTIONAL, "If provided, the benchmark will emit a list of individual block times during the CopyOut phase."},
  {"device", 'g', "CUDA device number", 0, "If provided, the benchmark will explicitly try to use the device with the given ID."},
  {0},
};

struct arguments {
  int data_size;
  uint64_t experiment_duration;
  uint32_t iteration_count;
  int show_block_times;
  int sync;
  int randsleep;
  int cuda_device;
};

// Returns the current system time in seconds. Exits if an error occurs while
// getting the time.
static double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
}

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = state->input;
  int iterations, duration;
  switch (key) {
  case 'd':
    duration = atoi(arg);
    if (duration < 0) {
      return EINVAL;
    }
    arguments->experiment_duration = (uint64_t) duration;
    break;
  case 'n':
    iterations = atoi(arg);
    if (iterations < 0) {
      return EINVAL;
    }
    arguments->iteration_count = iterations;
    break;
  case 'r':
    arguments->randsleep = 1;
    break;
  case 's':
    arguments->data_size = atoi(arg);
    if (arguments->data_size < 0) {
      return EINVAL;
    }
    break;
  case 'y':
    arguments->sync = atoi(arg);
    if (arguments->sync < 0 || arguments->sync > 2) {
      return EINVAL;
    }
    break;
  case 'b':
    arguments->show_block_times = 1;
    break;
  case 'g':
    arguments->cuda_device = atoi(arg);
    if (arguments->cuda_device < -1) {
      return EINVAL;
    }
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char **argv) {
  struct arguments arguments;
  GPUParameters benchmark_parameters;
  double experiment_start, iteration_start, exec_start, copy_out_start,
    iteration_end;
  int i;
  void *thread_data;
  struct timespec delay;
  delay.tv_sec = 0;
  arguments.data_size = DEFAULT_DATA_SIZE;
  arguments.experiment_duration = ((uint64_t) DEFAULT_EXPERIMENT_DURATION);
  arguments.iteration_count = (uint64_t) DEFAULT_ITERATION_COUNT;
  arguments.sync = DEFAULT_SYNC;
  arguments.randsleep = DEFAULT_RAND_SLEEP;
  arguments.cuda_device = DEFAULT_CUDA_DEVICE;
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  // Copy the command-line arguments to the parameters to actually pass the
  // benchmark.
  memset(&benchmark_parameters, 0, sizeof(benchmark_parameters));
  benchmark_parameters.sync_level = arguments.sync;
  benchmark_parameters.element_count = arguments.data_size;
  benchmark_parameters.show_block_times = arguments.show_block_times;
  benchmark_parameters.cuda_device = arguments.cuda_device;

  // Do initialization and allocation outside of the main loop.
  thread_data = Initialize(&benchmark_parameters);
  if (!thread_data) {
    printf("Benchmark does not support multithreading.\n");
  }
  MallocCPU(thread_data);
  MallocGPU(thread_data);
  if (!mlockall(MCL_CURRENT | MCL_FUTURE)) {
    printf("Error: failed locking pages in memory.\n");
    exit(1);
  }

  // Iterate until one of the exit conditions is met.
  printf("Program %s, PID %d\n", argv[0], (int) getpid());
  printf("Copy in start, Kernel start, Copy out start, Copy out end (s)\n");
  experiment_start = CurrentSeconds();
  iteration_end = experiment_start;
  for (i = 0; i < arguments.iteration_count; i++) {
    if ((iteration_end - experiment_start) >
      (double) arguments.experiment_duration) {
      break;
    }
    iteration_start = CurrentSeconds();
    CopyIn(thread_data);
    exec_start = CurrentSeconds();
    Exec(thread_data);
    copy_out_start = CurrentSeconds();
    CopyOut(thread_data);
    iteration_end = CurrentSeconds();
    printf("%f, %f, %f, %f\n", iteration_start, exec_start, copy_out_start,
      iteration_end);
    if (arguments.randsleep) {
      delay.tv_nsec = arguments.randsleep * (rand() % FIFTEEN_MS_IN_NS);
      nanosleep(&delay, NULL);
    }
  }
  FreeGPU(thread_data);
  FreeCPU(thread_data);
  Finish(thread_data);
  exit(EXIT_SUCCESS);
}
