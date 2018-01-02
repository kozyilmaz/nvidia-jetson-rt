#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "../gpusync.h"

float *matrix_a, *matrix_b, *result_matrix;
unsigned int mem_size, matrix_size, a_width, b_width;
int element_count;

static void ConstantInit(float *data, int size, float val) {
  int i;
  for (i = 0; i < size; i++) {
    data[i] = val;
  }
}

// For debugging/verification.
static void PrintMatrix(float *m, int w, int h) {
  int row, col;
  for (row = 0; row < h; row++) {
    for (col = 0; col < w; col++) {
      printf("%.04f ", m[(row * w) + col]);
    }
    printf("\n");
  }
}

// Assumes a's height = b's width, etc. Sets matrix c = a * b.
static void Multiply(float *a, float *b, float *c, int a_width, int b_width) {
  float sum, a_value, b_value;
  int row, col, b_row;
  for (row = 0; row < b_width; row++) {
    for (col = 0; col < a_width; col++) {
      sum = 0;
      for (b_row = 0; b_row < a_width; b_row++) {
        a_value = a[(a_width * row) + b_row];
        b_value = b[(b_width * b_row) + col];
        sum += a_value * b_value;
      }
      c[(a_width * row) + col] = sum;
    }
  }
}

void* Initialize(GPUParameters *parameters) {
  element_count = parameters->element_count;
  return NULL;
}

void MallocCPU(void *unused) {
  a_width = sqrt(element_count);
  b_width = sqrt(element_count);
  matrix_size = a_width * a_width;
  mem_size = sizeof(float) * matrix_size;
  matrix_a = (float *) malloc(mem_size);
  if (!matrix_a) {
    printf("Failed allocating matrix A.\n");
    exit(1);
  }
  matrix_b = (float *) malloc(mem_size);
  if (!matrix_b) {
    printf("Failed allocating matrix B.\n");
    exit(1);
  }
  result_matrix = (float *) malloc(mem_size);
  if (!result_matrix) {
    printf("Failed allocating result matrix.\n");
    exit(1);
  }
}

void MallocGPU(void *unused) {
}

void CopyIn(void *unused) {
  // Re-initialize memory in place of copying in for this demo.
  ConstantInit(matrix_a, matrix_size, 1.0f);
  ConstantInit(matrix_b, matrix_size, 0.01f);
}

void Exec(void *unused) {
  Multiply(matrix_a, matrix_b, result_matrix, a_width, b_width);
}

void CopyOut(void *unused) {
}

void FreeGPU(void *unused) {
}

void FreeCPU(void *unused) {
  free(matrix_a);
  matrix_a = NULL;
  free(matrix_b);
  matrix_b = NULL;
  free(result_matrix);
  result_matrix = NULL;
}

void Finish(void *unused) {
}
