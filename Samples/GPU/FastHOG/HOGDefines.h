#ifndef __HOG_DEFINES__
#define __HOG_DEFINES__

#define UNROLL_LOOPS

#ifndef CUDA_PIXEL
#define CUDA_PIXEL unsigned char
#endif

#ifndef CUDA_FLOAT
#define CUDA_FLOAT float
#endif

#ifndef CUDA_DT_PIXEL
#define CUDA_DT_PIXEL float
#endif

#ifndef CUDA_DT_PIXEL_INT
#define CUDA_DT_PIXEL_INT int
#endif

#ifndef THREAD_SIZE_W
#define THREAD_SIZE_W 16
#endif

#ifndef THREAD_SIZE_H
#define THREAD_SIZE_H 16
#endif

#ifndef BLOCK_SIZE_H
#define BLOCK_SIZE_H 16
#endif

#ifndef BLOCK_SIZE_W
#define BLOCK_SIZE_W 16
#endif

#ifndef MAX_HISTOGRAM_NO_BINS
#define MAX_HISTOGRAM_NO_BINS 9
#endif

#ifndef MAX_CELL_SIZE_Y
#define MAX_CELL_SIZE_Y 8
#endif

#ifndef MAX_CELL_SIZE_X
#define MAX_CELL_SIZE_X 8
#endif

#ifndef MAX_BLOCK_SIZE_X
#define MAX_BLOCK_SIZE_X 2
#endif

#ifndef MAX_BLOCK_SIZE_Y
#define MAX_BLOCK_SIZE_Y 2
#endif

#ifndef MAX_BLOCKS_PER_WINDOW_X
#define MAX_BLOCKS_PER_WINDOW_X 7
#endif

#ifndef MAX_BLOCKS_PER_WINDOW_Y
#define MAX_BLOCKS_PER_WINDOW_Y 15
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef MAX_BLOCKS_PER_DIM
#define MAX_BLOCKS_PER_DIM  65536
#endif

#ifndef MAX_RESULTS
#define MAX_RESULTS  1000
#endif

#ifndef IMUL
#define IMUL(a, b) __mul24(a, b)
#endif

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

#ifndef DEGTORAD
#define DEGTORAD 0.017453292519943295769236907684886
#endif

#ifndef RADTODEG
#define RADTODEG 57.2957795
#endif

#endif
