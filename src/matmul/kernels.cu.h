#ifndef MULT_KERNELS_H
#define MULT_KERNELS_H

#include <stdint.h>

#include "../shared_cuda.cu.h"

template<int isT, typename T>
__device__ inline 
T getElement(
    const uint32_t i, const uint32_t j, const T* array, 
    const uint32_t width, const uint32_t height
) {
    if(isT) {
        return array[j*height + i]; // array[j,i]
    } else {
        return array[i*width + j]; // array[i,j]
    }
}

// heightA = widthB
template <int isTA, int isTB, class T> 
__global__ void mmmNaiveKernel(
    T* matrixA, int widthA, int heightA, 
    T* matrixB, int widthB, int heightB, 
    T* matrixC, int widthC, int heightC
) {
    int i = blockIdx.y*blockDim.y + threadIdx.y; 
    int j = blockIdx.x*blockDim.x + threadIdx.x;

    if( (i >= heightC) || (j >= widthC) ) return;

    T accumulator = 0.0f;
    for(int k = 0; k < widthA; k ++) {
        T a = getElement<isTA, T>(i, k, matrixA, widthA, heightA);
        T b = getElement<isTB, T>(k, j, matrixB, widthB, heightB);
        accumulator += a*b;
    }

    matrixC[i*widthC + j] = accumulator;
}

// heightA = widthB
template <int isTA, int isTB, class T> 
__global__ void mmmNaiveKernelMulti(
    T* matrixA, int widthA, int heightA, 
    T* matrixB, int widthB, int heightB, 
    T* matrixC, int widthC, int heightC, 
    int device_height_offset
) {
    int i = blockIdx.y*blockDim.y + threadIdx.y; 
    int j = blockIdx.x*blockDim.x + threadIdx.x;

    if( (i >= heightC) || (j >= widthC) ) return;

    T accumulator = 0.0f;
    for(int k = 0; k < widthA; k ++) {
        T a = getElement<isTA, T>(i, k, matrixA, widthA, heightA);
        T b = getElement<isTB, T>(k, j, matrixB, widthB, heightB);
        accumulator += a*b;
    }

    matrixC[i*widthC + j] = accumulator;
}  

// heightA = widthB
template <int isTB, class T> 
__global__ void mmmPageTiledKernel(
    const T* matrixA, const int widthA, const int heightA, 
    const T* matrixB, const int widthB, const int heightB, 
    T* matrixC, const int widthC, const int heightC, 
    const int device
) { 
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x >= widthC) || (y >= heightC) ) return;

    T accumulator = 0.0f;
    for(int k = 0; k < widthA; k ++) {
        T a = getElement<false, T>(y, k, matrixA, widthA, heightA);
        T b = getElement<isTB, T>(k, x, matrixB, widthB, heightB);
        accumulator += a*b;
    }

    matrixC[y*widthC + x] = accumulator;
}

template <int isTB, class T> 
__global__ void mmmPageTiledKernelAdditive(
    const T* matrixA, const int widthA, const int heightA, 
    const T* matrixB, const int widthB, const int heightB, 
    T* matrixC, const int widthC, const int heightC
) { 
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x >= widthC) || (y >= heightC) ) return;

    T accumulator = 0.0f;
    for(int k = 0; k < widthA; k ++) {
        T a = getElement<false, T>(y, k, matrixA, widthA, heightA);
        T b = getElement<isTB, T>(k, x, matrixB, widthB, heightB);
        accumulator += a*b;
    }

    matrixC[y*widthC + x] += accumulator;
}

// heightA = widthB
template <int isTB, class T> 
__global__ void mmmPrefetchPageTiledKernel(
    const T* matrixA, const int widthA, const int heightA, 
    const T* matrixB, const int widthB, const int heightB, 
    T* const matrixC, const int widthC, const int heightC, 
    const int block_x_offset, const int block_y_offset, const int device_offset
) { 
    //if (block_x_offset = widthC) {
    //    block_x_offset -= widthC;
    //}
    int x = threadIdx.x + block_x_offset;
    int y = blockIdx.y*blockDim.y + block_y_offset;

    if (y >= heightC) {
        x += blockDim.x;
        y -= heightC;
    }

    if ((x >= widthC) || (y >= heightC)) return;

    x += device_offset;
    
    if (x >= widthC) {
        x -= widthC;
    }

    T accumulator = 0.0f;
    for(int k = 0; k < widthA; k ++) {
        T a = getElement<false, T>(y, k, matrixA, widthA, heightA);
        T b = getElement<isTB, T>(k, x, matrixB, widthB, heightB);
        accumulator += a*b;
    }

    matrixC[y*widthC + x] = accumulator;
}

template <int isTB, class T> 
__global__ void mmmPrefetchingKernel(
    const T* matrixA, T* matrixC, const int widthC, const int heightC,
    const int block_x_offset, const int block_y_offset, int PageSize, const int device_offset
) { 
    int x = block_x_offset;
    int y = block_y_offset + threadIdx.x;

    if (y >= heightC) {
        x += PageSize;
        y -= heightC;
    }

    if (x >= widthC) return;

    x += device_offset;
    
    if (x >= widthC) {
        x -= widthC;
    }

    matrixC[y*widthC + x] = matrixA[y*widthC + x];
}

// adapted from https://github.com/vogma/cannon_cuda_compression/blob/main/src/cudaMatrixMultiply.cu
template <class T, int cannon_block> 
__global__ void mmmCannon(
    const T *matrixA, const T *matrixB, T *matrixC, const int n
) {
  // Allocate shared memory for the two blocks aSub and bSub.
  // Use two-dimensional matrices of size BLOCK_SIZE * BLOCK_SIZE
  __shared__ double aSub[cannon_block][cannon_block];
  __shared__ double bSub[cannon_block][cannon_block];

  const int Bx_offset = blockIdx.x * cannon_block + threadIdx.x;
  const int Ay_offset = blockIdx.y * cannon_block + threadIdx.y;
  double tmp = 0;
  /* Go */
  for (int blocks = 0; blocks < gridDim.x; blocks += 1) {
    int Ax_offset = threadIdx.x + blocks * cannon_block;
    int By_offset = threadIdx.y + blocks * cannon_block;

    if (Ax_offset < n && Ay_offset < n)
      aSub[threadIdx.y][threadIdx.x] = matrixA[Ax_offset + Ay_offset * n];
    else
      aSub[threadIdx.y][threadIdx.x] = 0;
    if (Bx_offset < n && By_offset < n)
      bSub[threadIdx.y][threadIdx.x] = matrixB[Bx_offset + By_offset * n];
    else
      bSub[threadIdx.y][threadIdx.x] = 0;

    __syncthreads(); // Make sure that all threads had time to read the sub
                     // matrix.

    for (int i = 0; i < cannon_block; i++)
      tmp += aSub[threadIdx.y][i] * bSub[i][threadIdx.x];

    __syncthreads();
  }
  if ((Bx_offset < n) && (Ay_offset < n)) {
    matrixC[Bx_offset + n * Ay_offset] = tmp;
  }
}

// adapted from https://github.com/vogma/cannon_cuda_compression/blob/main/src/cudaMatrixMultiply.cu
template <class T, int cannon_block> 
__global__ void mmmCannonQuadrant(
    const T *matrixA, const T *matrixB, T *matrixC, 
    const unsigned int total_n, const unsigned int quadrant_n, 
    const unsigned int blocks, 
    const unsigned int offset_x, const unsigned int offset_y
) {
  // Allocate shared memory for the two blocks aSub and bSub.
  // Use two-dimensional matrices of size BLOCK_SIZE * BLOCK_SIZE
  __shared__ double aSub[cannon_block][cannon_block];
  __shared__ double bSub[cannon_block][cannon_block];

  const int Bx_offset = blockIdx.x * cannon_block + threadIdx.x + offset_x;
  const int Ay_offset = blockIdx.y * cannon_block + threadIdx.y + offset_y;
  double tmp = 0;
  /* Go */
  for (int block = 0; block < blocks; block += 1) {
    int Ax_offset = threadIdx.x + block * cannon_block;
    int By_offset = threadIdx.y + block * cannon_block;

    if (Ax_offset < total_n && Ay_offset < total_n)
      aSub[threadIdx.y][threadIdx.x] = matrixA[Ax_offset + Ay_offset * total_n];
    else
      aSub[threadIdx.y][threadIdx.x] = 0;
    if (Bx_offset < total_n && By_offset < total_n)
      bSub[threadIdx.y][threadIdx.x] = matrixB[Bx_offset + By_offset * total_n];
    else
      bSub[threadIdx.y][threadIdx.x] = 0;

    __syncthreads(); // Make sure that all threads had time to read the sub
                     // matrix.

    for (int i = 0; i < cannon_block; i++)
      tmp += aSub[threadIdx.y][i] * bSub[i][threadIdx.x];

    __syncthreads();
  }
  if (((Bx_offset) < total_n) && ((Ay_offset) < total_n)) {
    matrixC[(Bx_offset) + (total_n * (Ay_offset))] = tmp;
  }
}
#endif