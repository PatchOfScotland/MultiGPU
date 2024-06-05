#ifndef MULT_KERNELS_H
#define MULT_KERNELS_H

#include <stdint.h>

#include "../shared_cuda.cu.h"

template<int isT, typename T>
__device__ inline 
T getElement(
    uint32_t i, uint32_t j, T* array, uint32_t width, uint32_t height
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
    T* matrixA, int widthA, int heightA, 
    T* matrixB, int widthB, int heightB, 
    T* matrixC, int widthC, int heightC, 
    int device
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

// adapted from https://github.com/vogma/cannon_cuda_compression/blob/main/src/cudaMatrixMultiply.cu
template <class T> 
__global__ void mmmCannon(
    const T *matrixA, const T *matrixB, T *matrixC, const int n
) {
  // Allocate shared memory for the two blocks aSub and bSub.
  // Use two-dimensional matrices of size BLOCK_SIZE * BLOCK_SIZE
  __shared__ double aSub[CANNON_BLOCK][CANNON_BLOCK];
  __shared__ double bSub[CANNON_BLOCK][CANNON_BLOCK];

  const int Bx_offset = blockIdx.x * CANNON_BLOCK + threadIdx.x;
  const int Ay_offset = blockIdx.y * CANNON_BLOCK + threadIdx.y;
  double tmp = 0;
  /* Go */
  for (int blocks = 0; blocks < gridDim.x; blocks += 1) {
    int Ax_offset = threadIdx.x + blocks * CANNON_BLOCK;
    int By_offset = threadIdx.y + blocks * CANNON_BLOCK;

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

    for (int i = 0; i < CANNON_BLOCK; i++)
      tmp += aSub[threadIdx.y][i] * bSub[i][threadIdx.x];

    __syncthreads();
  }
  if ((Bx_offset < n) && (Ay_offset < n))

    matrixC[Bx_offset + n * Ay_offset] = tmp;
}

#endif