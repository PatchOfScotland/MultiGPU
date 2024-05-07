#ifndef MULT_KERNELS_H
#define MULT_KERNELS_H

#include <stdint.h>

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
    T* matrixA, T* matrixB, T* matrixC, int widthA, int heightA, int widthB, int heightB
) {
  int i = blockIdx.y*blockDim.y + threadIdx.y; 
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if( (i >= heightA) || (j >= widthB) ) return;

  T accumulator = 0.0f;
  for(int k = 0; k < widthA; k ++) {
      T a = getElement<isTA, T>(i, k, matrixA, widthA, heightA);
      T b = getElement<isTB, T>(k, j, matrixB, widthB, heightB);
      accumulator += a*b;
  }

  matrixC[i*widthB + j] = accumulator;
}

#endif