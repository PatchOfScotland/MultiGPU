#ifndef MULT_KERNELS_H
#define MULT_KERNELS_H

#include <stdint.h>

template<int isT, typename T>
__device__ inline 
T getElement(
    uint32_t i, uint32_t j, T* array, uint32_t height, uint32_t width
) {
    if(isT) {
        return array[j*height + i]; // array[j,i]
    } else {
        return array[i*width + j]; // array[i,j]
    }
}

// width_A = height_B
/**
 * Semantically,
 *   input_A: [height_A][width_A]T
 *   input_B: [height_B][width_A]T
 * but they may come from transpose arrays, i.e.,
 *   isTA == 1 => input_A = transpose(Ao), where Ao : [width_A][height_A]T
 *   isTB == 1 => input_B = transpose(Bo), where Bo : [widthA][height_B]T
 * Matrix Multiplication corresponds to the case when (isTA, isTB) == (0, 1) 
 */
template <int isTA, int isTB, class T> 
__global__ void mmmNaiveKernel(
    T* input_A, T* input_B, T* result_array, int height_A, 
    int height_B, int width_A
) {
  T accumulator = 0.0f;

  int index_x = blockIdx.x*blockDim.x + threadIdx.x;
  int index_y = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (index_x >= height_B) || (index_y >= height_A) ) return;

  for(int k = 0; k < width_A; k ++) {
      T a = getElement<isTA, T>(index_y, k, input_A, height_A, width_A);
      T b = getElement<isTB, T>(index_x, k, input_B, height_B, width_A);
      accumulator += a*b;
  }

  result_array[index_y*height_B + index_x] = accumulator;
}

#endif