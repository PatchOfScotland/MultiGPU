#ifndef COPYTEST_KERNELS_H
#define COPYTEST_KERNELS_H

#include "../shared_cuda.cu.h"

template<typename T>
__global__ void copyNaive(
    T* matrixA, T* matrixB, const unsigned int array_n
) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < (array_n * array_n)) {
        matrixB[index] = matrixA[index];
    }
}

#endif