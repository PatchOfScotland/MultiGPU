#ifndef SHARED_CUDA_H
#define SHARED_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>  

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

size_t block_size = 1024;

// Checking Cuda Call
#define CCC(call)                                       \
    {                                                   \
        cudaError_t cudaStatus = call;                  \
        if (cudaSuccess != cudaStatus) {                \
            std::cerr << "ERROR: CUDA RT call \""       \
                      << #call                          \
                      << "\" in line "                  \
                      << __LINE__                       \
                      << " of file "                    \
                      << __FILE__                       \
                      << " failed with "                \
                      << cudaGetErrorString(cudaStatus) \
                      << " ("                           \
                      << cudaStatus                     \
                      <<").\n",                         \
            exit(cudaStatus);                           \
        }                                               \
    }

int cuda_assert(cudaError_t code) {
  if(code != cudaSuccess) {
    std::cout << "GPU Error: " << cudaGetErrorString(code) << "\n";
    return 1;
  }
  return 0;
}

void check_device_count() {
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));
    if (device_count == 1) {
        std::cout << "!!! Only a single device detected !!!\n";
    }
}

#endif