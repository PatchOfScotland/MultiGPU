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

#define cuda_error_check(ans) { cuda_assert((ans), __FILE__, __LINE__); }

inline void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::cerr << "\nCUDA call at line " 
                  << line
                  << " of file " 
                  << file
                  << " failed: " 
                  << cudaGetErrorString(code) 
                  << "\n";
        if (abort == true) {
            exit(code);
        }
    }
}

void check_device_count() {
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));
    if (device_count == 1) {
        std::cout << "!!! Only a single device detected !!!\n";
    }
}

#endif