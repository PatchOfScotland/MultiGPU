#ifndef SHARED_CUDA_H
#define SHARED_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <iostream>
#include <unistd.h>  

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

const size_t block_size = 1024;

// Checking Cuda Call
#define CCC(ans) { cuda_assert((ans), __FILE__, __LINE__); }

inline void cuda_assert(
    cudaError_t code, const char *file, int line, bool abort=true
) {
    if (code != cudaSuccess) {
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