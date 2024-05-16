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

#define lgWARP      5
#define WARP        (1<<lgWARP)

const size_t BLOCK_SIZE = 1024;
const size_t PARALLEL_BLOCKS = 65535;
const size_t ELEMENTS_PER_THREAD = 12;
const size_t CANNON_BLOCK = 32;


uint32_t MAX_HARDWARE_WIDTH;
uint32_t MAX_SHARED_MEMORY;

void initialise_hardware() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    MAX_HARDWARE_WIDTH = 
        prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    MAX_SHARED_MEMORY = prop.sharedMemPerBlock;
}

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

void setup_events(
    cudaEvent_t** events_ptr, int origin_device, int device_count
) {
    cudaEvent_t* events = *events_ptr;
    
    for (int device=0; device<device_count; device++) {
        cudaSetDevice(device);

        cudaEvent_t event;
        CCC(cudaEventCreate(&event));
        events[device] = event;
    }

    cudaSetDevice(origin_device);
}

float get_runtime(
    cudaEvent_t start_event, cudaEvent_t end_event
) {
    float runtime_milliseconds;
    CCC(cudaEventElapsedTime(&runtime_milliseconds, start_event, end_event));
    return runtime_milliseconds;
}

// Not quite true runtime but will do for now
float get_mean_runtime(
    int devices, cudaEvent_t** start_events_ptr, cudaEvent_t** end_events_ptr
) {
    cudaEvent_t* start_events = *start_events_ptr;
    cudaEvent_t* end_events = *end_events_ptr;

    float total_milliseconds = 0;

    for (int device=0; device<devices; device++) {
        total_milliseconds += get_runtime(start_events[device], end_events[device]);
    }

    return total_milliseconds/devices;
}

/**
 * `array_len` is the input-array length
 * This function attempts to virtualize the computation so
 *   that it spawns at most 1024 CUDA blocks; otherwise an
 *   error is thrown. It should not throw an error for any
 *   B >= 64.
 * The return is the number of blocks, and `CHUNK * (*num_chunks)`
 *   is the number of elements to be processed sequentially by
 *   each thread so that the number of blocks is <= 1024. 
 */
template<int CHUNK>
uint32_t getNumBlocks(
    const unsigned long int array_len, uint32_t* num_chunks
) {
    const uint32_t max_input_threads = (array_len + CHUNK - 1) / CHUNK;
    const uint32_t num_threads_0 = min(max_input_threads, MAX_HARDWARE_WIDTH);

    const uint32_t min_elements_threads = num_threads_0 * CHUNK;
    *num_chunks = max((unsigned long int)1, array_len / min_elements_threads);

    const uint32_t sequential_chunk = (*num_chunks) * CHUNK;
    const uint32_t num_threads = 
        (array_len + sequential_chunk - 1) / sequential_chunk;
    const uint32_t num_blocks = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if(num_blocks > BLOCK_SIZE) {
        printf("Broken Assumption: number of blocks %d exceeds maximal block "
               "size: %d. BLOCK_SIZE: %d. Exiting!"
               , num_blocks, BLOCK_SIZE, BLOCK_SIZE);
        exit(1);
    }

    return num_blocks;
}

/**
 * Helper function that copies `CHUNK` input elements per thread from
 *   global to shared memory, in a way that optimizes spatial locality,
 *   i.e., (32) consecutive threads read consecutive input elements.
 *   This leads to "coalesced" access in the case when the element size
 *   is a word (or less). Coalesced access means that (groups of 32)
 *   consecutive threads access consecutive memory words.
 * 
 * `global_offset` is the offset in global-memory array `global_array`
 *    from where elements should be read.
 * `global_array` is the input array stored in global memory
 * `array_len` is the length of `global_array`
 * `neutral_element` is the neutral element of `T` (think zero). In case
 *    the index of the element to be read by the current thread
 *    is out of range, then place `neutral_element` in shared memory instead.
 * `shared_memory` is the shared memory. It has size 
 *     `blockDim.x*CHUNK*sizeof(T)`, where `blockDim.x` is the
 *     size of the CUDA block. `shared_memory` should be filled from
 *     index `0` to index `blockDim.x*CHUNK - 1`.
 *
 * As such, a CUDA-block B of threads executing this function would
 *   read `CHUNK*B` elements from global memory and store them to
 *   (fast) shared memory, in the same order in which they appear
 *   in global memory, but making sure that consecutive threads
 *   read consecutive elements of `global_array` in a SIMD instruction.
 */ 
template<class T, uint32_t CHUNK>
__device__ inline void from_global_to_shared_memory(
    const uint32_t global_offset, const unsigned long int array_len, 
    const T& neutral_element, T* global_array, volatile T* shared_memory
) {
    #pragma unroll
    for(uint32_t i=0; i<CHUNK; i++) {
        uint32_t local_index = i*blockDim.x + threadIdx.x;
        uint32_t global_index = global_offset + local_index;
        T element = neutral_element;
        if(global_index < array_len) { 
            element = global_array[global_index]; 
        }
        shared_memory[local_index] = element;
    }
    __syncthreads();
}

/**
 * A warp of threads cooperatively scan with generic-binop `Operation` a 
 *   number of warp elements stored in shared memory (`ptr`).
 * No synchronization is needed because the thread in a warp execute
 *   in lockstep.
 * `index` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename Operation::ReturnElement`
 */ 
template<class Operation>
__device__ inline typename Operation::ReturnElement scanIncWarp(
    volatile typename Operation::ReturnElement* ptr, const unsigned int index
) {
    const unsigned int lane = index & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) {
            ptr[index] = Operation::apply(ptr[index-p], ptr[index]);
        }
    }
    return Operation::remVolatile(ptr[index]);
}

/**
 * A CUDA-block of threads cooperatively scan with generic-binop `Operation`
 *   a CUDA-block number of elements stored in shared memory (`ptr`).
 * `index` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename Operation::ReturnElement`. Note that this is NOT published to shared memory!
 */ 
template<class Operation>
__device__ inline typename Operation::ReturnElement scanIncBlock(
    volatile typename Operation::ReturnElement* ptr, 
    const unsigned int index
) {
    const unsigned int lane = index & (WARP-1);
    const unsigned int warp_id = index >> lgWARP;

    // 1. perform scan at warp level
    typename Operation::ReturnElement res = scanIncWarp<Operation>(ptr,index);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == (WARP-1)) { ptr[warp_id] = res; } 
    __syncthreads();

    // 3. scan again the first warp
    if (warp_id == 0) scanIncWarp<Operation>(ptr, index);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warp_id > 0) {
        res = Operation::apply(ptr[warp_id-1], res);
    }

    return res;
}

uint32_t closestMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

#endif