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

uint32_t MAX_HWDTH;
uint32_t MAX_SHMEM;

void initHwd() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    MAX_HWDTH = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    MAX_SHMEM = prop.sharedMemPerBlock;
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

/**
 * `array_len` is the input-array length
 * `B` is the CUDA block size
 * This function attempts to virtualize the computation so
 *   that it spawns at most 1024 CUDA blocks; otherwise an
 *   error is thrown. It should not throw an error for any
 *   B >= 64.
 * The return is the number of blocks, and `CHUNK * (*num_chunks)`
 *   is the number of elements to be processed sequentially by
 *   each thread so that the number of blocks is <= 1024. 
 */
template<int CHUNK>
uint32_t getNumBlocks(const unsigned long int array_len, uint32_t* num_chunks) {
    const uint32_t max_inp_thds = (array_len + CHUNK - 1) / CHUNK;
    const uint32_t num_thds0    = min(max_inp_thds, MAX_HWDTH);

    const uint32_t min_elms_all_thds = num_thds0 * CHUNK;
    *num_chunks = max((unsigned long int)1, array_len / min_elms_all_thds);

    const uint32_t seq_chunk = (*num_chunks) * CHUNK;
    const uint32_t num_thds = (array_len + seq_chunk - 1) / seq_chunk;
    const uint32_t num_blocks = (num_thds + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if(num_blocks > BLOCK_SIZE) {
        printf("Broken Assumption: number of blocks %d exceeds maximal block size: %d. BLOCK_SIZE: %d. Exiting!"
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
 * `glb_offs` is the offset in global-memory array `d_inp`
 *    from where elements should be read.
 * `d_inp` is the input array stored in global memory
 * `N` is the length of `d_inp`
 * `ne` is the neutral element of `T` (think zero). In case
 *    the index of the element to be read by the current thread
 *    is out of range, then place `ne` in shared memory instead.
 * `shmem_inp` is the shared memory. It has size 
 *     `blockDim.x*CHUNK*sizeof(T)`, where `blockDim.x` is the
 *     size of the CUDA block. `shmem_inp` should be filled from
 *     index `0` to index `blockDim.x*CHUNK - 1`.
 *
 * As such, a CUDA-block B of threads executing this function would
 *   read `CHUNK*B` elements from global memory and store them to
 *   (fast) shared memory, in the same order in which they appear
 *   in global memory, but making sure that consecutive threads
 *   read consecutive elements of `d_inp` in a SIMD instruction.
 */ 
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const unsigned long int N
                  , const T& ne
                  , T* d_inp
                  , volatile T* shmem_inp
) {
    #pragma unroll
    for(uint32_t i=0; i<CHUNK; i++) {
        uint32_t lind = i*blockDim.x + threadIdx.x;
        uint32_t glb_ind = glb_offs + lind;
        T elm = ne;
        if(glb_ind < N) { 
            elm = d_inp[glb_ind]; 
        }
        shmem_inp[lind] = elm;
    }
    __syncthreads();
}

/**
 * A warp of threads cooperatively scan with generic-binop `OP` a 
 *   number of warp elements stored in shared memory (`ptr`).
 * No synchronization is needed because the thread in a warp execute
 *   in lockstep.
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`
 */ 
template<class Reduction>
__device__ inline typename Reduction::ReturnElement
scanIncWarp( volatile typename Reduction::ReturnElement* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) ptr[idx] = Reduction::apply(ptr[idx-p], ptr[idx]);
        // __syncwarp();
    }
    return Reduction::remVolatile(ptr[idx]);
}

/**
 * A CUDA-block of threads cooperatively scan with generic-binop `OP`
 *   a CUDA-block number of elements stored in shared memory (`ptr`).
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`. Note that this is NOT published to shared memory!
 */ 
template<class Reduction>
__device__ inline typename Reduction::ReturnElement
scanIncBlock(volatile typename Reduction::ReturnElement* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    // 1. perform scan at warp level
    typename Reduction::ReturnElement res = scanIncWarp<Reduction>(ptr,idx);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == (WARP-1)) { ptr[warpid] = res; } 
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0) scanIncWarp<Reduction>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = Reduction::apply(ptr[warpid-1], res);
    }

    return res;
}

uint32_t closestMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

#endif