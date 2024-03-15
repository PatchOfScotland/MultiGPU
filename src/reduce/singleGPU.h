#ifndef SINGLE_GPU_REDUCE
#define SINGLE_GPU_REDUCE

#include "../shared.h"

// based off solutions in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<typename Reduction>
__global__ void singleGpuReductionKernelInitial(
    typename Reduction::InputElement* input_array, 
    const unsigned long int array_len, 
    const int load_stride, 
    volatile typename Reduction::ReturnElement* global_results
) {
    size_t index = (blockDim.x * blockIdx.x + threadIdx.x);
    __shared__ typename Reduction::ReturnElement per_block_results[block_size];
    typename Reduction::ReturnElement per_thread_accumulator = 0;

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<array_len; i+=load_stride) {
        per_thread_accumulator = Reduction::apply(
            input_array[i], per_thread_accumulator
        );
    }

    __syncthreads();
 
    per_block_results[threadIdx.x] = per_thread_accumulator;

    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            per_block_results[threadIdx.x] = Reduction::apply(
                per_block_results[threadIdx.x], 
                per_block_results[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    // Record result to shared memory
    if (threadIdx.x == 0) {
        global_results[blockIdx.x] = per_block_results[0];
    }
}

// Note, designed to only ever be run as a single block
template<typename Reduction>
__global__ void singleGpuReductionKernelFinal(
    volatile typename Reduction::ReturnElement* input_array, 
    typename Reduction::ReturnElement* accumulator,
    const size_t array_len, 
    const size_t load_stride
) {
    size_t index = threadIdx.x;
    __shared__ typename Reduction::ReturnElement per_block_results[block_size];
    typename Reduction::ReturnElement per_thread_accumulator = 0;

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<array_len; i+=load_stride) {
        per_thread_accumulator = Reduction::apply(
            input_array[i], per_thread_accumulator
        );
    }

    __syncthreads();

    per_block_results[threadIdx.x] = per_thread_accumulator;

    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            per_block_results[threadIdx.x] = Reduction::apply(
                per_block_results[threadIdx.x], 
                per_block_results[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    *accumulator = per_block_results[0];
}

template<typename Reduction>
cudaError_t singleGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len,
    bool skip
) {
    // Only need half as many blocks, as each starts reducing
    size_t block_count = (((array_len + 1) / 2) + block_size - 1) / block_size;
    double datasize = 
        ((block_count*sizeof(typename Reduction::InputElement))/1e9); 

    typename Reduction::ReturnElement* global_results;
    CCC(cudaMallocManaged(&global_results, 
        block_count*sizeof(typename Reduction::ReturnElement))
    );

    // 2147483648 // max blocks
    // 2441407 too many blocks?

    std::cout << "single will run " 
              << block_count 
              << " blocks each of size " 
              << block_size 
              << ". Is reserving " 
              << block_count*sizeof(typename Reduction::ReturnElement) 
              << "(" 
              << ((block_count*sizeof(typename Reduction::ReturnElement))/1e9) 
              << "GB)\n";

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    if (skip == false) {
        singleGpuReductionKernelInitial<Reduction><<<
            block_count, block_size
        >>>(
            input_array, array_len, (block_count*block_size), global_results
        );
    }

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    if (skip == false) {
        singleGpuReductionKernelFinal<Reduction><<<1, block_size>>>(
            global_results, accumulator, block_count, block_size
        );
    }

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    cudaFree(global_results);

    return cudaGetLastError();
}

#endif