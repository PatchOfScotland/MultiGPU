#ifndef REDUCE_KERNELS_H
#define REDUCE_KERNELS_H

template<typename Reduction, typename InputElem>
__global__ void commutativeKernel(
    InputElem* input_array, 
    typename Reduction::ReturnElement* accumulator,
    const unsigned long int array_len, 
    const int load_stride
) {
    size_t index = (blockDim.x * blockIdx.x + threadIdx.x);
    __shared__ typename Reduction::ReturnElement per_block_results[BLOCK_SIZE];
    typename Reduction::ReturnElement thread_accumulator = Reduction::init();

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<array_len; i+=load_stride) {
        thread_accumulator = Reduction::apply(
            input_array[i], thread_accumulator
        );
    }

    __syncthreads();
 
    per_block_results[threadIdx.x] = thread_accumulator;

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
        accumulator[blockIdx.x] = per_block_results[0];
    }
}

#endif