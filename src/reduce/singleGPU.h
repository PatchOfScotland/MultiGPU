#ifndef SINGLE_GPU_REDUCE
#define SINGLE_GPU_REDUCE

#include "../shared_cuda.cu.h"
#include "../shared.h"

// based off solutions in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<typename Reduction>
__global__ void commutativeSingleGpuReductionKernelInitial(
    typename Reduction::InputElement* input_array, 
    const unsigned long int array_len, 
    const int load_stride, 
    volatile typename Reduction::ReturnElement* global_results
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
        global_results[blockIdx.x] = per_block_results[0];
    }
}

// Note, designed to only ever be run as a single block
template<typename Reduction>
__global__ void commutativeSingleGpuReductionKernelFinal(
    volatile typename Reduction::ReturnElement* input_array, 
    typename Reduction::ReturnElement* accumulator,
    const size_t array_len, 
    const size_t load_stride
) {
    size_t index = threadIdx.x;
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

    *accumulator = per_block_results[0];
}

template<typename Reduction>
cudaError_t commutativeSingleGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len,
    bool skip
) {
    size_t block_count = min(
        (array_len + BLOCK_SIZE) / BLOCK_SIZE, PARALLEL_BLOCKS
    );

    typename Reduction::ReturnElement* global_results;
    CCC(cudaMallocManaged(&global_results, 
        block_count*sizeof(typename Reduction::ReturnElement))
    );

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    if (skip == false) {
        commutativeSingleGpuReductionKernelInitial<Reduction><<<
            block_count, BLOCK_SIZE
        >>>(
            input_array, array_len, (block_count*BLOCK_SIZE), global_results
        );
    }

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    if (skip == false) {
        commutativeSingleGpuReductionKernelFinal<Reduction><<<1, BLOCK_SIZE>>>(
            global_results, accumulator, block_count, BLOCK_SIZE
        );
    }

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    cudaFree(global_results);

    return cudaGetLastError();
}

template<typename Reduction, int CHUNK>
__global__ void associativeSingleGpuReductionKernelInitial(
    typename Reduction::InputElement* input_array, 
    const unsigned long int array_len, 
    const int load_stride, 
    volatile typename Reduction::ReturnElement* global_results,
    uint32_t num_sequential_blocks
) {
    typename Reduction::ReturnElement result = Reduction::init();

    extern __shared__ char shared_memory[];
    volatile typename Reduction::InputElement* shared_memory_input = 
        (typename Reduction::InputElement*)shared_memory;
    volatile typename Reduction::ReturnElement* shared_memory_return = 
        (typename Reduction::ReturnElement*)shared_memory;

    uint32_t num_elems_per_block = num_sequential_blocks * CHUNK * blockDim.x;
    uint32_t input_block_offset = num_elems_per_block * blockIdx.x;
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    // `num_seq_chunks` is chosen such that it covers all N input elements
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {
        // 1. copy `CHUNK` input elements per thread from global to shared 
        //    memory in a coalesced fashion (for global memory)
        from_global_to_shared_memory<typename Reduction::InputElement, CHUNK>( 
            input_block_offset + seq, array_len, Reduction::init(), 
            input_array, shared_memory_input 
        );
    
        // 2. each thread sequentially reads its `CHUNK` elements from shared
        //     memory, applies the map function and reduces them.
        typename Reduction::ReturnElement accumulator = Reduction::init();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename Reduction::InputElement element = 
                shared_memory_input[shmem_offset + i];
            typename Reduction::ReturnElement red = Reduction::map(element);
            accumulator = Reduction::apply(accumulator, red);
        }
        __syncthreads();
        
        // 3. each thread publishes the previous result in shared memory
        shared_memory_return[threadIdx.x] = accumulator;
        __syncthreads();
    
        // 4. perform an intra-block reduction with the per-thread result
        //    from step 2; the last thread updates the per-block result `res`
        accumulator = scanIncBlock<Reduction>(
            shared_memory_return, threadIdx.x
        );
        if (threadIdx.x == blockDim.x-1) {
            result = Reduction::apply(result, accumulator);
        }
        __syncthreads();
        // rinse and repeat until all elements have been processed.
    }

    // Record result to shared memory
    if (threadIdx.x == blockDim.x-1) {
        global_results[blockIdx.x] = result;
    }
}

// Note, designed to only ever be run as a single block
template<typename Reduction, int CHUNK>
__global__ void associativeSingleGpuReductionKernelFinal(
    typename Reduction::ReturnElement* input_array,  
    typename Reduction::ReturnElement* accumulator,
    const size_t array_len
) {
    extern __shared__ char shared_memory[];
    volatile typename Reduction::ReturnElement* shared_memory_return = 
        (typename Reduction::ReturnElement*)shared_memory;
    typename Reduction::ReturnElement element = Reduction::init();
    if(threadIdx.x < array_len) {
        element = input_array[threadIdx.x];
    }
    shared_memory_return[threadIdx.x] = element;
    __syncthreads();
    element = scanIncBlock<Reduction>(shared_memory_return, threadIdx.x);
    if (threadIdx.x == blockDim.x-1) {
        *accumulator = element;
    }
}

template<typename Reduction>
cudaError_t associativeSingleGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len,
    bool skip
) {
    initialise_hardware();
    const uint32_t CHUNK = 
        ELEMENTS_PER_THREAD*4/sizeof(typename Reduction::InputElement);
    uint32_t num_sequential_blocks;
    uint32_t block_count = getNumBlocks<CHUNK>(
        array_len, &num_sequential_blocks
    );
    size_t shared_memory_size = BLOCK_SIZE * max(
        sizeof(typename Reduction::InputElement) * CHUNK, 
        sizeof(typename Reduction::ReturnElement)
    );

    typename Reduction::ReturnElement* global_results;
    CCC(cudaMallocManaged(
        &global_results, block_count*sizeof(typename Reduction::ReturnElement)
    ));

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    if (skip == false) {
        associativeSingleGpuReductionKernelInitial<Reduction, CHUNK><<<
            block_count, BLOCK_SIZE, shared_memory_size
        >>>(
            input_array, array_len, (block_count*BLOCK_SIZE), global_results, 
            num_sequential_blocks
        );
    }

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    if (skip == false) {
        const uint32_t block_size = closestMul32(block_count);
        shared_memory_size = 
            block_size * sizeof(typename Reduction::ReturnElement);

        associativeSingleGpuReductionKernelFinal<Reduction, CHUNK><<<
            1, block_size, shared_memory_size
        >>>(
            global_results, accumulator, block_count
        );
    }

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    cudaFree(global_results);

    return cudaGetLastError();
}

template<typename Reduction>
cudaError_t singleGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len,
    bool skip
) {
    if (Reduction::commutative == true) {
        return commutativeSingleGpuReduction<Reduction>(
            input_array, accumulator, array_len, skip
        );
    }
    else {
        return associativeSingleGpuReduction<Reduction>(
            input_array, accumulator, array_len, skip
        );
    }
}

#endif