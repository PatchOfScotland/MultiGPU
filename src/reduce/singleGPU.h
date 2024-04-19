#ifndef SINGLE_GPU_REDUCE
#define SINGLE_GPU_REDUCE

#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

template<typename Reduction>
float commutativeSingleGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len
) {
    size_t block_count = min(
        (array_len + BLOCK_SIZE) / BLOCK_SIZE, PARALLEL_BLOCKS
    );

    typename Reduction::ReturnElement* global_results;
    CCC(cudaMallocManaged(&global_results, 
        block_count*sizeof(typename Reduction::ReturnElement))
    );

    cudaEvent_t start_event;
    CCC(cudaEventCreate(&start_event));
    cudaEvent_t end_event;
    CCC(cudaEventCreate(&end_event));
    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    CCC(cudaEventRecord(start_event));
    commutativeKernel<Reduction,typename Reduction::InputElement><<<
        block_count, BLOCK_SIZE
    >>>(
        input_array, global_results, array_len, (block_count*BLOCK_SIZE)
    );

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    commutativeKernel<Reduction,typename Reduction::ReturnElement><<<
        1, BLOCK_SIZE
    >>>(
        global_results, accumulator, block_count, BLOCK_SIZE
    );
    CCC(cudaEventRecord(end_event));
    CCC(cudaEventSynchronize(end_event));

    cudaFree(global_results);

    return get_runtime(start_event, end_event);
}

template<typename Reduction>
float associativeSingleGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len
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

    cudaEvent_t start_event;
    CCC(cudaEventCreate(&start_event));
    cudaEvent_t end_event;
    CCC(cudaEventCreate(&end_event));
    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    CCC(cudaEventRecord(start_event));
    associativeKernelInitial<Reduction, CHUNK><<<
        block_count, BLOCK_SIZE, shared_memory_size
    >>>(
        input_array, global_results, array_len, num_sequential_blocks
    );

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    const uint32_t block_size = closestMul32(block_count);
    shared_memory_size = 
        block_size * sizeof(typename Reduction::ReturnElement);

    associativeKernelFinal<Reduction, CHUNK><<<
        1, block_size, shared_memory_size
    >>>(
        global_results, accumulator, block_count
    );
    CCC(cudaEventRecord(end_event));
    CCC(cudaEventSynchronize(end_event));

    cudaFree(global_results);

    return get_runtime(start_event, end_event);
}

template<typename Reduction>
float singleGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len
) {
    if (Reduction::commutative == true) {
        return commutativeSingleGpuReduction<Reduction>(
            input_array, accumulator, array_len
        );
    }
    else {
        return associativeSingleGpuReduction<Reduction>(
            input_array, accumulator, array_len
        );
    }
}

#endif