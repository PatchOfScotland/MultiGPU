#include "kernels.cu.h"

template<typename MappedFunction>
float singleGpuMapping(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array, 
    const int array_len
) {  
    size_t block_count = (array_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start_event;
    CCC(cudaEventCreate(&start_event));
    cudaEvent_t end_event;
    CCC(cudaEventCreate(&end_event));

    CCC(cudaEventRecord(start_event));
    mappingKernel<MappedFunction><<<block_count, BLOCK_SIZE>>>(
        input_array, x, output_array, array_len
    );
    CCC(cudaEventRecord(end_event));
    CCC(cudaEventSynchronize(end_event));

    float runtime_milliseconds = get_runtime(start_event, end_event);
    
    return runtime_milliseconds * 1e3;
}