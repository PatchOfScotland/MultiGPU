template<typename MappedFunction>
__global__ void singleGpuMappingKernel(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array, 
    const int array_len
) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < (blockDim.x * gridDim.x)) {
        output_array[index] = MappedFunction::apply(input_array[index], x);
    }
}

template<typename MappedFunction>
cudaError_t singleGpuMapping(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array, 
    const int array_len
) {  
    size_t block_count = (array_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    singleGpuMappingKernel<MappedFunction><<<block_count, BLOCK_SIZE>>>(
        input_array, x, output_array, array_len
    );

    return cudaGetLastError();
}