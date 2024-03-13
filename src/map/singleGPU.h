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
    size_t block_count = (array_len + block_size - 1) / block_size;
    
    singleGpuMappingKernel<MappedFunction><<<block_count, block_size>>>(
        input_array, x, output_array, array_len
    );

    return cudaGetLastError();
}