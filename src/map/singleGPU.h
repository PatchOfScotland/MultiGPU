template<typename MappedFunction>
__global__ void singleGpuMappingKernel(
    typename MappedFunction::InputElement* i, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* r, 
    const int array_len
) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < array_len) {
        r[index] = MappedFunction::apply(i[index], x);
    }
}

template<typename MappedFunction>
cudaError_t singleGpuMapping(
    //F mapped_function,
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