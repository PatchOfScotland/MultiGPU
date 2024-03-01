template<typename T>
__global__ void singleGpuMappingKernel(
    const T* input_array, const T x, T* output_array, const int array_len
) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < array_len) {
        output_array[index] = input_array[index] + x;
    }
}

template<typename F, typename T>
void singleGpuMapping(
    F mapped_kernel, const T* input_array, const T constant, T* output_array, 
    const int array_len
) {  
    size_t block_count = (array_len + block_size - 1) / block_size;

    mapped_kernel<<<block_count, block_size>>>(
        input_array, constant, output_array, array_len
    );
}