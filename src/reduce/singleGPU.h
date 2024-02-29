template<typename T>
__global__ void singleGpuKernel(
    const T* input_array, const T x, T* output_array, const int array_len
) {
    
}

template<typename F, typename T>
void singleGpuReduction(
    F mapped_kernel, const T* input_array, const T constant, T* output_array, 
    const int array_len
) {  
    std::cout << "TODO singleGpuReduction\n";
}