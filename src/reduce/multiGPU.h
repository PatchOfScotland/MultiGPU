#include "../shared.cu.h"

template<typename T>
__global__ void multiGpuKernel(
    const T* input_array, const T x, T* output_array, const int array_len, 
    const int device_num
) {
    
}

template<typename F, typename T>
void multiGpuReduction(
    F mapped_kernel, const T* input_array, const T constant, T* output_array, 
    const int array_len
) {
    std::cout << "TODO multiGpuReduction\n";
}