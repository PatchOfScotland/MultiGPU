#include "../shared_cuda.cu.h"

template<typename T>
__global__ void multiGpuStreamReductionKernel(
    const T* input_array, const T x, T* output_array, 
    const unsigned long int array_len, const int stream_num
) {
    
}

template<typename F, typename T>
void multiGpuStreamReduction(
    F mapped_kernel, const T* input_array, const T constant, T* output_array, 
    const unsigned long  int array_len, const cudaStream_t* streams, 
    const int stream_count
) {
    std::cout << "TODO multiGpuStreamReduction\n";
}