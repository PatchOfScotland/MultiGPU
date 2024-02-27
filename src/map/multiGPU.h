#include "../shared.cu.h"

template<typename T>
__global__ void multiGpuKernel(
    T* input_array, const T x, T* output_array, int array_len, int device_num
) {
    size_t index = device_num*blockDim.x*gridDim.x 
        + blockDim.x*blockIdx.x 
        + threadIdx.x;
    if (index < array_len) {
        output_array[index] = input_array[index] + x;
    }
}

template<typename F, typename T>
void multiGpuMapping(
    F mapped_kernel, T* input_array, const T constant, T* output_array, 
    int array_len
) {  
    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    size_t block_count = (array_len + block_size - 1) / block_size;
    size_t dev_block_count = (block_count + device_count - 1) / device_count;

    for (int device=0; device<device_count; device++) {
        CCC(cudaSetDevice(device));
        mapped_kernel<<<dev_block_count, block_size>>>(
            input_array, constant, output_array, array_len, device
        );
    }

    CCC(cudaSetDevice(origin_device));
}