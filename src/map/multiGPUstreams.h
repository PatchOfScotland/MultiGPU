#include "../shared.cu.h"

template<typename T>
__global__ void multiGpuStreamKernel(
    const T* input_array, const T x, T* output_array, const int array_len, 
    const int stream_num
) {
    size_t index = stream_num*blockDim.x*gridDim.x 
        + blockDim.x*blockIdx.x 
        + threadIdx.x;
    if (index < array_len) {
        output_array[index] = input_array[index] + x;
    }
}

template<typename F, typename T>
void multiGpuStreamMapping(
    F mapped_kernel, const T* input_array, const T constant, T* output_array, 
    const int array_len, const cudaStream_t* streams, const int stream_count
) {  
    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    size_t block_count = (array_len + block_size - 1) / block_size;
    size_t dev_block_count = 
        (block_count + (stream_count*device_count) - 1) 
        / (stream_count * device_count)
    ;

    for (int device=0; device<device_count; device++) {
        CCC(cudaSetDevice(device))
        for(int s = device*stream_count; s<(device+1)*stream_count; s++) {
            mapped_kernel<<<dev_block_count, block_size, 0, streams[s]>>>(
                input_array, constant, output_array, array_len, s
            );
        }
    }

    CCC(cudaSetDevice(origin_device));
}