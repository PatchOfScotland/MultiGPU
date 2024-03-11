#include "../shared.cu.h"

template<typename MappedFunction>
__global__ void multiGpuStreamMappingKernel(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array,
    const int array_len, 
    const int stream_num
) {
    size_t index = stream_num*blockDim.x*gridDim.x 
        + blockDim.x*blockIdx.x 
        + threadIdx.x;
    if (index < (blockDim.x * gridDim.x)) {
        output_array[index] = MappedFunction::apply(input_array[index], x);
    }
}

template<typename MappedFunction>
void multiGpuStreamMapping(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array, 
    const int array_len, 
    const cudaStream_t* streams, 
    const int stream_count
) {  
    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    size_t block_count = (array_len + block_size - 1) / block_size;
    size_t dev_block_count = (block_count + device_count - 1) / device_count;

    for (int device=0; device<device_count; device++) {
        CCC(cudaSetDevice(device))
        multiGpuStreamMappingKernel<MappedFunction><<<
            dev_block_count, block_size, 0, streams[device]
        >>>(
            input_array, x, output_array, array_len, device
        );
    }

    CCC(cudaSetDevice(origin_device));
}