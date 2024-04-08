#include "../shared_cuda.cu.h"

template<typename MappedFunction>
__global__ void multiGpuMappingKernel(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array, 
    const unsigned long int array_len,
    const int device_num
) {
    size_t index = device_num*blockDim.x*gridDim.x 
        + blockDim.x*blockIdx.x 
        + threadIdx.x;
    if (index < (blockDim.x * gridDim.x)) {
        output_array[index] = MappedFunction::apply(input_array[index], x);
    }
}

template<typename MappedFunction>
void multiGpuMapping(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array, 
    const unsigned long int array_len
) {  
    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    size_t block_count = (array_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t dev_block_count = (block_count + device_count - 1) / device_count;

    for (int device=0; device<device_count; device++) {
        CCC(cudaSetDevice(device));
        multiGpuMappingKernel<MappedFunction><<<dev_block_count, BLOCK_SIZE>>>(
            input_array, x, output_array, array_len, device
        );
    }

    CCC(cudaSetDevice(origin_device));
}