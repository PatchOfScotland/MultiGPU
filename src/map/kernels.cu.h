#ifndef MAP_KERNELS_H
#define MAP_KERNELS_H

template<typename MappedFunction>
__global__ void mappingKernel(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array, 
    const unsigned long array_len
) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    //if (index < (blockDim.x * gridDim.x)) {
    if (index < array_len) {
        output_array[index] = MappedFunction::apply(input_array[index], x);
    }
}

#endif