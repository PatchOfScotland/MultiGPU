
// based off solutions in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<typename T>
__global__ void singleGpuReductionKernel(
    const T* input_array, T* accumulator, const int array_len
) {
    // Each thread responsible for two indexes;
    size_t index = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    __shared__ T interim_results[block_size];   

    // must initialise shared data
    interim_results[threadIdx.x] = 0;

    __syncthreads();

    // Initial grab of values and placing them in shared memory. Could 
    if (index < array_len) {
        interim_results[threadIdx.x] = input_array[index] 
            + input_array[index + 1];
    }
    __syncthreads();

    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            interim_results[threadIdx.x] += interim_results[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Now each block should have an acculated result in interim_results[0]
    accumulator[blockIdx.x] = interim_results[0];
}

template<class T>
void pa(T* timing_array, size_t array_len) {
    for (int i=0; i<array_len; i++) {
        if (i==0) {
            std::cout << timing_array[i];
        }
        else if (i==array_len-1) {
            std::cout << ", " << timing_array[i] << "\n";
        }
        else {
            std::cout << ", " << timing_array[i];
        }
    }
}

template<typename F, typename T>
void singleGpuReduction(
    F mapped_kernel, const T* input_array, T* accumulator, const int array_len
) {
    // Only need half as many blocks, as each starts reducing
    size_t block_count = ((array_len / 2) + block_size - 1) / block_size;

    T* accumulator_array;
    CCC(cudaMallocManaged(&accumulator_array, sizeof(T)*block_count));

    if (block_count == 1) {
        mapped_kernel<<<block_count, block_size>>>(
            input_array, accumulator_array, array_len
        );

        *accumulator = accumulator_array[0];
    }
    else if (block_count < (block_size * 2)) {
 
        std::cout << "scheduling " << block_count << " in the first wave\n";
 
        mapped_kernel<<<block_count, block_size>>>(
            input_array, accumulator_array, array_len
        );

        cudaEvent_t sync_event;
        CCC(cudaEventCreate(&sync_event));
        CCC(cudaEventRecord(sync_event));
        CCC(cudaEventSynchronize(sync_event));

        pa(accumulator_array, block_count);

        mapped_kernel<<<1, block_count>>>(
            accumulator_array, accumulator, block_count
        );
        CCC(cudaEventRecord(sync_event));
        CCC(cudaEventSynchronize(sync_event));
    }
    else {
        std::cout << "Out of range ... for now\n";
    }

}