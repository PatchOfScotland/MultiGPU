
// based off solutions in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<typename T>
__global__ void singleGpuReductionKernel(
    const T* input_array, const int array_len, const int load_stride, 
    volatile T* global_results
) {
    size_t index = (blockDim.x * blockIdx.x + threadIdx.x);
    __shared__ T per_block_results[block_size];
    //volatile T* global_results = (T*)shared_memory;
    T per_thread_accumulator = 0;

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<array_len; i+=load_stride) {
        per_thread_accumulator += input_array[i];
    }

    __syncthreads();
 
    per_block_results[threadIdx.x] = per_thread_accumulator;

    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            per_block_results[threadIdx.x] += per_block_results[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Record result to shared memory
    if (threadIdx.x == 0) {
        global_results[blockIdx.x] = per_block_results[0];
    }
}

// Note, designed to only ever be run as a single block
template<typename T>
__global__ void singleGpuReductionKernelFinal(
    volatile T* input_array, const int array_len, const int load_stride, 
    T* accumulator
) {
    size_t index = threadIdx.x;
    __shared__ T per_block_results[block_size];
    T per_thread_accumulator = 0;

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<array_len; i+=load_stride) {
        per_thread_accumulator += input_array[i];
    
    }

    __syncthreads();

    per_block_results[threadIdx.x] = per_thread_accumulator;
    
    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            per_block_results[threadIdx.x] += per_block_results[threadIdx.x + stride];
        }
        __syncthreads();
    }

    *accumulator = per_block_results[0];
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

    T* global_results;
    CCC(cudaMallocManaged(&global_results, block_size*sizeof(T)));

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    mapped_kernel<<<block_count, block_size>>>(
        input_array, array_len, (block_count*block_size), global_results
    );
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    singleGpuReductionKernelFinal<<<1, block_size>>>(
        global_results, block_count, block_size, accumulator
    );

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));
}