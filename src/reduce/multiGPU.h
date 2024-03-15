#include "../shared_cuda.cu.h"
#include "../shared.h"
#include "singleGPU.h"

// based off solutions in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<typename Reduction>
__global__ void multiGpuReductionKernelInitial(
    typename Reduction::InputElement* input_array, 
    const unsigned long int end, 
    const unsigned long int offset,
    const int load_stride, 
    const int device_num, 
    volatile typename Reduction::ReturnElement* global_results
) {
    size_t index = (device_num*blockDim.x*gridDim.x 
        + blockDim.x*blockIdx.x 
        + threadIdx.x) 
        + offset;

    __shared__ typename Reduction::ReturnElement per_block_results[block_size];
    typename Reduction::ReturnElement per_thread_accumulator = 0;

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<end; i+=load_stride) {
        per_thread_accumulator = Reduction::apply(
            input_array[i], per_thread_accumulator
        );
    }
    __syncthreads();
 
    per_block_results[threadIdx.x] = per_thread_accumulator;

    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            per_block_results[threadIdx.x] = Reduction::apply(
                per_block_results[threadIdx.x], 
                per_block_results[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    // Record result to shared memory
    if (threadIdx.x == 0) {
        global_results[blockIdx.x] = per_block_results[0];
    }
}

// Note, designed to only ever be run as a single block
template<typename Reduction>
__global__ void multiGpuReductionKernelFinal(
    volatile typename Reduction::ReturnElement* input_array, 
    typename Reduction::ReturnElement* accumulator,
    const size_t array_len, 
    const size_t load_stride
) {
    size_t index = threadIdx.x;
    __shared__ typename Reduction::ReturnElement per_block_results[block_size];
    typename Reduction::ReturnElement per_thread_accumulator = 0;

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<array_len; i+=load_stride) {
        per_thread_accumulator = Reduction::apply(
            input_array[i], per_thread_accumulator
        );
    }

    __syncthreads();

    per_block_results[threadIdx.x] = per_thread_accumulator;

    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            per_block_results[threadIdx.x] = Reduction::apply(
                per_block_results[threadIdx.x], 
                per_block_results[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    *accumulator = per_block_results[0];
}

template<typename Reduction>
void per_device_management(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int device_start, 
    const unsigned long int device_end, 
    const size_t dev_block_count, 
    const int device
) {
    size_t index_start = device*dev_block_count*block_size ;
    unsigned long int offset = device_start - index_start;

    CCC(cudaSetDevice(device));

    typename Reduction::ReturnElement* global_results;
    CCC(cudaMallocManaged(&global_results, 
        dev_block_count*sizeof(typename Reduction::ReturnElement))
    );
    typename Reduction::ReturnElement* device_accumulator;
    CCC(cudaMallocManaged(&device_accumulator, 
        sizeof(typename Reduction::ReturnElement))
    );

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    multiGpuReductionKernelInitial<Reduction><<<
        dev_block_count, block_size
    >>>(
        input_array, device_end, offset, (dev_block_count*block_size), 
        device, global_results
    );
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    multiGpuReductionKernelFinal<Reduction><<<1, block_size>>>(
        global_results, device_accumulator, dev_block_count, block_size
    );

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    *accumulator = *device_accumulator;

    cudaFree(global_results);
    cudaFree(device_accumulator);
}

template<typename Reduction>
cudaError_t multiGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len,
    bool skip
) {  
    // For small enough jobs then just run on a single device
    // TODO derive this more programatically
    if (array_len < 2048) {
        std::cout << "Small enough input for just a single device\n";
        return singleGpuReduction<Reduction>(
            input_array, accumulator, array_len, skip
        );
    }

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    size_t block_count = min(
        (array_len + block_size) / block_size, parallel_blocks * device_count
    );
    size_t dev_block_count = (block_count + device_count - 1) / device_count;

    typename Reduction::ReturnElement accumulators[device_count];

    unsigned long int per_device = array_len / device_count;
    int remainder = array_len % device_count;
    unsigned long int running_total = 0;
    unsigned long int device_start;
    unsigned long int device_end;
    unsigned long int this_block;

    std::thread threads[device_count];
    for (int device=0; device<device_count; device++) {
        device_start = running_total;
        this_block = (remainder > 0) ? per_device + 1 : per_device;
        remainder -= 1;
        device_end = device_start + this_block;
        running_total += this_block;

        if (skip == false) {
            threads[device] = std::thread(
                per_device_management<Reduction>, input_array, 
                &accumulators[device], device_start, device_end, 
                dev_block_count, device 
            );
        }
    }

    if (skip == false) {
        for (int device=0; device<device_count; device++) {
            threads[device].join();        
        }
    }

    typename Reduction::ReturnElement total = 0;
    for (int device=0; device<device_count; device++) { 
        total += accumulators[device];
    }
    *accumulator = total;
 
    CCC(cudaSetDevice(origin_device));

    return cudaGetLastError();
}