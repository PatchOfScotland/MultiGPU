#include "../shared_cuda.cu.h"
#include "../shared.h"
#include "singleGPU.h"

// based off solutions in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<typename ReduceFunction, typename T>
__global__ void multiGpuReductionKernelInitial(
    typename ReduceFunction::InputElement* input_array, 
    const unsigned long int array_len, 
    const int load_stride, 
    const int device_num, 
    const int device_count,
    volatile T* global_results
) {
    size_t index = device_num*blockDim.x*gridDim.x 
        + blockDim.x*blockIdx.x 
        + threadIdx.x;
    size_t start = ((array_len + device_count - 1)/device_count) * device_num;
    size_t end = min(array_len, 
        ((array_len + device_count - 1)/device_count 
        + (device_num*(2*blockDim.x*gridDim.x)))
    );

    __shared__ T per_block_results[block_size];
    T per_thread_accumulator = 0;

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<end; i+=load_stride) {
        if (i >= start) {
            per_thread_accumulator = ReduceFunction::apply(
                input_array[i], per_thread_accumulator
            );
        }
    }
    __syncthreads();
 
    per_block_results[threadIdx.x] = per_thread_accumulator;

    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            per_block_results[threadIdx.x] = ReduceFunction::apply(
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
template<typename ReduceFunction, typename T>
__global__ void multiGpuReductionKernelFinal(
    volatile T* input_array, 
    typename ReduceFunction::ReturnElement* accumulator,
    const size_t array_len, 
    const size_t load_stride
) {
    size_t index = threadIdx.x;
    __shared__ T per_block_results[block_size];
    T per_thread_accumulator = 0;

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<array_len; i+=load_stride) {
        per_thread_accumulator = ReduceFunction::apply(
            input_array[i], per_thread_accumulator
        );
    }

    __syncthreads();

    per_block_results[threadIdx.x] = per_thread_accumulator;

    // Now start reducing further down until we have a unified result per block
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIdx.x < stride) {
            per_block_results[threadIdx.x] = ReduceFunction::apply(
                per_block_results[threadIdx.x], 
                per_block_results[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    *accumulator = per_block_results[0];
}

template<typename ReduceFunction, typename T>
void per_device_management(
    T* input_array, T* accumulator, const unsigned long int array_len, 
    const size_t dev_block_count, const int device, const int device_count 
) {
    CCC(cudaSetDevice(device));

    T* global_results;
    CCC(cudaMallocManaged(&global_results, dev_block_count*sizeof(T)));
    T* device_accumulator;
    CCC(cudaMallocManaged(&device_accumulator, sizeof(T)));

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    multiGpuReductionKernelInitial<ReduceFunction,T><<<
        dev_block_count, block_size
    >>>(
        input_array, array_len, (dev_block_count*block_size), device, device_count, global_results
    );
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));
    //std::cout << "load_stride: " << (dev_block_count*block_size) << "\n";

    //print_array(global_results, dev_block_count);

    multiGpuReductionKernelFinal<ReduceFunction,T><<<1, block_size>>>(
        global_results, device_accumulator, dev_block_count, block_size
    );

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    *accumulator = *device_accumulator;

    cudaFree(global_results);
    cudaFree(device_accumulator);

}

template<typename ReduceFunction, typename T>
cudaError_t multiGpuReduction(
    typename ReduceFunction::InputElement* input_array, 
    typename ReduceFunction::ReturnElement* accumulator, 
    const unsigned long int array_len
) {  
    // For small enough jobs then just run on a single device
    // TODO derive this more programatically
    if (array_len < 2048) {
        std::cout << "Small enough input for just a single device\n";
        return singleGpuReduction<ReduceFunction,T>(
            input_array, accumulator, array_len
        );
    }

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    // Only need half as many blocks, as each starts reducing
    size_t block_count = (((array_len + 1) / 2) + block_size - 1) / block_size;
    size_t dev_block_count = (block_count + device_count - 1) / device_count;

    //std::cout << "Scheduling " << block_count << " blocks,  " << dev_block_count << " per device\n";
    typename ReduceFunction::ReturnElement accumulators[device_count];

    std::thread threads[device_count];
    for (int device=0; device<device_count; device++) {
        threads[device] = std::thread(
            per_device_management<ReduceFunction,T>, input_array, 
            &accumulators[device], array_len, dev_block_count, device, 
            device_count 
        );

        
        
        //per_device_management<ReduceFunction,T>(input_array, 
        //    &accumulators[device], array_len, dev_block_count, device, 
        //    device_count);

        //per_device_management<ReduceFunction,T>(input_array, 
        //    accumulators, array_len, dev_block_count, device, 
        //    device_count);

        //std::cout << "escape: " << accumulators[device] << "\n";

    }

    for (int device=0; device<device_count; device++) {
        threads[device].join();        
    }

    T total = 0;
    for (int device=0; device<device_count; device++) { 
        total += accumulators[device];
    }
    *accumulator = total;

    //std::cout << "Final result: " << *accumulator << "\n";
 
    CCC(cudaSetDevice(origin_device));

    return cudaGetLastError();
}