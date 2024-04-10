#include "../shared_cuda.cu.h"
#include "../shared.h"
#include "singleGPU.h"

// based off solutions in https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<typename Reduction>
__global__ void commutativeMultiGpuReductionKernelInitial(
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

    __shared__ typename Reduction::ReturnElement per_block_results[BLOCK_SIZE];
    typename Reduction::ReturnElement thread_accumulator = Reduction::init();

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<end; i+=load_stride) {
        thread_accumulator = Reduction::apply(
            input_array[i], thread_accumulator
        );
    }
    __syncthreads();
 
    per_block_results[threadIdx.x] = thread_accumulator;

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
__global__ void commutativeMultiGpuReductionKernelFinal(
    volatile typename Reduction::ReturnElement* input_array, 
    typename Reduction::ReturnElement* accumulator,
    const size_t array_len, 
    const size_t load_stride
) {
    size_t index = threadIdx.x;
    __shared__ typename Reduction::ReturnElement per_block_results[BLOCK_SIZE];
    typename Reduction::ReturnElement thread_accumulator = Reduction::init();

    // Initial data grab. We traverse the input array by load_stride to 
    // maximise parallel thread locality
    for (size_t i=index; i<array_len; i+=load_stride) {
        thread_accumulator = Reduction::apply(
            input_array[i], thread_accumulator
        );
    }

    __syncthreads();

    per_block_results[threadIdx.x] = thread_accumulator;

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
void commutative_per_device_management(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int device_start, 
    const unsigned long int device_end, 
    const size_t dev_block_count, 
    const int device
) {
    size_t index_start = device*dev_block_count*BLOCK_SIZE ;
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

    commutativeMultiGpuReductionKernelInitial<Reduction><<<
        dev_block_count, BLOCK_SIZE
    >>>(
        input_array, device_end, offset, (dev_block_count*BLOCK_SIZE), 
        device, global_results
    );
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    commutativeMultiGpuReductionKernelFinal<Reduction><<<1, BLOCK_SIZE>>>(
        global_results, device_accumulator, dev_block_count, BLOCK_SIZE
    );

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    *accumulator = *device_accumulator;

    cudaFree(global_results);
    cudaFree(device_accumulator);
}

template<typename Reduction>
cudaError_t commutativeMultiGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len,
    bool skip
) {  
    //// For small enough jobs then just run on a single device
    //// TODO derive this more programatically
    //if (array_len < 2048) {
    //    std::cout << "Small enough input for just a single device\n";
    //    return singleGpuReduction<Reduction>(
    //        input_array, accumulator, array_len, skip
    //    );
    //}

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    size_t block_count = min(
        (array_len + BLOCK_SIZE) / BLOCK_SIZE, PARALLEL_BLOCKS * device_count
    );
    size_t dev_block_count = (block_count + device_count - 1) / device_count;

    typename Reduction::ReturnElement accumulators[device_count];

    unsigned long int per_device = array_len / device_count;
    int remainder = array_len % device_count;
    unsigned long int running_total = 0;
    unsigned long int device_start;
    unsigned long int this_block;

    std::thread threads[device_count];
    for (int device=0; device<device_count; device++) {
        device_start = running_total;
        this_block = (remainder > 0) ? per_device + 1 : per_device;
        remainder -= 1;
        running_total += this_block;

        if (skip == false) {
            threads[device] = std::thread(
                commutative_per_device_management<Reduction>, input_array, 
                &accumulators[device], device_start, device_start + this_block, 
                dev_block_count, device 
            );
        }
    }

    if (skip == false) {
        for (int device=0; device<device_count; device++) {
            threads[device].join();        
        }
    }

    typename Reduction::ReturnElement total;
    total = Reduction::init();
    for (int device=0; device<device_count; device++) { 
        total = Reduction::apply(total, accumulators[device]);
    }
    CCC(cudaSetDevice(origin_device));

    *accumulator = total;
 
    return cudaGetLastError();
}

/*
    template<typename Reduction, int CHUNK>
    __global__ void associativeMultiGpuReductionKernelInitial(
        typename Reduction::InputElement* input_array, 
        const unsigned long int end, 
        const unsigned long int offset,
        const int load_stride, 
        const int device_num, 
        volatile typename Reduction::ReturnElement* global_results,
        uint32_t num_sequential_blocks
    ) {
        typename Reduction::ReturnElement result = Reduction::init();
    
        extern __shared__ char shared_memory[];
        volatile typename Reduction::InputElement* shared_memory_input = 
            (typename Reduction::InputElement*)shared_memory;
        volatile typename Reduction::ReturnElement* shared_memory_return = 
            (typename Reduction::ReturnElement*)shared_memory;
    
        uint32_t num_elems_per_block = num_sequential_blocks * CHUNK * blockDim.x;
        uint32_t input_block_offset = num_elems_per_block * blockIdx.x;
        uint32_t num_elems_per_iter  = CHUNK * blockDim.x;
    
        // virtualization loop of count `num_seq_chunks`. Each iteration processes
        //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
        // `num_seq_chunks` is chosen such that it covers all N input elements
        for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {
            // 1. copy `CHUNK` input elements per thread from global to shared 
            //    memory in a coalesced fashion (for global memory)
            from_global_to_shared_memory<typename Reduction::InputElement, CHUNK>( 
                input_block_offset + seq, array_len, Reduction::init(), 
                input_array, shared_memory_input 
            );
        
            // 2. each thread sequentially reads its `CHUNK` elements from shared
            //     memory, applies the map function and reduces them.
            typename Reduction::ReturnElement accumulator = Reduction::init();
            uint32_t shmem_offset = threadIdx.x * CHUNK;
            #pragma unroll
            for (uint32_t i = 0; i < CHUNK; i++) {
                typename Reduction::InputElement element = 
                    shared_memory_input[shmem_offset + i];
                typename Reduction::ReturnElement red = Reduction::map(element);
                accumulator = Reduction::apply(accumulator, red);
            }
            __syncthreads();
            
            // 3. each thread publishes the previous result in shared memory
            shared_memory_return[threadIdx.x] = accumulator;
            __syncthreads();
        
            // 4. perform an intra-block reduction with the per-thread result
            //    from step 2; the last thread updates the per-block result `res`
            accumulator = scanIncBlock<Reduction>(
                shared_memory_return, threadIdx.x
            );
            if (threadIdx.x == blockDim.x-1) {
                result = Reduction::apply(result, accumulator);
            }
            __syncthreads();
            // rinse and repeat until all elements have been processed.
        }
    
        // Record result to shared memory
        if (threadIdx.x == blockDim.x-1) {
            global_results[blockIdx.x] = result;
        }
    }
    
    template<typename Reduction>
    __global__ void associativeMultiGpuReductionKernelFinal(
        volatile typename Reduction::ReturnElement* input_array, 
        typename Reduction::ReturnElement* accumulator,
        const size_t array_len, 
        const size_t load_stride
    ) {
        extern __shared__ char shared_memory[];
        volatile typename Reduction::ReturnElement* shared_memory_return = 
            (typename Reduction::ReturnElement*)shared_memory;
        typename Reduction::ReturnElement element = Reduction::init();
        if(threadIdx.x < array_len) {
            element = input_array[threadIdx.x];
        }
        shared_memory_return[threadIdx.x] = element;
        __syncthreads();
        element = scanIncBlock<Reduction>(shared_memory_return, threadIdx.x);
        if (threadIdx.x == blockDim.x-1) {
            *accumulator = element;
        }
    }
*/

template<typename Reduction>
void associative_per_device_management(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int device_start, 
    const unsigned long int device_end, 
    const size_t dev_block_count, 
    const int device
) {
    CCC(cudaSetDevice(device));
    
    const unsigned long int sub_array_len = device_end - device_start;
    typename Reduction::InputElement* sub_input_array = input_array + device_start;
    
    typename Reduction::ReturnElement* device_accumulator;
    CCC(cudaMallocManaged(&device_accumulator, 
        sizeof(typename Reduction::ReturnElement))
    );
    *device_accumulator = Reduction::init();

    typename Reduction::InputElement debug = 0;
    for (int i=0; i<sub_array_len; i++) {
        debug = debug + sub_input_array[i];
    }

    associativeSingleGpuReduction<Reduction>(
        sub_input_array, device_accumulator, sub_array_len, false
    );

    *accumulator = *device_accumulator;

    cudaFree(device_accumulator);
}

template<typename Reduction>
cudaError_t associativeMultiGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len,
    bool skip
) {
    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    size_t block_count = min(
        (array_len + BLOCK_SIZE) / BLOCK_SIZE, PARALLEL_BLOCKS * device_count
    );
    size_t dev_block_count = (block_count + device_count - 1) / device_count;

    typename Reduction::ReturnElement accumulators[device_count];

    unsigned long int per_device = array_len / device_count;
    int remainder = array_len % device_count;
    unsigned long int running_total = 0;
    unsigned long int device_start;
    unsigned long int this_block;

    std::thread threads[device_count];
    for (int device=0; device<device_count; device++) {
        device_start = running_total;
        this_block = (remainder > 0) ? per_device + 1 : per_device;
        remainder -= 1;
        running_total += this_block;

        if (skip == false) {
            threads[device] = std::thread(
                associative_per_device_management<Reduction>, input_array, 
                &accumulators[device], device_start, device_start + this_block, 
                dev_block_count, device 
            );
        }
    }

    if (skip == false) {
        for (int device=0; device<device_count; device++) {
            threads[device].join();        
        }
    }

    typename Reduction::ReturnElement total = Reduction::init();
    for (int device=0; device<device_count; device++) { 
        total = Reduction::apply(total, accumulators[device]);
    }
    *accumulator = total;
 
    CCC(cudaSetDevice(origin_device));

    return cudaGetLastError();
}

template<typename Reduction>
cudaError_t multiGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len,
    bool skip
) {
    if (Reduction::commutative == true) {
        return commutativeMultiGpuReduction<Reduction>(
            input_array, accumulator, array_len, skip
        );
    }
    else {
        return associativeMultiGpuReduction<Reduction>(
            input_array, accumulator, array_len, skip
        );
    }
}
