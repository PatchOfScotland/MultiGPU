#include "../shared_cuda.cu.h"
#include "../shared.h"
#include "kernels.cu.h"
#include "singleGPU.h"

template<typename Reduction>
void commutative_per_device_management(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* device_results,
    typename Reduction::ReturnElement* device_accumulator,
    typename Reduction::ReturnElement* global_accumulator, 
    const unsigned long int device_start, 
    const unsigned long int device_len, 
    const size_t dev_block_count, 
    const int device
) {
    CCC(cudaSetDevice(device));

    typename Reduction::InputElement* sub_input_array = 
        input_array + device_start;

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    commutativeKernel<Reduction,typename Reduction::InputElement><<<
        dev_block_count, BLOCK_SIZE
    >>>(
        sub_input_array, device_results, device_len, 
        (dev_block_count*BLOCK_SIZE)
    );
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    commutativeKernel<Reduction,typename Reduction::ReturnElement><<<
        1, BLOCK_SIZE
    >>>(
        device_results, device_accumulator, dev_block_count, BLOCK_SIZE
    );

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    *global_accumulator = *device_accumulator;
}

template<typename Reduction>
float commutativeMultiGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len
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
    typename Reduction::ReturnElement* all_device_results[device_count];
    typename Reduction::ReturnElement* all_device_accumulators[device_count];

    unsigned long int per_device = array_len / device_count;
    int remainder = array_len % device_count;
    unsigned long int running_total = 0;
    unsigned long int device_start;
    unsigned long int device_len;

    struct timeval cpu_start_time;
    struct timeval cpu_end_time;

    for (int device=0; device<device_count; device++) { 
        cudaSetDevice(device);

        typename Reduction::ReturnElement* device_results;
        CCC(cudaMallocManaged(&device_results, 
            dev_block_count*sizeof(typename Reduction::ReturnElement))
        );
        all_device_results[device] = device_results;

        typename Reduction::ReturnElement* device_accumulator;
        CCC(cudaMallocManaged(&device_accumulator, 
            sizeof(typename Reduction::ReturnElement))
        );
        all_device_accumulators[device] = device_accumulator;
    }
    cudaSetDevice(origin_device);

    // Not sold on this metod of timing. cudaEvents would be better, but final 
    // reduction is done at cpu level
    gettimeofday(&cpu_start_time, NULL); 

    std::thread threads[device_count];
    for (int device=0; device<device_count; device++) {
        device_start = running_total;
        device_len = (remainder > 0) ? per_device + 1 : per_device;
        remainder -= 1;
        running_total += device_len;

        threads[device] = std::thread(
            commutative_per_device_management<Reduction>, input_array, 
            all_device_results[device], all_device_accumulators[device],
            &accumulators[device], device_start, device_len, 
            dev_block_count, device 
        );
    }

    for (int device=0; device<device_count; device++) {
        threads[device].join();        
    }

    typename Reduction::ReturnElement total;
    total = Reduction::init();
    for (int device=0; device<device_count; device++) { 
        total = Reduction::apply(total, accumulators[device]);
    }

    gettimeofday(&cpu_end_time, NULL); 

    for (int device=0; device<device_count; device++) { 
        cudaSetDevice(device);
        CCC(cudaFree(all_device_results[device]));
        CCC(cudaFree(all_device_accumulators[device]));
    }

    CCC(cudaSetDevice(origin_device));

    *accumulator = total;
 
    float time_ms = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
    
    return time_ms;
}

template<typename Reduction>
void associative_per_device_management(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* device_results,
    typename Reduction::ReturnElement* device_accumulator, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int device_start, 
    const unsigned long int device_len, 
    const size_t dev_block_count, 
    const int device
) {
    CCC(cudaSetDevice(device));
    
    initialise_hardware();

    typename Reduction::InputElement* sub_input_array = input_array + device_start;

    const uint32_t CHUNK = 
        ELEMENTS_PER_THREAD*4/sizeof(typename Reduction::InputElement);
    uint32_t num_sequential_blocks;
    uint32_t block_count = getNumBlocks<CHUNK>(
        device_len, &num_sequential_blocks
    );
    size_t shared_memory_size = BLOCK_SIZE * max(
        sizeof(typename Reduction::InputElement) * CHUNK, 
        sizeof(typename Reduction::ReturnElement)
    );

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));

    associativeKernelInitial<Reduction, CHUNK><<<
        block_count, BLOCK_SIZE, shared_memory_size
    >>>(
        sub_input_array, device_results, device_len, num_sequential_blocks
    );

    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    const uint32_t block_size = closestMul32(block_count);
    shared_memory_size = 
        block_size * sizeof(typename Reduction::ReturnElement);

    associativeKernelFinal<Reduction, CHUNK><<<
        1, block_size, shared_memory_size
    >>>(
        device_results, device_accumulator, block_count
    );
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));

    *accumulator = *device_accumulator;
}

template<typename Reduction>
float associativeMultiGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len
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
    typename Reduction::ReturnElement* all_device_results[device_count];
    typename Reduction::ReturnElement* all_device_accumulators[device_count];

    unsigned long int per_device = array_len / device_count;
    int remainder = array_len % device_count;
    unsigned long int running_total = 0;
    unsigned long int device_start;
    unsigned long int device_len;

    struct timeval cpu_start_time;
    struct timeval cpu_end_time;

    for (int device=0; device<device_count; device++) {
        cudaSetDevice(device);

        typename Reduction::ReturnElement* device_results;
        CCC(cudaMallocManaged(&device_results, 
            dev_block_count*sizeof(typename Reduction::ReturnElement))
        );
        all_device_results[device] = device_results;

        typename Reduction::ReturnElement* device_accumulator;
        CCC(cudaMallocManaged(&device_accumulator, 
            sizeof(typename Reduction::ReturnElement))
        );
        *device_accumulator = Reduction::init();
        all_device_accumulators[device] = device_accumulator;
    }
    cudaSetDevice(origin_device);

    // Not sold on this metod of timing. cudaEvents would be better, but final 
    // reduction is done at cpu level
    gettimeofday(&cpu_start_time, NULL); 

    std::thread threads[device_count];
    for (int device=0; device<device_count; device++) {
        device_start = running_total;
        device_len = (remainder > 0) ? per_device + 1 : per_device;
        remainder -= 1;
        running_total += device_len;

        threads[device] = std::thread(
            associative_per_device_management<Reduction>, input_array, 
            all_device_results[device], all_device_accumulators[device], 
            &accumulators[device], device_start, device_len, 
            dev_block_count, device 
        );
    }

    for (int device=0; device<device_count; device++) {
        threads[device].join();        
    }

    typename Reduction::ReturnElement total = Reduction::init();
    for (int device=0; device<device_count; device++) { 
        total = Reduction::apply(total, accumulators[device]);
    }
    gettimeofday(&cpu_end_time, NULL); 

    *accumulator = total;

    for (int device=0; device<device_count; device++) {
        cudaSetDevice(device);
        CCC(cudaFree(all_device_results[device]));
        CCC(cudaFree(all_device_accumulators[device]));
    }
    cudaSetDevice(origin_device);

    CCC(cudaSetDevice(origin_device));
 
    float time_ms = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
    
    return time_ms;
}

template<typename Reduction>
float multiGpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* accumulator, 
    const unsigned long int array_len
) {
    if (Reduction::commutative == true) {
        return commutativeMultiGpuReduction<Reduction>(
            input_array, accumulator, array_len
        );
    }
    else {
        return associativeMultiGpuReduction<Reduction>(
            input_array, accumulator, array_len
        );
    }
}
