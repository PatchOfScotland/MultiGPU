#include <functional>

#include "shared.cu.h"
#include "shared.h"

typedef float arrayType;

// Toy function to be mapped accross an array. Just adds a constant x to an 
// element
template <typename T>
T PlusConst(const T inputElement, const T x) {
    return inputElement + x;
}

// Mapping function that takes a function and maps it across each element in an
// input array, with the output in a new output array. Opperates entirely on 
// the CPU. 
template<typename F, typename T>
void cpuMapping(F mapped_function, T* input_array, const T constant, T* output_array, int array_len) {  
    #pragma omp parallel for
    for (int i=0; i<array_len; i++) {
        output_array[i] = mapped_function(input_array[i], constant);
    }
}

template<typename T>
__global__ void singleGpuKernel(T* input_array, const T x, T* output_array, int array_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < array_len) {
        output_array[index] = input_array[index] + x;
    }
}

template<typename F, typename T>
void singleGpuMapping(F mapped_kernel, T* input_array, const T constant, T* output_array, int array_len) {  
    size_t block_size = 1024;
    size_t block_count = (array_len + block_size - 1) / block_size;

    mapped_kernel<<<block_count, block_size>>>(input_array, constant, output_array, array_len);
}

int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array length> <benchmark repeats> -v(optional)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned int array_len = atoi(argv[1]);
    unsigned int runs = atoi(argv[2]);
    bool validating = false;

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
    }

    float datasize = ((array_len*2*sizeof(arrayType))/1e9);
    std::cout << "Running array of length " 
              << array_len 
              << " (" 
              << datasize 
              <<"GB)\n";
    if (validating) {
        std::cout << "Will validate output\n";
    }
    else {
        std::cout << "Skipping output validation\n";
    }

    arrayType* input_array;
    arrayType* output_array;
    arrayType* validation_array;
    arrayType constant = 0.1;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float runtime;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(arrayType)));
    CCC(cudaMallocManaged(&output_array, array_len*sizeof(arrayType)));

    CCC(cudaEventCreate(&start_event));
    CCC(cudaEventCreate(&end_event));
    float* single_gpu_ms = (float*)calloc(runs, sizeof(float));

    init_array(input_array, array_len);

    if (validating) { // Populate validation array
        validation_array = (arrayType*)malloc(array_len*sizeof(arrayType));
        cpuMapping(PlusConst<arrayType>, input_array, constant, validation_array, array_len);

        std::cout << input_array[0] << ", " << input_array[1] << "\n";
        std::cout << validation_array[0] << ", " << validation_array[1] << "\n";
    }

    check_device_count();

    { // Warmup run
        std::cout << "Running 3 warmups\n";
        for (int i=0; i<3; i++) {
            CCC(cudaEventRecord(start_event));
            singleGpuMapping(singleGpuKernel<arrayType>, input_array, constant, output_array, array_len);
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event));
        }
    }

    { // Benchmark a single GPU
        std::cout << "*** Benchmarking single GPU map ***\n";

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            singleGpuMapping(singleGpuKernel<arrayType>, input_array, constant, output_array, array_len);
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event)); 

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            single_gpu_ms[run] = runtime;


            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(compare_arrays(validation_array, output_array, array_len)){
                    std::cout << "  Single GPU map is correct\n";
                } else {
                    std::cout << "  Single GPU map is incorrect. Skipping any subsequent runs\n";
                    break;
                }
            }
        }

        float mean = 0;
        float total = 0;
        get_timing_stats(single_gpu_ms, runs, &total, &mean);
        float gigabytes_per_second = (float)datasize / (mean * 1e-3f);
        std::cout << "  Total runtime: " << total <<"ms\n";
        std::cout << "  Mean runtime:  " << mean <<"ms\n";
        std::cout << "  Throughput:    " << gigabytes_per_second <<"GB/s\n";
    }

    { // Benchmark multiple GPUs
        std::cout << "*** Benchmarking multi GPU map ***\n";
        
        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            singleGpuMapping(singleGpuKernel<arrayType>, input_array, constant, output_array, array_len);
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event)); 

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            single_gpu_ms[run] = runtime;


            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(compare_arrays(validation_array, output_array, array_len)){
                    std::cout << "  Single GPU map is correct\n";
                } else {
                    std::cout << "  Single GPU map is incorrect. Skipping any subsequent runs\n";
                    break;
                }
            }
        }

        float mean = 0;
        float total = 0;
        get_timing_stats(single_gpu_ms, runs, &total, &mean);
        float gigabytes_per_second = (float)datasize / (mean * 1e-3f);
        std::cout << "  Total runtime: " << total <<"ms\n";
        std::cout << "  Mean runtime:  " << mean <<"ms\n";
        std::cout << "  Throughput:    " << gigabytes_per_second <<"GB/s\n";
    }

    if (validating) {
            free(validation_array);
        }
    cudaFree(input_array);
    cudaFree(output_array);
}
