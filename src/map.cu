#include <functional>

#include "map/cpu.h"
#include "map/singleGPU.h"
#include "map/multiGPU.h"
#include "shared.cu.h"
#include "shared.h"

typedef float arrayType;

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
    }

    check_device_count();

    { // Benchmark a single GPU
        std::cout << "*** Benchmarking single GPU map ***\n";

        std::cout << "  Running a warmup\n";
        singleGpuMapping(singleGpuKernel<arrayType>, input_array, constant, output_array, array_len);
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

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
        std::cout << "    Total runtime: " << total <<"ms\n";
        std::cout << "    Mean runtime:  " << mean <<"ms\n";
        std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/s\n";
    }

    { // Benchmark multiple GPUs
        std::cout << "*** Benchmarking multi GPU map ***\n";

        std::cout << "  Running a warmup\n";
        multiGpuMapping(multiGpuKernel<arrayType>, input_array, constant, output_array, array_len);
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));        

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuMapping(multiGpuKernel<arrayType>, input_array, constant, output_array, array_len);
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
        std::cout << "    Total runtime: " << total <<"ms\n";
        std::cout << "    Mean runtime:  " << mean <<"ms\n";
        std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/s\n";
    }

    if (validating) {
            free(validation_array);
        }
    cudaFree(input_array);
    cudaFree(output_array);
}
