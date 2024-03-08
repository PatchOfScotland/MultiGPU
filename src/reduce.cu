#include <functional>
#include <sys/time.h>

#include "reduce/cpu.h"
#include "reduce/multiGPU.h"
#include "reduce/multiGPUstreams.h"
#include "reduce/singleGPU.h"
#include "shared.cu.h"
#include "shared.h"

typedef float array_type;

template<typename T>
class Add {
    public:
        typedef T InputElement;
        typedef T ReturnElement;

        static __device__ __host__ ReturnElement apply(
            const InputElement i, const ReturnElement r
        ) {
            return i+r;
        };
};


int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array length> <benchmark repeats> -v(optional)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned long int array_len = strtoul(argv[1], NULL, 0);
    unsigned int runs = atoi(argv[2]);
    bool validating = false;

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
    }

    double datasize = ((array_len*sizeof(array_type))/1e9);
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

    array_type* input_array;
    array_type* output;
    array_type validation_result;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float runtime_ms;
    long int cpu_time_ms = -1;
    long int single_gpu_time_ms = -1;
    long int multi_gpu_time_ms = -1;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
    CCC(cudaMallocManaged(&output, sizeof(array_type)));

    CCC(cudaEventCreate(&start_event));
    CCC(cudaEventCreate(&end_event));
    float* timing_ms = (float*)calloc(runs, sizeof(float));

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    std::cout << "Initialising input array\n";
    init_array(input_array, array_len);

    if (validating) { // Populate validation array
        std::cout << "Getting CPU result for validation\n";

        struct timeval cpu_start_time;
        struct timeval cpu_end_time;

        gettimeofday(&cpu_start_time, NULL);

        cpuReduction(
            reduction<array_type>, input_array, &validation_result, array_len
        );    
        gettimeofday(&cpu_end_time, NULL); 

        cpu_time_ms = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
        std::cout << "CPU reduction took: " << cpu_time_ms << "ms\n";
    }

    std::cout << "Validation result is: " << validation_result << "\n";

    check_device_count();

    { // Benchmark a single GPU
        std::cout << "\nBenchmarking single GPU map **********************\n";

        std::cout << "  Running a warmup\n";
        singleGpuReduction<Add<array_type>,array_type>(
            input_array, output, array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            singleGpuReduction<Add<array_type>,array_type>(
                input_array, output, array_len
            );
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event));
            CCC(cudaPeekAtLastError());

            CCC(cudaEventElapsedTime(&runtime_ms, start_event, end_event));
            timing_ms[run] = runtime_ms;

            print_loop_feedback(run, runs);

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                array_type tolerance = array_len / 1e5;
                std::cout << "  Comparing " 
                          << std::setprecision(12) 
                          << validation_result 
                          << " and " 
                          << std::setprecision(12) 
                          << *output 
                          << " with tolerance of " 
                          << tolerance 
                          << "\n";
                // Very much rough guess
                if (in_range(validation_result, *output, tolerance)) {
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
            }
        }

         single_gpu_time_ms = print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms, 
            multi_gpu_time_ms
        );
    }

    cudaFree(input_array);
    cudaFree(output);
}
