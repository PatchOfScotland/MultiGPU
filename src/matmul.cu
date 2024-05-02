#include <sys/time.h>

#include "matmul/cpu.h"
#include "matmul/singleGPU.h"
#include "shared_cuda.cu.h"
#include "shared.h"

typedef float array_type;

int main(int argc, char** argv){
    if (argc < 5)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array A height> <array A width> <array B width> <benchmark repeats>-v(validation) -s(standalone) -r(reduced output)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned int height_A = strtoul(argv[1], NULL, 0);
    unsigned int width_A = strtoul(argv[2], NULL, 0);
    unsigned int width_B = strtoul(argv[3], NULL, 0);
    unsigned int height_B = width_A;
    unsigned int runs = atoi(argv[4]);
    bool validating = false;
    bool standalone = false;
    bool reduced_output = false;

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
        if (strcmp(argv[i], "-s") == 0) {
            standalone = true;
        }
        if (strcmp(argv[i], "-r") == 0) {
            reduced_output = true;
        }
    }

    unsigned long int size_A = height_A * width_A;
    unsigned long int size_B = height_B * width_B;
    unsigned long int size_result = width_A * height_B;

    double datasize = ((size_A + size_B + size_result)*sizeof(array_type)/1e9);
    std::cout << "Multiplying arrays of size " 
              << height_A
              << "x"
              << width_A
              << " and "
              << height_B
              << "x"
              << width_B
              << ", resulting in "
              << width_A
              << "x"
              << height_B
              << " (" 
              << datasize 
              <<"GB)\n";
    if (validating) {
        std::cout << "Will validate output\n";
    }
    else {
        std::cout << "Skipping output validation\n";
    }
    if (standalone) {
        std::cout << "Creating new datasets for each run\n";
    }

    array_type* input_array_A;
    array_type* input_array_B;
    array_type* result_array;
    
    float cpu_time_ms = -1;
    float single_gpu_time_ms = -1;
    float multi_gpu_time_ms = -1;

    CCC(cudaMallocManaged(&input_array_A, size_A*sizeof(array_type)));
    CCC(cudaMallocManaged(&input_array_B, size_B*sizeof(array_type)));
    CCC(cudaMallocManaged(&result_array, size_result*sizeof(array_type)));
    init_matrix<array_type>(input_array_A, size_A);
    init_matrix<array_type>(input_array_B, size_B);

    float* timing_ms = (float*)calloc(runs, sizeof(float));

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    check_device_count();

    { // Get CPU baseline
        std::cout << "Getting CPU result\n";

        std::cout << "Input A: \n";
        print_matrix(input_array_A, height_A, width_A);
        std::cout << "Input B: \n";
        print_matrix(input_array_B, height_B, width_B);

        cpu_time_ms = cpuMatMul<array_type>(
            input_array_A, width_A, height_A, 
            input_array_B, width_B, height_B, 
            result_array
        );

        std::cout << "Result: \n";
        print_matrix(result_array, height_B, width_A);

        if (standalone) {
            CCC(cudaFree(input_array_A));
            CCC(cudaFree(input_array_B));
            CCC(cudaFree(result_array));
        }

        std::cout << "CPU matrix multiplication took: " << cpu_time_ms << "ms\n";
        std::cout << "CPU throughput:     " << (float)datasize / cpu_time_ms << "GB/sec\n";
    }

    { // Benchmark a single GPU
        std::cout << "\nBenchmarking single GPU matrix multiplication *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array_A, size_A*sizeof(array_type)));
            CCC(cudaMallocManaged(&input_array_B, size_B*sizeof(array_type)));
            CCC(cudaMallocManaged(&result_array, size_result*sizeof(array_type)));
            init_matrix<array_type>(input_array_A, size_A);
            init_matrix<array_type>(input_array_B, size_B);
        }

        singleGpuMatMul<0, 1, array_type, 16>(
            input_array_A, width_A, height_A, 
            input_array_B, width_B, height_B, 
            result_array
        );

        if (standalone) {
            CCC(cudaFree(input_array_A));
            CCC(cudaFree(input_array_B));
            CCC(cudaFree(result_array));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array_A, size_A*sizeof(array_type)));
                CCC(cudaMallocManaged(&input_array_B, size_B*sizeof(array_type)));
                CCC(cudaMallocManaged(&result_array, size_result*sizeof(array_type)));
                init_matrix<array_type>(input_array_A, size_A);
                init_matrix<array_type>(input_array_B, size_B);
            }

            std::cout << "Input A: \n";
            print_matrix(input_array_A, height_A, width_A);
            std::cout << "Input B: \n";
            print_matrix(input_array_B, height_B, width_B);

            timing_ms[run] = singleGpuMatMul<0, 1, array_type, 16>(
                input_array_A, width_A, height_A, 
                input_array_B, width_B, height_B, 
                result_array
            );

            std::cout << "Result: \n";
            print_matrix(result_array, height_B, width_A);

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize as crude tolerance for now.
            if (validating && run==runs-1) {
                if(cpuValidation<array_type>(
                    input_array_A, width_A, height_A, 
                    input_array_B, width_B, height_B, 
                    result_array, datasize
                )){
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                }
            }

            if (standalone) {
                CCC(cudaFree(input_array_A));
                CCC(cudaFree(input_array_B));
                CCC(cudaFree(result_array));
            }
        }

        single_gpu_time_ms = print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms, 
            multi_gpu_time_ms
        );
    }
}