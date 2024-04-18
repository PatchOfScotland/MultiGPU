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
    unsigned long int size_result = height_A * height_B;

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

        cpu_time_ms = cpuMatMul<array_type>(
            input_array_A, width_A, height_A, 
            input_array_B, width_B, height_B, 
            result_array
        );

        if (standalone) {
            CCC(cudaFree(input_array_A));
            CCC(cudaFree(input_array_B));
            CCC(cudaFree(result_array));
        }

        std::cout << "CPU matrix multiplication took: " << cpu_time_ms << "ms\n";
        std::cout << "CPU throughput:     " << (float)datasize / cpu_time_ms << "GB/sec\n";
    }
}