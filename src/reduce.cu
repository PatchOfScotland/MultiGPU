#include <functional>
#include <sys/time.h>

#include "reduce/cpu.h"
#include "reduce/multiGPU.h"
#include "reduce/multiGPUstreams.h"
#include "reduce/singleGPU.h"
#include "shared.cu.h"
#include "shared.h"

typedef float array_type;

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
    array_type constant = 0.1;
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

    cudaFree(input_array);
    cudaFree(output);
}
