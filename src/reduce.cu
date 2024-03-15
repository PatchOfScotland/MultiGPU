#include <functional>
#include <sys/time.h>

#include "reduce/cpu.h"
#include "reduce/multiGPU.h"
#include "reduce/singleGPU.h"
#include "shared_cuda.cu.h"
#include "shared.h"

typedef float array_type;
typedef double return_type;

template<typename I, typename R>
class Add {
    public:
        typedef I InputElement;
        typedef R ReturnElement;

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
                  << " <array length> <benchmark repeats> -v(optional) -s(optional)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned long int array_len = strtoul(argv[1], NULL, 0);
    unsigned int runs = atoi(argv[2]);
    bool validating = false;
    bool skip = false;

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
        if (strcmp(argv[i], "-s") == 0) {
            skip = true;
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
    if (skip) {
        std::cout << "Skipping any significant processing\n";
    }

    array_type* input_array;
    return_type* output;
    return_type validation_result;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float runtime_ms;
    float cpu_time_ms = -1;
    float single_gpu_time_ms = -1;
    float multi_gpu_time_ms = -1;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
    CCC(cudaMallocManaged(&output, sizeof(return_type)));

    CCC(cudaEventCreate(&start_event));
    CCC(cudaEventCreate(&end_event));
    float* timing_ms = (float*)calloc(runs, sizeof(float));

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    std::cout << "Initialising input array\n";
    if (skip == false) {
        init_array(input_array, array_len);
    }

    if (validating) { // Populate validation array
        std::cout << "Getting CPU result for validation\n";

        struct timeval cpu_start_time;
        struct timeval cpu_end_time;

        gettimeofday(&cpu_start_time, NULL);

        if (skip == false) {
            cpuReduction(
                reduction<array_type,return_type>, input_array, 
                &validation_result, array_len
            );    
        }
        gettimeofday(&cpu_end_time, NULL); 

        cpu_time_ms = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
        std::cout << "CPU reduction took: " << cpu_time_ms << "ms\n";
    }

    check_device_count();

    { // Benchmark a single GPU
        std::cout << "\nBenchmarking single GPU reduce ********************\n";

        std::cout << "  Running a warmup\n";
        singleGpuReduction<Add<array_type,return_type>>(
            input_array, output, array_len, skip
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            singleGpuReduction<Add<array_type,return_type>>(
                input_array, output, array_len, skip
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
                if (in_range<double>(validation_result, *output, tolerance)) {
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

    { // Benchmark multi GPU
        std::cout << "\nBenchmarking multi GPU reduce *********************\n";

        unsigned long int per_device = array_len / device_count;
        int remainder = array_len % device_count;
        unsigned long int running_total = 0;
        unsigned long int device_start;
        unsigned long int this_block;
        for (int device=0; device<device_count; device++) {           
            device_start = running_total;
            this_block = (remainder > 0) ? per_device + 1 : per_device;
            remainder -= 1;
            running_total += this_block;
            
            std::cout << "  A:" << input_array+device_start << "\n";
            std::cout << "  B:" << this_block*sizeof(array_type) << "\n";
            std::cout << "  B.5:" << this_block << "\n";
            std::cout << "  C:" << cudaMemAdviseSetPreferredLocation << "\n";
            std::cout << "  D:" << device << "\n";

            CCC(cudaMemAdvise(
                input_array+device_start, 
                this_block*sizeof(array_type), 
                cudaMemAdviseSetPreferredLocation, 
                device
            ));
        }

        std::cout << "  Running a warmup\n";
        multiGpuReduction<Add<array_type,return_type>>(
            input_array, output, array_len, skip
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));


        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuReduction<Add<array_type,return_type>>(
                input_array, output, array_len, skip
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
                if (in_range<double>(validation_result, *output, tolerance)) {
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
            }
        }

         multi_gpu_time_ms = print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms, 
            multi_gpu_time_ms
        );
    }

    { // Benchmark multi GPU with hints
        std::cout << "\nBenchmarking multi GPU reduce with hints **********\n";

        std::cout << "  Running a warmup\n";
        multiGpuReduction<Add<array_type,return_type>>(
            input_array, output, array_len, skip
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuReduction<Add<array_type,return_type>>(
                input_array, output, array_len, skip
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
                if (in_range<double>(validation_result, *output, tolerance)) {
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
            }
        }

         multi_gpu_time_ms = print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms, 
            multi_gpu_time_ms
        );
    }

    cudaFree(input_array);
    cudaFree(output);
}
