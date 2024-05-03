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
        static const bool commutative = true;

        static __device__ __host__ ReturnElement apply(
            const InputElement i, const ReturnElement r
        ) {
            return i+r;
        };

        static __device__ __host__ ReturnElement init () {
            return (ReturnElement)0;
        }

        static __device__ __host__ ReturnElement map (
            const InputElement &i
        ) {
            return (ReturnElement)i;
        }

        static __device__ __host__ ReturnElement remVolatile (
            volatile ReturnElement &i
        ) {
            return i;
        }
};

template<typename I, typename R>
class AddNonCommutative {
    public:
        typedef I InputElement;
        typedef R ReturnElement;
        static const bool commutative = false;

        static __device__ __host__ ReturnElement apply(
            const InputElement i, const ReturnElement r
        ) {
            return i+r;
        };

        static __device__ __host__ ReturnElement init () {
            return (ReturnElement)0;
        }

        static __device__ __host__ ReturnElement map (
            const InputElement &i
        ) {
            return (ReturnElement)i;
        }

        static __device__ __host__ ReturnElement remVolatile (
            volatile ReturnElement &i
        ) {
            return i;
        }
};


int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array length> <benchmark repeats>-v(validation) -s(standalone) -r(reduced output)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned long int array_len = strtoul(argv[1], NULL, 0);
    unsigned int runs = atoi(argv[2]);
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

    unsigned long int datasize_bytes = (unsigned long int)((array_len*2*sizeof(array_type)));
    unsigned long int operations = (unsigned long int)array_len;
    std::cout << "Running array of length " 
              << array_len 
              << " (" 
              << datasize_bytes / 1e9 
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

    array_type* input_array;
    return_type* output;
    return_type validation_result;

    struct timing_stat cpu_time = timing_stat("CPU", operations, datasize_bytes);
    struct timing_stat commutative_single_gpu_time = timing_stat("commutative single GPU", operations, datasize_bytes);
    struct timing_stat commutative_multi_gpu_time = timing_stat("commutative multi GPU", operations, datasize_bytes);
    struct timing_stat commutative_hint_gpu_time = timing_stat("commutative multi GPU with hints", operations, datasize_bytes);
    struct timing_stat associative_single_gpu_time = timing_stat("associative single GPU", operations, datasize_bytes);
    struct timing_stat associative_multi_gpu_time = timing_stat("associative multi GPU", operations, datasize_bytes);
    struct timing_stat associative_hint_gpu_time = timing_stat("associative multi GPU with hints", operations, datasize_bytes);
    const struct timing_stat* all_timings[] = {
        &cpu_time,
        &commutative_single_gpu_time,
        &commutative_multi_gpu_time,
        &commutative_hint_gpu_time,
        &associative_single_gpu_time,
        &associative_multi_gpu_time,
        &associative_hint_gpu_time
    };
    int timings = sizeof(all_timings)/sizeof(all_timings[0]);

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
    CCC(cudaMallocManaged(&output, sizeof(return_type)));
    init_sparse_array(input_array, array_len, 10000);

    float* timing_ms = (float*)calloc(runs, sizeof(float));

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    check_device_count();

    { // Get CPU baseline
        std::cout << "Getting CPU result\n";

        cpu_time.timing_microseconds = cpuReduction<Add<array_type,return_type>>(
            input_array, &validation_result, array_len
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output));
        }

        std::cout << "CPU reduction took: " << cpu_time.timing_microseconds / 1e3 << "ms\n";
        std::cout << "CPU throughput:     " << cpu_time.throughput_gb() << "GB/sec\n";
    }

    { // Benchmark commutative single GPU
        std::cout << "\nBenchmarking commutative single GPU reduce ********\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output, sizeof(return_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        singleGpuReduction<Add<array_type,return_type>>(
            input_array, output, array_len
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output, sizeof(return_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = singleGpuReduction<Add<array_type,return_type>>(
                input_array, output, array_len
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

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

            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &commutative_single_gpu_time, all_timings, timings
        );
    }

    { // Benchmark commutative multi GPU
        std::cout << "\nBenchmarking commutative multi GPU reduce *********\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output, sizeof(return_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        std::cout << "  Running a warmup\n";
        multiGpuReduction<Add<array_type,return_type>>(
            input_array, output, array_len
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output, sizeof(return_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = multiGpuReduction<Add<array_type,return_type>>(
                input_array, output, array_len
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

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
            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &commutative_multi_gpu_time, all_timings, timings
        );
    }

    { // Benchmark commutative multi GPU with hints
        std::cout << "\nBenchmarking commutative multi GPU reduce with hints\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output, sizeof(return_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        std::cout << "  Running a warmup\n";
        multiGpuReduction<Add<array_type,return_type>>(
            input_array, output, array_len, true
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output, sizeof(return_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = multiGpuReduction<Add<array_type,return_type>>(
                input_array, output, array_len, true
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

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

            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &commutative_hint_gpu_time, all_timings, timings
        );
    }

    { // Benchmark associative single GPU
        std::cout << "\nBenchmarking associative single GPU reduce ****\n";


        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output, sizeof(return_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        std::cout << "  Running a warmup\n";
        singleGpuReduction<AddNonCommutative<array_type,return_type>>(
            input_array, output, array_len
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output, sizeof(return_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = singleGpuReduction<AddNonCommutative<array_type,return_type>>(
                input_array, output, array_len
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

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

            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &associative_single_gpu_time, all_timings, timings
        );
    }

    { // Benchmark associative multi GPU
        std::cout << "\nBenchmarking associative multi GPU reduce *****\n";
        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output, sizeof(return_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        std::cout << "  Running a warmup\n";
        multiGpuReduction<AddNonCommutative<array_type,return_type>>(
            input_array, output, array_len
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output, sizeof(return_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = multiGpuReduction<AddNonCommutative<array_type,return_type>>(
                input_array, output, array_len
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

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

            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &associative_multi_gpu_time, all_timings, timings
        );
    }

    { // Benchmark associative multi GPU with hints
        std::cout << "\nBenchmarking associative multi GPU reduce with hints\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output, sizeof(return_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        std::cout << "  Running a warmup\n";
        multiGpuReduction<AddNonCommutative<array_type,return_type>>(
            input_array, output, array_len, true
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output));
        }
        
        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output, sizeof(return_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = multiGpuReduction<AddNonCommutative<array_type,return_type>>(
                input_array, output, array_len, true
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

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
            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &associative_hint_gpu_time, all_timings, timings
        );
    }

    if (standalone == false) {
        CCC(cudaFree(input_array));
        CCC(cudaFree(output));
    }
}
