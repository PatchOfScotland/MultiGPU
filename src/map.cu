#include <functional>
#include <sys/time.h>

#include "map/cpu.h"
#include "map/multiGPU.h"
#include "map/multiGPUstreams.h"
#include "map/singleGPU.h"
#include "shared_cuda.cu.h"
#include "shared.h"

typedef float array_type;

template<typename T>
class PlusX {
    public:
        typedef T InputElement;
        typedef T X;
        typedef T ReturnElement;

        static __device__ __host__ ReturnElement apply(
            const InputElement i, const X x
        ) {
            return i+x;
        };
};


int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array length> <benchmark repeats> -d(devices) <devices> -v(validate) -r(reduced output) -s(standalone timings)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned long int array_len = strtoul(argv[1], NULL, 0);
    unsigned int runs = atoi(argv[2]);
    bool validating = false;
    bool reduced_output = false;
    bool standalone = false;

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int devices;
    CCC(cudaGetDeviceCount(&devices));

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
        if (strcmp(argv[i], "-r") == 0) {
            reduced_output = true;
        }
        if (strcmp(argv[i], "-s") == 0) {
            standalone = true;
        }
        if ((strcmp(argv[i], "-d") == 0 ) && (i+1<argc)) {
            devices = atoi(argv[i+1]);
        }
    }

    print_device_info(devices);
    initialise_hardware();

    unsigned long int datasize_bytes = (unsigned long int)((array_len*2*sizeof(array_type)));
    unsigned long int operations = (unsigned long int)array_len;
    std::cout << "Running array of length " 
              << array_len 
              << " (" 
              << datasize_bytes /1e9
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
    array_type* output_array;
    array_type constant = 0.1;

    struct timing_stat cpu_time = timing_stat("CPU", operations, datasize_bytes);
    struct timing_stat single_gpu_time = timing_stat("single GPU", operations, datasize_bytes);
    struct timing_stat multi_gpu_time = timing_stat("multi GPU", operations, datasize_bytes);
    struct timing_stat stream_gpu_time = timing_stat("multi GPU with streams", operations, datasize_bytes);
    struct timing_stat hint_gpu_time = timing_stat("multi GPU with hints", operations, datasize_bytes);
    const struct timing_stat* all_timings[] = {
        &cpu_time,
        &single_gpu_time,
        &multi_gpu_time,
        &stream_gpu_time,
        &hint_gpu_time
    };
    int timings = sizeof(all_timings)/sizeof(all_timings[0]);

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
    CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
    init_sparse_array(input_array, array_len, 10000);

    float* timing_ms = (float*)calloc(runs, sizeof(float));

    if (true) { // Get CPU baseline
        std::cout << "Getting CPU baseline\n";

        cpu_time.timing_microseconds = cpuMapping<PlusX<array_type>>(
            input_array, constant, output_array, array_len
        );    

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output_array));
        }

        std::cout << "CPU mapping took: " << cpu_time.timing_microseconds / 1e3 << "ms\n";
    }

    if (true) { // Benchmark a single GPU
        std::cout << "\nBenchmarking single GPU map ***********************\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        singleGpuMapping<PlusX<array_type>>(
            input_array, constant, output_array, array_len
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output_array));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = singleGpuMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(cpuValidation<PlusX<array_type>>(
                    input_array, constant, output_array, array_len
                )){
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
            }

            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output_array));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &single_gpu_time, all_timings, timings
        );
    }

    if (true) { // Benchmark multiple GPUs
        std::cout << "\nBenchmarking multi GPU map ************************\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        multiGpuMapping<PlusX<array_type>>(
            input_array, constant, output_array, array_len, devices
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output_array));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = multiGpuMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len, devices
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(cpuValidation<PlusX<array_type>>(
                        input_array, constant, output_array, array_len
                )){
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
            }

            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output_array));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &multi_gpu_time, all_timings, timings
        );
    }

    if (true) { // Benchmark multiple GPUs w/ Streams
        std::cout << "\nBenchmarking multi GPU w/ Steams map **************\n";

        cudaStream_t* streams = (cudaStream_t*)calloc(
            devices, sizeof(cudaStream_t)
        );
        for (int device=0; device<devices; device++) {
            CCC(cudaSetDevice(device));
            cudaStreamCreate(&streams[device]);
        }
        CCC(cudaSetDevice(origin_device));

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        multiGpuStreamMapping<PlusX<array_type>>(
            input_array, constant, output_array, array_len, streams, 
            devices
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output_array));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = multiGpuStreamMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len, streams, 
                devices
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(cpuValidation<PlusX<array_type>>(
                    input_array, constant, output_array, array_len
                )){
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
            }

            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output_array));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &stream_gpu_time, all_timings, timings
        );

        free(streams);
    }

    if (true) { // Benchmark multiple GPUs with hints
        std::cout << "\nBenchmarking multi GPU map with hints *************\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        std::cout << "  Running a warmup\n";
        multiGpuMapping<PlusX<array_type>>(
            input_array, constant, output_array, array_len, devices, true
        );

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output_array));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
                CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
                init_sparse_array(input_array, array_len, 10000);
            }

            timing_ms[run] = multiGpuMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len, devices, true
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(cpuValidation<PlusX<array_type>>(
                    input_array, constant, output_array, array_len
                )){
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
            }

            if (standalone) {
                CCC(cudaFree(input_array));
                CCC(cudaFree(output_array));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &hint_gpu_time, all_timings, timings
        );
    }

    if (standalone == false) {
        CCC(cudaFree(input_array));
        CCC(cudaFree(output_array));
    }

    std::cout << "\n";
}
