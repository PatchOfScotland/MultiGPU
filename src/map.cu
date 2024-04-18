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
                  << " <array length> <benchmark repeats> -v(validate) -r(reduced output) -s(standalone timings)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned long int array_len = strtoul(argv[1], NULL, 0);
    unsigned int runs = atoi(argv[2]);
    bool validating = false;
    bool reduced_output = false;
    bool standalone = false;

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
    }

    double datasize = ((array_len*2*sizeof(array_type))/1e9);
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
    if (standalone) {
        std::cout << "Creating new datasets for each run\n";
    }

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    array_type* input_array;
    array_type* output_array;
    array_type constant = 0.1;

    float cpu_time_ms = -1;
    float single_gpu_time_ms = -1;
    float multi_gpu_time_ms = -1;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
    CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
    init_sparse_array(input_array, array_len, 10000);

    float* timing_ms = (float*)calloc(runs, sizeof(float));

    initialise_hardware();
    check_device_count();

    { // Get CPU baseline
        std::cout << "Getting CPU baseline\n";

        cpu_time_ms = cpuMapping(
            PlusConst<array_type>, input_array, constant, output_array, 
            array_len
        );    

        if (standalone) {
            CCC(cudaFree(input_array));
            CCC(cudaFree(output_array));
        }

        std::cout << "CPU mapping took: " << cpu_time_ms << "ms\n";
    }

    { // Benchmark a single GPU
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
                if(cpuValidation(
                    PlusConst<array_type>, input_array, constant, 
                    output_array, array_len)
                ){
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

         single_gpu_time_ms = print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms, 
            multi_gpu_time_ms
        );
    }

    { // Benchmark multiple GPUs
        std::cout << "\nBenchmarking multi GPU map ************************\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        multiGpuMapping<PlusX<array_type>>(
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

            timing_ms[run] = multiGpuMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(cpuValidation(
                    PlusConst<array_type>, input_array, constant, 
                    output_array, array_len)
                ){
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

        multi_gpu_time_ms = print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms, 
            multi_gpu_time_ms
        );
    }

    { // Benchmark multiple GPUs w/ Streams
        std::cout << "\nBenchmarking multi GPU w/ Steams map **************\n";

        cudaStream_t* streams = (cudaStream_t*)calloc(
            device_count, sizeof(cudaStream_t)
        );
        for (int device=0; device<device_count; device++) {
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
            device_count
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
                device_count
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(cpuValidation(
                    PlusConst<array_type>, input_array, constant, 
                    output_array, array_len)
                ){
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

        print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms,
            multi_gpu_time_ms
        );

        free(streams);
    }

    { // Benchmark multiple GPUs with hints
        std::cout << "\nBenchmarking multi GPU map with hints *************\n";

        int origin_device;
        CCC(cudaGetDevice(&origin_device));
        int device_count;
        CCC(cudaGetDeviceCount(&device_count));

        size_t block_count = (array_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t dev_block_count = (block_count + device_count - 1) / device_count;

        unsigned long int array_offset = 0;
        unsigned long int block_range;

        if (standalone) {
            CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
            CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));
            init_sparse_array(input_array, array_len, 10000);
        }

        for (int device=0; device<device_count; device++) {
            block_range = min(dev_block_count, array_len-array_offset);

            CCC(cudaMemAdvise(
                input_array+array_offset, 
                block_range*sizeof(array_type), 
                cudaMemAdviseSetPreferredLocation, 
                device
            ));
            CCC(cudaMemAdvise(
                output_array+array_offset, 
                block_range*sizeof(array_type), 
                cudaMemAdviseSetPreferredLocation, 
                device
            ));
            array_offset = array_offset + block_range;
        }

        std::cout << "  Running a warmup\n";
        multiGpuMapping<PlusX<array_type>>(
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

            for (int device=0; device<device_count; device++) {
                block_range = min(dev_block_count, array_len-array_offset);

                CCC(cudaMemAdvise(
                    input_array+array_offset, 
                    block_range*sizeof(array_type), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
                CCC(cudaMemAdvise(
                    output_array+array_offset, 
                    block_range*sizeof(array_type), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
                array_offset = array_offset + block_range;
            }

            timing_ms[run] = multiGpuMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(cpuValidation(
                    PlusConst<array_type>, input_array, constant, 
                    output_array, array_len)
                ){
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

        print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms,
            multi_gpu_time_ms
        );
    }

    if (standalone == false) {
        CCC(cudaFree(input_array));
        CCC(cudaFree(output_array));
    }

    std::cout << "\n";
}
