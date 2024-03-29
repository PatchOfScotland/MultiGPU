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

    array_type* input_array;
    array_type* output_array;
    array_type* validation_array;
    array_type constant = 0.1;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float runtime_ms;
    float cpu_time_ms = -1;
    float single_gpu_time_ms = -1;
    float multi_gpu_time_ms = -1;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(array_type)));
    CCC(cudaMallocManaged(&output_array, array_len*sizeof(array_type)));

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
        validation_array = (array_type*)malloc(array_len*sizeof(array_type));
        cpuMapping(
            PlusConst<array_type>, input_array, constant, validation_array, 
            array_len
        );    
        gettimeofday(&cpu_end_time, NULL); 

        cpu_time_ms = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
        std::cout << "CPU mapping took: " << cpu_time_ms << "ms\n";
    }

    check_device_count();

    { // Benchmark a single GPU
        std::cout << "\nBenchmarking single GPU map ***********************\n";

        std::cout << "  Running a warmup\n";
        singleGpuMapping<PlusX<array_type>>(
            input_array, constant, output_array, array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            singleGpuMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len
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
                if(compare_arrays(validation_array, output_array, array_len)){
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

    { // Benchmark multiple GPUs
        std::cout << "\nBenchmarking multi GPU map ************************\n";

        std::cout << "  Running a warmup\n";
        multiGpuMapping<PlusX<array_type>>(
            input_array, constant, output_array, array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len
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
                if(compare_arrays(validation_array, output_array, array_len)){
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
        multiGpuStreamMapping<PlusX<array_type>>(
            input_array, constant, output_array, array_len, streams, 
            device_count
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuStreamMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len, streams, 
                device_count
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
                if(compare_arrays(validation_array, output_array, array_len)){
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
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

        size_t block_count = (array_len + block_size - 1) / block_size;
        size_t dev_block_count = (block_count + device_count - 1) / device_count;

        unsigned long int array_offset = 0;
        unsigned long int block_range;
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
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuMapping<PlusX<array_type>>(
                input_array, constant, output_array, array_len
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
                if(compare_arrays(validation_array, output_array, array_len)){
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any "
                              << "subsequent runs\n";
                    break;
                }
            }
        }

        print_timing_stats(
            timing_ms, runs, datasize, cpu_time_ms, single_gpu_time_ms,
            multi_gpu_time_ms
        );
    }

    std::cout << "\n";

    if (validating) {
        free(validation_array);
    }
    cudaFree(input_array);
    cudaFree(output_array);
}
