#include <functional>
#include <sys/time.h>

#include "map/cpu.h"
#include "map/multiGPU.h"
#include "map/multiGPUstreams.h"
#include "map/singleGPU.h"
#include "shared.cu.h"
#include "shared.h"

typedef float array_type;

int main(int argc, char** argv){
    if (argc < 4)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array length> <stream count> <benchmark repeats> "
                  << "-v(optional)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned long int array_len = strtoul(argv[1], NULL, 0);
    unsigned int stream_count = atoi(argv[2]);
    unsigned int runs = atoi(argv[3]);
    bool validating = false;
    struct timeval start_time;
    struct timeval end_time;

    gettimeofday(&start_time, NULL); 

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
              <<"GB) and " 
              << stream_count
              <<" streams\n";
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
    float runtime;

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

    gettimeofday(&end_time, NULL); 

    long int setup_time = end_time.tv_sec - start_time.tv_sec;
    std::cout << "Setup took " << setup_time << " seconds\n";

    if (validating) { // Populate validation array
        std::cout << "Getting CPU result for validation\n";
        validation_array = (array_type*)malloc(array_len*sizeof(array_type));
        cpuMapping(
            PlusConst<array_type>, input_array, constant, validation_array, 
            array_len
        );
    }

    check_device_count();

    { // Benchmark a single GPU
        std::cout << "*** Benchmarking single GPU map ***\n";

        std::cout << "  Running a warmup\n";
        singleGpuMapping(
            singleGpuKernel<array_type>, input_array, constant, output_array, 
            array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            singleGpuMapping(
                singleGpuKernel<array_type>, input_array, constant, 
                output_array, array_len
            );
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event));
            if (cuda_assert(cudaPeekAtLastError())) {
                std::cout << "\n";
                break;
            }

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            timing_ms[run] = runtime;

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

        print_timing_stats(timing_ms, runs, datasize);
    }

    { // Benchmark multiple GPUs
        std::cout << "*** Benchmarking multi GPU map ***\n";

        std::cout << "  Running a warmup\n";
        multiGpuMapping(
            multiGpuKernel<array_type>, input_array, constant, output_array, 
            array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuMapping(
                multiGpuKernel<array_type>, input_array, constant, output_array,
                array_len
            );
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event));
            if (cuda_assert(cudaPeekAtLastError())) {
                std::cout << "\n";
                break;
            }

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            timing_ms[run] = runtime;

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

        print_timing_stats(timing_ms, runs, datasize);
    }

    { // Benchmark multiple GPUs w/ Streams
        std::cout << "*** Benchmarking multi GPU w/ Steams map ***\n";

        cudaStream_t* streams = (cudaStream_t*)calloc(
            stream_count*device_count, sizeof(cudaStream_t)
        );
        for (int device=0; device<device_count; device++) {
            CCC(cudaSetDevice(device));
            for(int s = device*stream_count; s<(device+1)*stream_count; s++) {
                cudaStreamCreate(&streams[s]);
            }
        }
        CCC(cudaSetDevice(origin_device));

        std::cout << "  Running a warmup\n";
        multiGpuStreamMapping(
            multiGpuStreamKernel<array_type>, input_array, constant, 
            output_array, array_len, streams, stream_count
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuStreamMapping(
                multiGpuStreamKernel<array_type>, input_array, constant, 
                output_array, array_len, streams, stream_count
            );
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event)); 
            if (cuda_assert(cudaPeekAtLastError())) {
                std::cout << "\n";
                break;
            }

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            timing_ms[run] = runtime;

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

        print_timing_stats(timing_ms, runs, datasize);

        free(streams);
    }

    { // Benchmark multiple GPUs with hints
        std::cout << "*** Benchmarking multi GPU map with hints ***\n";

        int origin_device;
        CCC(cudaGetDevice(&origin_device));
        int device_count;
        CCC(cudaGetDeviceCount(&device_count));

        size_t block_count = (array_len + block_size - 1) / block_size;
        size_t dev_block_count = (block_count + device_count - 1) / device_count;

        size_t array_offset = 0;
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
        multiGpuMapping(
            multiGpuKernel<array_type>, input_array, constant, output_array, 
            array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuMapping(
                multiGpuKernel<array_type>, input_array, constant, output_array,
                array_len
            );
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event));
            if (cuda_assert(cudaPeekAtLastError())) {
                std::cout << "\n";
                break;
            }

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            timing_ms[run] = runtime;

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

        print_timing_stats(timing_ms, runs, datasize);
    }

    if (validating) {
            free(validation_array);
        }
    cudaFree(input_array);
    cudaFree(output_array);
}
