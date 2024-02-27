#include <functional>

#include "map/cpu.h"
#include "map/multiGPU.h"
#include "map/multiGPUstreams.h"
#include "map/singleGPU.h"
#include "shared.cu.h"
#include "shared.h"

typedef float arrayType;

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

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
    }

    double datasize = ((array_len*2*sizeof(arrayType))/1e9);
    std::cout << "Running array of length " 
              << array_len 
              << " (" 
              << datasize 
              <<"GB) and " 
              << stream_count
              <<" stream\n";
    if (validating) {
        std::cout << "Will validate output\n";
    }
    else {
        std::cout << "Skipping output validation\n";
    }

    arrayType* input_array;
    arrayType* output_array;
    arrayType* validation_array;
    arrayType constant = 0.1;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float runtime;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(arrayType)));
    CCC(cudaMallocManaged(&output_array, array_len*sizeof(arrayType)));

    CCC(cudaEventCreate(&start_event));
    CCC(cudaEventCreate(&end_event));
    float* single_gpu_ms = (float*)calloc(runs, sizeof(float));
    float* multi_gpu_ms = (float*)calloc(runs, sizeof(float));
    float* multi_gpu_stream_ms = (float*)calloc(runs, sizeof(float));

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    init_array(input_array, array_len);

    if (validating) { // Populate validation array
        validation_array = (arrayType*)malloc(array_len*sizeof(arrayType));
        cpuMapping(
            PlusConst<arrayType>, input_array, constant, validation_array, 
            array_len
        );
    }

    check_device_count();

    { // Benchmark a single GPU
        std::cout << "*** Benchmarking single GPU map ***\n";

        std::cout << "  Running a warmup\n";
        singleGpuMapping(
            singleGpuKernel<arrayType>, input_array, constant, output_array, 
            array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            singleGpuMapping(
                singleGpuKernel<arrayType>, input_array, constant, 
                output_array, array_len
            );
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event)); 

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            single_gpu_ms[run] = runtime;

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

        float mean = 0;
        float total = 0;
        get_timing_stats(single_gpu_ms, runs, &total, &mean);
        float gigabytes_per_second = (float)datasize / (mean * 1e-3f);
        std::cout << "    Total runtime: " << total <<"ms\n";
        std::cout << "    Mean runtime:  " << mean <<"ms\n";
        std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/s\n";
    }

    { // Benchmark multiple GPUs
        std::cout << "*** Benchmarking multi GPU map ***\n";

        std::cout << "  Running a warmup\n";
        multiGpuMapping(
            multiGpuKernel<arrayType>, input_array, constant, output_array, 
            array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));        

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuMapping(
                multiGpuKernel<arrayType>, input_array, constant, output_array,
                array_len
            );
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event)); 

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            multi_gpu_ms[run] = runtime;

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

        float mean = 0;
        float total = 0;
        get_timing_stats(multi_gpu_ms, runs, &total, &mean);
        float gigabytes_per_second = (float)datasize / (mean * 1e-3f);
        std::cout << "    Total runtime: " << total <<"ms\n";
        std::cout << "    Mean runtime:  " << mean <<"ms\n";
        std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/s\n";
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
        multiGpuStreamMapping(multiGpuStreamKernel<arrayType>, input_array, constant, output_array, array_len, streams, stream_count);
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));        

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuStreamMapping(multiGpuStreamKernel<arrayType>, input_array, constant, output_array, array_len, streams, stream_count);
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event)); 

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            multi_gpu_stream_ms[run] = runtime;

            print_loop_feedback(run, runs);

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(compare_arrays(validation_array, output_array, array_len)){
                    std::cout << "  Result is correct\n";
                } else {
                    std::cout << "  Result is incorrect. Skipping any subsequent runs\n";
                    break;
                }
            }
        }

        float mean = 0;
        float total = 0;
        get_timing_stats(multi_gpu_stream_ms, runs, &total, &mean);
        float gigabytes_per_second = (float)datasize / (mean * 1e-3f);
        std::cout << "    Total runtime: " << total <<"ms\n";
        std::cout << "    Mean runtime:  " << mean <<"ms\n";
        std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/s\n";

        free(streams);
    }

    { // Benchmark multiple GPUs with hints
        std::cout << "*** Benchmarking multi GPU map with hints ***\n";

        //hint1D<arrayType>(input_array, 1024, array_len);
        //hint1D<arrayType>(output_array, 1024, array_len);

        int origin_device;
        CCC(cudaGetDevice(&origin_device));
        int device_count;
        CCC(cudaGetDeviceCount(&device_count));

        for (int device=0; device<device_count; device++) {
            //CCC(cudaMemAdvise(input_array+, X, X, device));
            //CCC(cudaMemAdvise(output_array+, X, X, device));
        }

        std::cout << "  Running a warmup\n";
        multiGpuMapping(
            multiGpuKernel<arrayType>, input_array, constant, output_array, 
            array_len
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));        

        for (int run=0; run<runs; run++) {
            CCC(cudaEventRecord(start_event));
            multiGpuMapping(
                multiGpuKernel<arrayType>, input_array, constant, output_array,
                array_len
            );
            CCC(cudaEventRecord(end_event));
            CCC(cudaEventSynchronize(end_event)); 

            CCC(cudaEventElapsedTime(&runtime, start_event, end_event));
            multi_gpu_ms[run] = runtime;

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

        float mean = 0;
        float total = 0;
        get_timing_stats(multi_gpu_ms, runs, &total, &mean);
        float gigabytes_per_second = (float)datasize / (mean * 1e-3f);
        std::cout << "    Total runtime: " << total <<"ms\n";
        std::cout << "    Mean runtime:  " << mean <<"ms\n";
        std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/s\n";
    }

    if (validating) {
            free(validation_array);
        }
    cudaFree(input_array);
    cudaFree(output_array);
}
