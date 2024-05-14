#include <sys/time.h>
#include <limits>

#include "matmul/cpu.h"
#include "matmul/tiled.h"
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

    const unsigned int heightA = strtoul(argv[1], NULL, 0);
    const unsigned int widthA = strtoul(argv[2], NULL, 0);
    const unsigned int widthB = strtoul(argv[3], NULL, 0);
    const unsigned int heightB = widthA;
    const unsigned int widthC = widthB;
    const unsigned int heightC = heightA;
    const unsigned int runs = atoi(argv[4]);
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

    const unsigned long int sizeA = widthA * heightA;
    const unsigned long int sizeB = widthB * heightB;
    const unsigned long int sizeC = widthC * heightC;

    unsigned long int datasize_bytes = (unsigned long int)(((((device_count+1)*widthA*heightA)+((device_count+1)*widthB*heightB)+((device_count+1)*widthC*heightC))*sizeof(array_type)));
    unsigned long int operations = (unsigned long int)heightC * widthC * widthA * 2;
    std::cout << "Multiplying arrays of size " 
              << widthA
              << "x"
              << heightA
              << " and "
              << widthB
              << "x"
              << heightB
              << ", resulting in "
              << widthC
              << "x"
              << heightC
              << "\n";
    std::cout << "Using " 
              << datasize_bytes / 1e9
              << "GB of memory and "
              << (float)operations / 1e9
              << " GFLOPs per experiment\n";

    if (validating) {
        std::cout << "Will validate output\n";
    }
    else {
        std::cout << "Skipping output validation\n";
    }
    if (standalone) {
        std::cout << "Creating new datasets for each run\n";
    }

    array_type* matrixA;
    array_type* matrixB;
    array_type* matrixC;
    
    struct timing_stat cpu_time = 
        timing_stat("CPU", operations, datasize_bytes);
    struct timing_stat tiled_single_gpu_time = 
        timing_stat("tiled single GPU", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_raw_time = 
        timing_stat("tiled multi GPU raw", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_raw_hint_time = 
        timing_stat("tiled multi GPU raw w/ hints", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_raw_prefetch_time = 
        timing_stat("tiled multi GPU raw w/ prefetch", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_duplicate_time = 
        timing_stat("tiled multi GPU duplicate", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_duplicate_hint_time = 
        timing_stat("tiled multi GPU duplicate w/ hints", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_duplicate_prefetch_time = 
        timing_stat("tiled multi GPU duplicate w/ prefetch", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_split_time = 
        timing_stat("tiled multi GPU split", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_split_hint_time = 
        timing_stat("tiled multi GPU split w/ hints", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_split_prefetch_time = 
        timing_stat("tiled multi GPU split w/ prefetch", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_split_malloc_time = 
        timing_stat("tiled multi GPU split w/ malloc", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_split_reduce_time = 
        timing_stat("tiled multi GPU split w/ reduction", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_split_reduce_hint_time = 
        timing_stat("tiled multi GPU split w/ reduction and hints", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_split_reduce_prefetch_time = 
        timing_stat("tiled multi GPU split w/ reduction and prefetch", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_split_reduce_malloc_time = 
        timing_stat("tiled multi GPU split w/ reduction and malloc", operations, datasize_bytes);    
    struct timing_stat recursive_single_gpu_time = 
        timing_stat("recursive single GPU", operations, datasize_bytes);
    const struct timing_stat* all_timings[] = {
        &cpu_time,
        &tiled_single_gpu_time,
        &tiled_multi_gpu_raw_time,
        &tiled_multi_gpu_raw_hint_time,
        &tiled_multi_gpu_raw_prefetch_time,
        &tiled_multi_gpu_duplicate_time,
        &tiled_multi_gpu_duplicate_hint_time,
        &tiled_multi_gpu_duplicate_prefetch_time,
        &tiled_multi_gpu_split_time,
        &tiled_multi_gpu_split_hint_time,
        &tiled_multi_gpu_split_prefetch_time,
        &tiled_multi_gpu_split_malloc_time,
        &tiled_multi_gpu_split_reduce_time,
        &tiled_multi_gpu_split_reduce_hint_time,
        &tiled_multi_gpu_split_reduce_prefetch_time,
        &tiled_multi_gpu_split_reduce_malloc_time,
        &recursive_single_gpu_time
    };
    int timings = sizeof(all_timings)/sizeof(all_timings[0]);

    CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
    CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
    CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
    init_matrix<array_type>(matrixA, sizeA);
    init_matrix<array_type>(matrixB, sizeB);

    float* timing_ms = (float*)calloc(runs, sizeof(float));

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    check_device_count();

    if (false) { // Get CPU baseline
        std::cout << "Getting CPU result\n";

        cpu_time.timing_microseconds = cpuMatMul<array_type>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            CCC(cudaFree(matrixC));
        }

        //std::cout << "Input A: \n";
        //print_matrix(matrixA, widthA, heightA);
        //std::cout << "Input B: \n";
        //print_matrix(matrixB, widthB, heightB);
        //std::cout << "Result: \n";
        //print_matrix(matrixC, widthC, heightC);

        std::cout << "CPU matrix multiplication took: " << cpu_time.timing_microseconds / 1e3 << "ms\n";
        std::cout << "CPU throughput:     " << cpu_time.throughput_gb() << "GB/sec\n";
        std::cout << "CPU GFLOPS:         " << cpu_time.throughput_gf() << "ops/sec\n";
    }

    if (true) { // Benchmark a tiled single GPU
        std::cout << "\nBenchmarking tiled single GPU *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }

        tiled::singleGPU<false, false, array_type, 16>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            CCC(cudaFree(matrixC));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                init_matrix<array_type>(matrixA, sizeA);
                init_matrix<array_type>(matrixB, sizeB);
            }

            timing_ms[run] = tiled::singleGPU<false, false, array_type, 16>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC
            );


            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_single_gpu_time, all_timings, timings
        );
    }

    if (false) { // Benchmark a tiled multi GPU raw
        std::cout << "\nBenchmarking tiled multi GPU raw *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }

        tiled::multiGPU<false, false, array_type, 16>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            NO_HINTS
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            CCC(cudaFree(matrixC));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                init_matrix<array_type>(matrixA, sizeA);
                init_matrix<array_type>(matrixB, sizeB);
            }

            timing_ms[run] = tiled::multiGPU<false, false, array_type, 16>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                NO_HINTS
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {

                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_raw_time, all_timings, timings
        );
    }

    if (false) { // Benchmark a tiled multi GPU raw w/ hints
        std::cout << "\nBenchmarking tiled multi GPU raw w/ hints *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }

        tiled::multiGPU<false, false, array_type, 16>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            HINTS
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            CCC(cudaFree(matrixC));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                init_matrix<array_type>(matrixA, sizeA);
                init_matrix<array_type>(matrixB, sizeB);
            }

            timing_ms[run] = tiled::multiGPU<false, false, array_type, 16>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                HINTS
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {

                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_raw_hint_time, all_timings, timings
        );
    }

    if (false) { // Benchmark a tiled multi GPU raw w/ prefetch
        std::cout << "\nBenchmarking tiled multi GPU raw w/ prefetch *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }

        tiled::multiGPU<false, false, array_type, 16>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            PREFETCH
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            CCC(cudaFree(matrixC));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                init_matrix<array_type>(matrixA, sizeA);
                init_matrix<array_type>(matrixB, sizeB);
            }

            timing_ms[run] = tiled::multiGPU<false, false, array_type, 16>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                PREFETCH
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {

                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_raw_prefetch_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU duplicate B
        std::cout << "\nBenchmarking tiled multi GPU duplicate B *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixBs[device_count];
        matrixBs[0] = matrixB;
        for (int i=1; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
        }

        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[0], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixBs[0], sizeB);
        }
        for (int i=1; i<device_count; i++) {
            duplicate_matrix(matrixB, sizeB, matrixBs[i]);
        }

        tiled::multiGPUduplicate<false, false, array_type, 16>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            NO_HINTS
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixBs[i]));
            }   
            CCC(cudaFree(matrixC));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixBs[0], sizeB*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                init_matrix<array_type>(matrixA, sizeA);
                init_matrix<array_type>(matrixBs[0], sizeB);
                for (int i=1; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    duplicate_matrix(matrixBs[0], sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUduplicate<false, false, array_type, 16>(
                matrixA, widthA, heightA, 
                matrixBs, widthB, heightB, 
                matrixC, widthC, heightC,
                NO_HINTS
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {

                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixBs[0], widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixBs[i]));
                } 
                CCC(cudaFree(matrixC));
            }
        }

        if (!standalone) {
            for (int i=1; i<device_count; i++) {
                CCC(cudaFree(matrixBs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_duplicate_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU duplicate B w/ hints
        std::cout << "\nBenchmarking tiled multi GPU duplicate B w/ hints *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixBs[device_count];
        matrixBs[0] = matrixB;
        for (int i=1; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
        }

        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[0], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixBs[0], sizeB);
        }
        for (int i=1; i<device_count; i++) {
            duplicate_matrix(matrixBs[0], sizeB, matrixBs[i]);
        }

        tiled::multiGPUduplicate<false, false, array_type, 16>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            HINTS
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixBs[i]));
            }   
            CCC(cudaFree(matrixC));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixBs[0], sizeB*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                init_matrix<array_type>(matrixA, sizeA);
                init_matrix<array_type>(matrixBs[0], sizeB);
                for (int i=1; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    duplicate_matrix(matrixBs[0], sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUduplicate<false, false, array_type, 16>(
                matrixA, widthA, heightA, 
                matrixBs, widthB, heightB, 
                matrixC, widthC, heightC,
                HINTS
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {

                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixBs[0], widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixBs[i]));
                } 
                CCC(cudaFree(matrixC));
            }
        }

        if (!standalone) {
            for (int i=1; i<device_count; i++) {
                CCC(cudaFree(matrixBs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_duplicate_hint_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU duplicate B w/ prefetch
        std::cout << "\nBenchmarking tiled multi GPU duplicate B w/ prefetch *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixBs[device_count];
        matrixBs[0] = matrixB;
        for (int i=1; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
        }

        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[0], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixBs[0], sizeB);
        }
        for (int i=1; i<device_count; i++) {
            duplicate_matrix(matrixBs[0], sizeB, matrixBs[i]);
        }

        tiled::multiGPUduplicate<false, false, array_type, 2>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            PREFETCH
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixBs[i]));
            }   
            CCC(cudaFree(matrixC));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixBs[0], sizeB*sizeof(array_type)));
                CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                init_matrix<array_type>(matrixA, sizeA);
                init_matrix<array_type>(matrixBs[0], sizeB);
                for (int i=1; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    duplicate_matrix(matrixBs[0], sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUduplicate<false, false, array_type, 2>(
                matrixA, widthA, heightA, 
                matrixBs, widthB, heightB, 
                matrixC, widthC, heightC,
                PREFETCH
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {

                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixBs[0], widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixBs[i]));
                } 
                CCC(cudaFree(matrixC));
            }
        }

        if (!standalone) {
            for (int i=1; i<device_count; i++) {
                CCC(cudaFree(matrixBs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_duplicate_prefetch_time, all_timings, timings
        );
    }

    if (false) { // Benchmark a tiled multi GPU split
        std::cout << "\nBenchmarking tiled multi GPU split *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[device_count];
        array_type* matrixBs[device_count];
        array_type* matrixCs[device_count];
        const int sub_heightA = (heightA + device_count - 1) / device_count;
        const int sub_heightC = (heightC + device_count - 1) / device_count;
        const int sub_sizeA = sub_heightA * widthA;
        const int sub_sizeC = sub_heightC * widthC;
        
        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }
        for (int i=0; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
            duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
            duplicate_matrix(matrixB, sizeB, matrixBs[i]);
        }

        tiled::multiGPUsplit<false, false, array_type, 16>(
            matrixAs, widthA, sub_heightA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, sub_heightC,
            matrixC, widthC, heightC,
            NO_HINTS, NO_REDUCE
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            }   
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                if (standalone) {
                    CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                    init_matrix<array_type>(matrixA, sizeA);
                    init_matrix<array_type>(matrixB, sizeB);
                }
                for (int i=0; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
                    duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
                    duplicate_matrix(matrixB, sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, 16>(
                matrixAs, widthA, sub_heightA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, sub_heightC,
                matrixC, widthC, heightC,
                NO_HINTS, NO_REDUCE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            for (int i=0; i<device_count; i++) {
                duplicate_matrix(matrixCs[i], sub_sizeC, matrixC+(sub_sizeC*i));
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixAs[i]));
                    CCC(cudaFree(matrixBs[i]));
                    CCC(cudaFree(matrixCs[i]));
                } 
            }
        }

        if (!standalone) {
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_time, all_timings, timings
        );
    }

    if (false) { // Benchmark a tiled multi GPU split w/ hints
        std::cout << "\nBenchmarking tiled multi GPU split w/ hints *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[device_count];
        array_type* matrixBs[device_count];
        array_type* matrixCs[device_count];
        const int sub_heightA = (heightA + device_count - 1) / device_count;
        const int sub_heightC = (heightC + device_count - 1) / device_count;
        const int sub_sizeA = sub_heightA * widthA;
        const int sub_sizeC = sub_heightC * widthC;
        
        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }
        for (int i=0; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
            duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
            duplicate_matrix(matrixB, sizeB, matrixBs[i]);
        }

        tiled::multiGPUsplit<false, false, array_type, 16>(
            matrixAs, widthA, sub_heightA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, sub_heightC,
            matrixC, widthC, heightC,
            HINTS, NO_REDUCE
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            }   
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                if (standalone) {
                    CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                    init_matrix<array_type>(matrixA, sizeA);
                    init_matrix<array_type>(matrixB, sizeB);
                }
                for (int i=0; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
                    duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
                    duplicate_matrix(matrixB, sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, 16>(
                matrixAs, widthA, sub_heightA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, sub_heightC,
                matrixC, widthC, heightC,
                HINTS, NO_REDUCE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            for (int i=0; i<device_count; i++) {
                duplicate_matrix(matrixCs[i], sub_sizeC, matrixC+(sub_sizeC*i));
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixAs[i]));
                    CCC(cudaFree(matrixBs[i]));
                    CCC(cudaFree(matrixCs[i]));
                } 
            }
        }

        if (!standalone) {
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_hint_time, all_timings, timings
        );
    }

    if (false) { // Benchmark a tiled multi GPU split w/ prefetch
        std::cout << "\nBenchmarking tiled multi GPU split w/ prefetch *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[device_count];
        array_type* matrixBs[device_count];
        array_type* matrixCs[device_count];
        const int sub_heightA = (heightA + device_count - 1) / device_count;
        const int sub_heightC = (heightC + device_count - 1) / device_count;
        const int sub_sizeA = sub_heightA * widthA;
        const int sub_sizeC = sub_heightC * widthC;
        
        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeB*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }
        for (int i=0; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
            duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
            duplicate_matrix(matrixB, sizeB, matrixBs[i]);
        }

        tiled::multiGPUsplit<false, false, array_type, 16>(
            matrixAs, widthA, sub_heightA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, sub_heightC,
            matrixC, widthC, heightC,
            PREFETCH, NO_REDUCE
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            CCC(cudaFree(matrixC));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            }   
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                if (standalone) {
                    CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                    init_matrix<array_type>(matrixA, sizeA);
                    init_matrix<array_type>(matrixB, sizeB);
                }
                for (int i=0; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
                    duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
                    duplicate_matrix(matrixB, sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, 16>(
                matrixAs, widthA, sub_heightA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, sub_heightC,
                matrixC, widthC, heightC,
                PREFETCH, NO_REDUCE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            for (int i=0; i<device_count; i++) {
                duplicate_matrix(matrixCs[i], sub_sizeC, matrixC+(sub_sizeC*i));
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixAs[i]));
                    CCC(cudaFree(matrixBs[i]));
                    CCC(cudaFree(matrixCs[i]));
                } 
            }
        }

        if (!standalone) {
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_prefetch_time, all_timings, timings
        );
    }

    if (false) { // Benchmark a tiled multi GPU split w/ malloc
        std::cout << "\nBenchmarking tiled multi GPU split w/ malloc *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[device_count];
        array_type* matrixBs[device_count];
        array_type* matrixCs[device_count];
        if (!standalone) {
            cudaFree(matrixA);
            cudaFree(matrixB);
            cudaFree(matrixC);
        }
        matrixA = (array_type*)malloc(sizeA*sizeof(array_type));
        matrixB = (array_type*)malloc(sizeB*sizeof(array_type));
        matrixC = (array_type*)calloc(sizeC, sizeof(array_type));
        const int sub_heightA = (heightA + device_count - 1) / device_count;
        const int sub_heightC = (heightC + device_count - 1) / device_count;
        const int sub_sizeA = sub_heightA * widthA;
        const int sub_sizeC = sub_heightC * widthC;
        
        init_matrix<array_type>(matrixA, sizeA);
        init_matrix<array_type>(matrixB, sizeB);
 
        for (int i=0; i<device_count; i++) {
            CCC(cudaSetDevice(i));
            CCC(cudaMalloc(&matrixAs[i], sub_sizeA*sizeof(array_type)));
            CCC(cudaMalloc(&matrixBs[i], sizeB*sizeof(array_type)));
            CCC(cudaMalloc(&matrixCs[i], sub_sizeC*sizeof(array_type)));
            CCC(cudaMemcpy(matrixAs[i], matrixA+(sub_sizeA*i), sub_sizeA*sizeof(array_type), cudaMemcpyHostToDevice));
            CCC(cudaMemcpy(matrixBs[i], matrixB, sizeB*sizeof(array_type), cudaMemcpyHostToDevice));
        }

        CCC(cudaSetDevice(origin_device));

        tiled::multiGPUsplit<false, false, array_type, 16>(
            matrixAs, widthA, sub_heightA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, sub_heightC,
            matrixC, widthC, heightC,
            NO_HINTS, NO_REDUCE
        );

        if (standalone) {
            for (int i=0; i<device_count; i++) {
                CCC(cudaSetDevice(i));
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            }   
            CCC(cudaSetDevice(origin_device));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                for (int i=0; i<device_count; i++) {
                    CCC(cudaSetDevice(i));
                    CCC(cudaMalloc(&matrixAs[i], sub_sizeA*sizeof(array_type)));
                    CCC(cudaMalloc(&matrixBs[i], sizeB*sizeof(array_type)));
                    CCC(cudaMalloc(&matrixCs[i], sub_sizeC*sizeof(array_type)));
                    CCC(cudaMemcpy(matrixAs[i], matrixA+(sub_sizeA*i), sub_sizeA*sizeof(array_type), cudaMemcpyHostToDevice));
                    CCC(cudaMemcpy(matrixBs[i], matrixB, sizeB*sizeof(array_type), cudaMemcpyHostToDevice));
                }
                CCC(cudaSetDevice(origin_device));
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, 16>(
                matrixAs, widthA, sub_heightA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, sub_heightC,
                matrixC, widthC, heightC,
                NO_HINTS, NO_REDUCE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            for (int i=0; i<device_count; i++) {
                CCC(cudaSetDevice(i));
                CCC(cudaMemcpy(matrixC+(sub_sizeC*i), matrixCs[i], sub_sizeC*sizeof(array_type), cudaMemcpyDeviceToHost));
            }
            CCC(cudaSetDevice(origin_device));

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                for (int i=0; i<device_count; i++) {
                    CCC(cudaSetDevice(i));
                    CCC(cudaFree(matrixAs[i]));
                    CCC(cudaFree(matrixBs[i]));
                    CCC(cudaFree(matrixCs[i]));
                } 
                CCC(cudaSetDevice(origin_device));
            }
        }

        if (!standalone) {
            free(matrixA);
            free(matrixB);
            free(matrixC);
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            } 
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
        }
        
        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_malloc_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU split w/ reduction
        std::cout << "\nBenchmarking tiled multi GPU split w/ reduction *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[device_count];
        array_type* matrixBs[device_count];
        array_type* matrixCs[device_count];
        const int sub_heightA = (heightA + device_count - 1) / device_count;
        const int sub_heightC = (heightC + device_count - 1) / device_count;
        const int sub_sizeA = sub_heightA * widthA;
        const int sub_sizeC = sub_heightC * widthC;
        
        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }
        for (int i=0; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
            duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
            duplicate_matrix(matrixB, sizeB, matrixBs[i]);
        }

        tiled::multiGPUsplit<false, false, array_type, 16>(
            matrixAs, widthA, sub_heightA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, sub_heightC,
            matrixC, widthC, heightC,
            NO_HINTS, DUPLICATE
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            }   
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                if (standalone) {
                    CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                    init_matrix<array_type>(matrixA, sizeA);
                    init_matrix<array_type>(matrixB, sizeB);
                }
                for (int i=0; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
                    duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
                    duplicate_matrix(matrixB, sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, 16>(
                matrixAs, widthA, sub_heightA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, sub_heightC,
                matrixC, widthC, heightC,
                NO_HINTS, DUPLICATE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixAs[i]));
                    CCC(cudaFree(matrixBs[i]));
                    CCC(cudaFree(matrixCs[i]));
                } 
            }
        }

        if (!standalone) {
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_reduce_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU split w/ reduction and hints
        std::cout << "\nBenchmarking tiled multi GPU split w/ reduction and hints *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[device_count];
        array_type* matrixBs[device_count];
        array_type* matrixCs[device_count];
        const int sub_heightA = (heightA + device_count - 1) / device_count;
        const int sub_heightC = (heightC + device_count - 1) / device_count;
        const int sub_sizeA = sub_heightA * widthA;
        const int sub_sizeC = sub_heightC * widthC;
        
        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeB*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }
        for (int i=0; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
            duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
            duplicate_matrix(matrixB, sizeB, matrixBs[i]);
        }

        tiled::multiGPUsplit<false, false, array_type, 16>(
            matrixAs, widthA, sub_heightA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, sub_heightC,
            matrixC, widthC, heightC,
            HINTS, DUPLICATE,
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            CCC(cudaFree(matrixC));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            }   
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                if (standalone) {
                    CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                    init_matrix<array_type>(matrixA, sizeA);
                    init_matrix<array_type>(matrixB, sizeB);
                }
                for (int i=0; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
                    duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
                    duplicate_matrix(matrixB, sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, 16>(
                matrixAs, widthA, sub_heightA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, sub_heightC,
                matrixC, widthC, heightC,
                HINTS, DUPLICATE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixAs[i]));
                    CCC(cudaFree(matrixBs[i]));
                    CCC(cudaFree(matrixCs[i]));
                } 
            }
        }

        if (!standalone) {
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_reduce_hint_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU split w/ reduction and prefetch
        std::cout << "\nBenchmarking tiled multi GPU split w/ reduction and prefetch *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[device_count];
        array_type* matrixBs[device_count];
        array_type* matrixCs[device_count];
        const int sub_heightA = (heightA + device_count - 1) / device_count;
        const int sub_heightC = (heightC + device_count - 1) / device_count;
        const int sub_sizeA = sub_heightA * widthA;
        const int sub_sizeC = sub_heightC * widthC;
        
        if (standalone) {
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeB*sizeof(array_type)));
            init_matrix<array_type>(matrixA, sizeA);
            init_matrix<array_type>(matrixB, sizeB);
        }
        for (int i=0; i<device_count; i++) {
            CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
            duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
            duplicate_matrix(matrixB, sizeB, matrixBs[i]);
        }

        tiled::multiGPUsplit<false, false, array_type, 16>(
            matrixAs, widthA, sub_heightA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, sub_heightC,
            matrixC, widthC, heightC,
            PREFETCH, DUPLICATE
        );

        if (standalone) {
            CCC(cudaFree(matrixA));
            CCC(cudaFree(matrixB));
            CCC(cudaFree(matrixC));
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            }   
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                if (standalone) {
                    CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
                    init_matrix<array_type>(matrixA, sizeA);
                    init_matrix<array_type>(matrixB, sizeB);
                }
                for (int i=0; i<device_count; i++) {
                    CCC(cudaMallocManaged(&matrixAs[i], sub_sizeA*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
                    CCC(cudaMallocManaged(&matrixCs[i], sub_sizeC*sizeof(array_type)));
                    duplicate_matrix(matrixA+(sub_sizeA*i), sub_sizeA, matrixAs[i]);
                    duplicate_matrix(matrixB, sizeB, matrixBs[i]);
                }
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, 16>(
                matrixAs, widthA, sub_heightA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, sub_heightC,
                matrixC, widthC, heightC,
                PREFETCH, DUPLICATE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                CCC(cudaFree(matrixA));
                CCC(cudaFree(matrixB));
                CCC(cudaFree(matrixC));
                for (int i=0; i<device_count; i++) {
                    CCC(cudaFree(matrixAs[i]));
                    CCC(cudaFree(matrixBs[i]));
                    CCC(cudaFree(matrixCs[i]));
                } 
            }
        }

        if (!standalone) {
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            } 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_reduce_prefetch_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU split w/ reduction and malloc
        std::cout << "\nBenchmarking tiled multi GPU split w/ reduction and malloc *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[device_count];
        array_type* matrixBs[device_count];
        array_type* matrixCs[device_count];
        if (!standalone) {
            cudaFree(matrixA);
            cudaFree(matrixB);
            cudaFree(matrixC);
        }
        matrixA = (array_type*)malloc(sizeA*sizeof(array_type));
        matrixB = (array_type*)malloc(sizeB*sizeof(array_type));
        matrixC = (array_type*)calloc(sizeC, sizeof(array_type));
        const int sub_heightA = (heightA + device_count - 1) / device_count;
        const int sub_heightC = (heightC + device_count - 1) / device_count;
        const int sub_sizeA = sub_heightA * widthA;
        const int sub_sizeC = sub_heightC * widthC;
        
        init_matrix<array_type>(matrixA, sizeA);
        init_matrix<array_type>(matrixB, sizeB);
 
        for (int i=0; i<device_count; i++) {
            CCC(cudaSetDevice(i));
            CCC(cudaMalloc(&matrixAs[i], sub_sizeA*sizeof(array_type)));
            CCC(cudaMalloc(&matrixBs[i], sizeB*sizeof(array_type)));
            CCC(cudaMalloc(&matrixCs[i], sub_sizeC*sizeof(array_type)));
            CCC(cudaMemcpy(matrixAs[i], matrixA+(sub_sizeA*i), sub_sizeA*sizeof(array_type), cudaMemcpyHostToDevice));
            CCC(cudaMemcpy(matrixBs[i], matrixB, sizeB*sizeof(array_type), cudaMemcpyHostToDevice));
        }

        CCC(cudaSetDevice(origin_device));

        tiled::multiGPUsplit<false, false, array_type, 16>(
            matrixAs, widthA, sub_heightA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, sub_heightC,
            matrixC, widthC, heightC,
            NO_HINTS, MEMCPY
        );

        if (standalone) {
            for (int i=0; i<device_count; i++) {
                CCC(cudaSetDevice(i));
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            }   
            CCC(cudaSetDevice(origin_device));
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                for (int i=0; i<device_count; i++) {
                    CCC(cudaSetDevice(i));
                    CCC(cudaMalloc(&matrixAs[i], sub_sizeA*sizeof(array_type)));
                    CCC(cudaMalloc(&matrixBs[i], sizeB*sizeof(array_type)));
                    CCC(cudaMalloc(&matrixCs[i], sub_sizeC*sizeof(array_type)));
                    CCC(cudaMemcpy(matrixAs[i], matrixA+(sub_sizeA*i), sub_sizeA*sizeof(array_type), cudaMemcpyHostToDevice));
                    CCC(cudaMemcpy(matrixBs[i], matrixB, sizeB*sizeof(array_type), cudaMemcpyHostToDevice));
                }
                CCC(cudaSetDevice(origin_device));
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, 16>(
                matrixAs, widthA, sub_heightA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, sub_heightC,
                matrixC, widthC, heightC,
                NO_HINTS, MEMCPY
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            for (int i=0; i<device_count; i++) {
                CCC(cudaSetDevice(i));
                CCC(cudaMemcpy(matrixC+(sub_sizeC*i), matrixCs[i], sub_sizeC*sizeof(array_type), cudaMemcpyDeviceToHost));
            }
            CCC(cudaSetDevice(origin_device));

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-1) {
                    if(cpuValidation<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC, datasize_bytes/1e9
                    )){
                        std::cout << "  Result is correct\n";
                    } else {
                        std::cout << "  Result is incorrect. Skipping any "
                                << "subsequent runs\n";
                    }
                }
            }

            if (standalone) {
                for (int i=0; i<device_count; i++) {
                    CCC(cudaSetDevice(i));
                    CCC(cudaFree(matrixAs[i]));
                    CCC(cudaFree(matrixBs[i]));
                    CCC(cudaFree(matrixCs[i]));
                } 
                CCC(cudaSetDevice(origin_device));
            }
        }

        if (!standalone) {
            free(matrixA);
            free(matrixB);
            free(matrixC);
            for (int i=0; i<device_count; i++) {
                CCC(cudaFree(matrixAs[i]));
                CCC(cudaFree(matrixBs[i]));
                CCC(cudaFree(matrixCs[i]));
            } 
            CCC(cudaMallocManaged(&matrixA, sizeA*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixB, sizeB*sizeof(array_type)));
            CCC(cudaMallocManaged(&matrixC, sizeC*sizeof(array_type)));
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_reduce_malloc_time, all_timings, timings
        );
    }
}