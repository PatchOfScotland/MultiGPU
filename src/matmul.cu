#include <sys/time.h>
#include <limits>

#include "matmul/cannon.h"
#include "matmul/cpu.h"
#include "matmul/tiled.h"
#include "matmul/page_tile.h"
#include "matmul/prefetch_page_tile.h"
#include "shared_cuda.cu.h"
#include "shared.h"

void validate(
    array_type** matrixA, const unsigned int widthA, const unsigned int heightA,
    array_type** matrixB, const unsigned int widthB, const unsigned int heightB,
    array_type** matrixC, array_type tolerance
) {
    if(cpuValidation<array_type>(
        *matrixA, widthA, heightA, 
        *matrixB, widthB, heightB, 
        *matrixC, tolerance
    )){
        std::cout << "  Result is correct\n";
    } else {
        std::cout << "  Result is incorrect. Skipping any "
                << "subsequent runs\n";
    }
}

void validateZorder(
    array_type** matrixA, const unsigned int widthA, const unsigned int heightA,
    array_type** matrixB, const unsigned int widthB, const unsigned int heightB,
    array_type** matrixC, array_type tolerance, int split
) {
    array_type* matrixAz = (array_type*)malloc(widthA*heightA*sizeof(array_type));
    array_type* matrixBz = (array_type*)malloc(widthB*heightB*sizeof(array_type));
    array_type* matrixCz = (array_type*)malloc(widthB*heightB*sizeof(array_type));

    block<array_type>(*matrixA, matrixAz, widthA, heightA, split);
    block<array_type>(*matrixB, matrixBz, widthB, heightB, split);
    block<array_type>(*matrixC, matrixCz, widthB, heightB, split);

    if(cpuValidation<array_type>(
        matrixAz, widthA, heightA, 
        matrixBz, widthB, heightB, 
        matrixCz, tolerance
    )){
        std::cout << "  Result is correct\n";
    } else {
        std::cout << "  Result is incorrect. Skipping any "
                  << "subsequent runs\n";
    }

    free(matrixAz);
    free(matrixBz);
    free(matrixCz);
}

int main(int argc, char** argv){
    if (argc < 5)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array A height> <array A width> <array B width> <benchmark repeats> -d(devices) <devices> -v(validation) -s(standalone) -r(reduced output)\n";
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

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int devices;
    CCC(cudaGetDeviceCount(&devices));

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
        if ((strcmp(argv[i], "-d") == 0 ) && (i+1<argc)) {
            devices = atoi(argv[i+1]);
        }
    }

    print_device_info(devices);

    const unsigned long int sizeA = widthA * heightA;
    const unsigned long int sizeB = widthB * heightB;
    const unsigned long int sizeC = widthC * heightC;

    unsigned long int datasize_bytes = (unsigned long int)(
        ((((devices+1)*widthA*heightA)
        +((devices+1)*widthB*heightB)
        +((devices+1)*widthC*heightC))*sizeof(array_type))
    );
    unsigned long int operations = (unsigned long int)heightC*widthC*widthA*2;
    std::cout << "Multiplying arrays of size " 
              << widthA << "x" << heightA
              << " and "
              << widthB << "x" << heightB
              << ", resulting in "
              << widthC << "x" << heightC
              << "\n";
    std::cout << "Using " 
              << datasize_bytes / 1e9
              << "GB of memory and "
              << (float)operations / 1e9
              << " GFLOPs per experiment\n";

    if (widthA != heightB) {
        std::cout << "Invalid matrix shapes\n";
        exit(1);
    }

    if (validating) {
        std::cout << "Will validate output\n";
    }
    else {
        std::cout << "Skipping output validation\n";
    }
    if (standalone) {
        std::cout << "Creating new datasets for each run\n";
    }

    array_type* matrixA = NULL;
    array_type* matrixB = NULL;
    array_type* matrixTransB = NULL;
    array_type* matrixC = NULL;
    
    int timings = 0;
    struct timing_stat* all_timings = NULL;
    
    setup_managed(&matrixA, sizeA, validating);
    setup_managed(&matrixB, sizeB, validating);
    setup_managed(&matrixC, sizeC, false);
    setup_trans_managed(&matrixB, &matrixTransB, widthB, heightB, validating);

    float* timing_μs = (float*)calloc(runs, sizeof(float));

    if (false) { // Get CPU baseline
        std::cout << "Getting CPU result\n";

        struct timing_stat cpu_time =  
            timing_stat("tiled CPU", operations, datasize_bytes);
        cpu_time.timing_microseconds = cpuMatMul<array_type>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC
        );
        timing_μs[0] = cpu_time.timing_microseconds;
        
        update_timing_stats(
            timing_μs, 1, "CPU\0", &all_timings, &timings, operations, 
            datasize_bytes
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
        }

        if (false) {
            std::cout << "Input A: \n";
            print_matrix(matrixA, widthA, heightA);
            std::cout << "Input B: \n";
            print_matrix(matrixB, widthB, heightB);
            std::cout << "Result: \n";
            print_matrix(matrixC, widthC, heightC);
        }

        std::cout << "CPU matrix multiplication took: " << cpu_time.timing_microseconds / 1e3 << "ms\n";
        std::cout << "CPU throughput:     " << cpu_time.throughput_gb() << "GB/sec\n";
        std::cout << "CPU GFLOPS:         " << cpu_time.throughput_gf() << "ops/sec\n";
    }

    if (true) { // Benchmark a tiled single GPU
        std::cout << "\nBenchmarking tiled single GPU *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }

        tiled::singleGPU<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            timing_μs[run] = tiled::singleGPU<false, false, array_type, TILE_SIZE>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC
            );


            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                if (false) {
                    //std::cout << "Matrix A: \n";
                    //print_matrix(matrixA, widthA, heightA);
                    //std::cout << "Matrix B: \n";
                    //print_matrix(matrixB, widthB, heightB);
                    std::cout << "Result: \n";
                    print_matrix(matrixC, widthC, heightC);

                    cpuMatMul<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC
                    );
                    std::cout << "Reference: \n";
                    print_matrix(matrixC, widthC, heightC);
                }
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }
        }
        
        update_and_print_timing_stats(
            timing_μs, runs, "tiled single GPU\0", &all_timings, 
            &timings, operations, datasize_bytes
        );
    }
 
    if (true) { // Benchmark a tiled multi GPU raw
        std::cout << "\nBenchmarking tiled multi GPU raw *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {            
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }

        tiled::multiGPU<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices, NO_HINTS
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            timing_μs[run] = tiled::multiGPU<false, false, array_type, TILE_SIZE>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, NO_HINTS
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                if (false) {
                    std::cout << "Matrix A: \n";
                    print_matrix(matrixA, widthA, heightA);
                    std::cout << "Matrix B: \n";
                    print_matrix(matrixB, widthB, heightB);
                    std::cout << "Result: \n";
                    print_matrix(matrixC, widthC, heightC);

                    cpuMatMul<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC
                    );
                    std::cout << "Reference: \n";
                    print_matrix(matrixC, widthC, heightC);
                }
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU raw\0", &all_timings, &timings, 
            operations, datasize_bytes
        );
    }

    if (false) { // Benchmark a tiled multi GPU raw w/ hints
        std::cout << "\nBenchmarking tiled multi GPU raw w/ hints *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }

        tiled::multiGPU<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices, HINTS
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            timing_μs[run] = tiled::multiGPU<false, false, array_type, TILE_SIZE>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, HINTS
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU raw w/ hints\0", &all_timings, 
            &timings, operations, datasize_bytes
        );
    }

    if (false) { // Benchmark a tiled multi GPU raw w/ prefetch
        std::cout << "\nBenchmarking tiled multi GPU raw w/ prefetch *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }

        tiled::multiGPU<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices, PREFETCH
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            timing_μs[run] = tiled::multiGPU<false, false, array_type, TILE_SIZE>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, PREFETCH
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU raw w/ prefetch\0", &all_timings, 
            &timings, operations, datasize_bytes
        );

    }

    if (false) { // Benchmark a tiled multi GPU duplicate B
        std::cout << "\nBenchmarking tiled multi GPU duplicate B *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }
        array_type* matrixBs[devices];
        setup_managed_array(&matrixB, matrixBs, sizeB, devices, validating);

        tiled::multiGPUduplicate<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            devices, NO_HINTS
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
            free_managed_array(matrixBs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
                setup_managed_array(&matrixB, matrixBs, sizeB, devices, validating);
            }

            timing_μs[run] = tiled::multiGPUduplicate<false, false, array_type, TILE_SIZE>(
                matrixA, widthA, heightA, 
                matrixBs, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, NO_HINTS
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixBs[0], widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
                free_managed_array(matrixBs, devices);
            }
        }

        if (!standalone) {
            free_managed_array(matrixBs, devices);
        }
        
        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU duplicate raw\0", &all_timings, 
            &timings, operations, datasize_bytes
        );
    }

    if (false) { // Benchmark a tiled multi GPU duplicate B w/ hints
        std::cout << "\nBenchmarking tiled multi GPU duplicate B w/ hints *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }
        array_type* matrixBs[devices];
        setup_managed_array(&matrixB, matrixBs, sizeB, devices, validating);

        tiled::multiGPUduplicate<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            devices, HINTS
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
            free_managed_array(matrixBs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
                setup_managed_array(&matrixB, matrixBs, sizeB, devices, validating);
            }

            timing_μs[run] = tiled::multiGPUduplicate<false, false, array_type, TILE_SIZE>(
                matrixA, widthA, heightA, 
                matrixBs, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, HINTS
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixBs[0], widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
                free_managed_array(matrixBs, devices);
            }
        }

        if (!standalone) {
            free_managed_array(matrixBs, devices); 
        }
        
        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU duplicate w/ hints\0", 
            &all_timings, &timings, operations, datasize_bytes
        );
    }

    if (false) { // Benchmark a tiled multi GPU duplicate B w/ prefetch
        std::cout << "\nBenchmarking tiled multi GPU duplicate B w/ prefetch *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }
        array_type* matrixBs[devices];
        setup_managed_array(&matrixB, matrixBs, sizeB, devices, validating);

        tiled::multiGPUduplicate<false, false, array_type, 2>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            devices, PREFETCH
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
            free_managed_array(matrixBs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
                setup_managed_array(&matrixB, matrixBs, sizeB, devices, validating);
            }

            timing_μs[run] = tiled::multiGPUduplicate<false, false, array_type, 2>(
                matrixA, widthA, heightA, 
                matrixBs, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, PREFETCH
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixBs[0], widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
                free_managed_array(matrixBs, devices);
            }
        }

        if (!standalone) {
            free_managed_array(matrixBs, devices);
        }

        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU duplicated w/ prefetch\0", 
            &all_timings, &timings, operations, datasize_bytes
        );
    }

    if (false) { // Benchmark a tiled multi GPU split
        std::cout << "\nBenchmarking tiled multi GPU split *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[devices];
        array_type* matrixBs[devices];
        array_type* matrixCs[devices];
        const int heightSplitA = (heightA + devices - 1) / devices;
        const int heightSplitC = (heightC + devices - 1) / devices;
        const int sizeSplitA = heightSplitA * widthA;
        const int sizeSplitC = heightSplitC * widthC;
        
        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }
        setup_AsBsCs_managed(
            &matrixA, matrixAs, sizeSplitA,
            &matrixB, matrixBs, sizeB,
            matrixCs, sizeSplitC, devices, validating
        );

        tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
            matrixAs, widthA, heightSplitA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, heightSplitC,
            matrixC, widthC, heightC,
            devices, NO_HINTS, DUPLICATE
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
            free_managed_array(matrixAs, devices);
            free_managed_array(matrixBs, devices);
            free_managed_array(matrixCs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
                setup_AsBsCs_managed(
                    &matrixA, matrixAs, sizeSplitA,
                    &matrixB, matrixBs, sizeB,
                    matrixCs, sizeSplitC, devices, validating
                );
            }

            timing_μs[run] = tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
                matrixAs, widthA, heightSplitA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, heightSplitC,
                matrixC, widthC, heightC,
                devices, NO_HINTS, DUPLICATE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
                free_managed_array(matrixAs, devices);
                free_managed_array(matrixBs, devices);
                free_managed_array(matrixCs, devices);
            }
        }

        if (!standalone) {
            free_managed_array(matrixAs, devices);
            free_managed_array(matrixBs, devices);
            free_managed_array(matrixCs, devices);
        }
        
        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU split\0", &all_timings, &timings, 
            operations, datasize_bytes
        );
    }

    if (false) { // Benchmark a tiled multi GPU split w/ hints
        std::cout << "\nBenchmarking tiled multi GPU split w/ hints *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[devices];
        array_type* matrixBs[devices];
        array_type* matrixCs[devices];
        const int heightSplitA = (heightA + devices - 1) / devices;
        const int heightSplitC = (heightC + devices - 1) / devices;
        const int sizeSplitA = heightSplitA * widthA;
        const int sizeSplitC = heightSplitC * widthC;
        
        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }
        setup_AsBsCs_managed(
            &matrixA, matrixAs, sizeSplitA,
            &matrixB, matrixBs, sizeB,
            matrixCs, sizeSplitC, devices, validating
        );

        tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
            matrixAs, widthA, heightSplitA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, heightSplitC,
            matrixC, widthC, heightC,
            devices, HINTS, DUPLICATE
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
            free_managed_array(matrixAs, devices);
            free_managed_array(matrixBs, devices);
            free_managed_array(matrixCs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
                setup_AsBsCs_managed(
                    &matrixA, matrixAs, sizeSplitA,
                    &matrixB, matrixBs, sizeB,
                    matrixCs, sizeSplitC, devices, validating
                );
            }

            timing_μs[run] = tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
                matrixAs, widthA, heightSplitA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, heightSplitC,
                matrixC, widthC, heightC,
                devices, HINTS, DUPLICATE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
                free_managed_array(matrixAs, devices);
                free_managed_array(matrixBs, devices);
                free_managed_array(matrixCs, devices);
            }
        }

        if (!standalone) {
            free_managed_array(matrixAs, devices);
            free_managed_array(matrixBs, devices);
            free_managed_array(matrixCs, devices);
        }
        
        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU split w/ hints\0", &all_timings, 
            &timings, operations, datasize_bytes
        );
    }

    if (true) { // Benchmark a tiled multi GPU split w/ prefetch
        std::cout << "\nBenchmarking tiled multi GPU split w/ prefetch *****\n";

        std::cout << "  Running a warmup\n";

        array_type* matrixAs[devices];
        array_type* matrixBs[devices];
        array_type* matrixCs[devices];
        const int heightSplitA = (heightA + devices - 1) / devices;
        const int heightSplitC = (heightC + devices - 1) / devices;
        const int sizeSplitA = heightSplitA * widthA;
        const int sizeSplitC = heightSplitC * widthC;
        
        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }
        setup_AsBsCs_managed(
            &matrixA, matrixAs, sizeSplitA,
            &matrixB, matrixBs, sizeB,
            matrixCs, sizeSplitC, devices, validating
        );

        tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
            matrixAs, widthA, heightSplitA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, heightSplitC,
            matrixC, widthC, heightC,
            devices, PREFETCH, DUPLICATE
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
            free_managed_array(matrixAs, devices);
            free_managed_array(matrixBs, devices);
            free_managed_array(matrixCs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
                setup_AsBsCs_managed(
                    &matrixA, matrixAs, sizeSplitA,
                    &matrixB, matrixBs, sizeB,
                    matrixCs, sizeSplitC, devices, validating
                );
            }

            timing_μs[run] = tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
                matrixAs, widthA, heightSplitA, 
                matrixBs, widthB, heightB,
                matrixCs, widthC, heightSplitC,
                matrixC, widthC, heightC,
                devices, PREFETCH, DUPLICATE
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
                free_managed_array(matrixAs, devices);
                free_managed_array(matrixBs, devices);
                free_managed_array(matrixCs, devices);
            }
        }

        if (!standalone) {
            free_managed_array(matrixAs, devices);
            free_managed_array(matrixBs, devices);
            free_managed_array(matrixCs, devices);
        }

        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU split w/ prefetch", &all_timings, 
            &timings, operations, datasize_bytes
        );
    }

    if (false) { // Benchmark a tiled multi GPU split w/ malloc
        std::cout << "\nBenchmarking tiled multi GPU split w/ malloc *****\n";

        std::cout << "  Running a warmup\n";

        array_type* hostMatrixA;
        array_type* hostMatrixB;
        array_type* hostMatrixC;
        array_type* deviceMatrixAs[devices];
        array_type* deviceMatrixBs[devices];
        array_type* deviceMatrixCs[devices];

        setup_malloced(&hostMatrixA, sizeA, validating);
        setup_malloced(&hostMatrixB, sizeB, validating);
        setup_malloced(&hostMatrixC, sizeC, false);

        const int heightSplitA = (heightA + devices - 1) / devices;
        const int heightSplitC = (heightC + devices - 1) / devices;
        const int sizeSplitA = heightSplitA * widthA;
        const int sizeSplitC = heightSplitC * widthC;
 
        setup_AsBsCs_managed(
            &hostMatrixA, deviceMatrixAs, sizeSplitA,
            &hostMatrixB, deviceMatrixBs, sizeB,
            deviceMatrixCs, sizeSplitC, devices, validating
        );

        tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
            deviceMatrixAs, widthA, heightSplitA, 
            deviceMatrixBs, widthB, heightB,
            deviceMatrixCs, widthC, heightSplitC,
            hostMatrixC, widthC, heightC,
            devices, NO_HINTS, MEMCPY
        );

        if (standalone) {
            free_malloced_array(deviceMatrixAs, origin_device, devices);
            free_malloced_array(deviceMatrixBs, origin_device, devices);
            free_malloced_array(deviceMatrixCs, origin_device, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_AsBsCs_managed(
                    &hostMatrixA, deviceMatrixAs, sizeSplitA,
                    &hostMatrixB, deviceMatrixBs, sizeB,
                    deviceMatrixCs, sizeSplitC, devices, validating
                );
            }

            timing_μs[run] = tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
                deviceMatrixAs, widthA, heightSplitA, 
                deviceMatrixBs, widthB, heightB,
                deviceMatrixCs, widthC, heightSplitC,
                hostMatrixC, widthC, heightC,
                devices, NO_HINTS, MEMCPY
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &hostMatrixA, widthA, heightA, 
                    &hostMatrixB, widthB, heightB, 
                    &hostMatrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_malloced_array(deviceMatrixAs, origin_device, devices);
                free_malloced_array(deviceMatrixBs, origin_device, devices);
                free_malloced_array(deviceMatrixCs, origin_device, devices);
            }
        }

        if (!standalone) {
            free_malloced(&hostMatrixA);
            free_malloced(&hostMatrixB);
            free_malloced(&hostMatrixC);
            free_malloced_array(deviceMatrixAs, origin_device, devices);
            free_malloced_array(deviceMatrixBs, origin_device, devices);
            free_malloced_array(deviceMatrixCs, origin_device, devices);
        }
        
        update_and_print_timing_stats(
            timing_μs, runs, "tiled multi GPU split w/ malloc\0", &all_timings, 
            &timings, operations, datasize_bytes
        );
    }

    const unsigned int cannon_block = 32;
    const size_t quadrants_per_dim = 2;

    if ((widthA != heightA) || (widthA != widthB) || (widthA != heightB)) {
        std::cout << "Cannot run cannon algorithm for uneven matrix sizes\n";
    }
    else {
        if (true) { // Benchmark cannon single GPU
            std::cout << "\nBenchmarking cannon single GPU *****\n";

            std::cout << "  Running a warmup\n";

            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            cannon::singleGPU<array_type, cannon_block>(
                matrixA, matrixB, matrixC, widthC
            );

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }

            for (int run=0; run<runs; run++) {
                if (standalone) {
                    setup_managed(&matrixA, sizeA, validating);
                    setup_managed(&matrixB, sizeB, validating);
                    setup_managed(&matrixC, sizeC, false);  
                    zero_matrix(matrixC, widthC* heightC);
                }

                timing_μs[run] = cannon::singleGPU<array_type, cannon_block>(
                    matrixA, matrixB, matrixC, heightC
                );


                if (reduced_output == false) {
                    print_loop_feedback(run, runs);
                }

                // do this at the end as reading output array will shift it back to 
                // the host. Just use datasize_GB as crude tolerance for now.
                if ((validating) && (run==runs-1)) {
                    validate(
                        &matrixA, widthA, heightA, 
                        &matrixB, widthB, heightB, 
                        &matrixC, datasize_bytes/1e9
                    );
                    if (false) {
                        std::cout << "Input A: \n";
                        print_matrix(matrixA, widthA, heightA);
                        std::cout << "Input B: \n";
                        print_matrix(matrixB, widthB, heightB);
                        std::cout << "Result: \n";
                        print_matrix(matrixC, widthC, heightC);
                        cpuMatMul<array_type>(
                            matrixA, widthA, heightA, 
                            matrixB, widthB, heightB, 
                            matrixC
                        );
                        std::cout << "Reference: \n";
                        print_matrix(matrixC, widthC, heightC);
                    }
                }

                if (standalone) {
                    free_managed(&matrixA);
                    free_managed(&matrixB);
                    free_managed(&matrixC);
                }
            }
        
            update_and_print_timing_stats(
                timing_μs, runs, "cannon single GPU\0", &all_timings, &timings, 
                operations, datasize_bytes
            );
        }

        if (true) { // Benchmark cannon multi GPU on device basis
            std::cout << "\nBenchmarking cannon multi GPU on device basis *****\n";

            std::cout << "  Running a warmup\n";

            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            cannon::multiGPU<array_type, TILE_SIZE>(
                matrixA, matrixB, matrixC, widthC, devices, 
                quadrants_per_dim, false
            );

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }

            bool zero_c = false;

            for (int run=0; run<runs; run++) {
                if (standalone) {
                    setup_managed(&matrixA, sizeA, validating);
                    setup_managed(&matrixB, sizeB, validating);
                    setup_managed(&matrixC, sizeC, false);
                    zero_matrix(matrixC, widthC* heightC);
                }

                const size_t quadrants_per_dim = 2;

                if ((validating) && (run==runs-1)) {
                    zero_c = true;
                }

                timing_μs[run] = cannon::multiGPU<array_type, TILE_SIZE>(
                    matrixA, matrixB, matrixC, widthC, devices, 
                    quadrants_per_dim, zero_c
                );

                if (reduced_output == false) {
                    print_loop_feedback(run, runs);
                }

                // do this at the end as reading output array will shift it back to 
                // the host. Just use datasize_GB as crude tolerance for now.
                if ((validating) && (run==runs-1)) {
                    const int split = widthC / quadrants_per_dim;
                    validateZorder(
                        &matrixA, widthA, heightA, 
                        &matrixB, widthB, heightB, 
                        &matrixC, datasize_bytes/1e9, split
                    );
                    if (false) {
                        std::cout << "Input A: \n";
                        print_matrix_z(matrixA, widthA, quadrants_per_dim);
                        std::cout << "Input B: \n";
                        print_matrix_z(matrixB, widthB, quadrants_per_dim);
                        std::cout << "Result: \n";
                        print_matrix_z(matrixC, widthC, quadrants_per_dim);
                        cpuMatMulZorder<array_type>(
                            matrixA, widthA, heightA, 
                            matrixB, widthB, heightB, 
                            matrixC, split
                        );
                        std::cout << "Reference: \n";
                        print_matrix(matrixC, widthC, heightC);
                    }
                }

                if (standalone) {
                    free_managed(&matrixA);
                    free_managed(&matrixB);
                    free_managed(&matrixC);
                }
            }
        
            update_and_print_timing_stats(
                timing_μs, runs, "cannon device multi GPU\0", &all_timings, &timings, 
                operations, datasize_bytes
            );
        }        

        if (true) { // Benchmark cannon multi GPU on device basis with prefetch
            std::cout << "\nBenchmarking cannon multi GPU on device basis with prefetch *****\n";

            std::cout << "  Running a warmup\n";

            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            cannon::multiGPU<array_type, TILE_SIZE>(
                matrixA, matrixB, matrixC, widthC, devices, 
                quadrants_per_dim, false
            );

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }

            bool zero_c = false;

            for (int run=0; run<runs; run++) {
                if (standalone) {
                    setup_managed(&matrixA, sizeA, validating);
                    setup_managed(&matrixB, sizeB, validating);
                    setup_managed(&matrixC, sizeC, false);
                    zero_matrix(matrixC, widthC* heightC);
                }

                const size_t quadrants_per_dim = 2;

                if ((validating) && (run==runs-1)) {
                    zero_c = true;
                }

                timing_μs[run] = cannon::multiGPU<array_type, TILE_SIZE>(
                    matrixA, matrixB, matrixC, widthC, devices, 
                    quadrants_per_dim, zero_c
                );

                if (reduced_output == false) {
                    print_loop_feedback(run, runs);
                }

                // do this at the end as reading output array will shift it back to 
                // the host. Just use datasize_GB as crude tolerance for now.
                if ((validating) && (run==runs-1)) {
                    const int split = widthC / quadrants_per_dim;
                    validateZorder(
                        &matrixA, widthA, heightA, 
                        &matrixB, widthB, heightB, 
                        &matrixC, datasize_bytes/1e9, split
                    );
                    if (false) {
                        std::cout << "Input A: \n";
                        print_matrix_z(matrixA, widthA, quadrants_per_dim);
                        std::cout << "Input B: \n";
                        print_matrix_z(matrixB, widthB, quadrants_per_dim);
                        std::cout << "Result: \n";
                        print_matrix_z(matrixC, widthC, quadrants_per_dim);
                        cpuMatMulZorder<array_type>(
                            matrixA, widthA, heightA, 
                            matrixB, widthB, heightB, 
                            matrixC, split
                        );
                        std::cout << "Reference: \n";
                        print_matrix(matrixC, widthC, heightC);
                    }
                }

                if (standalone) {
                    free_managed(&matrixA);
                    free_managed(&matrixB);
                    free_managed(&matrixC);
                }
            }
        
            update_and_print_timing_stats(
                timing_μs, runs, "cannon prefetch multi GPU\0", &all_timings, &timings, 
                operations, datasize_bytes
            );
        }        
    }

    if (true) { // Benchmark a page-tiled multi GPU
        std::cout << "\nBenchmarking page tile multi GPU *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }
        
        const int page_size = PAGE_SIZE / sizeof(array_type);

        page_tiled::multiGPU<false, array_type, page_size>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            timing_μs[run] = page_tiled::multiGPU<
                false, array_type, page_size
            >(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
                if (false) {
                    std::cout << "Matrix A: \n";
                    print_matrix(matrixA, widthC, heightC);
                    std::cout << "Matrix B: \n";
                    print_matrix(matrixB, widthC, heightC);
                    std::cout << "Result: \n";
                    print_matrix(matrixC, widthC, heightC);
                }
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_μs, runs, "page tiled multi GPU\0", &all_timings, &timings, 
            operations, datasize_bytes
        );
    }

    if (false) { // Benchmark a page-tiled multi GPU w/trans B
        std::cout << "\nBenchmarking page tile multi GPU w/ trans B *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
            setup_trans_managed(&matrixB, &matrixTransB, widthB, heightB, validating);
        }
        
        const int page_size = PAGE_SIZE / sizeof(array_type);

        page_tiled::multiGPU<true, array_type, page_size>(
            matrixA, widthA, heightA, 
            matrixTransB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
            free(matrixTransB);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
                setup_trans_managed(&matrixB, &matrixTransB, widthB, heightB, validating);
            }

            timing_μs[run] = page_tiled::multiGPU<
                true, array_type, page_size
            >(
                matrixA, widthA, heightA, 
                matrixTransB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
                free(matrixTransB);
            }
        }

        update_and_print_timing_stats(
            timing_μs, runs, "page tiled multi GPU w/ trans B\0", &all_timings, &timings, 
            operations, datasize_bytes
        );
    }

    const int page_size = PAGE_SIZE / sizeof(array_type);
    // Get this more dynamically determined
    const int sm_count = 20; //20 aarhus, 84 hendrix03

    if (true) { // Benchmark a prefetching page-tiled multi GPU
        std::cout << "\nBenchmarking prefetching page tile multi GPU *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);
        }

        prefetch_page_tiled::multiGPU<false, array_type, page_size, sm_count>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices, false
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            timing_μs[run] = prefetch_page_tiled::multiGPU<
                false, array_type, page_size, sm_count
            >(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, false
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                if (false) {
                    std::cout << "Matrix A: \n";
                    print_matrix(matrixA, widthA, heightA);
                    std::cout << "Matrix B: \n";
                    print_matrix(matrixB, widthB, heightB);
                    std::cout << "Result: \n";
                    print_matrix(matrixC, widthC, heightC);
                }
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
                if (false) {
                    cpuMatMul<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC
                    );
                    std::cout << "Reference: \n";
                    print_matrix(matrixC, widthC, heightC);
                }
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_μs, runs, "prefetching page tiled multi GPU\0", &all_timings, &timings, 
            operations, datasize_bytes
        );
    }

    if (false) { // Benchmark an offset prefetching page-tiled multi GPU
        std::cout << "\nBenchmarking offset prefetching page tile multi GPU *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_managed(&matrixA, sizeA, validating);
            setup_managed(&matrixB, sizeB, validating);
            setup_managed(&matrixC, sizeC, false);  
        }
        
        prefetch_page_tiled::multiGPU<false, array_type, page_size, sm_count>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices, true
        );

        if (standalone) {
            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_managed(&matrixA, sizeA, validating);
                setup_managed(&matrixB, sizeB, validating);
                setup_managed(&matrixC, sizeC, false);
            }

            timing_μs[run] = prefetch_page_tiled::multiGPU<
                false, array_type, page_size, sm_count
            >(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, true
            );

            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if ((validating) && (run==runs-1)) {
                if (false) {
                    std::cout << "Matrix A: \n";
                    print_matrix(matrixA, widthA, heightA);
                    std::cout << "Matrix B: \n";
                    print_matrix(matrixB, widthB, heightB);
                    std::cout << "Result: \n";
                    print_matrix(matrixC, widthC, heightC);
                }
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
                if (false) {
                    cpuMatMul<array_type>(
                        matrixA, widthA, heightA, 
                        matrixB, widthB, heightB, 
                        matrixC
                    );
                    std::cout << "Reference: \n";
                    print_matrix(matrixC, widthC, heightC);
                }
            }

            if (standalone) {
                free_managed(&matrixA);
                free_managed(&matrixB);
                free_managed(&matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_μs, runs, "prefetching page tiled multi GPU\0", &all_timings, &timings, 
            operations, datasize_bytes
        );
    }
}