#include <sys/time.h>
#include <limits>

#include "matmul/cannon.h"
#include "matmul/cpu.h"
#include "matmul/tiled.h"
#include "shared_cuda.cu.h"
#include "shared.h"

typedef float array_type;

void setup_ABC_managed(
    array_type** matrixA, const unsigned long int sizeA,
    array_type** matrixB, const unsigned long int sizeB,
    array_type** matrixC, const unsigned long int sizeC
) {
    CCC(cudaMallocManaged(matrixA, sizeA*sizeof(array_type)));
    CCC(cudaMallocManaged(matrixB, sizeB*sizeof(array_type)));
    CCC(cudaMallocManaged(matrixC, sizeC*sizeof(array_type)));
    init_matrix<array_type>(*matrixA, sizeA);
    init_matrix<array_type>(*matrixB, sizeB);
}

void setup_ABC_malloced(
    array_type** matrixA, const unsigned long int sizeA,
    array_type** matrixB, const unsigned long int sizeB,
    array_type** matrixC, const unsigned long int sizeC
) {

    *matrixA = (array_type*)malloc(sizeA*sizeof(array_type));
    *matrixB = (array_type*)malloc(sizeB*sizeof(array_type));
    *matrixC = (array_type*)calloc(sizeC, sizeof(array_type));
    init_matrix<array_type>(*matrixA, sizeA);
    init_matrix<array_type>(*matrixB, sizeB);
}

void setup_Bs_managed(
    array_type** matrixB, array_type** matrixBs, 
    const unsigned long int sizeB, const int device_count
) {
    matrixBs[0] = *matrixB;
    for (int i=1; i<device_count; i++) {
        CCC(cudaMallocManaged(&matrixBs[i], sizeB*sizeof(array_type)));
        duplicate_matrix(matrixBs[0], sizeB, matrixBs[i]);
    }
}

void setup_AsBsCs_managed(
    array_type** matrixA, array_type** matrixAs, const int sizeSplitA,
    array_type** matrixB, array_type** matrixBs, const int sizeB,
    array_type** matrixCs, const int sizeSplitC, const int device_count
) {
    for (int device=0; device<device_count; device++) {
        CCC(cudaMallocManaged(&matrixAs[device], sizeSplitA*sizeof(array_type)));
        CCC(cudaMallocManaged(&matrixBs[device], sizeB*sizeof(array_type)));
        CCC(cudaMallocManaged(&matrixCs[device], sizeSplitC*sizeof(array_type)));
        duplicate_matrix(*matrixA+(sizeSplitA*device), sizeSplitA, matrixAs[device]);
        duplicate_matrix(*matrixB, sizeB, matrixBs[device]);
    }
}

void setup_AsBsCs_malloced(
    array_type** matrixA, array_type** matrixAs, const int sizeSplitA,
    array_type** matrixB, array_type** matrixBs, const int sizeB,
    array_type** matrixCs, const int sizeSplitC, 
    const int origin_device, const int device_count
) {
    for (int i=0; i<device_count; i++) {
        CCC(cudaSetDevice(i));
        CCC(cudaMalloc(&matrixAs[i], sizeSplitA*sizeof(array_type)));
        CCC(cudaMalloc(&matrixBs[i], sizeB*sizeof(array_type)));
        CCC(cudaMalloc(&matrixCs[i], sizeSplitC*sizeof(array_type)));
        CCC(cudaMemcpy(matrixAs[i], matrixA+(sizeSplitA*i), sizeSplitA*sizeof(array_type), cudaMemcpyHostToDevice));
        CCC(cudaMemcpy(matrixBs[i], matrixB, sizeB*sizeof(array_type), cudaMemcpyHostToDevice));
    }
    CCC(cudaSetDevice(origin_device));
}

void free_ABC_managed( 
    array_type** matrixA, array_type** matrixB, array_type** matrixC
) {
    CCC(cudaFree(*matrixA));
    CCC(cudaFree(*matrixB));
    CCC(cudaFree(*matrixC));
}

void free_ABC_malloced( 
    array_type** matrixA, array_type** matrixB, array_type** matrixC
) {
    free(*matrixA);
    free(*matrixB);
    free(*matrixC);
}

void free_Bs( 
    array_type** matrixBs, const int device_count
) {
    for (int i=1; i<device_count; i++) {
        CCC(cudaFree(matrixBs[i]));
    }
}

void free_AsBsCs_managed( 
    array_type** matrixAs, array_type** matrixBs, array_type** matrixCs, 
    const int device_count
) {
    for (int i=0; i<device_count; i++) {
        CCC(cudaFree(matrixAs[i]));
        CCC(cudaFree(matrixBs[i]));
        CCC(cudaFree(matrixCs[i]));
    }  
}

void free_AsBsCs_malloced( 
    array_type** matrixAs, array_type** matrixBs, array_type** matrixCs, 
    const int origin_device, const int device_count
) {
    for (int i=0; i<device_count; i++) {
        CCC(cudaSetDevice(i));
        CCC(cudaFree(matrixAs[i]));
        CCC(cudaFree(matrixBs[i]));
        CCC(cudaFree(matrixCs[i]));
    }   
    CCC(cudaSetDevice(origin_device));
}

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

    unsigned long int datasize_bytes = (unsigned long int)(((((devices+1)*widthA*heightA)+((devices+1)*widthB*heightB)+((devices+1)*widthC*heightC))*sizeof(array_type)));
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

    array_type* matrixA = NULL;
    array_type* matrixB = NULL;
    array_type* matrixC = NULL;
    
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
    struct timing_stat cannon_single_gpu_time = 
        timing_stat("cannon single GPU", operations, datasize_bytes);
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
        &cannon_single_gpu_time
    };
    int timings = sizeof(all_timings)/sizeof(all_timings[0]);

    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);

    float* timing_ms = (float*)calloc(runs, sizeof(float));

    if (true) { // Get CPU baseline
        std::cout << "Getting CPU result\n";

        cpu_time.timing_microseconds = cpuMatMul<array_type>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
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
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }

        tiled::singleGPU<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
            }

            timing_ms[run] = tiled::singleGPU<false, false, array_type, TILE_SIZE>(
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
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_single_gpu_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU raw
        std::cout << "\nBenchmarking tiled multi GPU raw *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }

        tiled::multiGPU<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices, NO_HINTS
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
            }

            timing_ms[run] = tiled::multiGPU<false, false, array_type, TILE_SIZE>(
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
                validate(
                    &matrixA, widthA, heightA, 
                    &matrixB, widthB, heightB, 
                    &matrixC, datasize_bytes/1e9
                );
            }

            if (standalone) {
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_raw_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU raw w/ hints
        std::cout << "\nBenchmarking tiled multi GPU raw w/ hints *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }

        tiled::multiGPU<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices, HINTS
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
            }

            timing_ms[run] = tiled::multiGPU<false, false, array_type, TILE_SIZE>(
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
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_raw_hint_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU raw w/ prefetch
        std::cout << "\nBenchmarking tiled multi GPU raw w/ prefetch *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }

        tiled::multiGPU<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC,
            devices, PREFETCH
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
            }

            timing_ms[run] = tiled::multiGPU<false, false, array_type, TILE_SIZE>(
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
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_raw_prefetch_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU duplicate B
        std::cout << "\nBenchmarking tiled multi GPU duplicate B *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }
        array_type* matrixBs[devices];
        setup_Bs_managed(&matrixB, matrixBs, sizeB, devices);

        tiled::multiGPUduplicate<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            devices, NO_HINTS
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
            free_Bs(matrixBs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                setup_Bs_managed(&matrixB, matrixBs, sizeB, devices);
            }

            timing_ms[run] = tiled::multiGPUduplicate<false, false, array_type, TILE_SIZE>(
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
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
                free_Bs(matrixBs, devices);
            }
        }

        if (!standalone) {
            free_Bs(matrixBs, devices);
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_duplicate_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU duplicate B w/ hints
        std::cout << "\nBenchmarking tiled multi GPU duplicate B w/ hints *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }
        array_type* matrixBs[devices];
        setup_Bs_managed(&matrixB, matrixBs, sizeB, devices);

        tiled::multiGPUduplicate<false, false, array_type, TILE_SIZE>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            devices, HINTS
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
            free_Bs(matrixBs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                setup_Bs_managed(&matrixB, matrixBs, sizeB, devices);
            }

            timing_ms[run] = tiled::multiGPUduplicate<false, false, array_type, TILE_SIZE>(
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
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
                free_Bs(matrixBs, devices);
            }
        }

        if (!standalone) {
            free_Bs(matrixBs, devices); 
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_duplicate_hint_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU duplicate B w/ prefetch
        std::cout << "\nBenchmarking tiled multi GPU duplicate B w/ prefetch *****\n";

        std::cout << "  Running a warmup\n";

        if (standalone) {
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }
        array_type* matrixBs[devices];
        setup_Bs_managed(&matrixB, matrixBs, sizeB, devices);

        tiled::multiGPUduplicate<false, false, array_type, 2>(
            matrixA, widthA, heightA, 
            matrixBs, widthB, heightB,
            matrixC, widthC, heightC,
            devices, PREFETCH
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
            free_Bs(matrixBs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                setup_Bs_managed(&matrixB, matrixBs, sizeB, devices);
            }

            timing_ms[run] = tiled::multiGPUduplicate<false, false, array_type, 2>(
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
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
                free_Bs(matrixBs, devices);
            }
        }

        if (!standalone) {
            free_Bs(matrixBs, devices);
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_duplicate_prefetch_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU split
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
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }
        setup_AsBsCs_managed(
            &matrixA, matrixAs, sizeSplitA,
            &matrixB, matrixBs, sizeB,
            matrixCs, sizeSplitC, devices
        );

        tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
            matrixAs, widthA, heightSplitA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, heightSplitC,
            matrixC, widthC, heightC,
            devices, NO_HINTS, DUPLICATE
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
            free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                setup_AsBsCs_managed(
                    &matrixA, matrixAs, sizeSplitA,
                    &matrixB, matrixBs, sizeB,
                    matrixCs, sizeSplitC, devices
                );
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
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
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
                free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
            }
        }

        if (!standalone) {
            free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU split w/ hints
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
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }
        setup_AsBsCs_managed(
            &matrixA, matrixAs, sizeSplitA,
            &matrixB, matrixBs, sizeB,
            matrixCs, sizeSplitC, devices
        );

        tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
            matrixAs, widthA, heightSplitA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, heightSplitC,
            matrixC, widthC, heightC,
            devices, HINTS, DUPLICATE
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
            free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                setup_AsBsCs_managed(
                    &matrixA, matrixAs, sizeSplitA,
                    &matrixB, matrixBs, sizeB,
                    matrixCs, sizeSplitC, devices
                );
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
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
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
                free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
            }
        }

        if (!standalone) {
            free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_hint_time, all_timings, timings
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
            setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
        }
        setup_AsBsCs_managed(
            &matrixA, matrixAs, sizeSplitA,
            &matrixB, matrixBs, sizeB,
            matrixCs, sizeSplitC, devices
        );

        tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
            matrixAs, widthA, heightSplitA, 
            matrixBs, widthB, heightB,
            matrixCs, widthC, heightSplitC,
            matrixC, widthC, heightC,
            devices, PREFETCH, DUPLICATE
        );

        if (standalone) {
            free_ABC_managed(&matrixA, &matrixB, &matrixC);
            free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                setup_AsBsCs_managed(
                    &matrixA, matrixAs, sizeSplitA,
                    &matrixB, matrixBs, sizeB,
                    matrixCs, sizeSplitC, devices
                );
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
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
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
                free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
            }
        }

        if (!standalone) {
            free_AsBsCs_managed(matrixAs, matrixBs, matrixCs, devices);
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_prefetch_time, all_timings, timings
        );
    }

    if (true) { // Benchmark a tiled multi GPU split w/ malloc
        std::cout << "\nBenchmarking tiled multi GPU split w/ malloc *****\n";

        std::cout << "  Running a warmup\n";

        array_type* hostMatrixA;
        array_type* hostMatrixB;
        array_type* hostMatrixC;
        array_type* deviceMatrixAs[devices];
        array_type* deviceMatrixBs[devices];
        array_type* deviceMatrixCs[devices];

        setup_ABC_malloced(&hostMatrixA, sizeA, &hostMatrixB, sizeB, &hostMatrixC, sizeC);

        const int heightSplitA = (heightA + devices - 1) / devices;
        const int heightSplitC = (heightC + devices - 1) / devices;
        const int sizeSplitA = heightSplitA * widthA;
        const int sizeSplitC = heightSplitC * widthC;
 
        setup_AsBsCs_managed(
            &hostMatrixA, deviceMatrixAs, sizeSplitA,
            &hostMatrixB, deviceMatrixBs, sizeB,
            deviceMatrixCs, sizeSplitC, devices
        );

        tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
            deviceMatrixAs, widthA, heightSplitA, 
            deviceMatrixBs, widthB, heightB,
            deviceMatrixCs, widthC, heightSplitC,
            hostMatrixC, widthC, heightC,
            devices, NO_HINTS, MEMCPY
        );

        if (standalone) {
            free_AsBsCs_malloced(deviceMatrixAs, deviceMatrixBs, deviceMatrixCs, origin_device, devices);
        }

        for (int run=0; run<runs; run++) {
            if (standalone) {
                setup_AsBsCs_managed(
                    &hostMatrixA, deviceMatrixAs, sizeSplitA,
                    &hostMatrixB, deviceMatrixBs, sizeB,
                    deviceMatrixCs, sizeSplitC, devices
                );
            }

            timing_ms[run] = tiled::multiGPUsplit<false, false, array_type, TILE_SIZE>(
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
                free_AsBsCs_malloced(deviceMatrixAs, deviceMatrixBs, deviceMatrixCs, origin_device, devices);
            }
        }

        if (!standalone) {
            free_ABC_malloced(&hostMatrixA, &hostMatrixB, &hostMatrixC);
            free_AsBsCs_malloced(deviceMatrixAs, deviceMatrixBs, deviceMatrixCs, origin_device, devices);
        }

        update_and_print_timing_stats(
            timing_ms, runs, &tiled_multi_gpu_split_malloc_time, all_timings, timings
        );
    }

    if ((widthA != heightA) || (widthA != widthB) || (widthA != heightB)) {
        std::cout << "Cannot run cannon algorithm for uneven matrix sizes\n";
    }
    else {
        if (true) { // Benchmark cannon single GPU
            std::cout << "\nBenchmarking cannon single GPU *****\n";

            std::cout << "  Running a warmup\n";

            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
            }

            cannon::singleGPU<array_type>(
                matrixA, matrixB, matrixC, widthA
            );

            if (standalone) {
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
            }

            for (int run=0; run<runs; run++) {
                if (standalone) {
                    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                }

                timing_ms[run] = cannon::singleGPU<array_type>(
                    matrixA, matrixB, matrixC, widthA
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
                    free_ABC_managed(&matrixA, &matrixB, &matrixC);
                }
            }

            update_and_print_timing_stats(
                timing_ms, runs, &cannon_single_gpu_time, all_timings, timings
            );
        }        
    }
}