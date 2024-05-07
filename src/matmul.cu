#include <sys/time.h>
#include <limits>

#include "matmul/cpu.h"
#include "matmul/tiled.h"
#include "shared_cuda.cu.h"
#include "shared.h"

typedef double array_type;

int main(int argc, char** argv){
    if (argc < 5)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array A height> <array A width> <array B width> <benchmark repeats>-v(validation) -s(standalone) -r(reduced output)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned int heightA = strtoul(argv[1], NULL, 0);
    unsigned int widthA = strtoul(argv[2], NULL, 0);
    unsigned int widthB = strtoul(argv[3], NULL, 0);
    unsigned int heightB = widthA;
    unsigned int widthC = widthB;
    unsigned int heightC = heightA;
    unsigned int runs = atoi(argv[4]);
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

    unsigned long int sizeA = widthA * heightA;
    unsigned long int sizeB = widthB * heightB;
    unsigned long int sizeC = widthC * heightC;

    unsigned long int datasize_bytes = (unsigned long int)((((widthA*heightA)+(2*widthB*heightB)+(2*widthC*heightC))*sizeof(array_type)));
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
    
    struct timing_stat cpu_time = timing_stat("CPU", operations, datasize_bytes);
    struct timing_stat tiled_single_gpu_time = timing_stat("tiled single GPU", operations, datasize_bytes);
    struct timing_stat tiled_multi_gpu_time = timing_stat("tiled multi GPU", operations, datasize_bytes);
    const struct timing_stat* all_timings[] = {
        &cpu_time,
        &tiled_single_gpu_time,
        &tiled_multi_gpu_time
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

    { // Get CPU baseline
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

    { // Benchmark a tiled single GPU
        std::cout << "\nBenchmarking tiled single GPU matrix multiplication *****\n";

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
            matrixC
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
                matrixC
            );


            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-2) {
                    zero_matrix(matrixC, sizeC);
                }
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

    { // Benchmark a tiled multi GPU
        std::cout << "\nBenchmarking tiled multi GPU matrix multiplication *****\n";

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
            matrixC
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
                matrixC
            );


            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (validating) {
                if (run==runs-2) {
                    zero_matrix(matrixC, sizeC);
                }
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
}