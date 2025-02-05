#include <sys/time.h>
#include <limits>

#include "matmul/cannon.h"
#include "matmul/cpu.h"
#include "matmul/tiled.h"
#include "matmul/page_tile.h"
#include "matmul/prefetch_page_tile.h"
#include "shared_cuda.cu.h"
#include "shared.h"

void init_AB(
    array_type** matrixA, const unsigned long int sizeA,
    array_type** matrixB, const unsigned long int sizeB
) {
    init_matrix_linear<array_type>(*matrixA, sizeA);
    init_matrix_linear<array_type>(*matrixB, sizeB);
}

void setup_ABC_managed(
    array_type** matrixA, const unsigned long int sizeA,
    array_type** matrixB, const unsigned long int sizeB,
    array_type** matrixC, const unsigned long int sizeC,
    bool validating
) {
    std::cout << "  Allocating " 
              << sizeA*sizeof(array_type) << ", " 
              << sizeB*sizeof(array_type) << ", and " 
              << sizeC*sizeof(array_type) <<"\n";

    CCC(cudaMallocManaged(matrixA, sizeA*sizeof(array_type)));
    CCC(cudaMallocManaged(matrixB, sizeB*sizeof(array_type)));
    CCC(cudaMallocManaged(matrixC, sizeC*sizeof(array_type)));
    if (validating) {
        init_AB(matrixA, sizeA, matrixB, sizeB);
    }

    cuda_error_check();
}

void free_ABC_managed( 
    array_type** matrixA, array_type** matrixB, array_type** matrixC
) {
    CCC(cudaFree(*matrixA));
    CCC(cudaFree(*matrixB));
    CCC(cudaFree(*matrixC));
}

void validateCPU(
    array_type** matrixA, const unsigned int widthA, const unsigned int heightA,
    array_type** matrixB, const unsigned int widthB, const unsigned int heightB,
    array_type** matrixC, array_type tolerance, const int split
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

void validateBlocked(
    array_type** matrixA, const unsigned int widthA, const unsigned int heightA,
    array_type** matrixB, const unsigned int widthB, const unsigned int heightB,
    array_type** matrixC, array_type tolerance, int split, bool cpu
) {
    if (cpu) {
        array_type* matrixAz = (array_type*)malloc(widthA*heightA*sizeof(array_type));
        array_type* matrixBz = (array_type*)malloc(widthB*heightB*sizeof(array_type));
        array_type* matrixCz = (array_type*)malloc(widthB*heightB*sizeof(array_type));

        to_block<array_type>(*matrixA, matrixAz, widthA, heightA, split);
        to_block<array_type>(*matrixB, matrixBz, widthB, heightB, split);
        to_block<array_type>(*matrixC, matrixCz, widthB, heightB, split);

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
    else {
        array_type* matrixValidating = *matrixC;
        array_type* matrixRef = NULL;
        cudaMallocManaged(&matrixRef, widthB*heightA*sizeof(array_type));

        cannon::singleGPU<array_type, BLOCK_N>(
            *matrixA, *matrixB, matrixRef, widthB
        );

        unsigned long int count = 0;
        //#pragma omp parallel for reduction(+:count)
        for(int i = 0; i < heightA*widthB; ++i) {
            if (abs(matrixRef[i] - matrixValidating[i]) > tolerance) {
                //std::cout << matrixRef[i] << " - " << matrixValidating[i] << "\n";
                count++;
            }
        }

        if (count > 0) {
            printf("  Got %ld of potential %d mismatches\n", count, heightA*widthB);
        }
        else {
            printf("  Result is correct\n");
        }
        cudaFree(matrixRef);
    }
}

int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array N> <benchmark repeats> -d(devices) <devices> -v(validation) -r(reduced output) -c (coldstart)\n";
        exit(EXIT_FAILURE);
    } 

    const unsigned int heightA = strtoul(argv[1], NULL, 0);
    const unsigned int widthA = strtoul(argv[1], NULL, 0);
    const unsigned int widthB = strtoul(argv[1], NULL, 0);
    const unsigned int heightB = widthA;
    const unsigned int widthC = widthB;
    const unsigned int heightC = heightA;
    const unsigned int runs = atoi(argv[2]);
    bool validating = false;
    bool reduced_output = false;
    bool coldstart = false;

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
        if ((strcmp(argv[i], "-d") == 0 ) && (i+1<argc)) {
            devices = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "-c") == 0) {
            coldstart = true;
        }
    }

    print_device_info(devices);

    const unsigned long int sizeA = widthA * heightA;
    const unsigned long int sizeB = widthB * heightB;
    const unsigned long int sizeC = widthC * heightC;

    double datasize_bytes = (double)(
        (1.0*widthA*heightA)
        +(1.0*widthB*heightB)
        +(1.0*widthC*heightC))*sizeof(array_type);
    double operations = (double)2.0*heightC*widthC*widthA;
    std::cout << "Multiplying arrays of size " 
              << widthA << "x" << heightA
              << " and "
              << widthB << "x" << heightB
              << ", resulting in "
              << widthC << "x" << heightC
              << "\n";
    std::cout << "Input arrays using " 
              << datasize_bytes / 1e9
              << "GB of memory and "
              << operations / 1e9
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

    array_type* matrixA_ref = NULL;
    array_type* matrixB_ref = NULL;

    setup_managed(&matrixA_ref, sizeA, validating);
    setup_managed(&matrixB_ref, sizeB, validating);

    int timings = 0;
    struct timing_stat* all_timings = NULL;
    
    double* timing_μs = (double*)calloc(runs, sizeof(double));

    if ((widthA != heightA) || (widthA != widthB) || (widthA != heightB)) {
        std::cout << "Cannot run cannon algorithm for uneven matrix sizes\n";
    }
    else {
        if (true) { // Benchmark cannon multi GPU on device basis
            int quadrants = 2;
            std::cout << "\nBenchmarking cannon multi GPU on device basis with " << quadrants << " quadrants per dimension *****\n";
                
            array_type* matrixA = NULL;
            array_type* matrixB = NULL;
            array_type* matrixC = NULL;
            map_managed(&matrixA, &matrixA_ref, widthA, heightA, validating, quadrants);
            map_managed(&matrixB, &matrixB_ref, widthB, heightB, validating, quadrants);
            zero_managed(&matrixC, sizeC, validating);

            if (!coldstart) {
                std::cout << "  Running a warmup\n";

                cannon::multiGPU<array_type, BLOCK_N>(
                    matrixA, matrixB, matrixC, widthC, devices, quadrants, validating
                );
            }
            else {
                std::cout << "  Skipping warmup\n";
            }

            bool validate = false;

            for (int run=0; run<runs; run++) {
                if ((validating) && (run==runs-1)) {
                    validate = true;
                }

                timing_μs[run] = cannon::multiGPU<array_type, BLOCK_N>(
                    matrixA, matrixB, matrixC, widthC, devices, quadrants, validate
                );

                if (reduced_output == false) {
                    print_loop_feedback(run, runs, timing_μs[run]);
                }

                // do this at the end as reading output array will shift it back to 
                // the host. Just use datasize_GB as crude tolerance for now.
                if (validate) {
                    printf("  Starting validation check\n");

                    array_type* matrixCz = (array_type*)malloc(sizeC*sizeof(array_type));
                    from_block(matrixC, matrixCz, widthC, heightC, widthC/quadrants);

                    validateCPU(
                        &matrixA_ref, widthA, heightA, 
                        &matrixB_ref, widthB, heightB, 
                        &matrixCz, datasize_bytes/1e9, widthA/quadrants
                    );
                    free(matrixCz);
                    if (false) {
                        std::cout << "Input A: \n";
                        print_matrix(matrixA, widthA, heightA);
                        std::cout << "Input B: \n";
                        print_matrix(matrixB, widthB, heightA);
                        std::cout << "Result: \n";
                        print_matrix_z(matrixC, widthC, quadrants);
                        cpuMatMul<array_type>(
                            matrixA_ref, widthA, heightA, 
                            matrixB_ref, widthB, heightB, 
                            matrixC
                        );
                        std::cout << "Reference: \n";
                        print_matrix(matrixC, widthC, heightC);
                    }
                }
            }

            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);

            update_and_print_timing_stats(
                timing_μs, runs, "cannon multi GPU", &all_timings, &timings, 
                operations, datasize_bytes
            );
        }  

        if (true) { // Benchmark cannon overlapping multi GPU on device basis
            int quadrants = 4;
            std::cout << "\nBenchmarking cannon overlapping multi GPU on device basis with " << quadrants << " quadrants per dimension *****\n";
                
            array_type* matrixA = NULL;
            array_type* matrixB = NULL;
            array_type* matrixC = NULL;
            map_managed(&matrixA, &matrixA_ref, widthA, heightA, validating, quadrants);
            map_managed(&matrixB, &matrixB_ref, widthB, heightB, validating, quadrants);
            zero_managed(&matrixC, sizeC, validating);

            if (!coldstart) {
                std::cout << "  Running a warmup\n";

                cannon::overlappingMultiGPU<array_type, BLOCK_N>(
                    matrixA, matrixB, matrixC, widthC, devices, quadrants, false
                );
            }
            else {
                std::cout << "  Skipping warmup\n";
            }

            bool validate = false;

            for (int run=0; run<runs; run++) {
                if ((validating) && (run==runs-1)) {
                    validate = true;
                }

                timing_μs[run] = cannon::overlappingMultiGPU<array_type, BLOCK_N>(
                    matrixA, matrixB, matrixC, widthC, devices, quadrants, validate
                );

                if (reduced_output == false) {
                    print_loop_feedback(run, runs, timing_μs[run]);
                }

                // do this at the end as reading output array will shift it back to 
                // the host. Just use datasize_GB as crude tolerance for now.
                if (validate) {
                    printf("  Starting validation check\n");

                    array_type* matrixCz = (array_type*)malloc(sizeC*sizeof(array_type));
                    from_block(matrixC, matrixCz, widthC, heightC, widthC/quadrants);

                    validateCPU(
                        &matrixA_ref, widthA, heightA, 
                        &matrixB_ref, widthB, heightB, 
                        &matrixCz, datasize_bytes/1e9, widthA/quadrants
                    );
                    free(matrixCz);
                    if (false) {
                        std::cout << "Input A: \n";
                        print_matrix(matrixA, widthA, heightA);
                        std::cout << "Input B: \n";
                        print_matrix(matrixB, widthB, heightA);
                        std::cout << "Result: \n";
                        print_matrix_z(matrixC, widthC, quadrants);
                        cpuMatMul<array_type>(
                            matrixA_ref, widthA, heightA, 
                            matrixB_ref, widthB, heightB, 
                            matrixC
                        );
                        std::cout << "Reference: \n";
                        print_matrix(matrixC, widthC, heightC);
                    }
                }
            }

            free_managed(&matrixA);
            free_managed(&matrixB);
            free_managed(&matrixC);

            update_and_print_timing_stats(
                timing_μs, runs, "cannon overlapping GPU", &all_timings, &timings, 
                operations, datasize_bytes
            );
        }  
    }

    free(timing_μs);
}