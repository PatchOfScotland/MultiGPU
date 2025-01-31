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
    array_type** matrixC, array_type tolerance, int split, bool cpu
) {
    if (cpu) {
        array_type* matrixAz = (array_type*)malloc(widthA*heightA*sizeof(array_type));
        array_type* matrixBz = (array_type*)malloc(widthB*heightB*sizeof(array_type));
        array_type* matrixCz = (array_type*)malloc(widthB*heightB*sizeof(array_type));

        z_order<array_type>(*matrixA, matrixAz, widthA, heightA, split);
        z_order<array_type>(*matrixB, matrixBz, widthB, heightB, split);
        z_order<array_type>(*matrixC, matrixCz, widthB, heightB, split);

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
                std::cout << matrixRef[i] << " - " << matrixValidating[i] << "\n";
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
                  << " <array N> <benchmark repeats> -d(devices) <devices> -v(validation) -s(standalone) -r(reduced output) -c (coldstart)\n";
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
    bool standalone = false;
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
        if (strcmp(argv[i], "-s") == 0) {
            standalone = true;
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
    if (standalone) {
        std::cout << "Creating new datasets for each run\n";
    }

    array_type* matrixA = NULL;
    array_type* matrixA_REF = NULL;
    array_type* matrixB = NULL;
    array_type* matrixC = NULL;

    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC, validating);
    setup_managed(&matrixA_REF, sizeA, validating);

    int timings = 0;
    struct timing_stat* all_timings = NULL;
    
    float* timing_μs = (float*)calloc(runs, sizeof(float));

    if ((widthA != heightA) || (widthA != widthB) || (widthA != heightB)) {
        std::cout << "Cannot run cannon algorithm for uneven matrix sizes\n";
    }
    else {
        if (true) { // Benchmark cannon multi GPU on device basis
            int quadrants = 2;
            std::cout << "\nBenchmarking cannon multi GPU on device basis with " << quadrants << " quadrants per dimension *****\n";
                
            array_type* matrixA_DUP = NULL;
            map_managed(&matrixA_DUP, &matrixA_REF, sizeA, validating);

            //printf("REF:\n");
            //print_matrix(matrixA_REF, widthA, heightA);
            //printf("DUP:\n");
            //print_matrix(matrixA_DUP, widthA, heightA);

            if (!coldstart) {
                std::cout << "  Running a warmup\n";

                if (standalone) {
                    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC, validating);
                }

                cannon::multiGPU<array_type, BLOCK_N>(
                    matrixA, matrixB, matrixC, widthC, devices, quadrants, validating
                );

                if (standalone) {
                    free_ABC_managed(&matrixA, &matrixB, &matrixC);
                }
            }
            else {
                std::cout << "  Skipping warmup\n";
            }

            bool validate = false;

            for (int run=0; run<runs; run++) {
                if (standalone) {
                    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC, validating);
                    zero_matrix(matrixC, widthC* heightC);
                }

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
                if ((validating) && (run==runs-1)) {
                    const int split = widthC / quadrants;
                    validateZorder(
                        &matrixA, widthA, heightA, 
                        &matrixB, widthB, heightB, 
                        &matrixC, datasize_bytes/1e9, split, true
                    );
                    if (false) {
                        std::cout << "Input A: \n";
                        print_matrix_z(matrixA, widthA, quadrants);
                        std::cout << "Input B: \n";
                        print_matrix_z(matrixB, widthB, quadrants);
                        std::cout << "Result: \n";
                        print_matrix_z(matrixC, widthC, quadrants);
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
                    free_ABC_managed(&matrixA, &matrixB, &matrixC);
                }
            }
            update_and_print_timing_stats(
                timing_μs, runs, "cannon multi GPU", &all_timings, &timings, 
                operations, datasize_bytes
            );

            free_managed(&matrixA_DUP);
        }  

        if (false) { // Benchmark cannon overlapping multi GPU on device basis
            int quadrants = 4;
            std::cout << "\nBenchmarking cannon overlapping multi GPU on device basis with " << quadrants << " quadrants per dimension *****\n";
                
            if (!coldstart) {
                std::cout << "  Running a warmup\n";

                if (standalone) {
                    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC, validating);
                }

                cannon::overlappingMultiGPU<array_type, BLOCK_N>(
                    matrixA, matrixB, matrixC, widthC, devices, quadrants, false
                );

                if (standalone) {
                    free_ABC_managed(&matrixA, &matrixB, &matrixC);
                }
            }
            else {
                std::cout << "  Skipping warmup\n";
            }

            bool validate = false;

            for (int run=0; run<runs; run++) {
                if (standalone) {
                    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC, validating);
                    zero_matrix(matrixC, widthC* heightC);
                }

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
                if ((validating) && (run==runs-1)) {
                    const int split = widthC / quadrants;
                    
                    validateZorder(
                        &matrixA, widthA, heightA, 
                        &matrixB, widthB, heightB, 
                        &matrixC, datasize_bytes/1e9, split, false
                    );
                    if (true) {
                        std::cout << "Input A: \n";
                        print_matrix(matrixA, widthA, heightA);
                        std::cout << "Input B: \n";
                        print_matrix_z(matrixB, widthB, quadrants);
                        std::cout << "Result: \n";
                        print_matrix_z(matrixC, widthC, quadrants);

                        cannon::singleGPU<array_type, BLOCK_N>(
                            matrixA, matrixB, matrixC, widthC
                        );
                        std::cout << "Reference Single GPU: \n";
                        print_matrix(matrixC, widthC, heightC);

                        cannon::multiGPU<array_type, BLOCK_N>(
                            matrixA, matrixB, matrixC, widthC, 4, 1, true
                        );
                        std::cout << "Reference Multi GPU (1): \n";
                        print_matrix(matrixC, widthC, heightC);

                        cannon::multiGPU<array_type, BLOCK_N>(
                            matrixA, matrixB, matrixC, widthC, 4, 2, true
                        );
                        std::cout << "Reference Multi GPU (2): \n";
                        print_matrix(matrixC, widthC, heightC);

                        //cannon::multiGPU<array_type, BLOCK_N>(
                        //    matrixA, matrixB, matrixC, widthC, 4, 4, true
                        //);
                        //std::cout << "Reference Multi GPU (4): \n";
                        //print_matrix(matrixC, widthC, heightC);


                    }
                }

                if (standalone) {
                    free_ABC_managed(&matrixA, &matrixB, &matrixC);
                }
            }
            update_and_print_timing_stats(
                timing_μs, runs, "cannon overlapping GPU", &all_timings, &timings, 
                operations, datasize_bytes
            );
        }  
    }

    free(timing_μs);
}