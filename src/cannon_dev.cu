#include <sys/time.h>
#include <limits>

#include "matmul/cannon.h"
#include "matmul/cpu.h"
#include "matmul/tiled.h"
#include "matmul/page_tile.h"
#include "matmul/prefetch_page_tile.h"
#include "shared_cuda.cu.h"
#include "shared.h"

typedef float array_type;

void setup_ABC_managed(
    array_type** matrixA, const unsigned long int sizeA,
    array_type** matrixB, const unsigned long int sizeB,
    array_type** matrixC, const unsigned long int sizeC,
    bool validating
) {
    CCC(cudaMallocManaged(matrixA, sizeA*sizeof(array_type)));
    CCC(cudaMallocManaged(matrixB, sizeB*sizeof(array_type)));
    CCC(cudaMallocManaged(matrixC, sizeC*sizeof(array_type)));
    if (validating) {
        init_matrix_linear<array_type>(*matrixA, sizeA);
        init_matrix_linear<array_type>(*matrixB, sizeB);
    }
}

void free_ABC_managed( 
    array_type** matrixA, array_type** matrixB, array_type** matrixC
) {
    CCC(cudaFree(*matrixA));
    CCC(cudaFree(*matrixB));
    CCC(cudaFree(*matrixC));
}

void validateZorder(
    array_type** matrixA, const unsigned int widthA, const unsigned int heightA,
    array_type** matrixB, const unsigned int widthB, const unsigned int heightB,
    array_type** matrixC, array_type tolerance, int split
) {
    array_type* matrixAz = (array_type*)malloc(widthA*heightA*sizeof(array_type));
    array_type* matrixBz = (array_type*)malloc(widthB*heightB*sizeof(array_type));

    z_order<array_type>(*matrixA, matrixAz, widthA, heightA, split);
    z_order<array_type>(*matrixB, matrixBz, widthB, heightB, split);

    if(cpuValidation<array_type>(
        matrixAz, widthA, heightA, 
        matrixBz, widthB, heightB, 
        *matrixC, tolerance
    )){
        std::cout << "  Result is correct\n";
    } else {
        std::cout << "  Result is incorrect. Skipping any "
                  << "subsequent runs\n";
    }

    free(matrixAz);
    free(matrixBz);
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
    array_type* matrixC = NULL;
    
    int timings = 0;
    struct timing_stat* all_timings = NULL;
    
    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC, validating);

    float* timing_ms = (float*)calloc(runs, sizeof(float));

    if ((widthA != heightA) || (widthA != widthB) || (widthA != heightB)) {
        std::cout << "Cannot run cannon algorithm for uneven matrix sizes\n";
    }
    else {
        if (true) { // Benchmark cannon multi GPU on device basis
            std::cout << "\nBenchmarking cannon multi GPU on device basis *****\n";

            std::cout << "  Running a warmup\n";

            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC, validating);
            }

            //cannon::multiGPU<array_type, cannon_block, quadrants_per_dim>(
            //    matrixA, matrixB, matrixC, widthC, devices
            //);

            if (standalone) {
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
            }

            for (int run=0; run<runs; run++) {
                if (standalone) {
                    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC, validating);
                    zero_matrix(matrixC, widthC* heightC);
                }

                const size_t quadrants_per_dim = 3;
        
                std::cout << "Input A: \n";
                print_matrix(matrixA, widthA, heightA);
                std::cout << "Input B: \n";
                print_matrix(matrixB, widthB, heightB);

                timing_ms[run] = cannon::multiGPU<array_type>(
                    matrixA, matrixB, matrixC, widthC, devices, quadrants_per_dim
                );

                std::cout << "Result: \n";
                print_matrix(matrixC, widthC, heightC);

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
                    if (true) {
                        //std::cout << "Input A: \n";
                        //print_matrix(matrixA, widthA, heightA);
                        //std::cout << "Input B: \n";
                        //print_matrix(matrixB, widthB, heightB);
                        //std::cout << "Result: \n";
                        //print_matrix(matrixC, widthC, heightC);
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
                timing_ms, runs, "cannon multi GPU\0", &all_timings, &timings, 
                operations, datasize_bytes
            );
        }        
    }
}