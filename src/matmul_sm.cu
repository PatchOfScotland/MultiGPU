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
    array_type** matrixC, const unsigned long int sizeC
) {
    CCC(cudaMallocManaged(matrixA, sizeA*sizeof(array_type)));
    CCC(cudaMallocManaged(matrixB, sizeB*sizeof(array_type)));
    CCC(cudaMallocManaged(matrixC, sizeC*sizeof(array_type)));
    init_matrix<array_type>(*matrixA, sizeA);
    init_matrix<array_type>(*matrixB, sizeB);
}

void setup_trans_managed(
    array_type** input_matrix, array_type** output_matrix, 
    const unsigned long int width, const unsigned long int height
) {
    CCC(cudaMallocManaged(output_matrix, width*height*sizeof(array_type)));
    transpose_matrix(*input_matrix, width, height, *output_matrix);
}

void free_ABC_managed( 
    array_type** matrixA, array_type** matrixB, array_type** matrixC
) {
    CCC(cudaFree(*matrixA));
    CCC(cudaFree(*matrixB));
    CCC(cudaFree(*matrixC));
}

void print_stats(float* timing_array, int* value_array, const char* name, const int runs) {
    std::cout << "Experiment: " << name << "\n"; 
    for (int run=0; run<runs; run++) {
        std::cout << value_array[run] << "," << timing_array[run] << "\n";
    }
}

int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array n> <benchmark repeats> -s(standalone)\n";
        exit(EXIT_FAILURE);
    } 

    const unsigned int heightA = strtoul(argv[1], NULL, 0);
    const unsigned int widthA = strtoul(argv[1], NULL, 0);
    const unsigned int widthB = strtoul(argv[1], NULL, 0);
    const unsigned int heightB = widthA;
    const unsigned int widthC = widthB;
    const unsigned int heightC = heightA;
    const unsigned int repeats = atoi(argv[2]);
    bool standalone = false;

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int devices;
    CCC(cudaGetDeviceCount(&devices));

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-s") == 0) {
            standalone = true;
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

    if (standalone) {
        std::cout << "Creating new datasets for each repeat\n";
    }

    array_type* matrixA = NULL;
    array_type* matrixB = NULL;
    array_type* matrixTransB = NULL;
    array_type* matrixC = NULL;
    
    float* timing_ms = (float*)calloc(repeats, sizeof(float));
    
    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
    setup_trans_managed(&matrixB, &matrixTransB, widthB, heightB);

    const int page_size = PAGE_SIZE / sizeof(array_type);
    // Get this more dynamically determined

    int runs = 26;
    int sm_counts[runs] = {1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500};

    float matched[runs] = {};
    float offset[runs] = {};

    if (true) { // Benchmark a prefetching page-tiled multi GPU
        for (int run=0; run<runs; run++)
        {
            const int sm_count = sm_counts[run];

            std::cout << "Benchmarking matched with SM of " << sm_count << "\n";

            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
            }

            prefetch_page_tiled_sm::multiGPU<false, array_type, page_size>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, false, sm_count
            );

            if (standalone) {
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
            }

            for (int repeat=0; repeat<repeats; repeat++) {
                if (standalone) {
                    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                }

                timing_ms[repeat] = prefetch_page_tiled_sm::multiGPU<
                    false, array_type, page_size
                >(
                    matrixA, widthA, heightA, 
                    matrixB, widthB, heightB, 
                    matrixC, widthC, heightC,
                    devices, false, sm_count
                );

                if (standalone) {
                    free_ABC_managed(&matrixA, &matrixB, &matrixC);
                }
            }

            float min = timing_ms[0];
            float max = timing_ms[0];
            float total = 0;

            for (int repeat=0; repeat<repeats; repeat++) {
                total = total + timing_ms[repeat];
                if (timing_ms[repeat] < min) {
                    min = timing_ms[repeat];
                }
                if (timing_ms[repeat] > max) {
                    max = timing_ms[repeat];
                }
            }
            matched[run] = total/repeats;
        }
    }


    if (true) { // Benchmark an offset prefetching page-tiled multi GPU
        for (int run=0; run<runs; run++)
        {
            const int sm_count = sm_counts[run]; //20 aarhus, 84 hendrix03
            std::cout << "Benchmarking offset with SM of " << sm_count << "\n";

            if (standalone) {
                setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
            }
            
            prefetch_page_tiled_sm::multiGPU<false, array_type, page_size>(
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, heightC,
                devices, true, sm_count
            );

            if (standalone) {
                free_ABC_managed(&matrixA, &matrixB, &matrixC);
            }

            for (int repeat=0; repeat<repeats; repeat++) {
                if (standalone) {
                    setup_ABC_managed(&matrixA, sizeA, &matrixB, sizeB, &matrixC, sizeC);
                }

                timing_ms[repeat] = prefetch_page_tiled_sm::multiGPU<
                    false, array_type, page_size
                >(
                    matrixA, widthA, heightA, 
                    matrixB, widthB, heightB, 
                    matrixC, widthC, heightC,
                    devices, true, sm_count
                );

                if (standalone) {
                    free_ABC_managed(&matrixA, &matrixB, &matrixC);
                }
            }

            float min = timing_ms[0];
            float max = timing_ms[0];
            float total = 0;

            for (int repeat=0; repeat<repeats; repeat++) {
                total = total + timing_ms[repeat];
                if (timing_ms[repeat] < min) {
                    min = timing_ms[repeat];
                }
                if (timing_ms[repeat] > max) {
                    max = timing_ms[repeat];
                }
            }
            offset[run] = total/repeats;
        }
    }

    print_stats(matched, sm_counts, "matched\0", runs);
    print_stats(offset, sm_counts, "offset\0", runs);
}