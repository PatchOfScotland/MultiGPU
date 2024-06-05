
#include "memtest/singlegpu.h"
#include "memtest/spatdat2d.h"
#include "shared_cuda.cu.h"
#include "shared.h"

typedef float array_type;

template<class T, int PageSize, int Splits>
int get_page_order_index(size_t x, size_t y, size_t width, size_t height) {
    int tileX = 1;
    int tileY = 1;

    int tileInnerX = 1;
    int tileInnerY = 1;

    return tileInnerY // offset within each tile for its X index
        + width*tileInnerX // offset within each tile for its Y index
        + PageSize*tileX // offset for each tile in X direction
        + tileY*height*width;
}

void setup_managed(
    array_type** matrix, const unsigned int array_h, const unsigned int array_w
) {
    size_t size = array_h*array_w;
    CCC(cudaMallocManaged(matrix, size*sizeof(array_type)));
}

void setup_init_managed(
    array_type** matrix, const unsigned int array_h, const unsigned int array_w
) {
    size_t size = array_h*array_w;

    setup_managed(matrix, array_h, array_w);
    init_matrix<array_type>(*matrix, size);
}

template<typename T>
void validate(
    const T* matrixA, const T* matrixB, const unsigned int array_n
) {
    bool match = true;
    #pragma omp parallel for collapse(2)
    for (int i=0; i<array_n; ++i) {
        for (int j=0; j<array_n; ++j) {
            unsigned long int index = i*array_n + j;
            //printf("%d %d %ld\n", i, j, index);

            if (matrixA[index] != matrixB[index]) {
                match = false;
            }
        }
    }
    if (match) {
        std::cout << "  Result is correct\n";
    } else {
        std::cout << "  Result is incorrect. Skipping any "
                << "subsequent runs\n";
    }
}

int main(int argc, char** argv){
    if (argc < 4) {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array h> <array w> <benchmark repeats> -d(devices) <devices>\n";
        exit(EXIT_FAILURE);
    } 
 
    const unsigned int array_h = strtoul(argv[1], NULL, 0);
    const unsigned int array_w = strtoul(argv[2], NULL, 0);
    //if (array_h > 37550) {
    //    std::cout << "array_n is too large. Max is 37550\n";
    //    exit(1);         
    //}
    const unsigned int runs = atoi(argv[3]);

    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int devices;
    CCC(cudaGetDeviceCount(&devices));

    for (int i=0; i<argc; i++) {
        if ((strcmp(argv[i], "-d") == 0 ) && (i+1<argc)) {
            devices = atoi(argv[i+1]);
        }
    }

    array_type* matrixA = NULL;
    array_type* matrixB = NULL;

    unsigned long int datasize_bytes = (unsigned long int)((array_h*array_w)*2*sizeof(array_type));
    unsigned long int operations = (unsigned long int)0;

    printf("Copying matrix of %dx%d (%fGB)\n", array_h, array_w, datasize_bytes*1e-9);
    printf("Running each test %d times\n", runs);

    int timings = 0;
    struct timing_stat* all_timings = NULL;

    float* timing_ms = (float*)calloc(runs, sizeof(float));

    setup_init_managed(&matrixA, array_h, array_w);
    setup_managed(&matrixB, array_h, array_w);

    for (int i=0; i<array_h; i++) {
        for (int j=0; j<array_w; j++) {
            matrixA[i*array_w + j] = i*array_w + j + 1;
        }
    }

    const int page_size = 3;
    const int splits = 2;
    z_order<array_type, page_size, splits>(matrixA, matrixB, 0, 0, array_w, array_h);

    if (true) {
        std::cout << "Matrix A: \n";
        print_matrix(matrixA, array_w, array_h);
        std::cout << "Matrix B: \n";
        print_matrix(matrixB, array_w, array_h);
    }

    std::cout << "\n";
    for (int x=0; x<array_h; x++) {
        for (int y=0; y<array_w; y++) {
            int i = get_page_order_index<array_type, page_size, splits>(
                    x, y, array_w, array_h
                );
            if (y==0) {
                std::cout << i;
            }
            else if (y==array_w-1) {
                std::cout << ", " << i << "\n";
            }
            else {
                std::cout << ", " << i;
            }
        }
    }

    if (false) { // Naive
        std::cout << "\n** Naive ******************************************\n";

        single_managed::copy_naive<array_type>(
            matrixA, matrixB, array_h
        );

        for (int run=0; run<runs; run++) {
            
            timing_ms[run] = single_managed::copy_naive<array_type>(
                matrixA, matrixB, array_h
            );
            print_loop_feedback(run, runs);

            // do this at the end as reading output array will shift it back to 
            // the host. Just use datasize_GB as crude tolerance for now.
            if (run==runs-1) {
                std::cout << "  Validating...\n";
                validate(matrixA, matrixB, array_h);
                std::cout << "  Validation complete\n";

                //std::cout << "Input A: \n";
                //print_matrix(matrixA, array_n, array_n);
                //std::cout << "Input B: \n";
                //print_matrix(matrixB, array_n, array_n);
            }
        }

        update_and_print_timing_stats(
            timing_ms, runs, "naive managed", &all_timings, &timings, 
            operations, datasize_bytes
        );
    }

    cudaFree(matrixA);
    cudaFree(matrixB);
}