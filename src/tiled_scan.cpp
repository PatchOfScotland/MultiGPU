#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>


typedef float array_type;

#define TILESIZE 2

float get_throughput(float mean_ms, double data_gigabytes) {
    float mean_seconds = mean_ms * 1e-6f;
    return (float)data_gigabytes / mean_seconds;
}

float get_flops(float mean_ms, long int operations) {
    float mean_seconds = mean_ms * 1e-6f;
    float giga_operations = operations /1e9f;
    return (float)giga_operations / mean_seconds;
}

float print_timing_stats(float* timing_array, size_t array_len, long int operations, double data_gigabytes, float naive_runtime_ms, float tiled_runtime_ms, float tiled_openmp_runtime_ms) {
    float mean_ms = 0;
    float min = timing_array[0];
    float max = timing_array[0];
    float total = 0;

    for (int i=0; i<array_len; i++) {
        total = total + timing_array[i];
        if (timing_array[i] < min) {
            min = timing_array[i];
        }
        if (timing_array[i] > max) {
            max = timing_array[i];
        }
    }
    mean_ms = total/array_len;

    float gigabytes_per_second = get_throughput(mean_ms, data_gigabytes);
    float gigaflops_per_second = get_flops(mean_ms, operations);
    printf("    Total runtime: %fms\n", total);
    printf("    Min runtime:   %fms\n", min);
    printf("    Max runtime:   %fms\n", max);
    printf("    Mean runtime:  %fms\n", mean_ms);
    printf("    Throughput:    %fGB/sec\n", gigabytes_per_second);
    printf("    GFLOPS/s:      %f/sec\n", gigaflops_per_second);
    
    if (naive_runtime_ms != -1) {
        printf("      Speedup vs naive:             x%f\n", naive_runtime_ms / mean_ms);
    }
    if (tiled_runtime_ms != -1) {
        printf("      Speedup vs tiled:             x%f\n", tiled_runtime_ms / mean_ms);
    }
    if (tiled_openmp_runtime_ms != -1) {
        printf("      Speedup vs tiled with openMP: x%f\n", tiled_openmp_runtime_ms / mean_ms);
    }
    return mean_ms;
}

void print_loop_feedback(int run, int runs) {
    if (run==0) {
        printf("  Completed run %d/%d", run+1, runs);
        fflush(stdout);
    }
    else {
        printf("\r  Completed run %d/%d", run+1, runs);
        fflush(stdout);
    }
    if (run==runs-1) {
        printf("\n");
    }
}

void print_array(array_type* matrix, size_t height, size_t width) {
    for (int row=0; row<height; row++) {
        for (int column=0; column<width; column++) {
            //printf("%d,%d: %f\n", row, column, matrix[row*width + column]);
            printf("%f", matrix[row*width + column]);
            if (column == width-1) {
                printf("\n");
            } else {
                printf(", ");
            }
        }
    }
}

void init_matrix(array_type* arr, unsigned int size) {
    for(int i=0; i<size; i++) {
        arr[i] = (array_type)rand() / RAND_MAX;
    }
}

void zero_matrix(array_type* arr, unsigned int size) {
    for(int i=0; i<size; i++) {
        arr[i] = 0;
    }
}

bool compare_arrays(array_type* array_1, array_type* array_2, size_t array_len){

    bool status = true;
    #pragma omp parallel for
    for(size_t i=0; i<array_len; i++){
        if (array_1[i] != array_2[i]){
            status = false;
        }
    }
    return status;
}

void runNaive(
    int height_A, int width_A, int width_B,
    array_type* matrix_A, array_type* matrix_B, array_type* matrix_C
) {
    for(int i = 0; i < height_A; ++i) {
        for(int j = 0; j < width_B; ++j) {
            array_type acc = 0;
            int c = i*width_B + j;
            for(int k = 0; k < width_A; ++k) {
                int a = i*width_A + k;
                int b = k*width_B + j;
                //printf("C[%d][%d] -> C[%d]\n", i, j, c);
                //printf("A[%d][%d] -> A[%d]\n", i, k, a);
                //printf("B[%d][%d] -> B[%d]\n", k, j, b);
                //printf("\n");
                acc += matrix_A[a] * matrix_B[b];
            }
            matrix_C[c] = acc;
        }
    }
}

template<int tileSize>
void runTiled( 
    int height_A, int width_A, int width_B,
    array_type* matrix_A, array_type* matrix_B, array_type* matrix_C
) {
    for (int rowTile = 0; rowTile < height_A; rowTile += 256) {
        
        for (int columnTile = 0; columnTile < width_A; columnTile += 256) {

            for (int innerTile = 0; innerTile < width_B; innerTile += tileSize) {
        
                for (int row = rowTile; row < rowTile + 256; row++) {
                    int innerTileEnd = std::min(width_B, innerTile + tileSize);
        
                    for (int inner = innerTile; inner < innerTileEnd; inner++) {
        
                        for (int col = columnTile; col < columnTile + 256; col++) {
                            matrix_C[row* width_A + col] += matrix_A[row * width_B + inner] * matrix_B[inner * width_A + col];
                        }
                    }
                }            
            }
        }
    } 
}

template<int tileSize>
void runTiledOpenMP( 
    int height_A, int width_A, int width_B,
    array_type* matrix_A, array_type* matrix_B, array_type* matrix_C
) {
    #pragma omp parallel for shared(matrix_C, matrix_A, matrix_B) collapse(2) num_threads(8)
    for (int rowTile = 0; rowTile < height_A; rowTile += 256) {
        
        for (int columnTile = 0; columnTile < width_A; columnTile += 256) {

            for (int innerTile = 0; innerTile < width_B; innerTile += tileSize) {
        
                for (int row = rowTile; row < rowTile + 256; row++) {
                    int innerTileEnd = std::min(width_B, innerTile + tileSize);
        
                    for (int inner = innerTile; inner < innerTileEnd; inner++) {
        
                        for (int col = columnTile; col < columnTile + 256; col++) {
                            matrix_C[row * width_A + col] += matrix_A[row * width_B + inner] * matrix_B[inner * width_A + col];
                        }
                    }
                }            
            }
        }
    } 
}

int main(int argc, char** argv){
    if (argc < 5)
    {
        printf("Usage: tiled_scan <rowsA> <columnsA> <columnsB> <benchmark repeats> -v(optional) -s(optional) -r(optional\n"); 
        exit(EXIT_FAILURE);
    } 

    unsigned int widthA = atoi(argv[2]);
    unsigned int heightA = atoi(argv[1]);
    unsigned int widthB = atoi(argv[3]);;
    unsigned int heightB = widthA;
    unsigned int widthC = widthB;
    unsigned int heightC = heightA;
    unsigned int runs = atoi(argv[4]);
    bool validating = false;
    bool skip = false;
    bool reduced_output = false;

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
        if (strcmp(argv[i], "-s") == 0) {
            skip = true;
        }
        if (strcmp(argv[i], "-r") == 0) {
            reduced_output = true;
        }
    }

    double datasize = (( ((widthA*heightA)+(widthB*heightB)+(widthC*heightC))*sizeof(array_type))/1e9);

    printf("Input arrays of %dx%d and %dx%d give output of %dx%d\n", widthA, heightA, widthB, heightB, widthC, heightC);

    printf("Using %fGB of memory\n", datasize);
    if (validating) {
        printf("Will validate output\n");
    }
    else {
        printf("Skipping output validation\n");
    }
    if (skip) {
        printf("Skipping any significant processing\n");
    }

    array_type* matrixA;
    array_type* matrixB;
    array_type* matrixC;
    array_type* reference;
    struct timeval start_event;
    struct timeval end_event;
    float runtime_microseconds;
    float naive_time_ms = -1;
    float tiled_time_ms = -1;
    float openmp_time_ms = -1;

    matrixA = (array_type*)malloc(sizeof(array_type) * (heightA * widthA));
    matrixB = (array_type*)malloc(sizeof(array_type) * (heightB * widthB));
    matrixC = (array_type*)malloc(sizeof(array_type) * (heightB * widthB));
    reference = (array_type*)malloc(sizeof(array_type) * (heightC * widthC));

    float* timing_ms = (float*)calloc(runs, sizeof(float));
    long int operations = heightC * widthC * heightB * 2;

    printf("Will be running %ld FLOPS per run\n", operations);

    printf("Initialising input arrays\n");
    if (skip == false) {
        init_matrix(matrixA, heightA * widthA);
        init_matrix(matrixB, heightB * widthB);
        //zero_matrix(matrixC, heightC * widthC);
    }

    //printf("Input A:\n");
    //print_array(matrixA, heightA, widthA);
    //printf("Input B:\n");
    //print_array(matrixB, heightB, widthB);

    { // Get CPU baseline
        printf("Getting CPU result\n");

        struct timeval cpu_start_time;
        struct timeval cpu_end_time;

        gettimeofday(&cpu_start_time, NULL);

        if (skip == false) {
            runNaive(heightA, widthA, widthB, matrixA, matrixB, reference);
        }
        gettimeofday(&cpu_end_time, NULL); 

        runtime_microseconds = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
        printf("CPU reduction took: %fms\n", runtime_microseconds);
        printf("CPU throughput:     %fGB/sec\n", (float)datasize / runtime_microseconds);
    }

    //printf("Refernce:\n");
    //print_array(reference, heightC, widthC);

    { // Benchmark commutative single GPU
        printf("\nBenchmarking naive matrix multiplication *****************\n");

        printf("  Running a warmup\n");
        runNaive(heightA, widthA, widthB, matrixA, matrixB, matrixC);

        for (int run=0; run<runs; run++) {
            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            //zero_matrix(matrixC, heightC * widthC);
            gettimeofday(&start_event, NULL);
            runNaive(heightA, widthA, widthB, matrixA, matrixB, matrixC);
            gettimeofday(&end_event, NULL);

            runtime_microseconds = (end_event.tv_usec+(1e6*end_event.tv_sec)) - (start_event.tv_usec+(1e6*start_event.tv_sec));
            timing_ms[run] = runtime_microseconds;

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(compare_arrays(reference, matrixC, widthC*heightC) == true
                ){
                    printf("  Result is correct\n");
                } else {
                    printf("  Result is incorrect. Skipping any subsequent runs\n");
                    break;
                }
            }
        }

         naive_time_ms = print_timing_stats(
            timing_ms, runs, operations, datasize, naive_time_ms, tiled_time_ms, 
            openmp_time_ms
        );
    }


    { // Benchmark commutative single GPU
        printf("\nBenchmarking tiled matrix multiplication ***************\n");

        printf("  Running a warmup\n");
        runTiled<TILESIZE>(heightA, widthA, widthB, matrixA, matrixB, matrixC);

        for (int run=0; run<runs; run++) {
            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            zero_matrix(matrixC, heightC * widthC);
            gettimeofday(&start_event, NULL);
            runTiled<TILESIZE>(heightA, widthA, widthB, matrixA, matrixB, matrixC);
            gettimeofday(&end_event, NULL);

            runtime_microseconds = (end_event.tv_usec+(1e6*end_event.tv_sec)) - (start_event.tv_usec+(1e6*start_event.tv_sec));
            timing_ms[run] = runtime_microseconds;

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(compare_arrays(reference, matrixC, widthC*heightC) == true
                ){
                    printf("  Result is correct\n");
                } else {
                    printf("  Result is incorrect. Skipping any subsequent runs\n");
                    break;
                }
            }
        }

         openmp_time_ms = print_timing_stats(
            timing_ms, runs, operations, datasize, naive_time_ms, tiled_time_ms, 
            openmp_time_ms
        );
    }

    { // Benchmark commutative single GPU
        printf("\nBenchmarking openmp tiled matrix multiplication *********\n");

        printf("  Running a warmup\n");
        runTiledOpenMP<TILESIZE>(heightA, widthA, widthB, matrixA, matrixB, matrixC);

        for (int run=0; run<runs; run++) {
            if (reduced_output == false) {
                print_loop_feedback(run, runs);
            }

            zero_matrix(matrixC, heightC * widthC);
            gettimeofday(&start_event, NULL);
            runTiledOpenMP<TILESIZE>(heightA, widthA, widthB, matrixA, matrixB, matrixC);
            gettimeofday(&end_event, NULL);

            runtime_microseconds = (end_event.tv_usec+(1e6*end_event.tv_sec)) - (start_event.tv_usec+(1e6*start_event.tv_sec));
            timing_ms[run] = runtime_microseconds;

            // do this at the end as reading output array will shift it back to 
            // the host
            if (validating && run==runs-1) {
                if(compare_arrays(reference, matrixC, widthC*heightC) == true
                ){
                    printf("  Result is correct\n");
                } else {
                    printf("  Result is incorrect. Skipping any subsequent runs\n");
                    break;
                }
            }
        }

         openmp_time_ms = print_timing_stats(
            timing_ms, runs, operations, datasize, naive_time_ms, tiled_time_ms, 
            openmp_time_ms
        );
    }

//    cudaFree(input_array);
}
