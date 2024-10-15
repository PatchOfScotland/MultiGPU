#ifndef SHARED_H
#define SHARED_H

#include <iostream>
#include <iomanip>
#include <string>
#include <thread>

#define NO_HINTS            0
#define HINTS               1
#define PREFETCH            2

#define NO_REDUCE           0
#define MEMCPY              1
#define DUPLICATE           2

typedef float array_type;

const auto processor_count = std::thread::hardware_concurrency();

float get_throughput(float timing_microseconds, double data_bytes) {
    float timing_seconds = timing_microseconds * 1e-6f;
    float data_gigabytes = data_bytes /1e9f;
    return (float)data_gigabytes / timing_seconds;
}

float get_flops(float timing_microseconds, long int operations) {
    float timing_seconds = timing_microseconds * 1e-6f;
    float giga_operations = operations /1e9f;
    return (float)giga_operations / timing_seconds;
}

typedef struct timing_stat {
    float timing_microseconds;
    unsigned long int flops;
    unsigned long int datasize_bytes;
    const char* type;

    timing_stat(
        const char* type, 
        long int flops, 
        long int datasize_bytes
    ): type(type), 
       flops(flops), 
       datasize_bytes(datasize_bytes)
    {
        timing_microseconds = -1;
    }

    float throughput_gb() {
        return get_throughput(timing_microseconds, datasize_bytes);
    }

    float throughput_gf() {
        return get_flops(timing_microseconds, flops);
    }
} timing_stat;

void init_array(float* arr, unsigned long int array_len) {
    srand(5454);
    for(int i=0; i<array_len; i++) {
        arr[i] = (float)rand() / RAND_MAX;
    }
}

void init_sparse_array(float* arr, unsigned long int array_len, int n) {
    srand(5454);
    for(int i=0; i<n; i++) {
        unsigned long int random_index = rand() % (array_len);
        arr[random_index] += 1;
    }
}

template<class T>
void init_matrix(T* data, uint64_t size) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = rand() / (T)RAND_MAX;
}

template<class T>
void init_matrix_linear(T* data, uint64_t size) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = i;
        
}

template<class T>
void duplicate_matrix(T* origin, uint64_t size, T* target) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < size; i++)
        target[i] = origin[i];
}

template<class T>
void transpose_matrix(T* input, size_t width, size_t height, T* output) {
    #pragma omp parallel for collapse(2)
    for (int w=0; w<width; w++) {
        for (int h=0; h<height; h++) {
            output[w*height + h] = input[h*width + w];
        }
    }
}

// T is element type
// split is the length of a 'Z' grouping in both dimensions
template<class T>
int z_order(T* input,  T* output, size_t width, size_t height, int split) {

    if (width%split) {
        printf("Cannot convert to z order. Width of %zu does not divide by %d\n", width, split);
        return 1;
    }
    if (height%split) {
        printf("Cannot convert to z order. Height of %zu does not divide by %d\n", height, split);
        return 1;
    }

    int tileCountX = width/split;
    int tileCountY = height/split;

    //#pragma omp parallel for collapse(4)
    for (int tileX=0; tileX<tileCountX; tileX++) {
        for (int tileY=0; tileY<tileCountY; tileY++) {
            for (int tileInnerX=0; tileInnerX<split; tileInnerX++) {
                for (int tileInnerY=0; tileInnerY<split; tileInnerY++) {
                    int input_offset = tileInnerY // offset within each tile for its X index
                         + split*tileInnerX // offset within each tile for its Y index
                         + tileX*split*split  // offset for each tile in X direction
                         + tileY*split*width;
                    int output_offset = tileInnerY // offset within each tile for its X index
                        + width*tileInnerX // offset within each tile for its Y index
                        + split*tileX // offset for each tile in X direction
                        + tileY*split*width; // offset for each tile in Y direction
                    output[output_offset] = input[input_offset];
                }
            }
        }
    }

    return 0;
}

template<class T>
void zero_matrix(T* data, uint64_t size) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < size; i++)
        data[i] = 0;
}

template<class T>
bool compare_arrays(T* array_1, T* array_2, size_t array_len){

    bool status = true;
    #pragma omp parallel for
    for(size_t i=0; i<array_len; i++){
        if (array_1[i] != array_2[i]){
            status = false;
        }
    }
    return status;
}

template<class T>
void print_array(T* timing_array, size_t array_len) {
    for (int i=0; i<array_len; i++) {
        if (i==0) {
            std::cout << timing_array[i];
        }
        else if (i==array_len-1) {
            std::cout << ", " << timing_array[i] << "\n";
        }
        else {
            std::cout << ", " << timing_array[i];
        }
    }
}

template<class T>
void print_matrix(T* matrix, size_t width, size_t height) {
    for (int c=0; c<height; c++) {
        for (int r=0; r<width; r++) {
            if (r==0) {
                printf("%8f", matrix[r+(width*c)]);
                //std::cout << matrix[r+(width*c)];
            }
            else if (r==width-1) {
                printf(", %8f\n", matrix[r+(width*c)]);
                //std::cout << ", " << matrix[r+(width*c)] << "\n";
            }
            else {
                printf(", %8f", matrix[r+(width*c)]);
                //std::cout << ", " << matrix[r+(width*c)];
            }
        }
    }
}

template<class T>
void print_matrix_z(T* matrix, size_t n, size_t quadrants_per_dim) {
    int split = n/quadrants_per_dim;

    for (int vert_quad=0; vert_quad<quadrants_per_dim; vert_quad++) {
        for (int quad_y=0; quad_y<split; quad_y++) {
            for (int hor_quad=0; hor_quad<quadrants_per_dim; hor_quad++) {
                int quadrant = hor_quad + (vert_quad*quadrants_per_dim); {
                    for (int quad_x=0; quad_x<split; quad_x++) {
                        std::cout << matrix[
                                (quadrant*split*split) // offset for whole quadrant
                                + quad_x // offset for horizontal
                                + (quad_y * split) // offset for vertical
                            ] << ", ";
                    }
                }
            }
            std::cout << "\n";
        }
    }
}

void _update_and_print_timing_stats(
    float* timing_array, size_t array_len, const char* title, 
    timing_stat** all_timings_ptr, int* timings, 
    unsigned long int operations, unsigned long int datasize_bytes, bool print
) {
    if (*timings == 0) {
        int malloc_size = (*timings+1)*sizeof(timing_stat);
        *all_timings_ptr = (timing_stat*)malloc(malloc_size);
    }
    else {
        timing_stat* all_timings_copy = *all_timings_ptr;
        int malloc_size = (*timings+1)*sizeof(timing_stat);
        *all_timings_ptr = (timing_stat*)realloc(*all_timings_ptr, malloc_size);
        if (*all_timings_ptr == NULL) {
            printf("Realloc failed\n");
            free(all_timings_copy);
            exit(1);
        }
    }
    timing_stat* all_timings = *all_timings_ptr;
    
    struct timing_stat to_update = 
        timing_stat(title, operations, datasize_bytes);

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

    if (print) {
        float gigabytes_per_second = get_throughput(mean_ms, to_update.datasize_bytes);
        float gigaflops_per_second = get_flops(mean_ms, to_update.flops);
        std::cout << "    Total runtime: " << total <<"ms\n";
        std::cout << "    Min runtime:   " << min <<"ms\n";
        std::cout << "    Max runtime:   " << max <<"ms\n";
        std::cout << "    Mean runtime:  " << mean_ms <<"ms\n";
        std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/sec\n";
        std::cout << "    GLFOPS:        " << gigaflops_per_second <<"/sec\n";

        for (int i=0; i<*timings; i++) {
            struct timing_stat timing = all_timings[i];
            if (timing.timing_microseconds != -1) {
                std::cout << "      Speedup vs "
                          << timing.type 
                          << ": x" 
                          << timing.timing_microseconds / mean_ms
                          << "\n";
            }
        }
    }
    to_update.timing_microseconds = mean_ms;   
    memcpy(&all_timings[*timings].timing_microseconds, &to_update.timing_microseconds, sizeof(to_update.timing_microseconds));
    memcpy(&all_timings[*timings].flops, &to_update.flops, sizeof(to_update.flops));
    memcpy(&all_timings[*timings].datasize_bytes, &to_update.datasize_bytes, sizeof(to_update.datasize_bytes));
    memcpy(&all_timings[*timings].type, &to_update.type, sizeof(to_update.type));

    *timings = *timings + 1;  
}

void update_timing_stats(
    float* timing_array, size_t array_len, const char* title, 
    timing_stat* all_timings[], int* timings, 
    unsigned long int operations, unsigned long int datasize_bytes
) {
    _update_and_print_timing_stats(
        timing_array, array_len, title, all_timings, timings, operations, 
        datasize_bytes, false
    );
}

void update_and_print_timing_stats(
    float* timing_array, size_t array_len, const char* title, 
    timing_stat** all_timings_ptr, int* timings, 
    unsigned long int operations, unsigned long int datasize_bytes
) {
    _update_and_print_timing_stats(
        timing_array, array_len, title, all_timings_ptr, timings, operations, 
        datasize_bytes, true
    );
}

void print_loop_feedback(int run, int runs) {
    if (run==0) {
        std::cout << "  Completed run " << run+1 << "/" << runs << std::flush;
    }
    else {
        std::cout << "\r  Completed run " << run+1 << "/" << runs << std::flush;
    }
    if (run==runs-1) {
        std::cout << "\n";
    }
}

template<typename T>
bool in_range(T num1, T num2, T tolerance) {
    T absolute_diff = (num1 > num2) ? (num1 - num2): (num2 - num1);

    if (absolute_diff <= tolerance) {
        return true;
    }
    return false;
}

template<class T>
void setup_managed(
    T** matrix, const unsigned long int size, bool validating
) {
    CCC(cudaMallocManaged(matrix, size*sizeof(T)));
    if (validating) {
        init_matrix<T>(*matrix, size);
    }
}

template<class T>
void setup_malloced(
    T** matrix, const unsigned long int size, bool validating
) {
    *matrix = (T*)calloc(size, sizeof(T));
    if (validating) {
        init_matrix<T>(*matrix, size);
    }
}

template<class T>
void setup_managed_array(
    T** matrix, T** array, 
    const unsigned long int size, const int device_count,
    bool validating
) {
    array[0] = *matrix;
    for (int i=1; i<device_count; i++) {
        CCC(cudaMallocManaged(&array[i], size*sizeof(T)));
        if (validating) {
            duplicate_matrix(array[0], size, array[i]);
        }
    }
}

template<class T>
void setup_AsBsCs_managed(
    T** matrixA, T** matrixAs, const int sizeSplitA,
    T** matrixB, T** matrixBs, const int sizeB,
    T** matrixCs, const int sizeSplitC, const int device_count,
    bool validating
) {
    for (int device=0; device<device_count; device++) {
        CCC(cudaMallocManaged(&matrixAs[device], sizeSplitA*sizeof(T)));
        CCC(cudaMallocManaged(&matrixBs[device], sizeB*sizeof(T)));
        CCC(cudaMallocManaged(&matrixCs[device], sizeSplitC*sizeof(T)));
        if (validating) {
            duplicate_matrix(*matrixA+(sizeSplitA*device), sizeSplitA, matrixAs[device]);
            duplicate_matrix(*matrixB, sizeB, matrixBs[device]);
        }
    }
}

template<class T>
void setup_trans_managed(
    T** input_matrix, T** output_matrix, 
    const unsigned long int width, const unsigned long int height,
    bool validating
) {
    setup_managed(output_matrix, width*height, false);
    transpose_matrix(*input_matrix, width, height, *output_matrix);
}

template<class T>
void free_managed( 
    T** matrix
) {
    CCC(cudaFree(*matrix));
}

template<class T>
void free_malloced( 
    T** matrix
) {
    free(*matrix);
}

template<class T>
void free_managed_array( 
    T** array, const int device_count
) {
    for (int i=1; i<device_count; i++) {
        CCC(cudaFree(array[i]));
    }
}

template<class T>
void free_malloced_array( 
    T** array, const int origin_device, const int device_count
) {
    for (int i=0; i<device_count; i++) {
        CCC(cudaSetDevice(i));
        CCC(cudaFree(array[i]));
    }   
    CCC(cudaSetDevice(origin_device));
}

#endif