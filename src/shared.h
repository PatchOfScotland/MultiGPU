#ifndef SHARED_H
#define SHARED_H

#include <iostream>
#include <iomanip>
#include <thread>

#define NO_HINTS    0
#define HINTS       1
#define PREFETCH    2

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

struct timing_stat {
    float timing_microseconds;
    unsigned long int flops;
    unsigned long int datasize_bytes;
    std::string type;

    timing_stat(
        std::string type, 
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
};

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
void duplicate_matrix(T* origin, uint64_t size, T* target) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < size; i++)
        target[i] = origin[i];
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
                std::cout << matrix[r+(width*c)];
            }
            else if (r==width-1) {
                std::cout << ", " << matrix[r+(width*c)] << "\n";
            }
            else {
                std::cout << ", " << matrix[r+(width*c)];
            }
        }
    }
}

void update_and_print_timing_stats(
    float* timing_array, size_t array_len, timing_stat* to_update, 
    const timing_stat* all_timings[], int timings
) {
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

    float gigabytes_per_second = get_throughput(mean_ms, to_update->datasize_bytes);
    float gigaflops_per_second = get_flops(mean_ms, to_update->flops);
    std::cout << "    Total runtime: " << total <<"ms\n";
    std::cout << "    Min runtime:   " << min <<"ms\n";
    std::cout << "    Max runtime:   " << max <<"ms\n";
    std::cout << "    Mean runtime:  " << mean_ms <<"ms\n";
    std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/sec\n";
    std::cout << "    GLFOPS:        " << gigaflops_per_second <<"/sec\n";
    
    for (int i=0; i<timings; i++) {
        struct timing_stat timing = *all_timings[i];
        if (timing.timing_microseconds != -1) {
            std::cout << "      Speedup vs "
                      << timing.type 
                      << ": x" 
                      << timing.timing_microseconds / mean_ms
                      << "\n";

        }
    }
    to_update->timing_microseconds = mean_ms;
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

#endif