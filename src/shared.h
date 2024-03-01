#ifndef SHARED_H
#define SHARED_H

#include <thread>

const auto processor_count = std::thread::hardware_concurrency();

void init_array(float* arr, size_t n) {
    srand(5454);
    #pragma omp parallel for
    for(int i=0; i<n; i++) {
        arr[i] = (float)rand() / RAND_MAX;
    }
}

template<class T>
bool compare_arrays(T* array_1, T* array_2, size_t array_len){

    bool status = true;
    #pragma omp parallel for
    for(size_t i=0; i<array_len; i++){
        if (array_1[i] != array_2[i]){
            //std::cout << "i:" 
            //          << i 
            //          << " array_1: " 
            //          << array_1[i] 
            //          << " array_2: " 
            //          << array_2[i] 
            //          <<"\n";
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

float print_timing_stats(
    float* timing_array, size_t array_len, double data_gigabytes, 
    float cpu_runtime_ms, float single_gpu_runtime_ms, 
    float multi_gpu_runtime_ms
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

    float mean_seconds = mean_ms * 1e-3f;
    float gigabytes_per_second = (float)data_gigabytes / mean_seconds;
    std::cout << "    Total runtime: " << total <<"ms\n";
    std::cout << "    Min runtime:   " << min <<"ms\n";
    std::cout << "    Max runtime:   " << max <<"ms\n";
    std::cout << "    Mean runtime:  " << mean_ms <<"ms\n";
    std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/sec\n";
    if (cpu_runtime_ms != -1) {
        std::cout << "      Speedup vs CPU:        x" 
                  << cpu_runtime_ms / mean_ms
                  <<"\n";
    }
    if (single_gpu_runtime_ms != -1) {
        std::cout << "      Speedup vs single GPU: x" 
                  << single_gpu_runtime_ms / mean_ms 
                  <<"\n";        
    }
    if (multi_gpu_runtime_ms != -1) {
        std::cout << "      Speedup vs multi GPU:  x" 
                  << multi_gpu_runtime_ms / mean_ms
                  <<"\n";
    }
    return mean_ms;
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

#endif