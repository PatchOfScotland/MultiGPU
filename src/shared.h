#ifndef SHARED_H
#define SHARED_H

#include <thread>

const auto processor_count = std::thread::hardware_concurrency();

void init_array(float* arr, size_t n) {
    srand(5454);
    #pragma omp parallel for
    for(int i=0; i<n; i++){
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
void print_timing_array(T* timing_array, size_t array_len, std::string units) {
    T total = 0;
    T min = timing_array[0];
    T max = timing_array[0];
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
        total = total + timing_array[i];
        if (timing_array[i] < min) {
            min = timing_array[i];
        }
        if (timing_array[i] > max) {
            max = timing_array[i];
        }
    }
    std::cout << "\tTotal: " << total << units << "\n";
    std::cout << "\tMin:   " << min << units << "\n";
    std::cout << "\tMax:   " << max << units << "\n";
    std::cout << "\tMean:  " << total/array_len << units << "\n";
}

void print_timing_stats(
    float* timing_array, size_t array_len, double data_gigabytes
) {
    float mean = 0;
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
    mean = total/array_len;

    float mean_seconds = mean * 1e-3f;
    float gigabytes_per_second = (float)data_gigabytes / mean_seconds;
    std::cout << "    Total runtime: " << total <<"ms\n";
    std::cout << "    Min runtime:   " << min <<"ms\n";
    std::cout << "    Max runtime:   " << max <<"ms\n";
    std::cout << "    Mean runtime:  " << mean <<"ms\n";
    std::cout << "    Throughput:    " << gigabytes_per_second <<"GB/sec\n";
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