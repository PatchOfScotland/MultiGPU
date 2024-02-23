#ifndef SHARED_H
#define SHARED_H

#include <thread>

const auto processor_count = std::thread::hardware_concurrency();

void init_array(float* arr, size_t n) {
    srand(5454);
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
            //std::cout << "i:" << i << " array_1: " << array_1[i] << " array_2: " << array_2[i] <<"\n";
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

#endif