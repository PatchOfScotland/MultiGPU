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

template<typename F, typename T>
void benchmarkFunction(F function, T* input_array, T constant, T* output_array, T array_len) {

}

#endif