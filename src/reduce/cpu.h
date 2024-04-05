#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <iostream>
#include <unistd.h>  
template <typename T, typename R>
R reduction(const T inputElement, R accumulator) {
    return inputElement + accumulator;
}

template<typename F, typename T, typename R>
void cpuReduction(
    F mapped_function, const T* input_array, R* output, 
    const unsigned long int array_len
) {  
    *output = 0;

    const auto processor_count = std::thread::hardware_concurrency();
    unsigned long int chunk_len = array_len / processor_count;

    #pragma omp parallel for
    for (int p=0; p<processor_count; p++) {
        R chunk_output = 0;
        for (int i=0; i<chunk_len+1; i++) {
            if (i+(p*(chunk_len+1)) < array_len) {
                chunk_output = mapped_function(
                    input_array[i+(p*(chunk_len+1))], chunk_output
                );
            }
        }
        #pragma omp atomic
        *output += chunk_output;
    }
}
