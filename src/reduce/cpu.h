template <typename T>
T reduction(const T inputElement, T accumulator) {
    return inputElement + accumulator;
}

template<typename F, typename T>
void cpuReduction(
    F mapped_function, const T* input_array, T* output, 
    const unsigned long int array_len
) {  
    *output = 0;

    #pragma omp parallel for reduction(+:*output)
    for (int i=0; i<array_len; i++) {
        *output = mapped_function(input_array[i], *output);
    }
}
