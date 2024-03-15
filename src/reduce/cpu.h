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

    #pragma omp parallel for reduction(+:*output)
    for (int i=0; i<array_len; i++) {
        *output = mapped_function(input_array[i], *output);
    }
}
