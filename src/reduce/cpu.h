template <typename T>
T PlusConst(const T inputElement, T accumulator) {
    return inputElement + accumulator;
}

template<typename F, typename T>
void cpuReduction(
    F mapped_function, const T* input_array, T* output, const int array_len
) {  
//    #pragma omp parallel for reduction(+:acc)
    for (int i=0; i<array_len; i++) {
        output = mapped_function(input_array[i], output);
    }
}
