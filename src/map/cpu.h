// Toy function to be mapped accross an array. Just adds a constant x to an 
// element
template <typename T>
T PlusConst(const T inputElement, const T x) {
    return inputElement + x;
}

// Mapping function that takes a function and maps it across each element in an
// input array, with the output in a new output array. Opperates entirely on 
// the CPU. 
template<typename F, typename T>
void cpuMapping(
    F mapped_function, T* input_array, const T constant, T* output_array, 
    int array_len
) {  
    #pragma omp parallel for
    for (int i=0; i<array_len; i++) {
        output_array[i] = mapped_function(input_array[i], constant);
    }
}
