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
    F mapped_function, const T* input_array, const T constant, T* output_array, 
    const unsigned long int array_len
) {  
    #pragma omp parallel for
    for (unsigned long int i=0; i<array_len; i++) {
        output_array[i] = mapped_function(input_array[i], constant);
    }
}

// Checking function, does not generate any data, but takes an input and 
// output, checking that the given input produces the given output. Used for 
// GPU validation.
template<typename F, typename T>
bool cpuValidation(
    F mapped_function, const T* input_array, const T constant, T* output_array, 
    const unsigned long int array_len
) {
    unsigned long int count = 0;
    #pragma omp parallel for
    for (unsigned long int i=0; i<array_len; i++) {
        if (mapped_function(input_array[i], constant) != output_array[i]) {
            count++;
        }
    }

    if (count == 0) {
        return true;
    }
    return false;
}