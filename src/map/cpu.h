// Mapping function that takes a function and maps it across each element in an
// input array, with the output in a new output array. Opperates entirely on 
// the CPU. 
template<typename MappedFunction>
float cpuMapping(
    typename MappedFunction::InputElement* input_array, 
    const typename MappedFunction::X constant, 
    typename MappedFunction::ReturnElement* output_array, 
    const unsigned long int array_len
) { 
    struct timeval cpu_start_time;
    struct timeval cpu_end_time;

    gettimeofday(&cpu_start_time, NULL); 

    #pragma omp parallel for
    for (unsigned long int i=0; i<array_len; i++) {
        output_array[i] = MappedFunction::apply(input_array[i], constant);
    }

    gettimeofday(&cpu_end_time, NULL); 

    float time_ms = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
    
    return time_ms;
}

// Checking function, does not generate any data, but takes an input and 
// output, checking that the given input produces the given output. Used for 
// GPU validation.
template<typename MappedFunction>
bool cpuValidation(
    typename MappedFunction::InputElement* input_array, 
    const typename MappedFunction::X constant, 
    typename MappedFunction::ReturnElement* output_array, 
    const unsigned long int array_len
) {
    unsigned long int count = 0;
    #pragma omp parallel for
    for (unsigned long int i=0; i<array_len; i++) {
        if (MappedFunction::apply(input_array[i], constant) != output_array[i]
        ) {
            count++;
        }
    }

    if (count == 0) {
        return true;
    }
    return false;
}