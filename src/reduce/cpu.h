#include <thread>

template<typename Reduction>
float cpuReduction(
    typename Reduction::InputElement* input_array, 
    typename Reduction::ReturnElement* output, 
    const unsigned long int array_len
) {  
    *output = Reduction::init();

    const auto processor_count = std::thread::hardware_concurrency();
    unsigned long int chunk_len = array_len / processor_count;

    struct timeval cpu_start_time;
    struct timeval cpu_end_time;

    gettimeofday(&cpu_start_time, NULL); 

    #pragma omp parallel for
    for (int p=0; p<processor_count; p++) {
        typename Reduction::ReturnElement chunk_output = Reduction::init();
        for (int i=0; i<chunk_len+1; i++) {
            if (i+(p*(chunk_len+1)) < array_len) {
                chunk_output = Reduction::apply(
                    input_array[i+(p*(chunk_len+1))], chunk_output
                );
            }
        }
        #pragma omp atomic
        *output += chunk_output;
    }

    gettimeofday(&cpu_end_time, NULL); 

    float time_ms = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
    
    return time_ms;
}
