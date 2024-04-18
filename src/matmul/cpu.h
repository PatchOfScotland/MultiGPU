#include <stdio.h>

template<class T>
float cpuMatMul(
    T* input_A, unsigned int width_A, unsigned int height_A, 
    T* input_B, unsigned int width_B, unsigned int height_B, 
    T* result
) {
    struct timeval cpu_start_time;
    struct timeval cpu_end_time;

    gettimeofday(&cpu_start_time, NULL); 

    // Performs matrix multiplication
    for (unsigned int x= 0; x < width_A; x++) {
        for (unsigned int y = 0; y < height_B; y++) {
            unsigned int index_xy = (x*height_B) + y;
            result[index_xy] = 0;
            for (unsigned int z = 0; z < width_B; z++) {
                unsigned int index_xz = (x*width_B) + z;
                unsigned int index_zy = (z*width_A) + y;
                result[index_xy] += input_A[index_xz] * input_B[index_zy];
            }
        }
    }

    gettimeofday(&cpu_end_time, NULL); 

    float time_ms = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
    
    return time_ms;
}
