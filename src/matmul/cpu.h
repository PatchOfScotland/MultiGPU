#include <stdio.h>
#include "kernels.cu.h"



int getIndex(bool isTrans, int i, int j, int heightX, int widthX) {
    if(isTrans) {
        return j*heightX + i; // A[j,i]
    } else {
        return i*widthX + j; // A[i,j]
    }
}

template<typename T>
float cpuMatMul(
    T* matrixA, unsigned int widthA, unsigned int heightA, 
    T* matrixB, unsigned int widthB, unsigned int heightB, 
    T* matrixC
) {
    struct timeval cpu_start_time;
    struct timeval cpu_end_time;

    gettimeofday(&cpu_start_time, NULL); 

    // Performs matrix multiplication
    #pragma omp parallel for shared(matrixC, matrixA, matrixB) collapse(2)
    for(int i = 0; i < heightA; ++i) {
        for(int j = 0; j < widthB; ++j) {
            T acc = 0;
            int c = i*widthB + j;
            for(int k = 0; k < widthA; ++k) {
                int a = getIndex(false, i, k, heightA, widthA);
                int b = getIndex(false, k, j, widthA, widthB);
                acc += matrixA[a] * matrixB[b];
            }
            matrixC[c] = acc;
        }
    }

    gettimeofday(&cpu_end_time, NULL); 

    float time_microseconds = (cpu_end_time.tv_usec+(1e6*cpu_end_time.tv_sec)) 
            - (cpu_start_time.tv_usec+(1e6*cpu_start_time.tv_sec));
    
    return time_microseconds * 1e3;
}

// Checking function, does not generate any data, but takes an input and 
// output, checking that the given input produces the given output. Used for 
// GPU validation.
template<typename T>
bool cpuValidation(
    T* matrixA, unsigned int widthA, unsigned int heightA, 
    T* matrixB, unsigned int widthB, unsigned int heightB, 
    T* validating, T tolerance
) {
    unsigned long int count = 0;
    #pragma omp parallel for collapse(2) reduction(+:count)
    for(int i = 0; i < heightA; ++i) {
        for(int j = 0; j < widthB; ++j) {
            T matrixC = 0;
            int c = i*widthB + j;
            for(int k = 0; k < widthA; ++k) {
                int a = getIndex(false, i, k, heightA, widthA);
                int b = getIndex(false, k, j, widthA, widthB);
                matrixC += matrixA[a] * matrixB[b];
            }
            if (abs(matrixC - validating[c]) > tolerance) {
                //printf("%f does not match %f at [%d][%d] with tolerance: %f\n", matrixC, validating[c], i, j, tolerance);
                count++;
            }
        }
    }

    if (count == 0) {
        return true;
    }
    printf("Got %ld of potential %d mismatches\n", count, heightA*widthB);
    return false;
}
