#include <stdio.h>
#include "kernels.cu.h"
#include "../shared.h"


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

template<typename T>
float cpuMatMulZorder(
    T* matrixA, unsigned int widthA, unsigned int heightA, 
    T* matrixB, unsigned int widthB, unsigned int heightB, 
    T* matrixC, const int tolerance, const int split
) {
    struct timeval cpu_start_time;
    struct timeval cpu_end_time;

    T* matrixAz = (T*)malloc(widthA*heightA*sizeof(T));
    T* matrixBz = (T*)malloc(widthB*heightB*sizeof(T));

    to_block<T>(matrixA, matrixAz, widthA, heightA, split);
    to_block<T>(matrixB, matrixBz, widthB, heightB, split);
    
    gettimeofday(&cpu_start_time, NULL); 

    // Performs matrix multiplication
    #pragma omp parallel for shared(matrixC, matrixAz, matrixBz) collapse(2)
    for(int i = 0; i < heightA; ++i) {
        for(int j = 0; j < widthB; ++j) {
            T acc = 0;
            int c = i*widthB + j;
            for(int k = 0; k < widthA; ++k) {
                int a = getIndex(false, i, k, heightA, widthA);
                int b = getIndex(false, k, j, widthA, widthB);
                acc += matrixAz[a] * matrixBz[b];
            }
            matrixC[c] = acc;
        }
    }

    gettimeofday(&cpu_end_time, NULL); 

    free(matrixAz);
    free(matrixBz);

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
    //#pragma omp parallel for collapse(2) reduction(+:count)
    for(int i = 0; i < heightA; ++i) {
        for(int j = 0; j < widthB; ++j) {
            T acc = 0;
            int c = i*widthB + j;
            for(int k = 0; k < widthA; ++k) {
                int a = getIndex(false, i, k, heightA, widthA);
                int b = getIndex(false, k, j, widthA, widthB);
                acc += matrixA[a] * matrixB[b];
            }
            if (abs(acc - validating[c]) > tolerance) {
                //printf("%f does not match %f at [%d][%d] with tolerance: %f\n", acc, validating[c], i, j, tolerance);
                count++;
                //debug[(i*widthB)+j] = 1;
            }
        }
    }

    if (count == 0) {
        return true;
    }
    printf("Got %ld of potential %d mismatches\n", count, heightA*widthB);
    return false;
}