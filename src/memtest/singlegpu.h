
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"

// note only suitable for square matrixes
namespace single_managed {
    template<typename T>
    float copy_naive(
        T* matrixA, T* matrixB, const unsigned int array_n
    ) {
        unsigned int dim = ((array_n*array_n) + BLOCK_SIZE - 1) / BLOCK_SIZE; 

        cudaEvent_t start_event;
        CCC(cudaEventCreate(&start_event));
        cudaEvent_t end_event;
        CCC(cudaEventCreate(&end_event));

        CCC(cudaEventRecord(start_event));
        copyNaive<T> <<<dim, BLOCK_SIZE>>>(matrixA, matrixB, array_n);
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        cudaError_t cudaError = cudaPeekAtLastError();
        if (cudaError != cudaSuccess) {
            std::cerr << "CUDA beansed it\n" << cudaError << "\n";
            std::cerr << cudaGetErrorString(cudaError) << "\n";
            exit(cudaError);
        }

        float runtime_milliseconds = get_runtime(start_event, end_event);
        
        return runtime_milliseconds * 1e3;
    }
}