
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

// note only suitable for square matrixes
namespace cannon {
    template<typename T>
    float singleGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n
    ) {
        unsigned int dim = (n + CANNON_BLOCK - 1) / CANNON_BLOCK; 

        dim3 dimGrid(dim, dim);
        dim3 dimBlock(CANNON_BLOCK, CANNON_BLOCK);

        printf(
            "  Scheduling %d blocks of %d threads\n", 
            dimGrid.x*dimGrid.y*dimGrid.z, 
            dimBlock.x*dimBlock.y*dimBlock.z
        );

        cudaEvent_t start_event;
        CCC(cudaEventCreate(&start_event));
        cudaEvent_t end_event;
        CCC(cudaEventCreate(&end_event));

        CCC(cudaEventRecord(start_event));
        mmmCannon<T> <<<dimGrid, dimBlock>>>(matrixA, matrixB, matrixC, n);
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