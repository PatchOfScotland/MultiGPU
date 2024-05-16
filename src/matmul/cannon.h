
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

// note only suitable for square matrixes
namespace cannon {
    template<typename T>
    float singleGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n
    ) {
        unsigned int dim = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 

        dim3 dimGrid(dim, dim);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        cudaEvent_t start_event;
        CCC(cudaEventCreate(&start_event));
        cudaEvent_t end_event;
        CCC(cudaEventCreate(&end_event));

        CCC(cudaEventRecord(start_event));
        mmmCannon<T> <<<dimGrid, dimBlock, 2*BLOCK_SIZE*BLOCK_SIZE>>>(matrixA, matrixB, matrixC, n);
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        float runtime_milliseconds = get_runtime(start_event, end_event);
        
        return runtime_milliseconds * 1e3;
    }
}