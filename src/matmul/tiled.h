
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

namespace tiled {
    /**
    * Naive kernel, i.e., the only tiling performed is on the grid;
    *   no shared or private memory is used.
    * It computes matrixC[i,j] = Sum_{k=0...widthA} matrixA[i,k] * matrixB[j,k]
    *   where matrixA: [heightA][widthA]T and
    *         matrixB: [heightA][widthA]T
    * except that matrixA and matrixB maybe transposed version of materialized
    * matrices, i.e.,
    *   isTA == 1 => matrixA = transpose(Ao), where Ao: [widthA][heightA]T
    *   isTB == 1 => matrixB = transpose(Bo), where Bo: [widthA][heightB]T
    * The case (isTA,isTB) = (0,1) corresponds to matrix multiplication
    *   where width_B is actually heightB
    */ 
    template<int isTA, int isTB, typename T, int TL>
    float singleGPU(
        T* matrixA, unsigned int widthA, unsigned int heightA, 
        T* matrixB, unsigned int widthB, unsigned int heightB, 
        T* matrixC
    ) {  
        // setup execution parameters
        unsigned int dim_x = (widthB + TL - 1) / TL; 
        unsigned int dim_y = (heightA + TL - 1) / TL;

        dim3 block(TL, TL, 1);      // blockcount
        dim3 grid(dim_x, dim_y, 1); // threads per block
        
        cudaEvent_t start_event;
        CCC(cudaEventCreate(&start_event));
        cudaEvent_t end_event;
        CCC(cudaEventCreate(&end_event));

        CCC(cudaEventRecord(start_event));
        mmmNaiveKernel<isTA, isTB, T> <<< grid, block >>>(
            matrixA, matrixB, matrixC, widthA, heightA, widthB, heightB
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        float runtime_milliseconds = get_runtime(start_event, end_event);
        
        return runtime_milliseconds * 1e3;
    }

    template<int isTA, int isTB, typename T, int TL>
    float multiGPU(
        T* matrixA, unsigned int widthA, unsigned int heightA, 
        T* matrixB, unsigned int widthB, unsigned int heightB, 
        T* matrixC
    ) {  
        // setup execution parameters
        unsigned int dim_x = (widthB + TL - 1) / TL; 
        unsigned int dim_y = (heightA + TL - 1) / TL;

        dim3 block(TL, TL, 1);      // blockcount
        dim3 grid(dim_x, dim_y, 1); // threads per block
        
        cudaEvent_t start_event;
        CCC(cudaEventCreate(&start_event));
        cudaEvent_t end_event;
        CCC(cudaEventCreate(&end_event));

        CCC(cudaEventRecord(start_event));
        mmmNaiveKernel<isTA, isTB, T> <<< grid, block >>>(
            matrixA, matrixB, matrixC, widthA, heightA, widthB, heightB
        );
        CCC(cudaEventRecord(end_event));
        CCC(cudaEventSynchronize(end_event));

        float runtime_milliseconds = get_runtime(start_event, end_event);
        
        return runtime_milliseconds * 1e3;
    }
}