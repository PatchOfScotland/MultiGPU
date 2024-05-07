
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
        T* matrixC, unsigned int widthC, unsigned int heightC
    ) {  
        // setup execution parameters
        unsigned int dim_x = (widthC + TL - 1) / TL; 
        unsigned int dim_y = (heightC + TL - 1) / TL;

        dim3 block(TL, TL, 1);      // blockcount
        dim3 grid(dim_x, dim_y, 1); // threads per block
        
        cudaEvent_t start_event;
        CCC(cudaEventCreate(&start_event));
        cudaEvent_t end_event;
        CCC(cudaEventCreate(&end_event));

        CCC(cudaEventRecord(start_event));
        mmmNaiveKernel<isTA, isTB, T> <<< grid, block >>>(
            matrixA, widthA, heightA, 
            matrixB, widthB, heightB, 
            matrixC, widthC, heightC
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
        T* matrixC, unsigned int widthC, unsigned int heightC
    ) {  
        int origin_device;
        CCC(cudaGetDevice(&origin_device));
        int device_count;
        CCC(cudaGetDeviceCount(&device_count));

        unsigned int per_device_heightC = (heightC + device_count - 1) / device_count;

        struct timeval start_time;
        struct timeval end_time;

        gettimeofday(&start_time, NULL); 

        for (int i=0; i<device_count; i++)
        {
            T* sub_matrixA = matrixA + (i * per_device_heightC * widthA);
            T* sub_matrixC = matrixC + (i * per_device_heightC * widthC);

            // setup execution parameters
            unsigned int dim_x = (widthC + TL - 1) / TL; 
            unsigned int dim_y = (per_device_heightC + TL - 1) / TL;

            dim3 block(TL, TL, 1);      // blockcount
            dim3 grid(dim_x, dim_y, 1); // threads per block

            cudaEvent_t sync_event;
            CCC(cudaEventCreate(&sync_event));
            mmmNaiveKernelMulti<isTA, isTB, T> <<< grid, block >>>(
                sub_matrixA, widthA, per_device_heightC,
                matrixB, widthB, heightB,
                sub_matrixC, widthC, per_device_heightC,
                i * per_device_heightC
            );
            CCC(cudaEventRecord(sync_event));
            CCC(cudaEventSynchronize(sync_event));
        }
        
        gettimeofday(&end_time, NULL); 

        float time_microseconds = (end_time.tv_usec+(1e6*end_time.tv_sec)) 
            - (start_time.tv_usec+(1e6*start_time.tv_sec));
    
        return time_microseconds;
    }
}