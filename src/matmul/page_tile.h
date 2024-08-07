
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

// Stores 2D matrix in partial z-order, with each tile being PageSize wide, and
// height/Splits tall.
template<class T, int PageSize, int Splits>
void page_order(T* input,  T* output, size_t width, size_t height) {

    int PageCountX = width/PageSize;
    int verticalSplitSize = height/Splits;
    
    #pragma omp parallel for collapse(4)
    for (int tileX=0; tileX<PageCountX; tileX++) {
        for (int tileY=0; tileY<Splits; tileY++) {
            for (int tileInnerX=0; tileInnerX<verticalSplitSize; tileInnerX++) {
                for (int tileInnerY=0; tileInnerY<PageSize; tileInnerY++) {
                    int input_offset = tileInnerY // offset within each tile for its X index
                        + PageSize*tileInnerX // offset within each tile for its Y index
                        + tileX*verticalSplitSize*PageSize  // offset for each tile in X direction
                        + tileY*verticalSplitSize*width;
                    int output_offset = tileInnerY // offset within each tile for its X index
                        + width*tileInnerX // offset within each tile for its Y index
                        + PageSize*tileX // offset for each tile in X direction
                        + tileY*verticalSplitSize*width; // offset for each tile in Y direction
                    output[output_offset] = input[input_offset];
                }
            }
        }
    }
}

namespace page_tiled {
    template<int isTB, typename T, int PageSize>
    void per_device_management(
            const T* matrixA, const unsigned int widthA, const unsigned int heightA, 
            const T* matrixB, const unsigned int widthB, const unsigned int heightB, 
            T* matrixC, const unsigned int widthC, const unsigned int per_device_heightC,
            const int device
    ) {
        cudaSetDevice(device);
    
        const T* sub_matrixA = matrixA + (device * per_device_heightC * widthA);
        T* sub_matrixC = matrixC + (device * per_device_heightC * widthC);
        
        // setup execution parameters
        unsigned int blocks_x = (widthC + PageSize - 1) / PageSize; 
    
        dim3 threadsPerBlock(PageSize, 1, 1);
        dim3 blockCount(blocks_x, per_device_heightC, 1);

        cudaEvent_t sync_event;
        CCC(cudaEventCreate(&sync_event));
        mmmPageTiledKernel<isTB, T> <<< blockCount, threadsPerBlock >>>(
            sub_matrixA, widthA, heightA,
            matrixB, widthB, heightB,
            sub_matrixC, widthC, per_device_heightC,
            device
        );
        CCC(cudaEventRecord(sync_event));
        CCC(cudaEventSynchronize(sync_event));

        cudaError_t cudaError = cudaPeekAtLastError();
        if (cudaError != cudaSuccess) {
            std::cerr << "CUDA beansed it\n" << cudaError << "\n";
            std::cerr << cudaGetErrorString(cudaError) << "\n";
            exit(cudaError);
        }
    }
    
    template<int isTB, typename T, int PageSize>
    float multiGPU(
        const T* matrixA, const unsigned int widthA, const unsigned int heightA, 
        const T* matrixB, const unsigned int widthB, const unsigned int heightB, 
        T* matrixC, const unsigned int widthC, const unsigned int heightC,
        const int device_count
    ) {  
        int origin_device;
        CCC(cudaGetDevice(&origin_device));

        unsigned int per_device_heightC = (heightC + device_count - 1) / device_count;

        struct timeval start_time;
        struct timeval end_time;

        gettimeofday(&start_time, NULL); 

        std::thread threads[device_count];
        for (int device=0; device<device_count; device++) {
            threads[device] = std::thread(
                per_device_management<isTB, T, PageSize>,
                matrixA, widthA, heightA, 
                matrixB, widthB, heightB, 
                matrixC, widthC, per_device_heightC,
                device
            );
        }

        for (int device=0; device<device_count; device++) {
            threads[device].join();        
        }
        
        gettimeofday(&end_time, NULL); 
        
        cudaSetDevice(origin_device);

        float time_microseconds = (end_time.tv_usec+(1e6*end_time.tv_sec)) 
            - (start_time.tv_usec+(1e6*start_time.tv_sec));
    
        return time_microseconds;
    }
}
