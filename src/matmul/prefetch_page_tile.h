
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

namespace prefetch_page_tiled {
    template<int isTB, typename T, int PageSize, int SM>
    void per_device_management(
            const T* matrixA, const unsigned int widthA, const unsigned int heightA, 
            const T* matrixB, const unsigned int widthB, const unsigned int heightB, 
            T* const matrixC, const unsigned int widthC, const unsigned int per_device_heightC,
            const int device
    ) {
        cudaSetDevice(device);

        cudaError_t cudaError = cudaPeekAtLastError();
        if (cudaError != cudaSuccess) {
            std::cout << "CUDA beansed it\n" << cudaError << "\n";
            std::cout << cudaGetErrorString(cudaError) << "\n";
            exit(cudaError);
        }

        cudaStream_t processing_stream;
        cudaStream_t loading_stream;

        cudaStreamCreate(&processing_stream);
        cudaStreamCreate(&loading_stream);
    
        const T* sub_matrixA = matrixA + (device * per_device_heightC * widthA);
        T* sub_matrixC = matrixC + (device * per_device_heightC * widthC);

        unsigned int block_x_offset = 0;
        unsigned int block_y_offset = 0;

        // setup execution parameters
        unsigned int blocks_x = (widthC + PageSize - 1) / PageSize; 
    
        dim3 threadsPerBlock(PageSize, 1, 1);
        dim3 blockCount(1, SM, 1);
        
        cudaEvent_t sync_event_processing;
        cudaEvent_t sync_event_loading;
        CCC(cudaEventCreate(&sync_event_processing));
        CCC(cudaEventCreate(&sync_event_loading));

        cudaEvent_t sync_event;
        CCC(cudaEventCreate(&sync_event));

        size_t runs = ((blocks_x*per_device_heightC) + SM - 1) / SM;
        //std::cout << "device " << device << " SM: " << SM << " PageSize: " << PageSize << "\n";
        //std::cout << "device " << device << " is going to run " << blocks_x*per_device_heightC << " blocks and has been told to run " << SM << " at once giving " << runs << " loops\n";
        
        for (int run=0; run<runs; run++) {
            mmmPrefetchPageTiledKernel<isTB, T> <<< blockCount, threadsPerBlock, 0, processing_stream >>>(
                sub_matrixA, widthA, heightA,
                matrixB, widthB, heightB,
                sub_matrixC, widthC, per_device_heightC,
                block_x_offset, block_y_offset
            );

            block_y_offset += SM;
            while (block_y_offset >= per_device_heightC) {
                block_x_offset += PageSize;
                block_y_offset -= per_device_heightC;
            }

            mmmPrefetchingKernel<isTB, T> <<< 1, SM, 0, loading_stream >>> (
                sub_matrixA, sub_matrixC, widthC, per_device_heightC,
                block_x_offset, block_y_offset, PageSize
            );

            CCC(cudaEventRecord(sync_event_processing, processing_stream));
            CCC(cudaEventRecord(sync_event_loading, loading_stream));
            CCC(cudaEventSynchronize(sync_event_processing));
            CCC(cudaEventSynchronize(sync_event_loading));
        }

        cudaError = cudaPeekAtLastError();
        if (cudaError != cudaSuccess) {
            std::cout << "CUDA beansed it\n" << cudaError << "\n";
            std::cout << cudaGetErrorString(cudaError) << "\n";
            exit(cudaError);
        }

        cudaStreamDestroy(processing_stream);
        cudaStreamDestroy(loading_stream);
    }
    
    template<int isTB, typename T, int PageSize, int SM>
    float multiGPU(
        const T* matrixA, const unsigned int widthA, const unsigned int heightA, 
        const T* matrixB, const unsigned int widthB, const unsigned int heightB, 
        T* const matrixC, const unsigned int widthC, const unsigned int heightC,
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
                per_device_management<isTB, T, PageSize, SM>,
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
