
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

// note only suitable for square matrixes
namespace cannon {
    template<typename T>
    void per_quadrant_management(
            T* matrixA, T* matrixB, T* matrixC, 
            const unsigned int n, const unsigned int quadrant_n, 
            const unsigned int offset_x, const unsigned int offset_y
    ) {
        //cudaSetDevice(device);

        unsigned int blocks_dim = (n + CANNON_BLOCK - 1) / CANNON_BLOCK;
        unsigned int blocks_quadrant = (blocks_dim + 1)  / 2;

        dim3 dimGrid(blocks_quadrant, blocks_quadrant);
        dim3 dimBlock(CANNON_BLOCK, CANNON_BLOCK);

        //printf(
        //    "  Scheduling %d (%dx%dx%d) blocks of %d (%dx%dx%d)threads\n", 
        //    dimGrid.x*dimGrid.y*dimGrid.z, 
        //    dimGrid.x, dimGrid.y, dimGrid.z, 
        //    dimBlock.x*dimBlock.y*dimBlock.z,
        //    dimBlock.x, dimBlock.y, dimBlock.z
        //);
                
        cudaEvent_t sync_event;
        CCC(cudaEventCreate(&sync_event));
        mmmCannonQuadrant<T> <<<dimGrid, dimBlock>>>(
            matrixA, matrixB, matrixC, 
            n, quadrant_n, blocks_dim, offset_x, offset_y
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

    template<typename T>
    float multiGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n,
        const int device_count
    ) {
        int origin_device;
        CCC(cudaGetDevice(&origin_device));

        const int quadrants = 4; // presumably should be only 4

        unsigned int quadrant_n = (n + 1) / 2;

        struct timeval start_time;
        struct timeval end_time;

        gettimeofday(&start_time, NULL); 

        std::thread threads[4] = {
            std::thread(
                per_quadrant_management<T>,
                matrixA, matrixB, matrixC, n, quadrant_n, 0, 0
            ),
            std::thread(
                per_quadrant_management<T>,
                matrixA, matrixB, matrixC, n, quadrant_n, quadrant_n, 0
            ),
            std::thread(
                per_quadrant_management<T>,
                matrixA, matrixB, matrixC, n, quadrant_n, 0, quadrant_n
            ),
            std::thread(
                per_quadrant_management<T>,
                matrixA, matrixB, matrixC, n, quadrant_n, quadrant_n, quadrant_n
            )          
        };

        for (int quadrant=0; quadrant<quadrants; quadrant++) {
            threads[quadrant].join();        
        }

        gettimeofday(&end_time, NULL); 
        
        cudaSetDevice(origin_device);

        float time_microseconds = (end_time.tv_usec+(1e6*end_time.tv_sec)) 
            - (start_time.tv_usec+(1e6*start_time.tv_sec));
    
        return time_microseconds;
    }

    template<typename T>
    float singleGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n
    ) {
        unsigned int dim = (n + CANNON_BLOCK - 1) / CANNON_BLOCK; 

        dim3 dimGrid(dim, dim);
        dim3 dimBlock(CANNON_BLOCK, CANNON_BLOCK);

        //printf(
        //    "  Scheduling %d (%dx%dx%d) blocks of %d (%dx%dx%d)threads\n", 
        //    dimGrid.x*dimGrid.y*dimGrid.z, 
        //    dimGrid.x, dimGrid.y, dimGrid.z, 
        //    dimBlock.x*dimBlock.y*dimBlock.z,
        //    dimBlock.x, dimBlock.y, dimBlock.z
        //);

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