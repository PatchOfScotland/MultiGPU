
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

// note only suitable for square matrixes
namespace cannon {
    template<typename T, int cannon_block, size_t quadrants_per_dim>
    void per_quadrant_management(
            T* matrixA, T* matrixB, T* matrixC, 
            const unsigned int total_n, const unsigned int quadrant_n, 
            const unsigned int offset_x, const unsigned int offset_y, 
            const int device
    ) {
        //cudaSetDevice(device);

        unsigned int blocks_dim = (total_n + cannon_block - 1) / cannon_block;
        unsigned int blocks_quadrant = (blocks_dim + quadrants_per_dim - 1)  / quadrants_per_dim;

        dim3 dimGrid(blocks_quadrant, blocks_quadrant);
        dim3 dimBlock(cannon_block, cannon_block);


        //printf(
        //    "  Scheduling %d (%dx%dx%d) blocks of %d (%dx%dx%d)threads\n", 
        //    dimGrid.x*dimGrid.y*dimGrid.z, 
        //    dimGrid.x, dimGrid.y, dimGrid.z, 
        //    dimBlock.x*dimBlock.y*dimBlock.z,
        //    dimBlock.x, dimBlock.y, dimBlock.z
        //);
                
        cudaEvent_t sync_event;
        CCC(cudaEventCreate(&sync_event));
        mmmCannonQuadrant<T, cannon_block> <<<dimGrid, dimBlock>>>(
            matrixA, matrixB, matrixC, 
            total_n, quadrant_n, blocks_dim, offset_x, offset_y
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

    template<typename T, int cannon_block, size_t quadrants_per_dim>
    float multiGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n,
        const int device_count
    ) {
        int origin_device;
        CCC(cudaGetDevice(&origin_device));

        if (cannon_block > 32) {
            std::cout << "Cannot scheduled a block of " << cannon_block 
                      << "x" << cannon_block 
                      << ". Largest acceptable value is 32\n";
            return 0;
        }

        const unsigned int quadrants = quadrants_per_dim * quadrants_per_dim;
        const unsigned int quadrant_n = (n + quadrants_per_dim - 1) / quadrants_per_dim;;

        //std::cout << "Quads: " << quadrants << "," <<  quadrants_per_dim << "," <<  quadrant_n << "\n";

        struct timeval start_time;
        struct timeval end_time;

        gettimeofday(&start_time, NULL); 

        unsigned int offset_x = 0;
        unsigned int offset_y = 0;

        unsigned int device = 0;
        std::thread threads[quadrants];
        for (int quadrant=0; quadrant<quadrants; quadrant++) {
            threads[quadrant] = std::thread(
                per_quadrant_management<T, cannon_block, quadrants_per_dim>,
                matrixA, matrixB, matrixC, 
                n, quadrant_n, offset_x, offset_y, device
            );

            //std::cout << "Scheduling with offsets: " << offset_x <<  "," <<  offset_y << "\n";
            //std::cout << "Scheduling on device: " << device << "\n";

            device += 1;
            if (device >= device_count) {
                device = 0;
            }           
            offset_x += quadrant_n;
            if (offset_x >= n) {
                offset_x = 0;
                offset_y += quadrant_n;
            }
        }

        for (int quadrant=0; quadrant<quadrants; quadrant++) {
            threads[quadrant].join();        
        }

        gettimeofday(&end_time, NULL); 
        
        cudaSetDevice(origin_device);

        float time_microseconds = (end_time.tv_usec+(1e6*end_time.tv_sec)) 
            - (start_time.tv_usec+(1e6*start_time.tv_sec));
    
        return time_microseconds;
    }

    template<typename T, int cannon_block>
    float singleGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n
    ) {
        unsigned int dim = (n + cannon_block - 1) / cannon_block; 

        dim3 dimGrid(dim, dim);
        dim3 dimBlock(cannon_block, cannon_block);

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
        mmmCannon<T, cannon_block> <<<dimGrid, dimBlock>>>(matrixA, matrixB, matrixC, n);
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