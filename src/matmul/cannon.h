
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

// note only suitable for square matrixes
namespace cannon {
    struct shared_mem {
        void* mem;
    };

    int* a_channels;
    int* b_channels;
    
    template<typename T, int cannon_block, size_t quadrants_per_dim>
    void per_quadrant_cannon_management(
            T* matrixA, T* matrixB, T* matrixC, 
            const unsigned int total_n, const unsigned int quadrant_n, 
            const unsigned int offset_x, const unsigned int offset_y, 
            const int device
    ) {
        cudaSetDevice(device);

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
    float blockedMultiGPU(
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
                per_quadrant_cannon_management<T, cannon_block, quadrants_per_dim>,
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


    template<typename T, int TL>
    void per_quadrant_management(
            T* matrixA, T* matrixB, T* matrixC, 
            const unsigned int total_n, const unsigned int quadrant_n, 
            const unsigned int quadrant_size, 
            const int device, size_t quadrants_per_dim,
            int a_channel_write, int b_channel_write,
            int a_channel_read, int b_channel_read,
            int quadrant, T* debugA, T* debugB, T* debugC 
    ) {
        //cudaSetDevice(device);

        unsigned int blocks_dim = (total_n + quadrant_n - 1) / quadrant_n;
        unsigned int blocks_quadrant = (blocks_dim + quadrants_per_dim - 1)  / quadrants_per_dim;

        dim3 dimGrid(blocks_quadrant, blocks_quadrant);
        dim3 dimBlock(quadrant_n, quadrant_n);

        unsigned int dim_x = (quadrant_n + TL - 1) / TL; 
        unsigned int dim_y = (quadrant_n + TL - 1) / TL;

        dim3 block(TL, TL, 1);      // blockcount
        dim3 grid(dim_x, dim_y, 1); // threads per block

        //printf(
        //    "  Scheduling %d (%dx%dx%d) blocks of %d (%dx%dx%d)threads\n", 
        //    dimGrid.x*dimGrid.y*dimGrid.z, 
        //    dimGrid.x, dimGrid.y, dimGrid.z, 
        //    dimBlock.x*dimBlock.y*dimBlock.z,
        //    dimBlock.x, dimBlock.y, dimBlock.z
        //);

        //printf("Quad %d starting with A: %p and B: %p\n", quadrant, matrixA, matrixB);
                
        for (int iteration=0; iteration<quadrants_per_dim; iteration++) {
            cudaEvent_t sync_event;
            CCC(cudaEventCreate(&sync_event));
            mmmNaiveKernelAdditive<T> <<<dimGrid, dimBlock>>>(
                matrixA, quadrant_n, quadrant_n,
                matrixB, quadrant_n, quadrant_n,
                matrixC, quadrant_n, quadrant_n,
                quadrant, iteration, debugA, debugB, debugC
            );
            CCC(cudaEventRecord(sync_event));
            CCC(cudaEventSynchronize(sync_event));


int c = write(a_channel_write, &matrixA, sizeof(T*));
if (c != sizeof(T*)) {
std::cerr << "write did not complete\n";
exit(EXIT_FAILURE); }
T* buf;
c = read(a_channel_read, &buf, sizeof(T*));
if (c != sizeof(T*)) {
std::cerr << "read did not complete\n";
exit(EXIT_FAILURE); }


            if (quadrant == 0) {
                std::cout << "Iteraction " << iteration << " Debug A: \n";
                print_matrix_z(debugA, total_n, quadrants_per_dim);
                std::cout << "Iteraction " << iteration << " Debug B: \n";
                print_matrix_z(debugB, total_n, quadrants_per_dim);
                std::cout << "Iteraction " << iteration << " Debug C: \n";
                print_matrix_z(debugC, total_n, quadrants_per_dim);

                for (int i=0; i<total_n*total_n; i++) {
                    debugA[i] = -1;
                    debugB[i] = -1;
                    debugC[i] = -1;
                }
            }

c = write(a_channel_write, &matrixA, sizeof(T*));
if (c != sizeof(T*)) {
std::cerr << "write did not complete\n";
exit(EXIT_FAILURE); }
c = read(a_channel_read, &buf, sizeof(T*));
if (c != sizeof(T*)) {
std::cerr << "read did not complete\n";
exit(EXIT_FAILURE); }

            cudaError_t cudaError = cudaPeekAtLastError();
            if (cudaError != cudaSuccess) {
                std::cerr << "CUDA beansed it\n" << cudaError << "\n";
                std::cerr << cudaGetErrorString(cudaError) << "\n";
                exit(cudaError);
            }

            if (iteration != quadrants_per_dim-1) {
                size_t count = write(a_channel_write, &matrixA, sizeof(T*));
                if (count != sizeof(T*)) {
                    std::cerr << "write did not complete\n";
                    exit(EXIT_FAILURE);
                }
                count = write(b_channel_write, &matrixB, sizeof(T*));
                if (count != sizeof(T*)) {
                    std::cerr << "write did not complete\n";
                    exit(EXIT_FAILURE);
                }


                T* buf_a;
                T* buf_b;

                count = read(a_channel_read, &buf_a, sizeof(T*));
                if (count != sizeof(T*)) {
                    std::cerr << "read did not complete\n";
                    exit(EXIT_FAILURE);
                }
                count = read(b_channel_read, &buf_b, sizeof(T*));
                if (count != sizeof(T*)) {
                    std::cerr << "read did not complete\n";
                    exit(EXIT_FAILURE);
                }

                matrixA = buf_a;
                matrixB = buf_b;

                //printf("Quad %d now got A: %p and B: %p\n", quadrant, matrixA, matrixB);
            }
        }
    }

    template<typename T, int TL>
    float multiGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n,
        const int device_count, const size_t quadrants_per_dim,
        T* debugA, T* debugB, T* debugC
    ) {
        int origin_device;
        CCC(cudaGetDevice(&origin_device));

        if (n % quadrants_per_dim) {
            perror("Cannot blocks evenly accross devices\n");
            exit(EXIT_FAILURE);
        }

        const unsigned int quadrants = quadrants_per_dim * quadrants_per_dim;
        const unsigned int quadrant_n = (n + quadrants_per_dim - 1) / quadrants_per_dim;;
        //std::cout << "Quads: " << quadrants << "," <<  quadrants_per_dim << "," <<  quadrant_n << "\n";

        unsigned int quadrant_size = quadrant_n * quadrant_n;

        unsigned int device = 0;
        std::thread threads[quadrants];
        a_channels = (int*)malloc(quadrants * 2 * sizeof(int));
        b_channels = (int*)malloc(quadrants * 2 * sizeof(int));
        for (int i=0; i<quadrants; i++) {
            if(pipe(a_channels + i*2) < 0) {
                perror("Couldn't create pipe\n");
                exit(EXIT_FAILURE);
            }
            if(pipe(b_channels + i*2) < 0) {
                perror("Couldn't create pipe\n");
                exit(EXIT_FAILURE);
            }
        }

        int x = 0;
        int y = 0;

        struct timeval start_time;
        struct timeval end_time;

        gettimeofday(&start_time, NULL); 

        for (int quadrant=0; quadrant<quadrants; quadrant++) {                
            int ax_write = x;
            int ay_write = (y + quadrant_n - 1) % quadrant_n;
            int bx_write = (x + quadrant_n - 1) % quadrant_n;
            int by_write = y;

            int ax_read = x;
            int ay_read = (y + 1) % quadrant_n;
            int bx_read = (x + 1) % quadrant_n;
            int by_read = y;

            unsigned int quadrant_a_write = (ax_write * quadrants_per_dim) + ay_write;
            unsigned int quadrant_b_write = (bx_write * quadrants_per_dim) + by_write;
            unsigned int quadrant_a_read = (ax_read * quadrants_per_dim) + ay_read;
            unsigned int quadrant_b_read = (bx_read * quadrants_per_dim) + by_read;
            unsigned int quadrant_offset = (quadrant_size * quadrant);

            threads[quadrant] = std::thread(
                per_quadrant_management<T, TL>,
                matrixA + quadrant_offset, 
                matrixB + quadrant_offset, 
                matrixC + quadrant_offset, 
                n, quadrant_n, quadrant_size, device, quadrants_per_dim,
                a_channels[(quadrant_a_write * 2) + 1], b_channels[(quadrant_b_write * 2) + 1],
                a_channels[quadrant_a_read * 2], b_channels[quadrant_b_read * 2],
                quadrant, 
                debugA + quadrant_offset, 
                debugB + quadrant_offset, 
                debugC + quadrant_offset
            );

            //std::cout << "Scheduling with offsets: " << offset_x <<  "," <<  offset_y << "\n";
            //std::cout << "Scheduling on device: " << device << "\n";

            y += 1;
            if (y >= quadrants_per_dim) {
                y = 0;
                x += 1;
            }

            device += 1;
            if (device >= device_count) {
                device = 0;
            }
        }

        for (int quadrant=0; quadrant<quadrants; quadrant++) {
            threads[quadrant].join();
        }

        gettimeofday(&end_time, NULL); 
        
        cudaSetDevice(origin_device);

        free(a_channels);
        free(b_channels);

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