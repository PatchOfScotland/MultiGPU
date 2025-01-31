
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

// note only suitable for square matrixes
namespace cannon {
    array_type** matrixAs;
    array_type** matrixBs;

    pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    unsigned int waiting = 0;

    void synchronize(const unsigned thread_total) {
        pthread_mutex_lock(&mutex);
        waiting += 1;
        if (waiting == thread_total) {
            pthread_cond_broadcast(&condition);
            waiting = 0;
        } else {
            pthread_cond_wait(&condition, &mutex);
        }
        pthread_mutex_unlock(&mutex);
    }

    template<typename T, int TL>
    void per_device_management(
            T* matrixA, T* matrixB, T* matrixC, const unsigned int quadrant_n, 
            const int device, const size_t quadrants_per_dim, int quadrants,
            const int a_write, const int b_write,
            const int a_read, const int b_read,
            const int debug_quadrant
    ) {
        cudaSetDevice(device);
        //int a = 0;
        //cudaDeviceGetAttribute(&a, cudaDevAttrConcurrentManagedAccess, device);
        //printf("Running on device %d with attr: %d\n", device, a);

        unsigned int dim_x = (quadrant_n + TL - 1) / TL; 
        unsigned int dim_y = (quadrant_n + TL - 1) / TL;

        dim3 dimBlock(TL, TL, 1);      // threads per block
        dim3 dimGrid(dim_x, dim_y, 1); // blockcount

        //printf("Quadrant: %d is scheduling %dx%dx%d (%d) blocks each of %d threads\n", debug_quadrant, dimGrid.x, dimGrid.y, dimGrid.z, dimGrid.x*dimGrid.y*dimGrid.z, dimBlock.x*dimBlock.y*dimBlock.z);
      
        for (int iteration=0; iteration<quadrants_per_dim; iteration++) {
            cudaEvent_t sync_event;
            CCC(cudaEventCreate(&sync_event));
            
            cuda_error_check();

            mmmNaiveKernelAdditive<T> <<<dimGrid, dimBlock>>>(
                matrixA, matrixB, matrixC, quadrant_n
            );
            
            cuda_error_check();

            CCC(cudaEventRecord(sync_event));
            
            cuda_error_check();

            CCC(cudaEventSynchronize(sync_event));

            cuda_error_check();

            if (iteration != quadrants_per_dim-1) {              
                synchronize(quadrants);
            
                matrixAs[a_write] = matrixA;
                matrixBs[b_write] = matrixB;
            
                synchronize(quadrants);
            
                matrixA = matrixAs[a_read];
                matrixB = matrixBs[b_read];
            }
        }
    }

    template<typename T, int TL>
    float overlappingMultiGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n,
        const int device_count, const size_t quadrants_per_dim, bool zero_c
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

        matrixAs = (T**)malloc(sizeof(T*)*quadrants);
        matrixBs = (T**)malloc(sizeof(T*)*quadrants);

        if (zero_c == true) {
            zero_matrix(matrixC, n*n);
        }

        unsigned int quadrant_size = quadrant_n * quadrant_n;

        unsigned int device = 0;
        std::thread threads[quadrants];

        int x = 0;
        int y = 0;

        struct timeval start_time;
        struct timeval end_time;

        gettimeofday(&start_time, NULL); 

        for (int quadrant=0; quadrant<quadrants; quadrant++) {  
            
            int ax_write = x;
            int ay_write = (y + quadrants_per_dim - 1) % quadrants_per_dim;
            int bx_write = (x + quadrants_per_dim - 1) % quadrants_per_dim;
            int by_write = y;

            //int ax_read = x;
            //int ay_read = (y + 1) % quadrants_per_dim;
            //int bx_read = (x + 1) % quadrants_per_dim;
            //int by_read = y;

            unsigned int quadrant_a_write = (ax_write * quadrants_per_dim) + ay_write;
            unsigned int quadrant_b_write = (bx_write * quadrants_per_dim) + by_write;
            //unsigned int quadrant_a_read = (ax_read * quadrants_per_dim) + ay_read;
            //unsigned int quadrant_b_read = (bx_read * quadrants_per_dim) + by_read;
            unsigned int quadrant_a_read = quadrant;
            unsigned int quadrant_b_read = quadrant;
            
            int k = (x + y) % quadrants_per_dim;

            unsigned int quadrant_offset_a = (quadrant_size * (k + (x * quadrants_per_dim)));
            unsigned int quadrant_offset_b = (quadrant_size * (y + (k * quadrants_per_dim)));
            unsigned int quadrant_offset_c = (quadrant_size * (y + (x * quadrants_per_dim)));
            
            //printf("Quadrant %d is: %d\n", quadrant, quadrant);
            //printf("Quadrant %d starts with A: %d, B: %d, C: %d\n", quadrant, quadrant_offset_a, quadrant_offset_b, quadrant_offset_c);
            //printf("Quadrant %d writing A to: %d, B to: %d, \n", quadrant, quadrant_a_write, quadrant_b_write);
            //printf("Quadrant %d reading A from: %d, B from: %d, \n", quadrant, quadrant_a_read, quadrant_b_read);
            //printf("Quadrant %d Should just read A from: %d, B from: %d, \n", quadrant, quadrant, quadrant);

            threads[quadrant] = std::thread(
                per_device_management<T, TL>,
                matrixA + quadrant_offset_a, 
                matrixB + quadrant_offset_b, 
                matrixC + quadrant_offset_c, 
                quadrant_n, device, quadrants_per_dim, quadrants,
                quadrant_a_write, quadrant_b_write, 
                quadrant_a_read, quadrant_b_read,
                quadrant
            );

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

        free(matrixAs);
        free(matrixBs);

        float time_microseconds = (end_time.tv_usec+(1e6*end_time.tv_sec)) 
            - (start_time.tv_usec+(1e6*start_time.tv_sec));
    
        return time_microseconds;
    }

    template<typename T, int TL>
    void per_quadrant_management(
            T* matrixA, T* matrixB, T* matrixC, const unsigned int quadrant_n, 
            const int device, const size_t quadrants_per_dim, int quadrants,
            const int a_write, const int b_write,
            const int a_read, const int b_read,
            const int debug_quadrant
    ) {
        cudaSetDevice(device);
        //int a = 0;
        //cudaDeviceGetAttribute(&a, cudaDevAttrConcurrentManagedAccess, device);
        //printf("Running on device %d with attr: %d\n", device, a);

        unsigned int dim_x = (quadrant_n + TL - 1) / TL; 
        unsigned int dim_y = (quadrant_n + TL - 1) / TL;

        dim3 dimBlock(TL, TL, 1);      // threads per block
        dim3 dimGrid(dim_x, dim_y, 1); // blockcount

        //printf("Quadrant: %d is scheduling %dx%dx%d (%d) blocks each of %d threads\n", debug_quadrant, dimGrid.x, dimGrid.y, dimGrid.z, dimGrid.x*dimGrid.y*dimGrid.z, dimBlock.x*dimBlock.y*dimBlock.z);
      
        for (int iteration=0; iteration<quadrants_per_dim; iteration++) {
            cudaEvent_t sync_event;
            CCC(cudaEventCreate(&sync_event));
            
            cuda_error_check();

            //printf("Device %d using matrices A:%f, B:%f and C:%f\n", device, matrixA[0], matrixB[0], matrixC[0]);

            mmmNaiveKernelAdditive<T> <<<dimGrid, dimBlock>>>(
                matrixA, matrixB, matrixC, quadrant_n
            );
            
            cuda_error_check();

            CCC(cudaEventRecord(sync_event));
            
            cuda_error_check();

            CCC(cudaEventSynchronize(sync_event));

            cuda_error_check();

            if (iteration != quadrants_per_dim-1) {              
                synchronize(quadrants);
            
                matrixAs[a_write] = matrixA;
                matrixBs[b_write] = matrixB;
            
                synchronize(quadrants);
            
                matrixA = matrixAs[a_read];
                matrixB = matrixBs[b_read];
            }
        }
    }

    template<typename T, int TL>
    float multiGPU(
        T* matrixA, T* matrixB, T* matrixC, unsigned int n,
        const int device_count, const size_t quadrants_per_dim, bool validate 
    ) {
        int origin_device;
        CCC(cudaGetDevice(&origin_device));


        if (n % quadrants_per_dim) {
            perror("Cannot blocks evenly accross devices\n");
            exit(EXIT_FAILURE);
        }

        //printf("A:\n");
        //print_matrix(matrixA, n, n);
        //printf("B:\n");
        //print_matrix(matrixB, n, n);
        //printf("C:\n");
        //print_matrix(matrixC, n, n);

        const unsigned int quadrants = quadrants_per_dim * quadrants_per_dim;
        const unsigned int quadrant_n = (n + quadrants_per_dim - 1) / quadrants_per_dim;;
        //std::cout << "Quads: " << quadrants << "," <<  quadrants_per_dim << "," <<  quadrant_n << "\n";

        matrixAs = (T**)malloc(sizeof(T*)*quadrants);
        matrixBs = (T**)malloc(sizeof(T*)*quadrants);

        if (validate == true) {
            zero_matrix(matrixC, n*n);
        }

        unsigned int quadrant_size = quadrant_n * quadrant_n;

        unsigned int device = 0;
        std::thread threads[quadrants];

        int x = 0;
        int y = 0;

        struct timeval start_time;
        struct timeval end_time;

        gettimeofday(&start_time, NULL); 

        for (int quadrant=0; quadrant<quadrants; quadrant++) {  
            
            int ax_write = x;
            int ay_write = (y + quadrants_per_dim - 1) % quadrants_per_dim;
            int bx_write = (x + quadrants_per_dim - 1) % quadrants_per_dim;
            int by_write = y;

            //int ax_read = x;
            //int ay_read = (y + 1) % quadrants_per_dim;
            //int bx_read = (x + 1) % quadrants_per_dim;
            //int by_read = y;

            unsigned int quadrant_a_write = (ax_write * quadrants_per_dim) + ay_write;
            unsigned int quadrant_b_write = (bx_write * quadrants_per_dim) + by_write;
            //unsigned int quadrant_a_read = (ax_read * quadrants_per_dim) + ay_read;
            //unsigned int quadrant_b_read = (bx_read * quadrants_per_dim) + by_read;
            unsigned int quadrant_a_read = quadrant;
            unsigned int quadrant_b_read = quadrant;
            
            int k = (x + y) % quadrants_per_dim;

            unsigned int quadrant_offset_a = (quadrant_size * (k + (x * quadrants_per_dim)));
            unsigned int quadrant_offset_b = (quadrant_size * (y + (k * quadrants_per_dim)));
            unsigned int quadrant_offset_c = (quadrant_size * (y + (x * quadrants_per_dim)));
            
            //printf("Device %d starts with:\n", device);
            //printf("A\n");
            //print_matrix(matrixA + quadrant_offset_a, quadrant_n, quadrant_n);
            //printf("B\n");
            //print_matrix(matrixB + quadrant_offset_b, quadrant_n, quadrant_n);
            //printf("C\n");
            //print_matrix(matrixC + quadrant_offset_c, quadrant_n, quadrant_n);
            //printf("Quadrant %d starts with A: %d, B: %d, C: %d\n", quadrant, quadrant_offset_a, quadrant_offset_b, quadrant_offset_c);
            //printf("Quadrant %d writing A to: %d, B to: %d, \n", quadrant, quadrant_a_write, quadrant_b_write);
            //printf("Quadrant %d reading A from: %d, B from: %d, \n", quadrant, quadrant_a_read, quadrant_b_read);
            //printf("Quadrant %d Should just read A from: %d, B from: %d, \n", quadrant, quadrant, quadrant);

            threads[quadrant] = std::thread(
                per_quadrant_management<T, TL>,
                matrixA + quadrant_offset_a, 
                matrixB + quadrant_offset_b, 
                matrixC + quadrant_offset_c, 
                quadrant_n, device, quadrants_per_dim, quadrants,
                quadrant_a_write, quadrant_b_write, 
                quadrant_a_read, quadrant_b_read,
                quadrant
            );

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

        free(matrixAs);
        free(matrixBs);

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

        cuda_error_check();

        float runtime_milliseconds = get_runtime(start_event, end_event);
        
        return runtime_milliseconds * 1e3;
    }



    // ----- Below here are legacy implementations --------------

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

        printf(
            "  Scheduling %d (%dx%dx%d) blocks of %d (%dx%dx%d)threads\n", 
            dimGrid.x*dimGrid.y*dimGrid.z, 
            dimGrid.x, dimGrid.y, dimGrid.z, 
            dimBlock.x*dimBlock.y*dimBlock.z,
            dimBlock.x, dimBlock.y, dimBlock.z
        );
                
        cudaEvent_t sync_event;
        CCC(cudaEventCreate(&sync_event));
        mmmCannonQuadrant<T, cannon_block> <<<dimGrid, dimBlock>>>(
            matrixA, matrixB, matrixC, 
            total_n, quadrant_n, blocks_dim, offset_x, offset_y
        );
        CCC(cudaEventRecord(sync_event));
        CCC(cudaEventSynchronize(sync_event));

        cuda_error_check();
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

}