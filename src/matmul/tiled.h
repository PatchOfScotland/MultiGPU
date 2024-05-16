
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

template<int isTA, int isTB, typename T, int TL>
void per_device_management(
        T* matrixA, unsigned int widthA, unsigned int heightA, 
        T* matrixB, unsigned int widthB, unsigned int heightB, 
        T* matrixC, unsigned int widthC, unsigned int per_device_heightC,
        int device
) {
    cudaSetDevice(device);

    T* sub_matrixA = matrixA + (device * per_device_heightC * widthA);
    T* sub_matrixC = matrixC + (device * per_device_heightC * widthC);

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
        device * per_device_heightC
    );
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));
}

template<int isTA, int isTB, typename T, int TL>
void per_device_management_split(
        T* matrixA, unsigned int widthA, unsigned int heightA, 
        T* matrixB, unsigned int widthB, unsigned int heightB, 
        T* matrixC, unsigned int widthC, unsigned int heightC,
        int device
) {
    cudaSetDevice(device);

    // setup execution parameters
    unsigned int dim_x = (widthC + TL - 1) / TL; 
    unsigned int dim_y = (heightC + TL - 1) / TL;

    dim3 block(TL, TL, 1);      // blockcount
    dim3 grid(dim_x, dim_y, 1); // threads per block

    cudaEvent_t sync_event;
    CCC(cudaEventCreate(&sync_event));
    mmmNaiveKernelMulti<isTA, isTB, T> <<< grid, block >>>(
        matrixA, widthA, heightC,
        matrixB, widthB, heightB,
        matrixC, widthC, heightC,
        device * heightC
    );
    CCC(cudaEventRecord(sync_event));
    CCC(cudaEventSynchronize(sync_event));
}

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
        T* matrixC, unsigned int widthC, unsigned int heightC,
        const int device_count, int hints
    ) {  
        int origin_device;
        CCC(cudaGetDevice(&origin_device));

        unsigned int per_device_heightC = (heightC + device_count - 1) / device_count;

        struct timeval start_time;
        struct timeval end_time;

        if (hints == HINTS) {
            for (int device=0; device<device_count; device++)
            {
                CCC(cudaMemAdvise(
                    matrixA + (device * per_device_heightC * widthA), 
                    widthA * per_device_heightC * sizeof(T), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
                CCC(cudaMemAdvise(
                    matrixC + (device * per_device_heightC * widthC), 
                    widthC *per_device_heightC * sizeof(T), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
            }
        }

        gettimeofday(&start_time, NULL); 

        std::thread threads[device_count];
        for (int device=0; device<device_count; device++) {
            threads[device] = std::thread(
                per_device_management<isTA, isTB, T, TL>,
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

    template<int isTA, int isTB, typename T, int TL>
    float multiGPUduplicate(
        T* matrixA, unsigned int widthA, unsigned int heightA, 
        T** matrixBs, unsigned int widthB, unsigned int heightB, 
        T* matrixC, unsigned int widthC, unsigned int heightC,
        const int device_count, int hints
    ) { 
        int origin_device;
        CCC(cudaGetDevice(&origin_device));

        unsigned int per_device_heightC = (heightC + device_count - 1) / device_count;

        struct timeval start_time;
        struct timeval end_time;

        if (hints == HINTS) {
            for (int device=0; device<device_count; device++)
            {
                CCC(cudaMemAdvise(
                    matrixA + (device * per_device_heightC * widthA), 
                    widthA * per_device_heightC * sizeof(T), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
                CCC(cudaMemAdvise(
                    matrixBs[device], 
                    widthB * heightB * sizeof(T), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
                CCC(cudaMemAdvise(
                    matrixC + (device * per_device_heightC * widthC), 
                    widthC *per_device_heightC * sizeof(T), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
            }
        }
        if (hints == PREFETCH) {
            for (int device=0; device<device_count; device++)
            {
                cudaSetDevice(device);
                cudaEvent_t sync_event;
                CCC(cudaEventCreate(&sync_event));
                CCC(cudaMemPrefetchAsync(
                    matrixA + (device * per_device_heightC * widthA), 
                    widthA * per_device_heightC * sizeof(T), 
                    device
                ));
                CCC(cudaMemPrefetchAsync(
                    matrixBs[device], 
                    widthB * heightB * sizeof(T), 
                    device
                ));
                CCC(cudaMemPrefetchAsync(
                    matrixC + (device * per_device_heightC * widthC), 
                    widthC * per_device_heightC * sizeof(T), 
                    device
                ));
                CCC(cudaEventRecord(sync_event));
                CCC(cudaEventSynchronize(sync_event));
            }
            cudaSetDevice(origin_device);
        }
        gettimeofday(&start_time, NULL); 

        std::thread threads[device_count];
        for (int device=0; device<device_count; device++) {
            threads[device] = std::thread(
                per_device_management<isTA, isTB, T, TL>,
                matrixA, widthA, heightA, 
                matrixBs[device], widthB, heightB, 
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

    template<int isTA, int isTB, typename T, int TL>
    float multiGPUsplit(
        T** matrixAs, unsigned int widthA, unsigned int split_heightA, 
        T** matrixBs, unsigned int widthB, unsigned int heightB, 
        T** matrixCs, unsigned int split_widthC, unsigned int split_heightC,
        T* matrixC, unsigned int widthC, unsigned int heightC,
        const int device_count, int hints, int reduce
    ) { 
        int origin_device;
        CCC(cudaGetDevice(&origin_device));

        struct timeval start_time;
        struct timeval end_time;

        if (hints == HINTS) {
            for (int device=0; device<device_count; device++)
            {
                CCC(cudaMemAdvise(
                    matrixAs[device], 
                    widthA * split_heightA * sizeof(T), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
                CCC(cudaMemAdvise(
                    matrixBs[device], 
                    widthB * heightB * sizeof(T), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
                CCC(cudaMemAdvise(
                    matrixCs[device], 
                    split_widthC * split_heightC * sizeof(T), 
                    cudaMemAdviseSetPreferredLocation, 
                    device
                ));
            }
        }
        if (hints == PREFETCH) {
            for (int device=0; device<device_count; device++)
            {
                cudaSetDevice(device);
                cudaEvent_t sync_event;
                CCC(cudaEventCreate(&sync_event));
                CCC(cudaMemPrefetchAsync(
                    matrixAs[device], 
                    widthA * split_heightA * sizeof(T), 
                    device
                ));
                CCC(cudaMemPrefetchAsync(
                    matrixBs[device], 
                    widthB * heightB * sizeof(T), 
                    device
                ));
                CCC(cudaMemPrefetchAsync(
                    matrixCs[device], 
                    split_widthC * split_heightC * sizeof(T), 
                    device
                ));
                CCC(cudaEventRecord(sync_event));
                CCC(cudaEventSynchronize(sync_event));
            }
            cudaSetDevice(origin_device);
        }
        gettimeofday(&start_time, NULL); 

        std::thread threads[device_count];
        for (int device=0; device<device_count; device++) {
            threads[device] = std::thread(
                per_device_management_split<isTA, isTB, T, TL>,
                matrixAs[device], widthA, split_heightA, 
                matrixBs[device], widthB, heightB, 
                matrixCs[device], split_widthC, split_heightC,
                device
            );
        }

        for (int device=0; device<device_count; device++) {
            threads[device].join();
            cudaSetDevice(device);
            if (reduce == MEMCPY) {
                CCC(cudaMemcpy(
                    matrixC + (split_widthC * split_heightC * device), 
                    matrixCs[device], 
                    split_widthC * split_heightC * sizeof(T), 
                    cudaMemcpyDeviceToHost
                ));
            } else if (reduce == DUPLICATE) {
                duplicate_matrix(
                    matrixCs[device], 
                    split_widthC * split_heightC,  
                    matrixC + (split_widthC * split_heightC * device)
                );
            }
        }
        cudaSetDevice(origin_device);
        
        gettimeofday(&end_time, NULL); 
        
        cudaSetDevice(origin_device);

        float time_microseconds = (end_time.tv_usec+(1e6*end_time.tv_sec)) 
            - (start_time.tv_usec+(1e6*start_time.tv_sec));
    
        return time_microseconds;
    }
}
