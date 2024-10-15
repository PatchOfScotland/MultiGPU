
#include "shared_cuda.cu.h"
#include "shared.h"

template <class T> 
__global__ void kernel(
    const T *matrixA, T* const matrixB, const int array_w, const int array_h, 
    const int offset
) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int len = array_h * array_w;
    if (index < len) {
        matrixB[index+offset] = matrixA[index+offset] + 1; 
    }
}

void setup_events_sanity(
    cudaEvent_t** events_ptr, int origin_device, int device_count
) {
    cudaEvent_t* events = *events_ptr;
    
    for (int device=0; device<device_count; device++) {
        cudaSetDevice(device);

        cudaEvent_t event;
        CCC(cudaEventCreate(&event));
        events[device] = event;
    }

    cudaSetDevice(origin_device);
}

void setup_managed(    
    array_type** matrix, const unsigned int array_h, const unsigned int array_n
) {
    size_t size = array_h*array_n;
    CCC(cudaMallocManaged(matrix, size*sizeof(array_type)));
}

void setup_init_managed(
    array_type** matrix, const unsigned int array_h, const unsigned int array_n
) {
    size_t size = array_h*array_n;

    setup_managed(matrix, array_h, array_n);
    init_matrix<array_type>(*matrix, size);
}

int main(int argc, char** argv){
    if (argc < 2) {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array n>\n";
        exit(EXIT_FAILURE);
    } 
 
    const unsigned int array_n = strtoul(argv[1], NULL, 0);

    int origin_device;
    CCC(cudaGetDevice(&origin_device));

    array_type* matrixA = NULL;
    array_type* matrixB = NULL;

    setup_init_managed(&matrixA, array_n, array_n);
    setup_managed(&matrixB, array_n, array_n);

    if (true) {
        std::cout << "Matrix A: \n";
        print_matrix(matrixA, 10, 10);
        std::cout << "Matrix B: \n";
        print_matrix(matrixB, 10, 10);
    }

    int array_len = array_n*array_n;
    int device_count = 2;

    cudaEvent_t* start_events = 
        (cudaEvent_t*)malloc(device_count * sizeof(cudaEvent_t));
    cudaEvent_t* end_events = 
        (cudaEvent_t*)malloc(device_count * sizeof(cudaEvent_t));

    setup_events_sanity(&start_events, origin_device, device_count);
    setup_events_sanity(&end_events, origin_device, device_count);

    int per_dev_n = array_n/device_count;
    for (int device=0; device<device_count; device++) {
        CCC(cudaSetDevice(device));

        CCC(cudaEventRecord(start_events[device]));
        kernel<array_type><<<array_len, BLOCK_SIZE>>>(
            matrixA, matrixB, array_n, per_dev_n, device*(per_dev_n*array_n)
        );
        CCC(cudaEventRecord(end_events[device]));
    }

    for (int device=0; device<device_count; device++) {
        CCC(cudaSetDevice(device));
        CCC(cudaEventSynchronize(end_events[device]));
    }  

    if (true) {
        std::cout << "Matrix A: \n";
        print_matrix(matrixA, 10, 10);
        std::cout << "Matrix B: \n";
        print_matrix(matrixB, 10, 10);
    }
}