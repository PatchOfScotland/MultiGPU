#include <functional>

#include "shared.cu.h"
#include "shared.h"

typedef float arrayType;

// Toy function to be mapped accross an array. Just adds a constant x to an 
// element
template <typename T>
T PlusConst(const T inputElement, const T x) {
    return inputElement + x;
}

// Mapping function that takes a function and maps it across each element in an
// input array, with the output in a new output array. Opperates entirely on 
// the CPU. 
template<typename F, typename T>
void cpuMapping(F mapped_function, T* input_array, const T constant, T* output_array, int array_len) {  
    #pragma omp parallel for
    for (int i=0; i<array_len; i++) {
        output_array[i] = mapped_function(input_array[i], constant);
    }
}

template<typename T>
__global__ void singleGpuKernel(T* input_array, const T x, T* output_array, int array_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < array_len) {
        output_array[index] = input_array[index] + x;
    }
}

template<typename F, typename T>
void singleGpuMapping(F mapped_kernel, T* input_array, const T constant, T* output_array, int array_len) {  
    size_t block_size = 1024;
    size_t block_count = (array_len + block_size - 1) / block_size;

    mapped_kernel<<<block_count, block_size>>>(input_array, constant, output_array, array_len);
}

int main(int argc, char** argv){
    if (argc < 2)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array length> -v(optional)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned int array_len = atoi(argv[1]);
    bool validating = false;

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
    }

    std::cout << "Running array of length " 
              << array_len 
              << " (" 
              << ((array_len*2*sizeof(arrayType))/1e9) 
              <<"GB)\n";
    if (validating) {
        std::cout << "Will validate output\n";
    }
    else {
        std::cout << "Skipping output validation\n";
    }

    arrayType* input_array;
    arrayType* output_array;
    arrayType* validation_array;
    arrayType constant = 0.1;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float runtime;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(arrayType)));
    CCC(cudaMallocManaged(&output_array, array_len*sizeof(arrayType)));
    CCC(cudaEventCreate(&start_event));
    CCC(cudaEventCreate(&end_event));

    init_array(input_array, array_len);

    if (validating) {
        validation_array = (arrayType*)malloc(array_len*sizeof(arrayType));
        cpuMapping(PlusConst<arrayType>, input_array, constant, validation_array, array_len);

        std::cout << input_array[0] << ", " << input_array[1] << "\n";
        std::cout << validation_array[0] << ", " << validation_array[1] << "\n";
    }


    // Warmup run

    { // Benchmark a single GPU
        std::cout << "*** Benchmarking single GPU map ***\n";
        CCC(cudaEventRecord(start_event));
        singleGpuMapping(singleGpuKernel<arrayType>, input_array, constant, output_array, array_len);
        CCC(cudaEventRecord(end_event));
        CCC(cudaDeviceSynchronize()); 
        CCC(cudaEventElapsedTime(&runtime, start_event, end_event));

        if (validating) {
            if(compare_arrays(validation_array, output_array, array_len)){
                std::cout << "  Single GPU map is correct\n";
            } else {
                std::cout << "  Single GPU map is incorrect\n";
            }
        }
        std::cout << "  Runtime was: " << runtime << "ms\n";

    }

//    for (int i=0; i<array_len; i++) {
//        std::cout << validation_array[i] << ", \t" << output_array[i] << "\n";
//    }

    if (validating) {
        free(validation_array);
    }
    cudaFree(input_array);
    cudaFree(output_array);
}
