#include <sys/time.h>

#include "matmul/cannon.h"
#include "shared_cuda.cu.h"
#include "shared.h"

int main(int argc, char** argv){
    const unsigned int heightA = strtoul(argv[1], NULL, 0);
    const unsigned int widthA = strtoul(argv[1], NULL, 0);
    const unsigned int widthB = strtoul(argv[1], NULL, 0);
    const unsigned int heightB = widthA;
    const unsigned int widthC = widthB;
    const unsigned int heightC = heightA;
    const unsigned int runs = atoi(argv[2]);
    const unsigned int quadrants = atoi(argv[3]);

    const unsigned long int sizeA = widthA * heightA;
    const unsigned long int sizeB = widthB * heightB;
    const unsigned long int sizeC = widthC * heightC;

    double datasize_bytes = (double)(
        (1.0*widthA*heightA)
        +(1.0*widthB*heightB)
        +(1.0*widthC*heightC))*sizeof(array_type);
    double operations = (double)2.0*heightC*widthC*widthA;

    std::cout << "\nBenchmarking cannon multi GPU on " 
              << heightA << "x" << heightA 
              << " across " << runs << " runs, with " 
              << quadrants << " quadrants\n";

    std::cout << "Multiplying arrays of size " 
              << widthA << "x" << heightA
              << " and "
              << widthB << "x" << heightB
              << ", resulting in "
              << widthC << "x" << heightC
              << "\n";

    std::cout << "Input arrays using " 
              << datasize_bytes / 1e9
              << "GB of memory and "
              << operations / 1e9
              << " GFLOPs per experiment\n";

    array_type* matrixA = NULL;
    array_type* matrixB = NULL;
    array_type* matrixC = NULL;
    setup_managed(&matrixA, sizeA, false);
    setup_managed(&matrixB, sizeB, false);
    zero_managed(&matrixC, sizeC, false);

    int devices;
    CCC(cudaGetDeviceCount(&devices));

    for (int run=0; run<runs; run++) {
        double timing_μs = cannon::multiGPU<array_type, BLOCK_N>(
            matrixA, matrixB, matrixC, widthC, devices, quadrants, false
        );
        printf("%lf\n", timing_μs);
    }

    free_managed(&matrixA);
    free_managed(&matrixB);
    free_managed(&matrixC);
}