
#include "kernels.cu.h"
#include "../shared_cuda.cu.h"
#include "../shared.h"

/**
 * Naive kernel, i.e., the only tiling performed is on the grid;
 *   no shared or private memory is used.
 * It computes result[i,j] = Sum_{k=0...width_A} array_A[i,k] * array_B[j,k]
 *   where array_A: [height_A][width_A]T and
 *         array_B: [height_A][width_A]T
 * except that array_A and array_B maybe transposed version of materialized
 * matrices, i.e.,
 *   isTA == 1 => array_A = transpose(Ao), where Ao: [width_A][height_A]T
 *   isTB == 1 => array_B = transpose(Bo), where Bo: [width_A][height_B]T
 * The case (isTA,isTB) = (0,1) corresponds to matrix multiplication
 *   where width_B is actually height_B
 */ 
template<int isTA, int isTB, typename T, int TL>
cudaError_t singleGpuMatMul(int height_A, int width_A, int height_B,
                T* array_A, T* array_B, T* result
) {  
    // setup execution parameters
    int  dim_y = (height_A + TL - 1) / TL; 
    int  dim_x = (height_B + TL - 1) / TL;

    dim3 block(TL, TL, 1);
    dim3 grid(dim_x, dim_y, 1);
    
    mmmNaiveKernel<isTA, isTB, T> <<< grid, block >>>(
        array_A, array_B, result, height_A, height_B, width_A
    );

    return cudaGetLastError();
}