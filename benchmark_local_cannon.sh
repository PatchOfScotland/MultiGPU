#!/bin/bash

N=$1
REPEATS=$2
BUILD_DIR=$3

hostname
echo $CUDA_VISIBLE_DEVICES

make cannon_dev

## Matmul
#module load cuda/12.2

echo "Benchmarking Matrix Multiplication ${N} using ${BUILD_DIR}/cannon_dev"

./${BUILD_DIR}/cannon_dev ${N} $REPEATS -v
# > ./results/cannon_${N}.out
#./${BUILD_DIR}/cannon_dev ${N} $REPEATS -v -s
#> ./results/cannon_${N}_standalone.out

echo "All tests complete"
