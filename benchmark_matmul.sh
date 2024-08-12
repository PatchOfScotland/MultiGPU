#!/bin/bash

REPEATS=$1
DEVICES=$2
N=$3
FLOPS=$4
NODE=$5
BUILD_DIR=$6

hostname
echo $CUDA_VISIBLE_DEVICES

## Matmul
module load cuda/12.2

echo "Benchmarking Matrix Multiplication ${N} (${DEVICES} devices) using ${BUILD_DIR}/matmul"

./${BUILD_DIR}/matmul ${N} ${N} ${N} $REPEATS -d ${DEVICES} -r > ./results/hendrixgpu${NODE}fl/${DEVICES}/matmul_${FLOPS}.out
./${BUILD_DIR}/matmul ${N} ${N} ${N} $REPEATS -d ${DEVICES} -r -s > ./results/hendrixgpu${NODE}fl/${DEVICES}/matmul-no-repeat_${FLOPS}.out

echo "All tests complete"
