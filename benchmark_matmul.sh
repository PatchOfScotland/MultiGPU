#!/bin/bash

REPEATS=$1
DEVICES=$2
N=$3
FLOPS=$4
NODE=$5

hostname
echo $CUDA_VISIBLE_DEVICES

## Matmul
module load cuda/12.2
make hendrix

echo "Benchmarking Matrix Multiplication ${N} (${DEVICES} devices)"
./build/matmul ${N} ${N} ${N} $REPEATS -d ${DEVICES} -r > ./results/hendrixgpu${NODE}fl/${DEVICES}/matmul_${FLOPS}.out
./build/matmul ${N} ${N} ${N} $REPEATS -d ${DEVICES} -r -s > ./results/hendrixgpu${NODE}fl/${DEVICES}/matmul-no-repeat_${FLOPS}.out

echo "All tests complete"
