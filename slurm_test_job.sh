#!/bin/bash

echo "SETTING UP $1"

hostname
echo $CUDA_VISIBLE_DEVICES
module load cuda/12.2

make matmul_sm
./build/matmul_sm 8192 10 -s