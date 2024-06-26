#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

## Matmul
module load cuda/12.2
make sanity_check

echo "Benchmarking sanity check 30000 (2 devices)"
./build/sanity_check 30000 1 -d 2 -r > ./results/sanity.out

echo "All tests complete"
