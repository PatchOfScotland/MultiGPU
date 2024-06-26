#!/bin/bash

echo "SETTING UP $1"

hostname
echo $CUDA_VISIBLE_DEVICES
module load cuda/12.2

make hendrix
./build/map 1000 1 -d 1 -r > ./results/hendrix_test_$1.out