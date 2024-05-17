#!/bin/bash

REPEATS=$1
DEVICES=$2

## Map
make matmul

for device in $(seq 1 $DEVICES)
do
    for i in "32 1GFLOPS" \
             "2048 68GFLOPS" \
             "4096 137GFLOPS" \
             "8192 274GFLOPS" \
             "16384 549GFLOPS" \
             "32768 1099GFLOPS"
    do
        set -- $i
        echo "Benchmarking Matrix Multiplication $2 ($device devices)"
        ./build/matmul 4096 $1 4096 $REPEATS -d $device -r > ./results/matmul_${device}_${2}.out
        ./build/matmul 4096 $1 4096 $REPEATS -d $device -r -s > ./results/matmul-no-repeat_${device}_${2}.out
    done
done