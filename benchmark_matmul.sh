#!/bin/bash

REPEATS=$1

## Map
make matmul
for i in "128 1GFLOPS" \
         "256 2GFLOPS" \
         "512 4GFLOPS" \
         "1024 8GFLOPS" \
         "2048 17GFLOPS" \
         "4096 34GFLOPS" \
         "6144 51GFLOPS" \
         "8192 68GFLOPS" \
         "10240 85GFLOPS" \
         "12288 103GFLOPS" \
         "14336 120GFLOPS" \
         "16384 137GFLOPS"
do
    set -- $i
    echo "Benchmarking Matrix Multiplication $2"
    ./build/matmul 2048 $1 2048 $REPEATS -v -r > ./results/matmul_$2.out
    ./build/matmul 2048 $1 2048 $REPEATS -v -r -s > ./results/matmul-no-repeat_$2.out
done