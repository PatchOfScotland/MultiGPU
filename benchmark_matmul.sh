#!/bin/bash

REPEATS=$1

## Map
make matmul
for i in "32 1GFLOPS" \
         "2048 68GFLOPS" \
         "4096 137GFLOPS" \
         "8192 274GFLOPS" \
         "12288 412GFLOPS" \
         "16384 549GFLOPS" \
         "20480 687GFLOPS" \
         "24576 824GFLOPS" \
         "28672 962GFLOPS" \
         "32768 1099GFLOPS" \
         "40960 1374GFLOPS" \
         "49152 1649GFLOPS" \
         "57344 1924GFLOPS" \
         "65536 2199GFLOPS"
do
    set -- $i
    echo "Benchmarking Matrix Multiplication $2"
    ./build/matmul 4096 $1 4096 $REPEATS -v -r > ./results/matmul_$2.out
    ./build/matmul 4096 $1 4096 $REPEATS -v -r -s > ./results/matmul-no-repeat_$2.out
done