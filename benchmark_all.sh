#!/bin/bash

REPEATS=$1

make clean
mkdir -p build

./benchmark_map.sh $REPEATS
./benchmark_reduce.sh $REPEATS
./benchmark_matmul.sh $REPEATS

python3 assemble_graph.py