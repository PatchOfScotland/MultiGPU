#!/bin/bash

REPEATS=$1

make clean
mkdir -p build

./benchmark_map.sh
./benchmark_reduce.sh
./benchmark_matmul.sh

python3 assemble_graph.py