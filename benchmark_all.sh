#!/bin/bash

REPEATS=$1
DEVICES=$2

make clean
mkdir -p build

./benchmark_map.sh $REPEATS $DEVICES
./benchmark_reduce.sh $REPEATS $DEVICES
./benchmark_matmul.sh $REPEATS $DEVICES

python3 assemble_csv_and_graphs.py
