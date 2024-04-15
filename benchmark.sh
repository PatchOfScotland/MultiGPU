#!/bin/bash

REPEATS=25

make clean
mkdir -p build

## Map
make map
for i in "1000 8KB" \
         "10000 80KB" \
         "100000 800KB" \
         "1000000 8MB" \
         "10000000 80MB" \
         "100000000 800MB" \
         "1000000000 8GB" \
         "2000000000 16GB" \
         "3000000000 24GB" \
         "4000000000 32GB" \
         "5000000000 40GB" \
         "6000000000 48GB" \
         "7000000000 56GB" \
         "8000000000 64GB" \
         "9000000000 72GB" \
         "10000000000 80GB"
do
    set -- $i
    echo "Benchmarking Map $2"
    ./build/map $1 $REPEATS -v -r > ./results/map_$2.out
    ./build/map $1 $REPEATS -v -r -s > ./results/map-no-repeat_$2.out
done

## Reduce
make reduce
for i in "1000 4KB" \
         "10000 40KB" \
         "100000 400KB" \
         "1000000 4MB" \
         "10000000 40MB" \
         "100000000 400MB" \
         "250000000 1GB" \
         "1000000000 4GB" \
         "2500000000 10GB" \
         "5000000000 20GB" \
         "7500000000 30GB" \
         "10000000000 40GB" \
         "12500000000 50GB" \
         "15000000000 60GB" \
         "17500000000 70GB" \
         "20000000000 80GB"
do
    set -- $i
    echo "Benchmarking Reduce $2"
    ./build/reduce $1 $REPEATS -v -r > ./results/reduce_$2.out
    ./build/reduce $1 $REPEATS -v -r -s > ./results/reduce-no-repeat_$2.out
done

python3 assemble_graph.py