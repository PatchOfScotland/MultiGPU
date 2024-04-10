#!/bin/bash

REPEATS=100

make clean
mkdir -p build

### Map
#make map
#for i in "500 4KB"         "1000 8KB" \
#         "5000 40KB"       "10000 80KB" \
#         "50000 400KB"     "100000 800KB" \
#         "500000 4MB"      "1000000 8MB" \
#         "5000000 40MB"    "10000000 80MB" \
#         "50000000 400MB"  "100000000 800MB" \
#         "500000000 4GB"   "1000000000 8GB" \
#         "5000000000 40GB" "10000000000 80GB"
#do
#    set -- $i
#    echo "Benchmarking Map $2"
#    ./build/map $1 $REPEATS -v -r > ./results/map_$2.out
#done

## Reduce
make reduce
for i in "1000 4KB"         "2000 8KB" \
         "10000 40KB"       "20000 80KB" \
         "100000 400KB"     "200000 800KB" \
         "1000000 4MB"      "2000000 8MB" \
         "10000000 40MB"    "20000000 80MB" \
         "100000000 400MB"  "200000000 800MB" \
         "1000000000 4GB"   "2000000000 8GB" \
         "10000000000 40GB" "20000000000 80GB"
do
    set -- $i
    echo "Benchmarking Reduce $2"
    ./build/reduce $1 $REPEATS -v -r > ./results/reduce_$2.out
done