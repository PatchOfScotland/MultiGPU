#!/bin/bash

REPEATS=$1
DEVICES=$2

## Map
make map


for device in $(seq 1 $DEVICES)
do
    for i in "2000000000 16GB" \
             "4000000000 32GB" \
             "6000000000 48GB" \
             "8000000000 64GB" \
             "10000000000 80GB"
    do
        set -- $i
        echo "Benchmarking Map $2 ($device devices)"
        ./build/map $1 $REPEATS -d $device -r > ./results/map_${device}_${2}.out
        ./build/map $1 $REPEATS -d $device -r -s > ./results/map-no-repeat_${device}_${2}.out
    done
done