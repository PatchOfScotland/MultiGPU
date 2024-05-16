#!/bin/bash

REPEATS=$1
DEVICES=$2

## Map
make map


for device in $(seq 1 $DEVICES)
do
    for i in "1000 8KB" \
             "10000 80KB" \
             "100000 800KB" #\
    #         "1000000 8MB" \
    #         "10000000 80MB" \
    #         "100000000 800MB" \
    #         "1000000000 8GB" \
    #         "2000000000 16GB" \
    #         "3000000000 24GB" \
    #         "4000000000 32GB" \
    #         "5000000000 40GB" \
    #         "6000000000 48GB" \
    #         "7000000000 56GB" \
    #         "8000000000 64GB" \
    #         "9000000000 72GB" \
    #         "10000000000 80GB"
    do
        set -- $i
        echo "Benchmarking Map $2 ($device devices)"
        ./build/map $1 $REPEATS -d $device -r > ./results/map_${device}_${2}.out
        ./build/map $1 $REPEATS -d $device -r -s > ./results/map-no-repeat_${device}_${2}.out
    done
done