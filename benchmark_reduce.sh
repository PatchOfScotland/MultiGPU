#!/bin/bash

REPEATS=$1
DEVICES=$2

## Reduce
make reduce


for device in $(seq 1 $DEVICES)
do
    for i in "1000 4KB" \
             "10000 40KB" \
             "100000 400KB" \
    #         "1000000 4MB" \
    #         "10000000 40MB" \
    #         "100000000 400MB" \
    #         "250000000 1GB" \
    #         "1000000000 4GB" \
    #         "2500000000 10GB" \
    #         "5000000000 20GB" \
    #         "7500000000 30GB" \
    #         "10000000000 40GB" \
    #         "12500000000 50GB" \
    #         "15000000000 60GB" \
    #         "17500000000 70GB" \
    #         "20000000000 80GB"
    do
        set -- $i
        echo "Benchmarking Reduce $2 ($device devices)"
        ./build/reduce $1 $REPEATS -d $device -r > ./results/reduce_${device}_${2}.out
        ./build/reduce $1 $REPEATS -d $device -r -s > ./results/reduce-no-repeat_${device}_${2}.out
    done
done