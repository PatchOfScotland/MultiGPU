#!/bin/bash

REPEATS=$1
DEVICES=$2

## Reduce
make reduce


for device in $(seq 1 $DEVICES)
do
    for i in "250000000 1GB" \
             "1000000000 4GB" \
             "2500000000 10GB" \
             "5000000000 20GB" \
             "10000000000 40GB" \
             "20000000000 80GB"
    do
        set -- $i
        echo "Benchmarking Reduce $2 ($device devices)"
        ./build/reduce $1 $REPEATS -d $device -r > ./results/reduce_${device}_${2}.out
        ./build/reduce $1 $REPEATS -d $device -r -s > ./results/reduce-no-repeat_${device}_${2}.out
    done
done