#!/bin/bash

mkdir -p results
mkdir -p slurm

REPEATS=${1:-10}

#12 13 19 20
for node in 20
do
    ./schedule_benchmark_node.sh ${node} $REPEATS
done