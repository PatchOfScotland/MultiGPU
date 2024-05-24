#!/bin/bash

mkdir -p slurm

REPEATS=${1:-10}

for i in {1,2,4}
do
    sbatch -o "slurm/map_${i}_%j.out" -p gpu ./benchmark_map.sh $REPEATS $i
    sbatch -o "slurm/reduce_${i}_%j.out" -p gpu ./benchmark_reduce.sh $REPEATS $i
    sbatch -o "slurm/matmul_${i}_%j.out" -p gpu ./benchmark_matmul.sh $REPEATS $i
done
