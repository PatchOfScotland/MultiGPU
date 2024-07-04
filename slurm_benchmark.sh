#!/bin/bash

mkdir -p results
mkdir -p slurm

REPEATS=${1:-10}

#19
for node in 03 20
do
    mkdir -p results/hendrixgpu${node}fl
    mkdir -p slurm/hendrixgpu${node}fl
    for gpus in {1,2,4,8}
    do
        mkdir -p results/hendrixgpu${node}fl/${gpus}
        mkdir -p slurm/hendrixgpu${node}fl/${gpus}
        #sbatch -o "slurm/map_${i}_%j.out" -p gpu ./benchmark_map.sh $REPEATS $i
        #sbatch -o "slurm/reduce_${i}_%j.out" -p gpu ./benchmark_reduce.sh $REPEATS $i
        for i in "1024 2GFLOPS" \
                 "2048 17GFLOPS" \
                 "3072 77GFLOPS" \
                 "4096 137GFLOPS" \
                 "5120 268GFLOPS" \
                 "6144 463GFLOPS" \
                 "7168 736GFLOPS" \
                 "8192 1099GFLOPS"
        do
            set -- $i
            sbatch -o "slurm/hendrixgpu${node}fl/${gpus}/matmul_${i}.out" -p gpu --ntasks=1 --cpus-per-task=8 --mem=8G --gres=gpu:${gpus} -w hendrixgpu${node}fl --time=2-00:00:00 ./benchmark_matmul.sh $REPEATS ${gpus} $1 $2 ${node}
        done
    done
done

#sbatch -J test -o "slurm/sanity_check.out" -p gpu --ntasks=1 --cpus-per-task=8 --mem=8G --gres=gpu:2 -w hendrixgpu03fl --time=2-00:00:00 ./benchmark_sanity_check.sh