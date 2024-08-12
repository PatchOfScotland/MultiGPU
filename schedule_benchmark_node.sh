#!/bin/bash

mkdir -p results
mkdir -p slurm

node=$1
REPEATS=${2:-10}

mkdir -p results/hendrixgpu${node}fl
mkdir -p slurm/hendrixgpu${node}fl
build_dir=build/hendrixgpu${node}fl

echo "Compiling on hendrix device: hendrixgpu${node}fl"

mkdir -p build/hendrixgpu${node}fl
srun -o "slurm/compile/hendrixgpu${node}fl.out" -J COM_${node} -p gpu --ntasks=1 --mem=1G -w hendrixgpu${node}fl --time=0-01:00:00 ./slurm_compile.sh ${node}

echo "Compiling complete"

for gpus in {1,2,4,8}
do
    mkdir -p results/hendrixgpu${node}fl/${gpus}
    mkdir -p slurm/hendrixgpu${node}fl/${gpus}
    #sbatch -o "slurm/map_${i}_%j.out" -p gpu ./benchmark_map.sh $REPEATS $i
    #sbatch -o "slurm/reduce_${i}_%j.out" -p gpu ./benchmark_reduce.sh $REPEATS $i
    for i in "2048 17GFLOPS" \
             "4096 137GFLOPS" \
             "6144 463GFLOPS" \
             "8192 1099GFLOPS" \
             "10240 2147GFLOPS" \
             "12288 3710GFLOPS" \
             "14336 5892GFLOPS" \
             "16384 8796GFLOPS"
    do
        set -- $i
        sbatch -J MM_${node}_${gpus}_${1} -o "slurm/hendrixgpu${node}fl/${gpus}/matmul_${i}.out" -p gpu --ntasks=1 --cpus-per-task=8 --mem=8G --gres=gpu:${gpus} -w hendrixgpu${node}fl --time=2-00:00:00 ./benchmark_matmul.sh $REPEATS ${gpus} $1 $2 ${node} ${build_dir}
    done
done
