#!/bin/bash

node=${1:-03}
gpus=${2:-8}
repeats=${3:-10}
dim=${4:-8192}
flops=$(((${dim}*${dim}*${dim}*2)/1000000000))GFLOPS
build_dir=build/hendrixgpu${node}fl

mkdir -p results
mkdir -p slurm
mkdir -p results/hendrixgpu${node}fl
mkdir -p slurm/hendrixgpu${node}fl
mkdir -p results/hendrixgpu${node}fl/${gpus}
mkdir -p slurm/hendrixgpu${node}fl/${gpus}

if [ $node = "-h" ];
then
    echo "Args are: node[03] gpus[8] repeats[10] dim[8192]"
fi

echo "Running command:"
echo sbatch -o "slurm/hendrixgpu${node}fl/${gpus}/matmul_${i}.out" -p gpu --ntasks=1 --cpus-per-task=8 --mem=8G --gres=gpu:${gpus} -w hendrixgpu${node}fl --time=2-00:00:00 ./benchmark_matmul.sh $repeats ${gpus} $dim $flops ${node} ${build_dir}
sbatch -o "slurm/hendrixgpu${node}fl/${gpus}/matmul_${i}.out" -p gpu --ntasks=1 --cpus-per-task=8 --mem=8G --gres=gpu:${gpus} -w hendrixgpu${node}fl --time=2-00:00:00 ./benchmark_matmul.sh $repeats ${gpus} $dim $flops ${node} ${build_dir}
