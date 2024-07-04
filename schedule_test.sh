#!/bin/bash

mkdir -p slurm

# 11 - not available
# 19 - possibly not available either?
for i in 03 05 12 13 20
do
    sbatch -o "slurm/matmul_sm_${i}.out" -J SM_${i} -p gpu --ntasks=1 --cpus-per-task=8 --mem=8G --gres=gpu:4 -w hendrixgpu${i}fl --time=2-00:00:00 ./slurm_test_job.sh $i
done
