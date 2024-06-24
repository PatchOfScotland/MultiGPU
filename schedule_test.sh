#!/bin/bash

mkdir -p slurm

# 11 - not available
# 19 - possibly not available either?
for i in 03 05 12 13 19 20
do
    echo "submitting ${i}"
    sbatch -o "slurm/test_${i}.out" -p gpu --ntasks=1 --cpus-per-task=1 --mem=8G --gres=gpu:8 -w hendrixgpu${i}fl --time=1-00:00:00 ./slurm_test_job.sh $i
done
