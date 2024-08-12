#!/bin/bash

mkdir -p build
mkdir -p slurm
mkdir -p slurm/compile

for node in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22
do
    mkdir -p build/hendrixgpu${node}fl
    sbatch -o "slurm/compile/hendrixgpu${node}fl.out" -J COM_${node} -p gpu --ntasks=1 --mem=1G -w hendrixgpu${node}fl --time=0-01:00:00 ./slurm_compile.sh ${node}
done
