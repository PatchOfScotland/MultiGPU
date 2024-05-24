#!/bin/bash
#SBATCH -p gpu --ntasks=1 --cpus-per-task=32 --mem=64G
#SBATCH --job-name=MultiGPUBenchmarks
#SBATCH -p gpu --gres=gpu:4
#SBATCH --time=2-00:00:00

REPEATS=$1
DEVICES=$2

make clean
mkdir -p build
make

./benchmark_map.sh $REPEATS $DEVICES
./benchmark_reduce.sh $REPEATS $DEVICES
./benchmark_matmul.sh $REPEATS $DEVICES

python3 assemble_csv_and_graphs.py
