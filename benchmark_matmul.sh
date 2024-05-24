#!/bin/bash
#SBATCH -p gpu --ntasks=1 --cpus-per-task=1 --mem=8G
#SBATCH --job-name=BenchMatmul4
#SBATCH -p gpu --gres=gpu:4
#SBATCH --time=1-00:00:00

REPEATS=$1
DEVICES=$2

hostname
echo $CUDA_VISIBLE_DEVICES

## Map
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
    echo "Benchmarking Matrix Multiplication $2 (${DEVICES} devices)"
    ./build/matmul $1 $1 $1 $REPEATS -d ${DEVICES} -r > ./results/matmul_${DEVICES}_${2}.out
    ./build/matmul $1 $1 $1 $REPEATS -d ${DEVICES} -r -s > ./results/matmul-no-repeat_${DEVICES}_${2}.out
done

echo "All tests complete"
