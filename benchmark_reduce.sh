#!/bin/bash
#SBATCH -p gpu --ntasks=1 --cpus-per-task=1 --mem=8G
#SBATCH --job-name=BenchReduce8
#SBATCH -p gpu --gres=gpu:8 -w hendrixgpu03fl
#SBATCH --time=1-00:00:00

REPEATS=$1
DEVICES=$2

hostname
echo $CUDA_VISIBLE_DEVICES

## Reduce
module load cuda/12.2
make hendrix

for i in "1000 8KB" \
         "10000 80KB" \
         "100000 800KB" \
         "1000000 8MB" \
         "10000000 80MB" \
         "100000000 800MB" \
         "1000000000 8GB"
do
    set -- $i
    echo "Benchmarking Reduce $2 (${DEVICES} devices)"
    ./build/reduce $1 $REPEATS -d ${DEVICES} -r > ./results/reduce_${DEVICES}_${2}.out
    ./build/reduce $1 $REPEATS -d ${DEVICES} -r -s > ./results/reduce-no-repeat_${DEVICES}_${2}.out
done

echo "All tests complete"
