#!/bin/bash
#SBATCH -p gpu --ntasks=1 --cpus-per-task=1 --mem=8G
#SBATCH --job-name=BenchMap4
#SBATCH -p gpu --gres=gpu:4
#SBATCH --time=1-00:00:00

REPEATS=$1
DEVICES=$2

hostname
echo $CUDA_VISIBLE_DEVICES

## Map
module load cuda/12.2
make map

for i in "100000 1GB" \
         "200000 2GB" \
         "400000 4GB" \
         "600000 6GB" \
         "800000 8GB"
do
    set -- $i
    echo "Benchmarking Map $2 (${DEVICES} devices)"
    ./build/map $1 $REPEATS -d ${DEVICES} -r > ./results/map_${DEVICES}_${2}.out
    ./build/map $1 $REPEATS -d ${DEVICES} -r -s > ./results/map-no-repeat_${DEVICES}_${2}.out
done

echo "All tests complete"
