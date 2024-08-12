#!/bin/bash

hostname
module load cuda/12.2

mkdir -p build
mkdir -p build/hendrixgpu${1}fl

make BUILD_DIR=hendrixgpu${1}fl