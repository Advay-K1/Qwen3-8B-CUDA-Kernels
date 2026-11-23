#!/bin/bash
#SBATCH -J qwen3-kernels-test
#SBATCH -A bcjw-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 2:00:00
#SBATCH -o logs/test_%j.out
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
unset TORCH_CUDA_ARCH_LIST
python tests/test_ops.py && python tests/test_model.py && python bench/bench.py --decode
