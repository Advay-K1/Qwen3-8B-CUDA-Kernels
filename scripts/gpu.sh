#!/bin/bash
# Run a command inside an already-running SLURM GPU allocation:
#   scripts/gpu.sh <jobid> python tests/test_ops.py
JOB=$1; shift
cd "$(dirname "$0")/.."
srun --jobid="$JOB" --overlap -n1 bash -c "source scripts/env.sh; unset TORCH_CUDA_ARCH_LIST; export TORCH_EXTENSIONS_DIR=\$HOME/.cache/torch_ext_\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .); nvidia-smi -L; $*"
