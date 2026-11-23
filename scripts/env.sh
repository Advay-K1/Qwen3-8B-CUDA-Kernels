# Source this on Delta (login or compute node) to get torch 2.12 + CUDA 13.
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
module load pytorch-conda/2.12 >/dev/null 2>&1
export QWEN3_PATH=${QWEN3_PATH:-$(ls -d /work/hdd/bcjw/zwang33/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;9.0"}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions_qwen3}
