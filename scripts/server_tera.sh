#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
stamp="$(date +%Y%m%d_%H%M%S)_$$"
python -m egc.tera_server run \
  --model /mnt/yanghui/models/Qwen/Qwen2.5-7B \
  --archive data/learned_v2_screened/experiment.zip \
  --output "runs/tera_qwen25_7b_seed42_${stamp}" \
  --seed 42 --epochs 3 --max-length 8192 "$@"
