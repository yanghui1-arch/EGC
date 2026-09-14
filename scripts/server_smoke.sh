#!/usr/bin/env bash
# Run only on the user's server. Synthetic software check, not benchmark training.
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-4b}"
case "$profile" in
  4b) model=/mnt/yanghui/models/Qwen/Qwen3-4B; mode=full; artifact=model ;;
  8b) model=/mnt/yanghui/models/Qwen/Qwen3-8B; mode=lora; artifact=adapter ;;
  *) echo 'Usage: bash scripts/server_smoke.sh [4b|8b]' >&2; exit 2 ;;
esac
stamp="$(date +%Y%m%d_%H%M%S)_$$"
work="data/smoke_${profile}_${stamp}"
run="runs/smoke_${profile}_${stamp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python scripts/smoke.py --output-dir "$work"
python -m egc train --model "$model" --training-mode "$mode" \
  --train "$work/base/train/sft.jsonl" --dev "$work/base/dev/sft.jsonl" \
  --output "$run" --max-length 1024 --batch-size 1 --grad-accum 1 \
  --max-steps 4 --eval-steps 2 --learning-rate 1e-5
if [[ "$mode" == full ]]; then
  python -m egc infer --model "$run/$artifact" \
    --jobs "$work/base/test/jobs.jsonl" --output "$run/predictions.jsonl" \
    --max-model-len 2048 --max-new-tokens 256 --batch-size 1
else
  python -m egc infer --model "$model" --adapter "$run/$artifact" \
    --jobs "$work/base/test/jobs.jsonl" --output "$run/predictions.jsonl" \
    --max-model-len 2048 --max-new-tokens 256 --batch-size 1
fi
echo "Completed synthetic server check: $run"
echo "Return completion.json, run_manifest.json and predictions.jsonl from that directory."
