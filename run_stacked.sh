#!/bin/bash
# Stacked training handoff, unattended.
#
#   stage 1 (already running): human-labelled selection data teaches general
#           move patterns
#   stage 2 (this script):      continue those same adapters on Stockfish-labelled
#           data, which has no human blunders and sharper targets
#
# Waits for stage 1 to reach STAGE1_ITERS, stops it, preserves the stage-1
# adapters for comparison, then starts stage 2 from that checkpoint.

set -u
cd "$(dirname "$0")"

STAGE1_ITERS=${STAGE1_ITERS:-6000}
STAGE2_ITERS=${STAGE2_ITERS:-4000}
CKPT="adapters/selection/$(printf '%07d' "$STAGE1_ITERS")_adapters.safetensors"

echo "[stacked] waiting for stage-1 checkpoint: $CKPT"
while [ ! -f "$CKPT" ]; do sleep 60; done
echo "[stacked] stage-1 checkpoint reached at $(date)"

echo "[stacked] waiting for Stockfish-labelled data"
while [ ! -f data/train_sfsel.jsonl ]; do sleep 60; done
echo "[stacked] Stockfish data ready at $(date)"

echo "[stacked] stopping stage-1 training"
pkill -f "mlx_lm lora" 2>/dev/null
pkill -f "src/train.py" 2>/dev/null
sleep 10

# Preserve stage-1 adapters so the two stages can be compared.
rm -rf adapters/selection_human
cp -r adapters/selection adapters/selection_human
echo "[stacked] stage-1 adapters preserved at adapters/selection_human"

echo "[stacked] starting stage 2 from $CKPT at $(date)"
uv run python src/train.py \
  --model models/Qwen2.5-0.5B \
  --data-suffix _sfsel \
  --adapter-name selection_sf \
  --resume "$CKPT" \
  --mask-prompt \
  --lora-rank 32 --lora-layers -1 --learning-rate 5e-5 \
  --iters "$STAGE2_ITERS" --save-every 1000 --steps-per-eval 1000

echo "[stacked] stage 2 finished at $(date)"
