#!/bin/bash
# Final evaluation suite.
#
# Reports the model against a stated Stockfish configuration at several
# strengths, and separates pure model play from SEE-filtered play so the
# contribution of the harness is always visible.

set -u
cd "$(dirname "$0")"

ADAPTER=${ADAPTER:-adapters/selection_sf2}
OUT=${OUT:-/tmp/final_eval.log}
: > "$OUT"

say() { echo -e "\n===== $* =====" | tee -a "$OUT"; }

say "MOVE MATCH vs Stockfish target (150 held-out positions)"
uv run python src/eval_match.py --adapter-path "$ADAPTER" \
  --data data/valid_sfsel2.jsonl --n 150 2>&1 \
  | grep -E "positions|top1 match|top3 match|MRR|random top1|lift" | tee -a "$OUT"

say "PURE MODEL (constrained ranking, no SEE) vs Skill 0 / depth 1 - 10 games"
uv run python src/evaluate.py --model models/Qwen2.5-0.5B --constrained \
  --adapter-path "$ADAPTER" --games-per-level 10 --elo-levels 0 \
  --sf-skill 0 --sf-depth 1 2>&1 \
  | grep -E "^ +0 \||Estimated Elo" | tee -a "$OUT"

say "MODEL + SEE FILTER vs Skill 0 / depth 1 - 20 games"
uv run python src/evaluate.py --model models/Qwen2.5-0.5B --safe \
  --adapter-path "$ADAPTER" --games-per-level 20 --elo-levels 0 \
  --sf-skill 0 --sf-depth 1 2>&1 \
  | grep -E "^ +0 \||Estimated Elo" | tee -a "$OUT"

say "MODEL + SEE FILTER vs Skill 0 / 0.1s per move - 10 games"
uv run python src/evaluate.py --model models/Qwen2.5-0.5B --safe \
  --adapter-path "$ADAPTER" --games-per-level 10 --elo-levels 0 \
  --sf-skill 0 2>&1 \
  | grep -E "^ +0 \||Estimated Elo" | tee -a "$OUT"

say "MODEL + SEE FILTER vs Skill 3 / depth 1 - 10 games"
uv run python src/evaluate.py --model models/Qwen2.5-0.5B --safe \
  --adapter-path "$ADAPTER" --games-per-level 10 --elo-levels 3 \
  --sf-skill 3 --sf-depth 1 2>&1 \
  | grep -E "^ +3 \||Estimated Elo" | tee -a "$OUT"

say "DONE $(date +%H:%M)"
