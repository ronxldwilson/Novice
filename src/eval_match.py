"""Measure move-choice quality on held-out positions.

Playing full games is slow and noisy; this scores the model directly against
what a 2000+ Elo human played, using constrained ranking over the legal moves.

Reported metrics:
  top1    — model's first choice equals the human move
  top3    — human move is in the model's top 3
  mrr     — mean reciprocal rank of the human move
  random  — 1/(legal moves), the score for guessing

top1 is the headline. In selection mode legal% is ~100% by construction and
therefore meaningless as a measure of skill.
"""

import argparse
import json
import random
import time
from pathlib import Path

import chess
import mlx_lm

from selection_infer import score_moves
from utils import fmt_time, log

DATA_DIR = Path(__file__).parent.parent / "data"
LOCAL_MODEL = Path(__file__).parent.parent / "models" / "Qwen2.5-0.5B"


def parse_fen(prompt: str) -> str | None:
    for line in prompt.split("\n"):
        if line.startswith("FEN: "):
            body = line[5:].strip()
            parts = body.split()
            if len(parts) < 4:
                return None
            board_fen, turn, castling, ep = parts[0], parts[1], parts[2], parts[3]
            castling = castling if castling != "-" else "-"
            ep = ep if ep != "-" else "-"
            return f"{board_fen} {turn} {castling} {ep} 0 1"
    return None


def main():
    parser = argparse.ArgumentParser(description="Measure top-1 move match against human play")
    parser.add_argument("--model", default=str(LOCAL_MODEL))
    parser.add_argument("--adapter-path", default="adapters/selection")
    parser.add_argument("--data", default=str(DATA_DIR / "valid_selection.jsonl"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-length-norm", action="store_true")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    log("NOVICE — MOVE MATCH EVALUATION")
    print("=" * 70, flush=True)

    ap = Path(args.adapter_path)
    if ap.exists() and any(ap.glob("*.safetensors")):
        log(f"Loading {args.model} + {ap}")
        model, tokenizer = mlx_lm.load(args.model, adapter_path=str(ap))
    else:
        log(f"Loading {args.model} (no adapters — baseline)")
        model, tokenizer = mlx_lm.load(args.model)

    rows = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]
    log(f"Evaluating {len(rows)} held-out positions")

    top1 = top3 = 0
    rr_sum = 0.0
    rand_sum = 0.0
    used = 0
    start = time.time()

    for i, row in enumerate(rows):
        fen = parse_fen(row["prompt"])
        if not fen:
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if board.is_game_over():
            continue

        target = row["completion"].strip()
        legal = [board.san(m) for m in board.legal_moves]
        if target not in legal or len(legal) < 2:
            continue

        ranked = score_moves(model, tokenizer, board,
                             length_normalize=not args.no_length_norm,
                             rng=rng)
        order = [s for s, _ in ranked]
        rank = order.index(target) + 1

        used += 1
        if rank == 1:
            top1 += 1
        if rank <= 3:
            top3 += 1
        rr_sum += 1.0 / rank
        rand_sum += 1.0 / len(legal)

        if used % 25 == 0:
            el = time.time() - start
            log(f"  {used}/{len(rows)}  top1 {top1/used*100:5.1f}%  "
                f"top3 {top3/used*100:5.1f}%  ({el/used:.2f}s/pos)")

    if used == 0:
        log("No usable positions.")
        return

    print("\n" + "=" * 70, flush=True)
    print(f"  positions      {used}", flush=True)
    print(f"  top1 match     {top1 / used * 100:.1f}%", flush=True)
    print(f"  top3 match     {top3 / used * 100:.1f}%", flush=True)
    print(f"  MRR            {rr_sum / used:.3f}", flush=True)
    print(f"  random top1    {rand_sum / used * 100:.1f}%", flush=True)
    lift = (top1 / used) / (rand_sum / used) if rand_sum else 0
    print(f"  lift vs random {lift:.1f}x", flush=True)
    print(f"  time           {fmt_time(time.time() - start)}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
