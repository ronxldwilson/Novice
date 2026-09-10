"""Probe how much prompt engineering alone helps an untrained model pick legal moves.

Cheap alternative to full-game evaluation: sample positions from real games, ask
the model to choose a move under several prompt formats, and measure

  legal%  — the raw output parses to a legal move (no fuzzy match, no fallback)
  match%  — the output equals the move a 2000+ Elo player actually played

A trained model should beat these numbers by a wide margin; if a prompt format
alone closes the gap, the problem is format rather than knowledge.
"""

import argparse
import json
import random
import time
from pathlib import Path

import chess
import mlx_lm
from mlx_lm.sample_utils import make_sampler

from utils import build_selection_prompt

DATA_DIR = Path(__file__).parent.parent / "data"
LOCAL_MODEL = Path(__file__).parent.parent / "models" / "Qwen2.5-0.5B"
MODEL = "Qwen/Qwen2.5-0.5B"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def move_history(moves: list[str], upto: int) -> str:
    parts = []
    for i in range(upto):
        parts.append(f"{i // 2 + 1}.{moves[i]}" if i % 2 == 0 else moves[i])
    return " ".join(parts)


def sample_positions(n: int, seed: int) -> list[dict]:
    """Pull n random positions from downloaded games."""
    games_path = DATA_DIR / "games_checkpoint.jsonl"
    rng = random.Random(seed)
    positions = []

    with open(games_path) as f:
        for line_no, line in enumerate(f):
            if len(positions) >= n:
                break
            if line_no % 37 != 0:  # spread across the file
                continue
            game = json.loads(line)
            moves = game["moves"]
            if len(moves) < 12:
                continue

            i = rng.randrange(4, min(len(moves), 60))
            board = chess.Board()
            ok = True
            for m in moves[:i]:
                try:
                    board.push(board.parse_san(m))
                except ValueError:
                    ok = False
                    break
            if not ok or board.is_game_over():
                continue

            legal = [board.san(m) for m in board.legal_moves]
            if moves[i] not in legal:
                continue

            positions.append({
                "history": move_history(moves, i),
                "recent": moves[max(0, i - 12):i],
                "side": "White" if board.turn == chess.WHITE else "Black",
                "legal": legal,
                "played": moves[i],
                "fen": board.fen(),
            })

    return positions


# Each builder takes a position dict and returns the prompt string.

def p_bare(p, rng):
    legal = p["legal"][:]
    rng.shuffle(legal)
    return (
        f"[Position] {p['history']}\n"
        f"[Side] {p['side']} to move\n"
        f"[Legal] {', '.join(legal)}\n"
        f"[Best]"
    )


def p_instructed(p, rng):
    legal = p["legal"][:]
    rng.shuffle(legal)
    return (
        "You are a chess grandmaster. Choose the strongest move for the side to "
        "move. You must answer with exactly one move copied from the list of legal "
        "moves, in standard algebraic notation, and nothing else.\n\n"
        f"Position (moves so far): {p['history']}\n"
        f"Side to move: {p['side']}\n"
        f"Legal moves: {', '.join(legal)}\n\n"
        "Best move:"
    )


def p_numbered(p, rng):
    legal = p["legal"][:]
    rng.shuffle(legal)
    listing = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(legal))
    return (
        f"Chess position after: {p['history']}\n"
        f"{p['side']} to move. The legal moves are:\n{listing}\n\n"
        f"The strongest move is:"
    )


def p_fewshot(p, rng):
    legal = p["legal"][:]
    rng.shuffle(legal)
    return (
        "[Position] 1.e4\n[Side] Black to move\n"
        "[Legal] c5, e5, e6, d5, Nf6, c6, g6, d6\n[Best] c5\n\n"
        "[Position] 1.d4 Nf6 2.c4\n[Side] Black to move\n"
        "[Legal] e6, g6, c5, e5, d5, d6, b6, Nc6\n[Best] e6\n\n"
        f"[Position] {p['history']}\n"
        f"[Side] {p['side']} to move\n"
        f"[Legal] {', '.join(legal)}\n[Best]"
    )


def p_continuation(p, rng):
    """No scaffolding at all — just continue the game score."""
    n = len(p["history"].split()) if p["history"] else 0
    move_no = n // 2 + 1
    if p["side"] == "White":
        return f"{p['history']} {move_no}." if p["history"] else "1."
    return p["history"]


def p_v2(p, rng):
    """The trained selection format: FEN + recent history + legal moves, chat-templated."""
    legal = p["legal"][:]
    rng.shuffle(legal)
    board = chess.Board(p["fen"])
    return build_selection_prompt(board, p["recent"], legal)


PROMPTS = {
    "v2_selection": p_v2,
    "bare": p_bare,
    "instructed": p_instructed,
    "numbered": p_numbered,
    "fewshot": p_fewshot,
    "continuation": p_continuation,
}


def first_token(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text.split()[0].strip().rstrip(".,;:").lstrip("*-")


def main():
    parser = argparse.ArgumentParser(description="Probe prompt formats on a model")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--positions", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-samples", type=int, default=3)
    args = parser.parse_args()

    print("=" * 78, flush=True)
    log("NOVICE — PROMPT PROBE")
    print("=" * 78, flush=True)

    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else args.model
    if args.adapter_path and Path(args.adapter_path).exists():
        log(f"Loading {model_path} + adapters {args.adapter_path}")
        model, tokenizer = mlx_lm.load(model_path, adapter_path=args.adapter_path)
    else:
        log(f"Loading {model_path} (no adapters — untrained baseline)")
        model, tokenizer = mlx_lm.load(model_path)

    log(f"Sampling {args.positions} positions from real games...")
    positions = sample_positions(args.positions, args.seed)
    log(f"Got {len(positions)} positions")
    avg_legal = sum(len(p["legal"]) for p in positions) / len(positions)
    log(f"Average legal moves per position: {avg_legal:.1f} "
        f"(random guess ≈ {100 / avg_legal:.1f}% match)")

    sampler = make_sampler(temp=args.temperature)
    results = {}
    samples = {}

    for name, builder in PROMPTS.items():
        rng = random.Random(args.seed)
        legal_hits = 0
        match_hits = 0
        shown = []
        start = time.time()

        for p in positions:
            prompt = builder(p, rng)
            if name == "v2_selection":
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=False,
                )
            out = mlx_lm.generate(model, tokenizer, prompt=prompt,
                                  max_tokens=8, sampler=sampler)
            tok = first_token(out)

            is_legal = tok in p["legal"]
            if is_legal:
                legal_hits += 1
                if tok == p["played"]:
                    match_hits += 1

            if len(shown) < args.show_samples:
                shown.append((tok, p["played"], is_legal))

        n = len(positions)
        results[name] = {
            "legal_pct": legal_hits / n * 100,
            "match_pct": match_hits / n * 100,
            "secs": time.time() - start,
        }
        samples[name] = shown
        r = results[name]
        log(f"  {name:<14} legal {r['legal_pct']:5.1f}%   "
            f"match {r['match_pct']:5.1f}%   ({r['secs']:.0f}s)")

    print("\n" + "=" * 78, flush=True)
    print(f"{'Prompt format':<16} | {'Legal %':>8} | {'Match %':>8} | {'Time':>7}", flush=True)
    print("-" * 78, flush=True)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["legal_pct"]):
        print(f"{name:<16} | {r['legal_pct']:>7.1f}% | {r['match_pct']:>7.1f}% | "
              f"{r['secs']:>6.0f}s", flush=True)
    print("=" * 78, flush=True)
    print(f"\nRandom-guess baseline for match%: ~{100 / avg_legal:.1f}%", flush=True)

    if args.show_samples:
        print("\nSample outputs (model → played, legal?):", flush=True)
        for name, shown in samples.items():
            preview = "  ".join(
                f"{repr(t)}→{pl}{'✓' if ok else '✗'}" for t, pl, ok in shown
            )
            print(f"  {name:<14} {preview}", flush=True)


if __name__ == "__main__":
    main()
