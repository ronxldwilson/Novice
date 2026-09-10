"""Prepare training data in move-selection format.

The model receives ALL legal moves and learns to select the best one — like a
chess UI showing which moves are available. At inference time python-chess
provides the legal moves, so no engine is needed as a crutch.

Training format:
    [Position] 1.e4 e5 2.Nf3 Nc6
    [Side] White to move
    [Legal] d3, Bc4, Nc3, Bb5, d4, Be2, ...
    [Best] Bb5

Two label sources:
  --label played     (default) the move actually played by a 2000+ Elo player.
                     Free — no engine calls, so millions of examples in minutes.
  --label stockfish  Stockfish's best move at --depth. Higher quality labels but
                     ~0.25s per position.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine

from utils import count_lines, fmt_num, fmt_time, log, progress_bar

DATA_DIR = Path(__file__).parent.parent / "data"
STOCKFISH_PATH = "stockfish"
STOCKFISH_DEPTH = 12
POSITIONS_PER_GAME = 10
TRAIN_SPLIT = 0.95
CHECKPOINT_SIZE = 20000


def move_history_to_san(moves: list[str], upto: int) -> str:
    parts = []
    for i in range(upto):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.{moves[i]}")
        else:
            parts.append(moves[i])
    return " ".join(parts)


def stream_games(path: Path, max_games: int):
    count = 0
    with open(path) as f:
        for line in f:
            if count >= max_games:
                break
            line = line.strip()
            if line:
                yield json.loads(line)
                count += 1


def process_game(
    game: dict,
    positions_per_game: int,
    engine: chess.engine.SimpleEngine | None,
    depth: int,
) -> list[dict]:
    """Replay a game and emit move-selection examples at sampled positions."""
    moves = game["moves"]
    total = len(moves)
    if total < 2:
        return []

    if total <= positions_per_game:
        selected = set(range(total))
    else:
        selected = set(random.sample(range(total), positions_per_game))

    board = chess.Board()
    examples = []

    for i, played_san in enumerate(moves):
        if i in selected and not board.is_game_over():
            legal = [board.san(m) for m in board.legal_moves]

            if len(legal) >= 2:
                if engine is not None:
                    try:
                        info = engine.analyse(board, chess.engine.Limit(depth=depth))
                        pv = info.get("pv", [])
                        label = board.san(pv[0]) if pv else None
                    except Exception:
                        label = None
                else:
                    label = played_san

                if label in legal:
                    random.shuffle(legal)
                    history = move_history_to_san(moves, i)
                    side = "White" if board.turn == chess.WHITE else "Black"
                    examples.append({
                        "text": (
                            f"[Position] {history}\n"
                            f"[Side] {side} to move\n"
                            f"[Legal] {', '.join(legal)}\n"
                            f"[Best] {label}"
                        )
                    })

        try:
            board.push(board.parse_san(played_san))
        except ValueError:
            break

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare move-selection training data")
    parser.add_argument("--max-games", type=int, default=100_000)
    parser.add_argument("--label", choices=["played", "stockfish"], default="played")
    parser.add_argument("--depth", type=int, default=STOCKFISH_DEPTH)
    parser.add_argument("--positions-per-game", type=int, default=POSITIONS_PER_GAME)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    log("NOVICE — MOVE-SELECTION DATA PREPARATION")
    print("=" * 60, flush=True)

    games_path = DATA_DIR / "games_checkpoint.jsonl"
    if not games_path.exists():
        log("ERROR: No games found. Run prepare_data.py first.")
        sys.exit(1)

    total_games = min(args.max_games, count_lines(games_path))
    log(f"Source:   {fmt_num(total_games)} games")
    log(f"Label:    {args.label}" + (f" (depth {args.depth})" if args.label == "stockfish" else " (move played in game)"))
    log(f"Sampling: {args.positions_per_game} positions/game")
    log(f"Estimate: ~{fmt_num(total_games * args.positions_per_game)} examples")

    engine = None
    if args.label == "stockfish":
        try:
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            engine.configure({"Threads": 1, "Hash": 64})
            log(f"Stockfish OK: {engine.id.get('name', 'unknown')}")
        except Exception as e:
            log(f"ERROR: cannot start Stockfish: {e}")
            sys.exit(1)

    print("─" * 60, flush=True)

    random.seed(args.seed)
    checkpoint_path = DATA_DIR / "selection_checkpoint.jsonl"
    checkpoint_path.unlink(missing_ok=True)

    start = time.time()
    written = 0
    buffer = []

    with open(checkpoint_path, "a") as out:
        for i, game in enumerate(stream_games(games_path, total_games)):
            buffer.extend(process_game(game, args.positions_per_game, engine, args.depth))

            if len(buffer) >= CHECKPOINT_SIZE:
                for ex in buffer:
                    out.write(json.dumps(ex) + "\n")
                written += len(buffer)
                buffer = []

            if (i + 1) % 10_000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = total_games - i - 1
                log(
                    f"  {progress_bar(i + 1, total_games)}  "
                    f"{fmt_num(i + 1)}/{fmt_num(total_games)} games  |  "
                    f"{fmt_num(written + len(buffer))} examples  |  "
                    f"{rate:.0f} games/s  |  "
                    f"ETA: {fmt_time(remaining / rate) if rate > 0 else '...'}"
                )

        for ex in buffer:
            out.write(json.dumps(ex) + "\n")
        written += len(buffer)

    if engine:
        engine.quit()

    log(f"Generated {fmt_num(written)} examples in {fmt_time(time.time() - start)}")

    # Shuffle line indices only, then stream-split to keep memory flat
    print("─" * 60, flush=True)
    log(f"Shuffling and splitting (seed={args.seed})...")
    indices = list(range(written))
    random.shuffle(indices)
    split = int(written * TRAIN_SPLIT)
    train_idx = set(indices[:split])

    train_path = DATA_DIR / "train_selection.jsonl"
    valid_path = DATA_DIR / "valid_selection.jsonl"

    sample_text = None
    with open(checkpoint_path) as inp, \
         open(train_path, "w") as tf, \
         open(valid_path, "w") as vf:
        for i, line in enumerate(inp):
            (tf if i in train_idx else vf).write(line)
            if i == 0:
                sample_text = json.loads(line)["text"]

    print("─" * 60, flush=True)
    log("SAMPLE EXAMPLE:")
    print("─" * 60, flush=True)
    print(sample_text, flush=True)

    print("=" * 60, flush=True)
    log("ALL DONE!")
    log(f"  Train: {fmt_num(split)} examples")
    log(f"  Valid: {fmt_num(written - split)} examples")
    log(f"  Files: data/train_selection.jsonl, data/valid_selection.jsonl")
    print("=" * 60, flush=True)
    log("Next: uv run python src/train.py --selection")


if __name__ == "__main__":
    main()
