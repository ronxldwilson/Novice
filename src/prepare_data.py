"""Download and process Lichess games into training data for chess SLM (basic mode).

Streams games from the checkpoint file and writes training examples directly
to disk to avoid holding everything in memory.
"""

import argparse
import json
import random
import time
from pathlib import Path

from utils import (
    download_games,
    fmt_num,
    fmt_time,
    log,
    progress_bar,
    save_jsonl,
)

DATA_DIR = Path(__file__).parent.parent / "data"
LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
MIN_ELO = 2000
MAX_GAMES = 500_000
TRAIN_SPLIT = 0.95


def game_to_training_examples(game: dict) -> list[dict]:
    """Convert a game into next-move prediction training examples."""
    examples = []
    moves = game["moves"]

    for i in range(1, len(moves)):
        prompt_parts = []
        for j in range(i):
            if j % 2 == 0:
                move_num = j // 2 + 1
                prompt_parts.append(f"{move_num}.{moves[j]}")
            else:
                prompt_parts.append(moves[j])

        prompt = " ".join(prompt_parts)
        completion = f" {moves[i]}"
        examples.append({"text": f"{prompt}{completion}"})

    return examples


def stream_games(checkpoint_path: Path, max_games: int):
    """Yield games one at a time from the checkpoint file."""
    count = 0
    with open(checkpoint_path) as f:
        for line in f:
            if count >= max_games:
                break
            line = line.strip()
            if line:
                yield json.loads(line)
                count += 1


def convert_streaming(checkpoint_path: Path, train_path: Path, valid_path: Path, max_games: int, split: float, seed: int):
    """Convert games to training examples using streaming to keep memory low.

    Writes examples to a temp file, then shuffles by reading line indices
    and splitting into train/valid.
    """
    temp_path = train_path.parent / "examples_temp.jsonl"

    # Pass 1: stream games -> write examples to temp file
    log("Pass 1: Converting games to training examples (streaming)...")
    total_examples = 0
    total_games = 0
    start = time.time()

    with open(temp_path, "w") as out:
        for game in stream_games(checkpoint_path, max_games):
            examples = game_to_training_examples(game)
            for ex in examples:
                out.write(json.dumps(ex) + "\n")
            total_examples += len(examples)
            total_games += 1

            if total_games % 50000 == 0:
                elapsed = time.time() - start
                log(
                    f"  {progress_bar(total_games, max_games)}  "
                    f"{fmt_num(total_games)} games  |  "
                    f"{fmt_num(total_examples)} examples  |  "
                    f"elapsed: {fmt_time(elapsed)}"
                )

    elapsed = time.time() - start
    avg = total_examples / total_games if total_games else 0
    log(f"  Done: {fmt_num(total_examples)} examples from {fmt_num(total_games)} games ({avg:.0f}/game) in {fmt_time(elapsed)}")

    # Pass 2: shuffle line indices and split (only indices in memory, not content)
    log(f"Pass 2: Shuffling {fmt_num(total_examples)} examples (indices only, seed={seed})...")
    random.seed(seed)
    indices = list(range(total_examples))
    random.shuffle(indices)

    split_idx = int(total_examples * split)
    train_indices = set(indices[:split_idx])

    # Pass 3: stream temp file and split into train/valid
    log(f"Pass 3: Writing train ({fmt_num(split_idx)}) and valid ({fmt_num(total_examples - split_idx)}) files...")

    with open(temp_path) as inp, open(train_path, "w") as train_f, open(valid_path, "w") as valid_f:
        for i, line in enumerate(inp):
            if i in train_indices:
                train_f.write(line)
            else:
                valid_f.write(line)

    temp_path.unlink()
    log(f"  Train: {fmt_num(split_idx)}, Valid: {fmt_num(total_examples - split_idx)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare chess training data (basic)")
    parser.add_argument("--url", default=LICHESS_DB_URL)
    parser.add_argument("--min-elo", type=int, default=MIN_ELO)
    parser.add_argument("--max-games", type=int, default=MAX_GAMES)
    parser.add_argument("--chunk-size", type=int, default=5000, help="Games per checkpoint save")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    log("NOVICE — BASIC DATA PREPARATION")
    print("=" * 60, flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = DATA_DIR / "games_checkpoint.jsonl"

    # Step 1: Download with checkpointing (skips if already done)
    download_games(
        args.url, args.max_games, args.min_elo,
        checkpoint_path=checkpoint_path,
        chunk_size=args.chunk_size,
    )

    # Step 2+3: Convert, shuffle, split — all streaming
    print("=" * 60, flush=True)
    log("CONVERTING, SHUFFLING & SPLITTING")
    print("─" * 60, flush=True)

    convert_streaming(
        checkpoint_path,
        DATA_DIR / "train.jsonl",
        DATA_DIR / "valid.jsonl",
        args.max_games,
        TRAIN_SPLIT,
        args.seed,
    )

    # Summary
    print("=" * 60, flush=True)
    log("ALL DONE!")
    log(f"  Files: data/train.jsonl, data/valid.jsonl")
    print("=" * 60, flush=True)
    log("Next step: uv run python src/train.py")


if __name__ == "__main__":
    main()
