"""Download and process Lichess games into training data for chess SLM (basic mode)."""

import argparse
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

    # Step 1: Download with checkpointing
    games = download_games(
        args.url, args.max_games, args.min_elo,
        checkpoint_path=checkpoint_path,
        chunk_size=args.chunk_size,
    )

    # Step 2: Convert to training examples
    print("=" * 60, flush=True)
    log("CONVERTING TO TRAINING EXAMPLES")
    print("─" * 60, flush=True)

    all_examples = []
    start = time.time()
    total_games = len(games)

    for i, game in enumerate(games):
        all_examples.extend(game_to_training_examples(game))

        if (i + 1) % 50000 == 0 or i + 1 == total_games:
            elapsed = time.time() - start
            pbar = progress_bar(i + 1, total_games)
            log(
                f"{pbar}  {fmt_num(i + 1)}/{fmt_num(total_games)} games  |  "
                f"{fmt_num(len(all_examples))} examples so far  |  "
                f"elapsed: {fmt_time(elapsed)}"
            )

    avg_examples = len(all_examples) / len(games) if games else 0
    log(f"Generated {fmt_num(len(all_examples))} training examples ({avg_examples:.0f} per game)")

    # Step 3: Shuffle & split
    print("=" * 60, flush=True)
    log("SHUFFLING & SPLITTING DATA")
    print("─" * 60, flush=True)

    random.seed(args.seed)
    log(f"Shuffling {fmt_num(len(all_examples))} examples (seed={args.seed})...")
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * TRAIN_SPLIT)
    train = all_examples[:split_idx]
    valid = all_examples[split_idx:]

    log(f"Split: {TRAIN_SPLIT:.0%} train / {1 - TRAIN_SPLIT:.0%} validation")
    save_jsonl(train, DATA_DIR / "train.jsonl")
    save_jsonl(valid, DATA_DIR / "valid.jsonl")

    # Summary
    print("=" * 60, flush=True)
    log("ALL DONE!")
    log(f"  Games:      {fmt_num(len(games))}")
    log(f"  Train:      {fmt_num(len(train))} examples")
    log(f"  Validation: {fmt_num(len(valid))} examples")
    log(f"  Files:      data/train.jsonl, data/valid.jsonl")
    print("=" * 60, flush=True)
    log("Next step: uv run python src/train.py")


if __name__ == "__main__":
    main()
