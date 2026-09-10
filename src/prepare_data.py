"""Download and process Lichess games into training data for chess SLM."""

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import requests
import zstandard


DATA_DIR = Path(__file__).parent.parent / "data"
LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
MIN_ELO = 2000
MAX_GAMES = 500_000
TRAIN_SPLIT = 0.95


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


def fmt_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def progress_bar(current: int, total: int, width: int = 30) -> str:
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.1%}"


def download_and_stream_pgn(url: str, max_games: int, min_elo: int):
    """Stream PGN games from a zstd-compressed Lichess database."""
    log(f"Connecting to Lichess database...")
    log(f"URL: {url}")
    log(f"Filters: both players >= {min_elo} Elo")
    log(f"Target: {fmt_num(max_games)} games")
    print("─" * 60, flush=True)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    if total_size:
        log(f"Download size: {total_size / (1024**3):.1f} GB (compressed)")
    else:
        log("Download size: unknown (streaming)")

    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(resp.raw)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    games = []
    seen = 0
    kept = 0
    skipped_elo = 0
    skipped_incomplete = 0
    skipped_short = 0
    start_time = time.time()
    last_log_time = start_time

    log("Streaming and filtering games...")

    while kept < max_games:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            log("Reached end of database file.")
            break

        seen += 1
        now = time.time()

        if now - last_log_time >= 5:
            elapsed = now - start_time
            rate = kept / elapsed if elapsed > 0 else 0
            scan_rate = seen / elapsed if elapsed > 0 else 0

            if rate > 0:
                eta = (max_games - kept) / rate
                eta_str = f"ETA: {fmt_time(eta)}"
            else:
                eta_str = "ETA: calculating..."

            pbar = progress_bar(kept, max_games)
            log(
                f"{pbar}  {fmt_num(kept)}/{fmt_num(max_games)} kept  |  "
                f"scanned: {fmt_num(seen)}  |  "
                f"rate: {rate:.0f} games/s  |  "
                f"elapsed: {fmt_time(elapsed)}  |  {eta_str}"
            )
            last_log_time = now

        white_elo = game.headers.get("WhiteElo", "?")
        black_elo = game.headers.get("BlackElo", "?")
        if white_elo == "?" or black_elo == "?":
            skipped_incomplete += 1
            continue
        if int(white_elo) < min_elo or int(black_elo) < min_elo:
            skipped_elo += 1
            continue

        result = game.headers.get("Result", "*")
        if result == "*":
            skipped_incomplete += 1
            continue

        moves = []
        board = game.board()
        for move in game.mainline_moves():
            moves.append(board.san(move))
            board.push(move)

        if len(moves) < 10:
            skipped_short += 1
            continue

        games.append({"moves": moves, "result": result})
        kept += 1

    elapsed = time.time() - start_time
    print("─" * 60, flush=True)
    log(f"Download & filter complete!")
    log(f"  Total scanned:    {fmt_num(seen)}")
    log(f"  Kept:             {fmt_num(kept)}")
    log(f"  Skipped (low Elo):{fmt_num(skipped_elo)}")
    log(f"  Skipped (incomplete): {fmt_num(skipped_incomplete)}")
    log(f"  Skipped (< 10 moves): {fmt_num(skipped_short)}")
    log(f"  Keep rate:        {kept / seen * 100:.1f}%")
    log(f"  Time:             {fmt_time(elapsed)}")
    log(f"  Speed:            {seen / elapsed:.0f} games/s scanned, {kept / elapsed:.0f} games/s kept")

    return games


def game_to_training_examples(game: dict) -> list[dict]:
    """Convert a game into next-move prediction training examples.

    Format: the model sees moves so far and predicts the next move.
    Example prompt: "1.e4 e5 2.Nf3 Nc6 3.Bb5"
    Example completion: " a6"
    """
    examples = []
    moves = game["moves"]

    for i in range(1, len(moves)):
        prompt_parts = []
        for j in range(i):
            move_in_pair = j % 2
            if move_in_pair == 0:
                move_num = j // 2 + 1
                prompt_parts.append(f"{move_num}.{moves[j]}")
            else:
                prompt_parts.append(moves[j])

        prompt = " ".join(prompt_parts)
        completion = f" {moves[i]}"

        examples.append({"text": f"{prompt}{completion}"})

    return examples


def save_jsonl(examples: list[dict], path: Path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log(f"Saved {fmt_num(len(examples))} examples to {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare chess training data")
    parser.add_argument("--url", default=LICHESS_DB_URL, help="Lichess PGN database URL")
    parser.add_argument("--min-elo", type=int, default=MIN_ELO)
    parser.add_argument("--max-games", type=int, default=MAX_GAMES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    log("CHESS SLM DATA PREPARATION")
    print("=" * 60, flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "games.json"

    # Step 1: Download & filter
    if raw_path.exists():
        log(f"Found cached games at {raw_path.name}, skipping download")
        with open(raw_path) as f:
            games = json.load(f)
        log(f"Loaded {fmt_num(len(games))} games from cache")
    else:
        games = download_and_stream_pgn(args.url, args.max_games, args.min_elo)
        log("Saving raw games to cache...")
        with open(raw_path, "w") as f:
            json.dump(games, f)
        log(f"Cached {fmt_num(len(games))} games to {raw_path.name}")

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

    avg_examples = len(all_examples) / len(games)
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
