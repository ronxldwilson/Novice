"""Prepare Stockfish-annotated chess training data with checkpoint/resume support.

Instead of raw move sequences, each training example includes:
- The position (move history)
- Stockfish evaluation of the position
- Top candidate moves with their evaluations
- The best move with a short tactical tag
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine

from utils import (
    download_games,
    fmt_num,
    fmt_time,
    load_checkpoint,
    log,
    progress_bar,
    save_jsonl,
)

DATA_DIR = Path(__file__).parent.parent / "data"
LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
MIN_ELO = 2000
MAX_GAMES = 100_000
TRAIN_SPLIT = 0.95
STOCKFISH_PATH = "stockfish"
STOCKFISH_DEPTH = 12
NUM_CANDIDATES = 3
POSITIONS_PER_GAME = 8
ANNOTATION_CHUNK_SIZE = 500


def format_eval(score: chess.engine.PovScore, board: chess.Board) -> str:
    white_score = score.white()
    if white_score.is_mate():
        mate_in = white_score.mate()
        return f"M{mate_in}" if mate_in > 0 else f"M{mate_in}"
    cp = white_score.score()
    return f"{cp / 100:+.1f}"


def classify_position(eval_before: float | None, eval_after: float | None, is_white: bool) -> str:
    if eval_before is None or eval_after is None:
        return ""
    delta = eval_after - eval_before
    if not is_white:
        delta = -delta
    if delta < -2.0:
        return "blunder"
    elif delta < -0.8:
        return "mistake"
    elif delta < -0.3:
        return "inaccuracy"
    elif delta > 0.3:
        return "good"
    elif delta > 1.0:
        return "excellent"
    return "book"


def get_eval_score_numeric(score: chess.engine.PovScore) -> float | None:
    white_score = score.white()
    if white_score.is_mate():
        return 100.0 if white_score.mate() > 0 else -100.0
    cp = white_score.score()
    return cp / 100.0 if cp is not None else None


def move_history_to_san(board: chess.Board) -> str:
    temp = board.copy()
    moves = list(temp.move_stack)
    temp.reset()
    parts = []
    for i, move in enumerate(moves):
        san = temp.san(move)
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.{san}")
        else:
            parts.append(san)
        temp.push(move)
    return " ".join(parts)


def annotate_game(game: dict, engine: chess.engine.SimpleEngine, depth: int, num_candidates: int, positions_per_game: int) -> list[dict]:
    moves = game["moves"]
    board = chess.Board()
    examples = []

    total_positions = len(moves) - 1
    if total_positions <= positions_per_game:
        selected = list(range(1, len(moves)))
    else:
        selected = sorted(random.sample(range(1, len(moves)), positions_per_game))

    prev_eval = None

    for i in range(len(moves)):
        board.push(board.parse_san(moves[i]))

        if i not in selected:
            if i > 0:
                try:
                    info = engine.analyse(board, chess.engine.Limit(depth=max(6, depth // 2)))
                    prev_eval = get_eval_score_numeric(info["score"])
                except Exception:
                    prev_eval = None
            continue

        if board.is_game_over():
            break

        try:
            analysis = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=num_candidates)
        except Exception:
            continue

        if not analysis:
            continue

        current_eval = get_eval_score_numeric(analysis[0]["score"])
        is_white = board.turn == chess.WHITE

        position_eval = format_eval(analysis[0]["score"], board)
        candidates = []
        for info in analysis:
            pv = info.get("pv", [])
            if pv:
                candidates.append(f"{board.san(pv[0])} ({format_eval(info['score'], board)})")

        best_move_san = board.san(analysis[0]["pv"][0]) if analysis[0].get("pv") else None
        if not best_move_san:
            continue

        tag = classify_position(prev_eval, current_eval, not is_white)
        history = move_history_to_san(board)
        side = "White" if is_white else "Black"

        if tag:
            prompt = f"[Position] {history}\n[Eval] {position_eval}\n[Side] {side} to move\n[Context] {tag}\n[Candidates] {', '.join(candidates)}\n[Best]"
        else:
            prompt = f"[Position] {history}\n[Eval] {position_eval}\n[Side] {side} to move\n[Candidates] {', '.join(candidates)}\n[Best]"

        examples.append({"text": f"{prompt} {best_move_san}"})
        prev_eval = current_eval

    return examples


def annotate_with_checkpoints(
    games: list[dict],
    checkpoint_path: Path,
    depth: int,
    num_candidates: int,
    positions_per_game: int,
    chunk_size: int,
) -> list[dict]:
    """Annotate games with Stockfish, checkpointing every chunk_size games."""
    existing = load_checkpoint(checkpoint_path)
    already_done = len(existing)

    if already_done >= len(games):
        log(f"Annotation checkpoint complete: {fmt_num(already_done)} examples already saved")
        return existing

    if already_done > 0:
        log(f"Resuming annotation: {fmt_num(already_done)} examples from previous run")
        games_done = 0
        count = 0
        for game in games:
            n = min(positions_per_game, len(game["moves"]) - 1)
            count += n
            games_done += 1
            if count >= already_done:
                break
        log(f"Skipping ~{fmt_num(games_done)} already-annotated games")
    else:
        games_done = 0

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 64})

    chunk_buffer = []
    total_new = 0
    start = time.time()

    for i in range(games_done, len(games)):
        examples = annotate_game(games[i], engine, depth, num_candidates, positions_per_game)
        chunk_buffer.extend(examples)
        total_new += len(examples)

        if (i - games_done + 1) % 100 == 0:
            elapsed = time.time() - start
            done_in_run = i - games_done + 1
            rate = done_in_run / elapsed
            remaining = len(games) - i - 1
            log(
                f"  Annotating: {progress_bar(i + 1, len(games))}  "
                f"{i + 1}/{len(games)} games  |  "
                f"{fmt_num(already_done + total_new)} total examples  |  "
                f"{rate:.1f} games/s  |  "
                f"ETA: {fmt_time(remaining / rate) if rate > 0 else '...'}"
            )

        if len(chunk_buffer) >= chunk_size:
            with open(checkpoint_path, "a") as f:
                for ex in chunk_buffer:
                    f.write(json.dumps(ex) + "\n")
            log(f"  Checkpoint: +{len(chunk_buffer)} examples saved (total: {fmt_num(already_done + total_new)})")
            chunk_buffer = []

    if chunk_buffer:
        with open(checkpoint_path, "a") as f:
            for ex in chunk_buffer:
                f.write(json.dumps(ex) + "\n")
        log(f"  Final checkpoint: +{len(chunk_buffer)} examples (total: {fmt_num(already_done + total_new)})")

    engine.quit()
    return load_checkpoint(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare Stockfish-annotated chess training data")
    parser.add_argument("--url", default=LICHESS_DB_URL)
    parser.add_argument("--min-elo", type=int, default=MIN_ELO)
    parser.add_argument("--max-games", type=int, default=MAX_GAMES)
    parser.add_argument("--depth", type=int, default=STOCKFISH_DEPTH)
    parser.add_argument("--candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--positions-per-game", type=int, default=POSITIONS_PER_GAME)
    parser.add_argument("--chunk-size", type=int, default=ANNOTATION_CHUNK_SIZE, help="Examples per annotation checkpoint")
    parser.add_argument("--download-chunk-size", type=int, default=5000, help="Games per download checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cached-games", action="store_true", help="Use games from previous download")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    log("NOVICE — STOCKFISH-ANNOTATED DATA PREPARATION")
    print("=" * 60, flush=True)

    log(f"Checking Stockfish at '{STOCKFISH_PATH}'...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        log(f"Stockfish OK: {engine.id.get('name', 'unknown')}")
        engine.quit()
    except Exception as e:
        log(f"ERROR: Cannot start Stockfish: {e}")
        log("Install with: brew install stockfish")
        sys.exit(1)

    log(f"Settings: depth={args.depth}, candidates={args.candidates}, positions/game={args.positions_per_game}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    games_checkpoint = DATA_DIR / "games_checkpoint.jsonl"
    annotation_checkpoint = DATA_DIR / "annotations_checkpoint.jsonl"

    # Step 1: Get games (with checkpointing)
    print("=" * 60, flush=True)
    log("STEP 1: ACQUIRE GAMES")
    print("─" * 60, flush=True)

    if args.use_cached_games and games_checkpoint.exists():
        games = load_checkpoint(games_checkpoint)
        log(f"Using cached games: {fmt_num(len(games))}")
        if len(games) > args.max_games:
            games = games[:args.max_games]
            log(f"Trimmed to {fmt_num(len(games))} games")
    else:
        games = download_games(
            args.url, args.max_games, args.min_elo,
            checkpoint_path=games_checkpoint,
            chunk_size=args.download_chunk_size,
        )

    # Step 2: Annotate with Stockfish (with checkpointing)
    print("=" * 60, flush=True)
    log("STEP 2: STOCKFISH ANNOTATION")
    log(f"Annotating {fmt_num(len(games))} games ({args.positions_per_game} positions each, depth {args.depth})")
    log(f"Estimated examples: ~{fmt_num(len(games) * args.positions_per_game)}")
    log(f"Checkpoint: saving every {args.chunk_size} examples to {annotation_checkpoint.name}")
    print("─" * 60, flush=True)

    random.seed(args.seed)
    start = time.time()
    all_examples = annotate_with_checkpoints(
        games, annotation_checkpoint,
        args.depth, args.candidates, args.positions_per_game,
        args.chunk_size,
    )
    elapsed = time.time() - start

    log(f"Annotation complete: {fmt_num(len(all_examples))} examples in {fmt_time(elapsed)}")

    # Step 3: Shuffle & split
    print("=" * 60, flush=True)
    log("STEP 3: SHUFFLE & SPLIT")
    print("─" * 60, flush=True)

    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * TRAIN_SPLIT)
    train = all_examples[:split_idx]
    valid = all_examples[split_idx:]

    save_jsonl(train, DATA_DIR / "train_annotated.jsonl")
    save_jsonl(valid, DATA_DIR / "valid_annotated.jsonl")

    # Sample
    print("=" * 60, flush=True)
    log("SAMPLE TRAINING EXAMPLE:")
    print("─" * 60, flush=True)
    if all_examples:
        print(random.choice(all_examples)["text"], flush=True)
    print("─" * 60, flush=True)

    # Summary
    print("=" * 60, flush=True)
    log("ALL DONE!")
    log(f"  Games:      {fmt_num(len(games))}")
    log(f"  Train:      {fmt_num(len(train))} examples")
    log(f"  Validation: {fmt_num(len(valid))} examples")
    log(f"  Files:      data/train_annotated.jsonl, data/valid_annotated.jsonl")
    print("=" * 60, flush=True)
    log("Next step: uv run python src/train.py --annotated")


if __name__ == "__main__":
    main()
