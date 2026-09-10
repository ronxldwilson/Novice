"""Prepare Stockfish-annotated chess training data.

Instead of raw move sequences, each training example includes:
- The position (move history)
- Stockfish evaluation of the position
- Top candidate moves with their evaluations
- The best move with a short tactical tag

This teaches the model to reason about positions, not just memorize move patterns.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import requests
import zstandard


DATA_DIR = Path(__file__).parent.parent / "data"
LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
MIN_ELO = 2000
MAX_GAMES = 100_000
TRAIN_SPLIT = 0.95
STOCKFISH_PATH = "stockfish"
STOCKFISH_DEPTH = 12
NUM_CANDIDATES = 3
POSITIONS_PER_GAME = 8


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
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.1%}"


def format_eval(score: chess.engine.PovScore, board: chess.Board) -> str:
    """Format engine score from white's perspective."""
    white_score = score.white()
    if white_score.is_mate():
        mate_in = white_score.mate()
        return f"M{mate_in}" if mate_in > 0 else f"M{mate_in}"
    cp = white_score.score()
    return f"{cp / 100:+.1f}"


def classify_position(eval_before: float | None, eval_after: float | None, is_white: bool) -> str:
    """Classify a move based on eval change."""
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
    """Convert PovScore to a numeric value from white's perspective."""
    white_score = score.white()
    if white_score.is_mate():
        mate_in = white_score.mate()
        return 100.0 if mate_in > 0 else -100.0
    cp = white_score.score()
    return cp / 100.0 if cp is not None else None


def move_history_to_san(board: chess.Board) -> str:
    """Convert board's move history to SAN string."""
    temp = board.copy()
    moves = list(temp.move_stack)
    temp.reset()
    parts = []
    for i, move in enumerate(moves):
        san = temp.san(move)
        if i % 2 == 0:
            move_num = i // 2 + 1
            parts.append(f"{move_num}.{san}")
        else:
            parts.append(san)
        temp.push(move)
    return " ".join(parts)


def download_and_stream_pgn(url: str, max_games: int, min_elo: int):
    """Stream PGN games from a zstd-compressed Lichess database."""
    log(f"Connecting to Lichess database...")
    log(f"Filters: both players >= {min_elo} Elo, target: {fmt_num(max_games)} games")
    print("─" * 60, flush=True)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    if total_size:
        log(f"Download size: {total_size / (1024**3):.1f} GB (compressed)")

    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(resp.raw)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    games = []
    seen = 0
    kept = 0
    start_time = time.time()
    last_log_time = start_time

    log("Streaming and filtering games...")

    while kept < max_games:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break

        seen += 1
        now = time.time()

        if now - last_log_time >= 5:
            elapsed = now - start_time
            rate = kept / elapsed if elapsed > 0 else 0
            eta_str = f"ETA: {fmt_time((max_games - kept) / rate)}" if rate > 0 else "ETA: ..."
            pbar = progress_bar(kept, max_games)
            log(f"{pbar}  {fmt_num(kept)}/{fmt_num(max_games)} kept  |  scanned: {fmt_num(seen)}  |  {rate:.0f}/s  |  {eta_str}")
            last_log_time = now

        white_elo = game.headers.get("WhiteElo", "?")
        black_elo = game.headers.get("BlackElo", "?")
        if white_elo == "?" or black_elo == "?":
            continue
        if int(white_elo) < min_elo or int(black_elo) < min_elo:
            continue

        result = game.headers.get("Result", "*")
        if result == "*":
            continue

        moves = []
        board = game.board()
        for move in game.mainline_moves():
            moves.append(board.san(move))
            board.push(move)

        if len(moves) < 10:
            continue

        games.append({"moves": moves, "result": result})
        kept += 1

    elapsed = time.time() - start_time
    print("─" * 60, flush=True)
    log(f"Download complete: {fmt_num(kept)} games from {fmt_num(seen)} scanned in {fmt_time(elapsed)}")
    return games


def annotate_game(game: dict, engine: chess.engine.SimpleEngine, depth: int, num_candidates: int, positions_per_game: int) -> list[dict]:
    """Annotate selected positions from a game with Stockfish analysis."""
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
            analysis = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=num_candidates,
            )
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
                move_san = board.san(pv[0])
                move_eval = format_eval(info["score"], board)
                candidates.append(f"{move_san} ({move_eval})")

        best_move_san = board.san(analysis[0]["pv"][0]) if analysis[0].get("pv") else None
        if not best_move_san:
            continue

        tag = classify_position(prev_eval, current_eval, not is_white)

        history = move_history_to_san(board)
        side = "White" if is_white else "Black"

        prompt = f"[Position] {history}\n[Eval] {position_eval}\n[Side] {side} to move\n[Candidates] {', '.join(candidates)}\n[Best]"
        completion = f" {best_move_san}"

        if tag:
            prompt_with_ctx = f"[Position] {history}\n[Eval] {position_eval}\n[Side] {side} to move\n[Context] {tag}\n[Candidates] {', '.join(candidates)}\n[Best]"
        else:
            prompt_with_ctx = prompt

        examples.append({"text": f"{prompt_with_ctx}{completion}"})

        prev_eval = current_eval

    return examples


def annotate_games_batch(games: list[dict], depth: int, num_candidates: int, positions_per_game: int) -> list[dict]:
    """Annotate a batch of games with Stockfish."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 64})

    all_examples = []
    start = time.time()

    for i, game in enumerate(games):
        examples = annotate_game(game, engine, depth, num_candidates, positions_per_game)
        all_examples.extend(examples)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            log(
                f"  Annotating: {progress_bar(i + 1, len(games))}  "
                f"{i + 1}/{len(games)} games  |  "
                f"{fmt_num(len(all_examples))} examples  |  "
                f"{rate:.1f} games/s  |  "
                f"ETA: {fmt_time((len(games) - i - 1) / rate)}"
            )

    engine.quit()
    return all_examples


def save_jsonl(examples: list[dict], path: Path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log(f"Saved {fmt_num(len(examples))} examples to {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Stockfish-annotated chess training data")
    parser.add_argument("--url", default=LICHESS_DB_URL)
    parser.add_argument("--min-elo", type=int, default=MIN_ELO)
    parser.add_argument("--max-games", type=int, default=MAX_GAMES)
    parser.add_argument("--depth", type=int, default=STOCKFISH_DEPTH, help="Stockfish search depth")
    parser.add_argument("--candidates", type=int, default=NUM_CANDIDATES, help="Number of candidate moves")
    parser.add_argument("--positions-per-game", type=int, default=POSITIONS_PER_GAME, help="Positions to annotate per game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cached-games", action="store_true", help="Use games.json from previous download")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    log("CHESS SLM — STOCKFISH-ANNOTATED DATA PREPARATION")
    print("=" * 60, flush=True)

    # Verify Stockfish
    log(f"Checking Stockfish at '{STOCKFISH_PATH}'...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        sf_name = engine.id.get("name", "unknown")
        log(f"Stockfish OK: {sf_name}")
        engine.quit()
    except Exception as e:
        log(f"ERROR: Cannot start Stockfish: {e}")
        log("Install with: brew install stockfish")
        sys.exit(1)

    log(f"Settings: depth={args.depth}, candidates={args.candidates}, positions/game={args.positions_per_game}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "games.json"

    # Step 1: Get games
    print("=" * 60, flush=True)
    log("STEP 1: ACQUIRE GAMES")
    print("─" * 60, flush=True)

    if args.use_cached_games and raw_path.exists():
        log(f"Using cached games from {raw_path.name}")
        with open(raw_path) as f:
            games = json.load(f)
        log(f"Loaded {fmt_num(len(games))} games")
        if len(games) > args.max_games:
            games = games[:args.max_games]
            log(f"Trimmed to {fmt_num(len(games))} games")
    else:
        games = download_and_stream_pgn(args.url, args.max_games, args.min_elo)
        with open(raw_path, "w") as f:
            json.dump(games, f)
        log(f"Cached {fmt_num(len(games))} games")

    # Step 2: Annotate with Stockfish
    print("=" * 60, flush=True)
    log("STEP 2: STOCKFISH ANNOTATION")
    log(f"Annotating {fmt_num(len(games))} games ({args.positions_per_game} positions each, depth {args.depth})")
    log(f"Estimated examples: ~{fmt_num(len(games) * args.positions_per_game)}")
    print("─" * 60, flush=True)

    random.seed(args.seed)
    start = time.time()
    all_examples = annotate_games_batch(games, args.depth, args.candidates, args.positions_per_game)
    elapsed = time.time() - start

    log(f"Annotation complete: {fmt_num(len(all_examples))} examples in {fmt_time(elapsed)}")
    log(f"Average: {len(all_examples) / len(games):.1f} examples/game, {len(games) / elapsed:.1f} games/s")

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

    # Show example
    print("=" * 60, flush=True)
    log("SAMPLE TRAINING EXAMPLE:")
    print("─" * 60, flush=True)
    if all_examples:
        sample = random.choice(all_examples)
        print(sample["text"], flush=True)
    print("─" * 60, flush=True)

    # Summary
    print("=" * 60, flush=True)
    log("ALL DONE!")
    log(f"  Games:      {fmt_num(len(games))}")
    log(f"  Train:      {fmt_num(len(train))} examples")
    log(f"  Validation: {fmt_num(len(valid))} examples")
    log(f"  Files:      data/train_annotated.jsonl, data/valid_annotated.jsonl")
    print("=" * 60, flush=True)
    log("Next step: uv run python src/train.py --data-suffix _annotated")


if __name__ == "__main__":
    main()
