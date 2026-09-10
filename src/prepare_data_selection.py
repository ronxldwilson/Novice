"""Prepare move-selection training data.

The model is shown the board and every legal move, and picks one — like a chess
UI highlighting available squares. Legal moves come from `python-chess`, so no
engine is needed at inference time.

Design decisions that matter for a 0.5B model:

* **FEN in the prompt.** Without it the model must mentally replay the whole game
  to know where the pieces are. FEN states the board directly for ~30 tokens.
* **prompt/completion format.** Combined with `--mask-prompt` at training time,
  loss is computed only on the chosen move rather than spread across the legal
  move list. Nearly all gradient goes into the actual decision.
* **Length filtering.** Truncation cuts the *end* of a sequence, which is where
  the label lives. Over-long examples are dropped rather than silently corrupted.

Label sources:
  --label played     (default) the move a 2000+ Elo player actually made. Free.
  --label stockfish  Stockfish's best move at --depth. Better labels, ~0.25s/position.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine

from utils import build_selection_prompt, count_lines, fmt_num, fmt_time, log, progress_bar

DATA_DIR = Path(__file__).parent.parent / "data"
STOCKFISH_PATH = "stockfish"
STOCKFISH_DEPTH = 12
POSITIONS_PER_GAME = 8
HISTORY_PLIES = 12
TRAIN_SPLIT = 0.97
FLUSH_EVERY = 20000

# Rough token budget. Measured against the Qwen tokenizer: ~3.6 chars/token for
# this content, plus ~25 tokens of chat-template overhead.
MAX_PROMPT_CHARS = 1500


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
    rng: random.Random,
) -> list[dict]:
    """Replay a game and emit selection examples at sampled positions."""
    moves = game["moves"]
    total = len(moves)
    if total < 4:
        return []

    if total <= positions_per_game:
        selected = set(range(total))
    else:
        selected = set(rng.sample(range(total), positions_per_game))

    board = chess.Board()
    examples = []
    history: list[str] = []

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
                    shuffled = legal[:]
                    rng.shuffle(shuffled)
                    prompt = build_selection_prompt(board, history[-HISTORY_PLIES:], shuffled)
                    if len(prompt) <= MAX_PROMPT_CHARS:
                        examples.append({"prompt": prompt, "completion": label})

        try:
            board.push(board.parse_san(played_san))
        except ValueError:
            break
        history.append(played_san)

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare move-selection training data")
    parser.add_argument("--max-games", type=int, default=100_000)
    parser.add_argument("--label", choices=["played", "stockfish"], default="played")
    parser.add_argument("--depth", type=int, default=STOCKFISH_DEPTH)
    parser.add_argument("--positions-per-game", type=int, default=POSITIONS_PER_GAME)
    parser.add_argument("--out-prefix", default="selection")
    parser.add_argument("--max-examples", type=int, default=400_000,
                        help="Cap total examples to keep training inside RAM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    log("NOVICE — MOVE-SELECTION DATA (v2: FEN + masked completion)")
    print("=" * 60, flush=True)

    games_path = DATA_DIR / "games_checkpoint.jsonl"
    if not games_path.exists():
        log("ERROR: No games found. Run prepare_data.py first.")
        sys.exit(1)

    total_games = min(args.max_games, count_lines(games_path))
    log(f"Source:   {fmt_num(total_games)} games")
    log(f"Label:    {args.label}")
    log(f"Sampling: {args.positions_per_game} positions/game")
    log(f"Cap:      {fmt_num(args.max_examples)} examples")

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

    rng = random.Random(args.seed)
    raw_path = DATA_DIR / f"{args.out_prefix}_raw.jsonl"
    raw_path.unlink(missing_ok=True)

    start = time.time()
    written = 0
    buffer = []
    stop = False

    with open(raw_path, "a") as out:
        for i, game in enumerate(stream_games(games_path, total_games)):
            buffer.extend(process_game(game, args.positions_per_game, engine, args.depth, rng))

            if len(buffer) >= FLUSH_EVERY:
                for ex in buffer:
                    out.write(json.dumps(ex) + "\n")
                written += len(buffer)
                buffer = []
                if written >= args.max_examples:
                    log(f"  Reached example cap at game {fmt_num(i + 1)}")
                    stop = True

            if (i + 1) % 10_000 == 0 or stop:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                log(
                    f"  {progress_bar(written, args.max_examples)}  "
                    f"{fmt_num(i + 1)} games  |  "
                    f"{fmt_num(written + len(buffer))} examples  |  "
                    f"{rate:.0f} games/s"
                )
            if stop:
                break

        for ex in buffer:
            out.write(json.dumps(ex) + "\n")
        written += len(buffer)

    if engine:
        engine.quit()

    log(f"Generated {fmt_num(written)} examples in {fmt_time(time.time() - start)}")

    # Shuffle line indices only, then stream-split, so memory stays flat.
    print("─" * 60, flush=True)
    log("Shuffling and splitting...")
    indices = list(range(written))
    rng.shuffle(indices)
    split = int(written * TRAIN_SPLIT)
    train_idx = set(indices[:split])

    train_path = DATA_DIR / f"train_{args.out_prefix}.jsonl"
    valid_path = DATA_DIR / f"valid_{args.out_prefix}.jsonl"

    first = None
    with open(raw_path) as inp, open(train_path, "w") as tf, open(valid_path, "w") as vf:
        for i, line in enumerate(inp):
            (tf if i in train_idx else vf).write(line)
            if i == 0:
                first = json.loads(line)

    raw_path.unlink(missing_ok=True)

    print("─" * 60, flush=True)
    log("SAMPLE EXAMPLE:")
    print("─" * 60, flush=True)
    print(first["prompt"], flush=True)
    print(f"--> completion: {first['completion']}", flush=True)

    print("=" * 60, flush=True)
    log("ALL DONE!")
    log(f"  Train: {fmt_num(split)} examples ({train_path.stat().st_size / 1e6:.0f} MB)")
    log(f"  Valid: {fmt_num(written - split)} examples")
    print("=" * 60, flush=True)
    log(f"Next: uv run python src/train.py --selection --mask-prompt")


if __name__ == "__main__":
    main()
