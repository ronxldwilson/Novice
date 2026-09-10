"""Shared utilities for data preparation scripts."""

import io
import json
import random
import time
from pathlib import Path

import chess
import chess.pgn
import requests
import zstandard


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


def save_jsonl(examples: list[dict], path: Path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log(f"Saved {fmt_num(len(examples))} examples to {path.name}")


HISTORY_PLIES = 12


def build_selection_prompt(board: chess.Board, recent: list[str], legal: list[str]) -> str:
    """Render a position as a move-selection prompt.

    Shared by data generation and inference — keep them identical or the model
    sees a different format at test time than it trained on.
    """
    side = "White" if board.turn == chess.WHITE else "Black"
    parts = [
        f"FEN: {board.board_fen()} {'w' if board.turn else 'b'} "
        f"{board.castling_xfen() if board.castling_rights else '-'} "
        f"{chess.SQUARE_NAMES[board.ep_square] if board.ep_square else '-'}",
        f"Move {board.fullmove_number}, {side} to play.",
    ]
    if recent:
        parts.append(f"Recent: {' '.join(recent)}")
    parts.append(f"Legal moves: {', '.join(legal)}")
    parts.append("Choose the strongest move.")
    return "\n".join(parts)


def selection_prompt_for_board(board: chess.Board, rng=None) -> tuple[str, list[str]]:
    """Build a selection prompt from a live board. Returns (prompt, legal_sans)."""
    legal = [board.san(m) for m in board.legal_moves]
    shuffled = legal[:]
    (rng or random).shuffle(shuffled)

    temp = board.copy()
    hist = []
    stack = list(temp.move_stack)
    temp.reset()
    for mv in stack:
        hist.append(temp.san(mv))
        temp.push(mv)

    return build_selection_prompt(board, hist[-HISTORY_PLIES:], shuffled), legal


def count_lines(path: Path) -> int:
    """Count lines in a file without loading it into memory."""
    if not path.exists():
        return 0
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def load_checkpoint(checkpoint_path: Path) -> list[dict]:
    """Load games from a JSONL checkpoint file."""
    games = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
    return games


def append_checkpoint(games: list[dict], checkpoint_path: Path):
    """Append games to a JSONL checkpoint file."""
    with open(checkpoint_path, "a") as f:
        for game in games:
            f.write(json.dumps(game) + "\n")


def download_games(
    url: str,
    max_games: int,
    min_elo: int,
    checkpoint_path: Path,
    chunk_size: int = 5000,
) -> list[dict]:
    """Stream and filter games from Lichess with checkpoint/resume support.

    Saves progress every `chunk_size` games to `checkpoint_path` (JSONL format).
    On restart, resumes from the last checkpoint — but must re-scan from the
    beginning of the database since the PGN stream isn't seekable. Already-saved
    games are loaded instantly; only new games require network.
    """
    existing_count = count_lines(checkpoint_path)
    if existing_count >= max_games:
        log(f"Checkpoint already has {fmt_num(existing_count)} games (target: {fmt_num(max_games)}), skipping download")
        return []

    already_kept = existing_count
    remaining = max_games - already_kept

    if already_kept > 0:
        log(f"Resuming from checkpoint: {fmt_num(already_kept)} games already saved, need {fmt_num(remaining)} more")
    else:
        log(f"Starting fresh download, target: {fmt_num(max_games)} games")

    log(f"Filters: both players >= {min_elo} Elo")
    log(f"Checkpoint: saving every {fmt_num(chunk_size)} games to {checkpoint_path.name}")
    print("─" * 60, flush=True)

    log(f"Connecting to Lichess database...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    if total_size:
        log(f"Download size: {total_size / (1024**3):.1f} GB (compressed)")

    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(resp.raw)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    seen = 0
    kept = 0
    skip_until = already_kept
    chunk_buffer = []
    skipped_elo = 0
    skipped_incomplete = 0
    skipped_short = 0
    start_time = time.time()
    last_log_time = start_time

    if skip_until > 0:
        log(f"Fast-forwarding past first {fmt_num(skip_until)} qualifying games (re-scanning stream)...")

    total_target = max_games

    while (already_kept + kept) < max_games:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            log("Reached end of database file.")
            break

        seen += 1
        now = time.time()

        if now - last_log_time >= 5:
            elapsed = now - start_time
            total_kept = already_kept + kept
            rate = total_kept / elapsed if elapsed > 0 else 0
            if rate > 0:
                eta = (max_games - total_kept) / rate
                eta_str = f"ETA: {fmt_time(eta)}"
            else:
                eta_str = "ETA: calculating..."

            pbar = progress_bar(total_kept, total_target)
            status = f"{pbar}  {fmt_num(total_kept)}/{fmt_num(total_target)} kept  |  scanned: {fmt_num(seen)}  |  {rate:.0f}/s  |  {eta_str}"
            if skip_until > 0:
                status += f"  (fast-fwd: {fmt_num(skip_until)} left)"
            log(status)
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

        if skip_until > 0:
            skip_until -= 1
            continue

        game_data = {"moves": moves, "result": result}
        chunk_buffer.append(game_data)
        kept += 1

        if len(chunk_buffer) >= chunk_size:
            append_checkpoint(chunk_buffer, checkpoint_path)
            log(f"  Checkpoint saved: +{len(chunk_buffer)} games (total: {fmt_num(already_kept + kept)})")
            chunk_buffer = []

    if chunk_buffer:
        append_checkpoint(chunk_buffer, checkpoint_path)
        log(f"  Final checkpoint: +{len(chunk_buffer)} games (total: {fmt_num(already_kept + kept)})")

    elapsed = time.time() - start_time
    total_kept = already_kept + kept
    print("─" * 60, flush=True)
    log(f"Download complete!")
    log(f"  Total scanned:        {fmt_num(seen)}")
    log(f"  New games kept:       {fmt_num(kept)}")
    log(f"  Resumed from:         {fmt_num(already_kept)}")
    log(f"  Total games:          {fmt_num(total_kept)}")
    log(f"  Skipped (low Elo):    {fmt_num(skipped_elo)}")
    log(f"  Skipped (incomplete): {fmt_num(skipped_incomplete)}")
    log(f"  Skipped (< 10 moves): {fmt_num(skipped_short)}")
    log(f"  Time:                 {fmt_time(elapsed)}")

    return []
