"""Evaluate the chess SLM by playing games against Stockfish at various Elo levels."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import chess
import chess.engine
import mlx_lm
from mlx_lm.sample_utils import make_sampler

from utils import selection_prompt_for_board
from selection_infer import best_move

MODEL = "Qwen/Qwen2.5-0.5B"
ADAPTERS_BASIC = Path(__file__).parent.parent / "adapters" / "basic"
ADAPTERS_ANNOTATED = Path(__file__).parent.parent / "adapters" / "annotated"
ADAPTERS_SELECTION = Path(__file__).parent.parent / "adapters" / "selection"
LOCAL_MODEL = Path(__file__).parent.parent / "models" / "Qwen2.5-0.5B"
STOCKFISH_PATH = "stockfish"
RESULTS_DIR = Path(__file__).parent.parent / "results"

ELO_LEVELS = [1320, 1500, 1800, 2000, 2200]
GAMES_PER_LEVEL = 20


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


def log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def board_to_move_history(board: chess.Board) -> str:
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


def get_model_move_basic(model, tokenizer, board: chess.Board, temperature: float) -> tuple[chess.Move, bool]:
    """Basic mode: predict next move. Returns (move, was_legal)."""
    history = board_to_move_history(board)
    if not history:
        history = "1."

    move_num = len(board.move_stack) // 2 + 1
    is_white = board.turn == chess.WHITE
    if is_white and history != "1.":
        prompt = f"{history} {move_num}."
    else:
        prompt = history

    response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=10, sampler=make_sampler(temp=temperature))
    move_text = response.strip().split()[0].rstrip(".") if response.strip() else ""

    try:
        move = board.parse_san(move_text)
        return move, True
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        pass

    for legal_move in board.legal_moves:
        san = board.san(legal_move)
        if len(move_text) >= 2 and san.startswith(move_text[:2]):
            return legal_move, False

    return random.choice(list(board.legal_moves)), False


def get_model_move_selection(model, tokenizer, board: chess.Board, temperature: float) -> tuple[chess.Move, bool]:
    """Selection mode: present all legal moves, model picks the best.

    Uses the same prompt builder and chat template as training, so the model
    sees at test time exactly the format it was trained on.
    """
    prompt, _legal = selection_prompt_for_board(board)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )

    response = mlx_lm.generate(model, tokenizer, prompt=text, max_tokens=8,
                               sampler=make_sampler(temp=temperature))
    move_text = response.strip().split()[0].rstrip(".") if response.strip() else ""

    try:
        return board.parse_san(move_text), True
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        pass

    for legal_move in board.legal_moves:
        if len(move_text) >= 2 and board.san(legal_move).startswith(move_text[:2]):
            return legal_move, False

    return random.choice(list(board.legal_moves)), False


def get_model_move_annotated(model, tokenizer, board: chess.Board, temperature: float, engine: chess.engine.SimpleEngine) -> tuple[chess.Move, bool]:
    """Annotated mode with Stockfish context. Returns (move, was_legal)."""
    history = board_to_move_history(board)
    side = "White" if board.turn == chess.WHITE else "Black"

    eval_str = "0.0"
    candidates_str = ""

    try:
        analysis = engine.analyse(board, chess.engine.Limit(depth=12), multipv=3)
        if analysis:
            score = analysis[0]["score"].white()
            if score.is_mate():
                eval_str = f"M{score.mate()}"
            elif score.score() is not None:
                eval_str = f"{score.score() / 100:+.1f}"

            cands = []
            for info in analysis:
                pv = info.get("pv", [])
                if pv:
                    ms = board.san(pv[0])
                    s = info["score"].white()
                    if s.is_mate():
                        es = f"M{s.mate()}"
                    elif s.score() is not None:
                        es = f"{s.score() / 100:+.1f}"
                    else:
                        es = "?"
                    cands.append(f"{ms} ({es})")
            candidates_str = ", ".join(cands)
    except Exception:
        pass

    prompt = f"[Position] {history}\n[Eval] {eval_str}\n[Side] {side} to move\n[Candidates] {candidates_str}\n[Best]"
    response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=10, sampler=make_sampler(temp=temperature))
    move_text = response.strip().split()[0].rstrip(".") if response.strip() else ""

    try:
        move = board.parse_san(move_text)
        return move, True
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        pass

    for legal_move in board.legal_moves:
        san = board.san(legal_move)
        if len(move_text) >= 2 and san.startswith(move_text[:2]):
            return legal_move, False

    return random.choice(list(board.legal_moves)), False


def play_game(
    model, tokenizer,
    sf_engine: chess.engine.SimpleEngine,
    sf_elo: int,
    model_is_white: bool,
    mode: str,
    eval_engine: chess.engine.SimpleEngine | None,
    temperature: float,
    sf_limit: chess.engine.Limit,
    max_moves: int = 200,
) -> dict:
    """Play one game between the model and Stockfish. Returns game stats."""
    board = chess.Board()
    legal_moves = 0
    total_model_moves = 0
    move_times = []

    while not board.is_game_over() and len(board.move_stack) < max_moves * 2:
        is_model_turn = (board.turn == chess.WHITE) == model_is_white

        if is_model_turn:
            t0 = time.time()
            if mode in ("constrained", "safe"):
                move = best_move(model, tokenizer, board, temperature=0.0,
                                 safe=(mode == "safe"))
                was_legal = True
            elif mode == "selection":
                move, was_legal = get_model_move_selection(model, tokenizer, board, temperature)
            elif mode == "annotated" and eval_engine:
                move, was_legal = get_model_move_annotated(model, tokenizer, board, temperature, eval_engine)
            else:
                move, was_legal = get_model_move_basic(model, tokenizer, board, temperature)
            move_times.append(time.time() - t0)

            total_model_moves += 1
            if was_legal:
                legal_moves += 1
        else:
            move = sf_engine.play(board, sf_limit).move

        board.push(move)

    result = board.result()
    if result == "1-0":
        model_result = "win" if model_is_white else "loss"
    elif result == "0-1":
        model_result = "loss" if model_is_white else "win"
    else:
        model_result = "draw"

    return {
        "sf_elo": sf_elo,
        "model_color": "white" if model_is_white else "black",
        "result": model_result,
        "board_result": result,
        "total_moves": len(board.move_stack),
        "model_moves": total_model_moves,
        "legal_moves": legal_moves,
        "legal_rate": legal_moves / total_model_moves if total_model_moves > 0 else 0,
        "avg_move_time": sum(move_times) / len(move_times) if move_times else 0,
        "pgn": board_to_move_history(board),
        "termination": (
            "checkmate" if board.is_checkmate() else
            "stalemate" if board.is_stalemate() else
            "insufficient" if board.is_insufficient_material() else
            "fifty_move" if board.is_fifty_moves() else
            "repetition" if board.is_repetition() else
            "max_moves" if len(board.move_stack) >= max_moves * 2 else
            "other"
        ),
    }


def estimate_elo(results_by_level: dict[int, list[dict]]) -> float | None:
    """Estimate model Elo from results against known Elo levels.

    Uses the performance rating formula:
    Elo = avg_opponent_elo + 400 * log10(W / L)
    For draws, count as 0.5 win + 0.5 loss.
    """
    total_score = 0
    total_games = 0
    weighted_elo = 0

    for elo, games in results_by_level.items():
        for g in games:
            total_games += 1
            weighted_elo += elo
            if g["result"] == "win":
                total_score += 1.0
            elif g["result"] == "draw":
                total_score += 0.5

    if total_games == 0:
        return None

    avg_opponent = weighted_elo / total_games
    win_pct = total_score / total_games

    if win_pct <= 0:
        return avg_opponent - 400
    if win_pct >= 1:
        return avg_opponent + 400

    return avg_opponent + 400 * math.log10(win_pct / (1 - win_pct))


def print_results_table(results_by_level: dict[int, list[dict]]):
    """Print a formatted results table."""
    print("\n" + "=" * 80, flush=True)
    print(f"{'Elo':>6} | {'Games':>5} | {'W':>3} {'D':>3} {'L':>3} | {'Win%':>6} | {'Legal%':>7} | {'Avg Moves':>9} | {'Avg Time':>8}", flush=True)
    print("-" * 80, flush=True)

    for elo in sorted(results_by_level.keys()):
        games = results_by_level[elo]
        n = len(games)
        wins = sum(1 for g in games if g["result"] == "win")
        draws = sum(1 for g in games if g["result"] == "draw")
        losses = sum(1 for g in games if g["result"] == "loss")
        win_pct = (wins + 0.5 * draws) / n * 100 if n > 0 else 0
        legal_pct = sum(g["legal_rate"] for g in games) / n * 100 if n > 0 else 0
        avg_moves = sum(g["total_moves"] for g in games) / n if n > 0 else 0
        avg_time = sum(g["avg_move_time"] for g in games) / n if n > 0 else 0

        print(
            f"{elo:>6} | {n:>5} | {wins:>3} {draws:>3} {losses:>3} | "
            f"{win_pct:>5.1f}% | {legal_pct:>6.1f}% | {avg_moves:>9.0f} | {avg_time:>7.2f}s",
            flush=True,
        )

    print("=" * 80, flush=True)

    estimated = estimate_elo(results_by_level)
    if estimated is not None:
        print(f"\nEstimated model Elo: {estimated:.0f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess SLM against Stockfish")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--annotated", action="store_true", help="Use annotated model")
    parser.add_argument("--selection", action="store_true", help="Use selection model (legal moves)")
    parser.add_argument("--constrained", action="store_true",
                        help="Score every legal move and take argmax (always legal)")
    parser.add_argument("--safe", action="store_true",
                        help="Constrained ranking plus SEE blunder filter")
    parser.add_argument("--sf-depth", type=int, default=None,
                        help="Cap Stockfish search depth instead of using UCI_Elo. "
                             "Depth 1-3 are genuinely weak opponents.")
    parser.add_argument("--sf-time", type=float, default=0.1,
                        help="Stockfish seconds per move when using UCI_Elo")
    parser.add_argument("--sf-skill", type=int, default=None,
                        help="Stockfish Skill Level 0-20. Weaker than the "
                             "UCI_Elo floor of 1320; 0 plays deliberately badly.")
    parser.add_argument("--sf-nodes", type=int, default=None,
                        help="Cap Stockfish nodes per move")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--elo-levels", type=int, nargs="+", default=ELO_LEVELS)
    parser.add_argument("--games-per-level", type=int, default=GAMES_PER_LEVEL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 80, flush=True)
    log("NOVICE — MODEL EVALUATION")
    print("=" * 80, flush=True)

    # Load model
    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
    elif args.selection or args.constrained or args.safe:
        adapter_path = ADAPTERS_SELECTION
    elif args.annotated:
        adapter_path = ADAPTERS_ANNOTATED
    else:
        adapter_path = ADAPTERS_BASIC

    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else args.model

    if adapter_path.exists():
        log(f"Loading model {model_path} with adapters from {adapter_path}")
        model, tokenizer = mlx_lm.load(model_path, adapter_path=str(adapter_path))
    else:
        log(f"No adapters at {adapter_path}, using base model (pre-training baseline)")
        model, tokenizer = mlx_lm.load(model_path)

    # Start Stockfish for playing
    log("Starting Stockfish opponent...")
    sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Separate engine for annotated mode eval context
    eval_engine = None
    if args.annotated:
        eval_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        eval_engine.configure({"Threads": 1, "Hash": 64})

    random.seed(args.seed)

    mode = ("safe" if args.safe else "constrained" if args.constrained
            else "selection" if args.selection
            else "annotated" if args.annotated else "basic")
    log(f"Mode: {mode}")
    log(f"Elo levels: {args.elo_levels}")
    log(f"Games per level: {args.games_per_level} ({args.games_per_level // 2} as white, {args.games_per_level // 2} as black)")
    log(f"Temperature: {args.temperature}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    total_games = len(args.elo_levels) * args.games_per_level
    games_played = 0
    overall_start = time.time()

    for elo in args.elo_levels:
        print(f"\n{'─' * 80}", flush=True)
        log(f"Playing against Stockfish Elo {elo}")
        print("─" * 80, flush=True)

        if args.sf_skill is not None:
            sf_engine.configure({"UCI_LimitStrength": False, "Skill Level": args.sf_skill,
                                 "Threads": 1, "Hash": 64})
            if args.sf_nodes:
                sf_limit = chess.engine.Limit(nodes=args.sf_nodes)
            elif args.sf_depth:
                sf_limit = chess.engine.Limit(depth=args.sf_depth)
            else:
                sf_limit = chess.engine.Limit(time=args.sf_time)
        elif args.sf_depth:
            sf_engine.configure({"UCI_LimitStrength": False, "Threads": 1, "Hash": 64})
            sf_limit = chess.engine.Limit(depth=args.sf_depth)
        else:
            sf_engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": elo,
                "Threads": 1,
                "Hash": 64,
            })
            sf_limit = chess.engine.Limit(time=args.sf_time)

        level_results = []

        for i in range(args.games_per_level):
            model_is_white = i < args.games_per_level // 2
            games_played += 1

            try:
                game_start = time.time()
                result = play_game(
                    model, tokenizer, sf_engine, elo,
                    model_is_white, mode, eval_engine,
                    args.temperature, sf_limit,
                )
                game_time = time.time() - game_start

                level_results.append(result)

                color = "W" if model_is_white else "B"
                legal_pct = result["legal_rate"] * 100
                elapsed_total = time.time() - overall_start
                eta = elapsed_total / games_played * (total_games - games_played) if games_played > 0 else 0

                log(
                    f"  Game {i + 1}/{args.games_per_level} [{color}]: {result['result']:>4}  "
                    f"({result['termination']}, {result['total_moves']} moves, "
                    f"legal: {legal_pct:.0f}%, {game_time:.1f}s)  "
                    f"[{games_played}/{total_games} total, ETA: {fmt_time(eta)}]"
                )
            except Exception as e:
                color = "W" if model_is_white else "B"
                log(f"  Game {i + 1}/{args.games_per_level} [{color}]: ERROR — {type(e).__name__}: {e}")
                level_results.append({
                    "sf_elo": elo,
                    "model_color": "white" if model_is_white else "black",
                    "result": "loss",
                    "board_result": "*",
                    "total_moves": 0,
                    "model_moves": 0,
                    "legal_moves": 0,
                    "legal_rate": 0,
                    "avg_move_time": 0,
                    "pgn": "",
                    "termination": "error",
                    "error": str(e),
                })

        all_results[elo] = level_results

        # Print running table
        print_results_table(all_results)

    # Cleanup
    sf_engine.quit()
    if eval_engine:
        eval_engine.quit()

    # Save detailed results
    results_file = RESULTS_DIR / f"eval_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Detailed results saved to {results_file}")

    # Final summary
    total_time = time.time() - overall_start
    estimated_elo = estimate_elo(all_results)

    print("\n" + "=" * 80, flush=True)
    log("EVALUATION COMPLETE")
    log(f"  Total games:    {games_played}")
    log(f"  Total time:     {fmt_time(total_time)}")
    if estimated_elo is not None:
        log(f"  Estimated Elo:  {estimated_elo:.0f}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
