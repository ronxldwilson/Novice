"""Play chess against the fine-tuned model."""

import argparse
from pathlib import Path

import chess
import chess.engine
import mlx_lm
from mlx_lm.sample_utils import make_sampler


MODEL = "Qwen/Qwen2.5-0.5B"
ADAPTERS_BASIC = Path(__file__).parent.parent / "adapters" / "basic"
ADAPTERS_ANNOTATED = Path(__file__).parent.parent / "adapters" / "annotated"
STOCKFISH_PATH = "stockfish"


def board_to_move_history(board: chess.Board) -> str:
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


def get_model_move_basic(model, tokenizer, board: chess.Board, temperature: float) -> tuple[chess.Move, str]:
    """Basic mode: predict next move from move history."""
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
    move_text = response.strip().split()[0].rstrip(".")

    return resolve_move(board, move_text), ""


def get_model_move_annotated(model, tokenizer, board: chess.Board, temperature: float, engine: chess.engine.SimpleEngine | None) -> tuple[chess.Move, str]:
    """Annotated mode: use structured reasoning format with Stockfish eval context."""
    history = board_to_move_history(board)
    side = "White" if board.turn == chess.WHITE else "Black"

    eval_str = "0.0"
    candidates_str = ""

    if engine:
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
    move_text = response.strip().split()[0].rstrip(".")

    reasoning = f"Eval: {eval_str} | Candidates: {candidates_str}"
    return resolve_move(board, move_text), reasoning


def resolve_move(board: chess.Board, move_text: str) -> chess.Move:
    """Try to parse a move, with fuzzy matching and random fallback."""
    try:
        return board.parse_san(move_text)
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        pass

    for legal_move in board.legal_moves:
        san = board.san(legal_move)
        if len(move_text) >= 2 and san.startswith(move_text[:2]):
            print(f"  (fuzzy matched '{move_text}' -> {san})")
            return legal_move

    import random
    fallback = random.choice(list(board.legal_moves))
    print(f"  (invalid '{move_text}', falling back to random: {board.san(fallback)})")
    return fallback


def print_board(board: chess.Board):
    print()
    print(board.unicode(borders=True))
    print()


def main():
    parser = argparse.ArgumentParser(description="Play chess against the fine-tuned model")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--annotated", action="store_true", help="Use annotated (reasoning) model")
    parser.add_argument("--adapter-path", default=None, help="Override adapter path")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--play-as", choices=["white", "black"], default="white")
    parser.add_argument("--show-reasoning", action="store_true", help="Show model's reasoning")
    args = parser.parse_args()

    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
    elif args.annotated:
        adapter_path = ADAPTERS_ANNOTATED
    else:
        adapter_path = ADAPTERS_BASIC

    if adapter_path.exists():
        print(f"Loading model {args.model} with adapters from {adapter_path}")
        model, tokenizer = mlx_lm.load(args.model, adapter_path=str(adapter_path))
    else:
        print(f"No adapters found at {adapter_path}, using base model")
        model, tokenizer = mlx_lm.load(args.model)

    engine = None
    if args.annotated:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            engine.configure({"Threads": 1, "Hash": 64})
            print(f"Stockfish loaded for position evaluation")
        except Exception:
            print("Warning: Stockfish not available, annotated mode will run without eval context")

    board = chess.Board()
    human_is_white = args.play_as == "white"
    mode = "annotated (reasoning)" if args.annotated else "basic"

    print(f"\nChess SLM ({mode}) — type moves in SAN (e.g., e4, Nf3, O-O)")
    print("Commands: 'quit', 'board', 'undo'")
    print(f"You are playing as {'White' if human_is_white else 'Black'}\n")

    def do_model_move():
        if args.annotated:
            move, reasoning = get_model_move_annotated(model, tokenizer, board, args.temperature, engine)
            if args.show_reasoning and reasoning:
                print(f"  Reasoning: {reasoning}")
        else:
            move, _ = get_model_move_basic(model, tokenizer, board, args.temperature)
        print(f"Model plays: {board.san(move)}")
        board.push(move)

    if not human_is_white:
        print("Model is thinking...")
        do_model_move()

    print_board(board)

    while not board.is_game_over():
        user_input = input("Your move: ").strip()

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "board":
            print_board(board)
            continue
        if user_input.lower() == "undo":
            if len(board.move_stack) >= 2:
                board.pop()
                board.pop()
                print_board(board)
            continue

        try:
            move = board.parse_san(user_input)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
            print(f"Invalid move: '{user_input}'. Legal: {', '.join(board.san(m) for m in board.legal_moves)}")
            continue

        board.push(move)
        print_board(board)

        if board.is_game_over():
            break

        print("Model is thinking...")
        do_model_move()
        print_board(board)

    if engine:
        engine.quit()

    print(f"\nGame over: {board.result()}")
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate! {winner} wins.")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")

    print(f"\nFull game: {board_to_move_history(board)}")


if __name__ == "__main__":
    main()
