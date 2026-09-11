"""Show what the model actually believes about a position.

The model was trained with masked loss on the move token alone, so it has no
chain-of-thought to print — inventing a verbal rationale would be fiction. What
it does have is a probability distribution over every legal move, and that is
its real opinion. This renders it, alongside the static-exchange evaluation, so
you can see where the model and the blunder filter agree or disagree.

Columns:
  P(move)   softmax over the model's length-normalised log-probabilities
  material  SEE: net centipawns after the best enemy reply (0 = nothing hangs)
  flag      how the blunder filter treated the move
"""

import argparse
import math
from pathlib import Path

import chess
import mlx_lm

from selection_infer import score_moves
from tactics import filter_blunders, material_delta

LOCAL_MODEL = Path(__file__).parent.parent / "models" / "Qwen2.5-0.5B"
MODEL = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "Qwen/Qwen2.5-0.5B"
DEFAULT_ADAPTER = Path(__file__).parent.parent / "adapters" / "selection_sf"


def softmax(values: list[float]) -> list[float]:
    hi = max(values)
    exp = [math.exp(v - hi) for v in values]
    total = sum(exp)
    return [e / total for e in exp]


def bar(p: float, width: int = 24) -> str:
    filled = int(round(p * width))
    return "█" * filled + "·" * (width - filled)


def explain(model, tokenizer, board: chess.Board, top: int = 8, use_filter: bool = True):
    ranked = score_moves(model, tokenizer, board)
    if not ranked:
        print("No legal moves — game is over.")
        return None

    probs = softmax([s for _, s in ranked])
    prob_of = {san: p for (san, _), p in zip(ranked, probs)}

    model_choice = ranked[0][0]
    final = filter_blunders(board, ranked) if use_filter else ranked
    final_choice = final[0][0]

    print()
    print(board.unicode(borders=True, empty_square="."))
    side = "White" if board.turn == chess.WHITE else "Black"
    print(f"\n{side} to move — {len(ranked)} legal moves\n")

    print(f"  {'move':<8} {'P(move)':>8}  {'material':>8}  {'':<26}")
    print("  " + "-" * 56)

    for san, _score in ranked[:top]:
        p = prob_of[san]
        try:
            delta = material_delta(board, board.parse_san(san))
        except ValueError:
            delta = 0

        if delta >= 10000:
            mat = "  MATE"
        elif delta == 0:
            mat = "     0"
        else:
            mat = f"{delta:+6d}"

        marks = []
        if san == model_choice:
            marks.append("model's pick")
        if san == final_choice and san != model_choice:
            marks.append("PROMOTED by filter")
        if use_filter and san == model_choice and final_choice != model_choice:
            marks.append("VETOED — hangs material")
        note = ("  <- " + ", ".join(marks)) if marks else ""

        print(f"  {san:<8} {p*100:7.1f}%  {mat}  {bar(p)}{note}")

    print()
    if use_filter and final_choice != model_choice:
        print(f"  Filter overrode the model: {model_choice} -> {final_choice}")
        print(f"  ({model_choice} loses material to the best reply.)")
    else:
        print(f"  Playing {final_choice} — model and filter agree.")
    print()
    return final_choice


def main():
    parser = argparse.ArgumentParser(
        description="Show the model's move distribution for a position")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--fen", default=None, help="Position as FEN")
    parser.add_argument("--moves", default=None,
                        help="Position as SAN moves, e.g. 'e4 e5 Nf3'")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--play", type=int, default=0,
                        help="Continue N more plies, explaining each")
    args = parser.parse_args()

    ap = Path(args.adapter_path)
    if ap.exists() and any(ap.glob("*.safetensors")):
        print(f"Loading {args.model}\n     + adapters {ap}")
        model, tokenizer = mlx_lm.load(args.model, adapter_path=str(ap))
    else:
        print(f"Loading {args.model} (no adapters — untrained baseline)")
        model, tokenizer = mlx_lm.load(args.model)

    board = chess.Board(args.fen) if args.fen else chess.Board()
    if args.moves:
        for tok in args.moves.replace(",", " ").split():
            board.push_san(tok)

    choice = explain(model, tokenizer, board, args.top, not args.no_filter)

    for _ in range(args.play):
        if choice is None or board.is_game_over():
            break
        board.push_san(choice)
        if board.is_game_over():
            print(f"Game over: {board.result()}")
            break
        print("=" * 60)
        choice = explain(model, tokenizer, board, args.top, not args.no_filter)


if __name__ == "__main__":
    main()
