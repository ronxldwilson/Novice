"""Shallow alpha-beta search over the moves the model proposes.

Diagnosis that motivated this: the model's top-ranked move loses at least a pawn
in ~40% of positions. It has learned what a plausible move looks like but has no
lookahead, so it cannot see that a natural developing move drops a piece. That is
a search problem, and no amount of extra supervised data fixes it.

The split of labour is the usual policy-plus-search one:

  * the **model** proposes — it ranks the legal moves and we search only its top
    k, which is what makes a 0.5B policy useful at all. Full-width search would
    make the network irrelevant.
  * the **search** verifies — plain material alpha-beta confirms the candidate
    does not hang anything within the horizon.

Only the root consults the model. Deeper plies use a material evaluation, so a
whole search costs one model call plus cheap board arithmetic.
"""

from __future__ import annotations

import chess

from selection_infer import score_moves
from tactics import PIECE_VALUE

MATE = 1_000_000


def material(board: chess.Board, side: bool) -> int:
    """Material balance in centipawns from `side`'s point of view."""
    total = 0
    for piece_type, value in PIECE_VALUE.items():
        if piece_type == chess.KING:
            continue
        total += value * len(board.pieces(piece_type, side))
        total -= value * len(board.pieces(piece_type, not side))
    return total


def _quiesce(board: chess.Board, side: bool, alpha: int, beta: int, depth: int) -> int:
    """Resolve pending captures so the evaluation is not read mid-exchange."""
    stand = material(board, side)
    if depth <= 0:
        return stand

    maximizing = board.turn == side
    if maximizing:
        if stand >= beta:
            return stand
        alpha = max(alpha, stand)
    else:
        if stand <= alpha:
            return stand
        beta = min(beta, stand)

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        board.push(move)
        try:
            score = _quiesce(board, side, alpha, beta, depth - 1)
        finally:
            board.pop()

        if maximizing:
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        else:
            beta = min(beta, score)
            if beta <= alpha:
                break

    return alpha if maximizing else beta


def _negamax(board: chess.Board, side: bool, depth: int, alpha: int, beta: int) -> int:
    if board.is_checkmate():
        return -MATE if board.turn == side else MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0:
        return _quiesce(board, side, alpha, beta, 4)

    maximizing = board.turn == side
    best = -MATE if maximizing else MATE

    # Captures first — cheap ordering that makes alpha-beta actually prune.
    moves = sorted(board.legal_moves, key=board.is_capture, reverse=True)

    for move in moves:
        board.push(move)
        try:
            score = _negamax(board, side, depth - 1, alpha, beta)
        finally:
            board.pop()

        if maximizing:
            best = max(best, score)
            alpha = max(alpha, best)
        else:
            best = min(best, score)
            beta = min(beta, best)
        if alpha >= beta:
            break

    return best


def search_best_move(
    model,
    tokenizer,
    board: chess.Board,
    top_k: int = 6,
    depth: int = 2,
    prior_weight: float = 40.0,
    rng=None,
) -> tuple[chess.Move, list[tuple[str, int, float]]]:
    """Pick a move: model proposes top_k, alpha-beta verifies them.

    Returns (move, [(san, search_score, model_prob)]) sorted best first.
    `prior_weight` converts the model's preference into centipawns so it breaks
    ties between tactically equal moves without overriding tactics.
    """
    ranked = score_moves(model, tokenizer, board, rng=rng)
    if not ranked:
        raise ValueError("no legal moves")

    # Softmax the model's log-probabilities into a usable prior.
    import math
    hi = max(s for _, s in ranked)
    exps = [math.exp(s - hi) for _, s in ranked]
    total = sum(exps)
    priors = {san: e / total for (san, _), e in zip(ranked, exps)}

    side = board.turn
    candidates = ranked[:top_k]
    scored: list[tuple[str, int, float]] = []

    for san, _ in candidates:
        try:
            move = board.parse_san(san)
        except ValueError:
            continue
        board.push(move)
        try:
            if board.is_checkmate():
                value = MATE
            else:
                value = _negamax(board, side, depth - 1, -MATE, MATE)
        finally:
            board.pop()
        scored.append((san, value, priors.get(san, 0.0)))

    if not scored:
        return board.parse_san(ranked[0][0]), []

    scored.sort(key=lambda t: -(t[1] + prior_weight * t[2]))
    return board.parse_san(scored[0][0]), scored
