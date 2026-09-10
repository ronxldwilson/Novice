"""Static exchange evaluation, used as a blunder filter over the model's ranking.

The model has no search, so it will happily leave a queen en prise if the move
looks plausible. SEE answers one narrow question with pure chess rules — "if I
play this, what material do I lose to the best capture sequence?" — and lets us
drop moves that simply hang material.

This is deliberately *not* an engine oracle. It does no lookahead beyond forced
capture sequences on a single square, has no positional model, and never tells
the model which move is best — it only vetoes moves that lose material outright.
Results using it are reported separately from pure model play.
"""

from __future__ import annotations

import chess

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def _value(piece: chess.Piece | None) -> int:
    return PIECE_VALUE[piece.piece_type] if piece else 0


def _least_valuable_attacker(board: chess.Board, square: int, color: bool):
    best, best_val = None, None
    for sq in board.attackers(color, square):
        v = _value(board.piece_at(sq))
        if best_val is None or v < best_val:
            best, best_val = sq, v
    return best


def see_capture(board: chess.Board, square: int, side: bool) -> int:
    """Material swing on `square` if `side` initiates the capture sequence."""
    attacker_sq = _least_valuable_attacker(board, square, side)
    if attacker_sq is None:
        return 0

    target = board.piece_at(square)
    if target is None:
        return 0

    captured_value = _value(target)
    moving = board.piece_at(attacker_sq)

    move = chess.Move(attacker_sq, square)
    if moving.piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
        move = chess.Move(attacker_sq, square, promotion=chess.QUEEN)
    if move not in board.legal_moves:
        return 0

    board.push(move)
    try:
        gain = captured_value - max(0, see_capture(board, square, not side))
    finally:
        board.pop()
    return gain


def material_delta(board: chess.Board, move: chess.Move) -> int:
    """Net material for the mover after `move` and the best enemy reply capture.

    Positive means the move wins material; negative means it loses material.
    """
    mover = board.turn

    gained = 0
    if board.is_capture(move):
        if board.is_en_passant(move):
            gained = PIECE_VALUE[chess.PAWN]
        else:
            gained = _value(board.piece_at(move.to_square))

    board.push(move)
    try:
        if board.is_checkmate():
            return 100000
        worst = 0
        for reply in board.legal_moves:
            if not board.is_capture(reply):
                continue
            sq = reply.to_square
            swing = see_capture(board, sq, board.turn)
            worst = max(worst, swing)
    finally:
        board.pop()

    return gained - worst


def filter_blunders(
    board: chess.Board,
    ranked: list[tuple[str, float]],
    top_k: int = 8,
    tolerance: int = 100,
) -> list[tuple[str, float]]:
    """Reorder the model's top_k so material-losing moves fall behind safe ones.

    Only the model's own top_k candidates are considered, so the model still
    chooses; SEE only vetoes moves among them that hang material. Ordering
    within each group preserves the model's preference.
    """
    head = ranked[:top_k]
    if not head:
        return ranked

    scored = []
    for san, score in head:
        try:
            mv = board.parse_san(san)
        except ValueError:
            continue
        scored.append((san, score, material_delta(board, mv)))

    if not scored:
        return ranked

    best_delta = max(d for _, _, d in scored)
    safe = [(s, sc) for s, sc, d in scored if d >= best_delta - tolerance]
    risky = [(s, sc) for s, sc, d in scored if d < best_delta - tolerance]

    return safe + risky + ranked[top_k:]
