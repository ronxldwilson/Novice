"""Constrained move selection: score every legal move, pick the best.

Free generation can emit an illegal move, or a token sequence that merely looks
like one. Instead we enumerate the legal moves, score each under the model, and
take the argmax. Illegal output becomes structurally impossible, and we use the
model's whole distribution rather than a single sampled guess.

Scoring is the summed log-probability of the move's tokens given the prompt.
Length normalisation is optional: SAN moves differ in token count ("e4" vs
"Qxd5+"), and summed logprob quietly favours shorter moves.
"""

from __future__ import annotations

import math

import chess
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from utils import selection_prompt_for_board

CHUNK = 8  # candidates scored per forward pass; keeps peak memory flat


def _prompt_tokens(tokenizer, board: chess.Board, rng=None) -> tuple[list[int], list[str]]:
    prompt, legal = selection_prompt_for_board(board, rng)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(text), legal


def score_moves(
    model,
    tokenizer,
    board: chess.Board,
    length_normalize: bool = True,
    rng=None,
) -> list[tuple[str, float]]:
    """Return [(san, score)] for every legal move, best first.

    The prompt dominates the sequence (~250 tokens vs ~3 for a move), so it is
    encoded once into a KV cache and reused. Each candidate then costs at most
    its own length, and single-token moves cost nothing extra — their score is
    already in the prompt's final logits.
    """
    prompt_ids, legal = _prompt_tokens(tokenizer, board, rng)
    if not legal:
        return []

    cand_ids = [tokenizer.encode(san, add_special_tokens=False) for san in legal]

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids]), cache=cache).astype(mx.float32)
    tail = logits[0, -1]
    first_lp = tail - mx.logsumexp(tail)
    first_lp = mx.array(first_lp)  # materialise once; indexed per candidate below

    scores: list[float] = []
    for cand in cand_ids:
        total = float(first_lp[cand[0]])

        if len(cand) > 1:
            out = model(mx.array([cand[:-1]]), cache=cache).astype(mx.float32)
            lp = out - mx.logsumexp(out, axis=-1, keepdims=True)
            for j in range(1, len(cand)):
                total += float(lp[0, j - 1, cand[j]])
            trim_prompt_cache(cache, len(cand) - 1)

        scores.append(total / len(cand) if length_normalize else total)

    mx.clear_cache()
    return sorted(zip(legal, scores), key=lambda kv: -kv[1])


def best_move(
    model,
    tokenizer,
    board: chess.Board,
    temperature: float = 0.0,
    length_normalize: bool = True,
    rng=None,
) -> chess.Move:
    """Pick a legal move by constrained scoring. Always returns a legal move."""
    ranked = score_moves(model, tokenizer, board, length_normalize, rng)
    if not ranked:
        raise ValueError("no legal moves")

    if temperature <= 0:
        san = ranked[0][0]
    else:
        sans = [s for s, _ in ranked]
        raw = [v / temperature for _, v in ranked]
        hi = max(raw)
        weights = [math.exp(v - hi) for v in raw]
        total = sum(weights)
        r = (rng.random() if rng else __import__("random").random()) * total
        acc = 0.0
        san = sans[-1]
        for s, w in zip(sans, weights):
            acc += w
            if r <= acc:
                san = s
                break

    return board.parse_san(san)
