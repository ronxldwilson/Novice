# Results log

Experiment record for Novice. Newest findings first.

Headline metric is **top1 match** — how often the model's first choice equals the
move a 2000+ Elo human actually played, on held-out positions, using constrained
ranking over the legal moves. Random choice scores ~4%.

`legal%` is deliberately *not* the headline: in selection mode it reads ~100% for
any model, trained or not, so it measures the harness rather than the model.

---

## Selection model — training curve

Qwen 2.5 0.5B + LoRA (rank 32, all 24 layers), masked completion loss,
388K move-selection examples labelled with human moves.

| Checkpoint | top1 | top3 | MRR | lift vs random |
|---|---|---|---|---|
| untrained (control) | 5.0% | 17.5% | 0.155 | 1.2× — chance |
| 1000 iters | 11.7% | 30.0% | 0.268 | 2.7× |
| 3000 iters | 15.0% | 36.7% | 0.297 | 3.5× |

Measured on 60 held-out positions each, constrained ranking, greedy.

The untrained control runs through the *identical* harness, so the gain cannot be
attributed to scaffolding.

---

## Baselines

### Untrained model, no scaffolding

30 games vs Stockfish, 10 per level:

| Stockfish Elo | W | D | L | Legal % |
|---|---|---|---|---|
| 1320 | 0 | 0 | 10 | 30.9% |
| 1500 | 0 | 0 | 10 | 22.7% |
| 1800 | 0 | 0 | 10 | 29.7% |

Effective Elo ~0. Checkmated in ~40 moves every game.

### Untrained model + legal-move list

| Stockfish Elo | W | D | L | Legal % |
|---|---|---|---|---|
| 1320 | 0 | 0 | 4 | 0% |
| 1500 | 0 | 0 | 4 | 0% |

0% legal — *lower* than with no scaffolding. In basic mode the prompt is a move
sequence, so the model continues with chess-like tokens and occasionally lands on
something valid. Given a `[Best]` slot it has never seen, it emits garbage every
time. Games still complete because the harness falls back to a random legal move,
which is why the result is a meaningful "random play" floor.

---

## Negative result: the annotated model was reading the answer

| Setup | Legal % | Result |
|---|---|---|
| Annotated + Stockfish eval & top-3 candidates | 100% | **30W–0D–0L**, incl. 5–0 vs Stockfish 3190 (max) |
| Same adapters, Stockfish removed | 28–40% | **0W–0D–15L** |

Handed Stockfish's ranked candidates, the model learned one rule — *output
candidate #1* — and rode the engine to an apparent 3190 Elo. Strip the engine and
it returns to baseline.

**Lesson:** a harness that feeds the model ranked engine output measures the
engine, not the model. Any scaffolding must be answer-free. Selection mode
satisfies this: the legal move list comes from `python-chess`, is free to compute,
and contains no ranking.

---

## Negative result: prompting fixes legality, not skill

80 held-out positions, untrained base model, free generation:

| Prompt format | Legal % | Match % |
|---|---|---|
| instructed ("answer with one move from the list") | **100.0%** | 1.2% |
| few-shot (2 worked examples) | 95.0% | 1.2% |
| continuation (no scaffolding) | 25.0% | 1.2% |
| numbered list | 1.2% | 1.2% |
| bare `[Legal] … [Best]` | 0.0% | 0.0% |

Random baseline for match: ~2.9%.

A firm instruction takes legality from 0% to 100%. Match stays at or *below*
random for every format — sample outputs were `Qf4`, `Qa5`, `Kd7`: all legal, all
unrelated to the position.

**Lesson:** legality is a formatting problem and is free. Move quality is a
knowledge problem and only training buys it. These are separable, and conflating
them makes a model look far better than it is.

---

## Engineering notes

- **19 GB memory blowup** in data conversion, fixed by streaming games and
  shuffling line *indices* rather than buffering 37.7M examples in RAM.
- **Truncation was destroying labels.** 2.4% of v1 selection examples exceeded
  `max_seq_length`, and truncation cuts the end of a sequence — exactly where the
  label lives. v2 caps prompt length at generation time: max 340 tokens, 0% over.
- **Masked loss cut memory 7.96 → 5.78 GB** at rank 16, since activations are only
  needed for the ~4.7 completion tokens.
- **Constrained ranking: 5.9s → 2.3s per position** by encoding the prompt once
  into a KV cache and trimming between candidates. Single-token moves need no
  extra forward pass.
- **Download checkpointing paid for itself** — a connection drop at 180K games
  cost nothing.
