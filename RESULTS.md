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
| 6000 iters | 11.7% | 23.3% | 0.245 | 2.7× |

Measured on 60 held-out positions each, constrained ranking, greedy.

The untrained control runs through the *identical* harness, so the gain over
chance cannot be attributed to scaffolding.

**Caveat on the last two rows:** n=60 gives roughly ±9pp, so 15.0% (9/60) and
11.7% (7/60) are not distinguishable. The honest reading is that human-label
training rises steeply to ~1000 iters and then **plateaus around 12–15%**, not
that it regressed. This plateau is the motivation for stage 2 — if more of the
same data stops helping, the labels themselves are the ceiling. Human moves at
2000 Elo contain blunders and stylistic noise; Stockfish labels do not.

### Play strength at the plateau

The 3000-iter model, constrained ranking, vs Stockfish:

| Opponent | Result | Notes |
|---|---|---|
| UCI_Elo 1320 | 0W–2L | 100% legal, 38 moves avg |
| UCI_Elo 1320 + SEE filter | 0W–2L | survived **102 moves** as White vs 38 |
| depth-1 Stockfish + SEE | 0W–2L | 36, 37 moves |

Not yet beating Stockfish at any setting. The SEE filter clearly extends
survival — the model stops shedding material in the opening — but a blunder
filter cannot manufacture a plan.

---

## Stage 2 — Stockfish labels

Stage-1 adapters continued on 58K Stockfish-labelled positions (lr 5e-5, 4000
iters). Measured against *Stockfish's* choice on held-out positions:

| Model | top1 | top3 | MRR | lift |
|---|---|---|---|---|
| stage 1 (human labels, plateaued) | 11.7–15.0% | 23–37% | 0.245 | 2.7× |
| stage 2 (Stockfish labels) | **20.0%** | **40.0%** | **0.352** | **4.8×** |

Changing the labels moved the number that more training could not.

### First wins against Stockfish

10 games vs **Stockfish Skill Level 0, depth 1**, constrained ranking + SEE filter:

| W | D | L | Score | Legal % |
|---|---|---|---|---|
| 2 | 1 | 7 | **25%** | 100% |

Both wins were checkmates. An earlier 4-game probe showed 50%, which the larger
sample corrected to 25% — small samples flatter, and the 10-game figure is the
one to trust.

`UCI_Elo` floors at 1320, which initially hid the fact that a usable difficulty
ladder exists below it. `Skill Level` is a separate knob; at 0 the engine
deliberately plays inferior moves. The opponent configuration is always stated
because "beat Stockfish" is meaningless without it.

## Final results

Best model: **`adapters/selection_sf`** — Qwen 2.5 0.5B + LoRA (rank 32, all 24
layers), trained on 388K human-labelled selection examples then continued on 58K
Stockfish-labelled ones.

### Does it beat Stockfish?

**It wins games, but it does not win matches.** 20 games per row, model plays
half as White and half as Black, constrained ranking, 100% legal moves throughout.

| Opponent | Harness | W | D | L | Score |
|---|---|---|---|---|---|
| Skill 0, depth 1 | model + SEE | **3** | **2** | 15 | **20.0%** |
| Skill 0, depth 1 | model only | 0 | 1 | 9 | 5.0% |
| Skill 0, 0.1s/move | model + SEE | 0 | 1 | 9 | 5.0% |
| Skill 3, depth 1 | model + SEE | 0 | 0 | 10 | 0.0% |
| UCI_Elo 1320, 0.1s | model + SEE | 0 | 0 | 2 | 0.0% |

The three wins were genuine checkmates delivered by the model, not timeouts or
adjudications. But at 20% aggregate against Stockfish's weakest available
setting, the honest summary is: **beats Stockfish occasionally at Skill Level 0,
loses to it overall, and loses to everything above it.**

### Checkpoint comparison — and the pooled figure

Three stage-2 checkpoints against the same opponent, to check we were not just
reading noise off one lucky save:

| Checkpoint | games | W | D | L | Score |
|---|---|---|---|---|---|
| @2000 | 12 | 1 | 4 | 7 | 25.0% |
| @3000 | 12 | 1 | 2 | 9 | 16.7% |
| @4000 | 20 | 3 | 2 | 15 | 20.0% |
| **pooled** | **44** | **5** | **8** | **31** | **20.5%** |

No checkpoint is meaningfully better than the others — the spread is noise. The
pooled 44-game figure of **20.5%** is the number to quote, rather than the
best-looking single run.

### How much is the model and how much is the harness?

| Configuration | Score |
|---|---|
| model only (constrained ranking) | 5.0% |
| model + SEE blunder filter | 20.0% |

Most of the winning comes from the blunder filter, not the network. This is the
same trap as the annotated experiment in a milder form, which is why the two are
always reported separately. The SEE filter is defensible — it uses only chess
rules and can only reorder the model's own candidates — but it must never be
folded into a headline number.

### More engine data made play worse

| Stage | Data | top1 (n=150) | Score vs Skill 0 (20 games) |
|---|---|---|---|
| stage 2 | 58K Stockfish labels | 20.0% | **20.0%** |
| stage 3 | +135K more, continued | **21.3%** | 7.5% |

Stage 3 improved the proxy metric and degraded the thing we actually care about.
Move-match agreement and playing strength came apart: agreeing slightly more
often with the engine while blundering more decisively in the games that matter.
Stage 2 is shipped as the best model on the strength of the games, not the
match%.

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
