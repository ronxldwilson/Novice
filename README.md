# Novice

Fine-tune a small language model to play chess, using Apple Silicon (MLX) for fast local training and inference.

## Where it got to

Qwen 2.5 0.5B + LoRA, trained on Lichess games then on Stockfish-labelled
positions. Against **Stockfish Skill Level 0 at depth 1**, over 20 games:

**3 wins, 2 draws, 15 losses — 20%.** All three wins were checkmates. Legal move
rate 100%. Pooled across three checkpoints (44 games) the score is **20.5%**,
so this is a stable figure rather than one lucky run.

So it beats Stockfish sometimes, at Stockfish's weakest available setting, and
loses to it overall. It loses every game at Skill Level 3 and above. That is the
honest summary; the full picture, including two instructive negative results and
an ablation showing how much of the play comes from the harness rather than the
network, is in [RESULTS.md](RESULTS.md).

Baseline for comparison: the untrained model produces a legal move under 30% of
the time and is checkmated in ~40 moves every game.

Two results from the run are worth more than the win count:

**Move-match agreement does not predict playing strength.** Four models were
trained; agreement with Stockfish's chosen move rose from 15% to 25%, while
game results got *worse*. The model that best imitates the engine plays
materially worse than the one that imitates it least. Every model was
ultimately selected on games, not on the proxy.

**The model hangs material in 40% of positions.** It has learned what a
plausible move looks like — its Ruy Lopez choices are all book moves — but it
has no lookahead. That is a search problem, and no quantity of supervised data
addressed it.

## Overview

This project trains [Qwen 2.5 0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) to play chess via LoRA fine-tuning on high-quality games from the [Lichess open database](https://database.lichess.org/). Three training modes are available:

- **Basic** — next-move prediction from raw game transcripts. The model must generate valid algebraic notation from scratch.
- **Annotated** — structured reasoning with Stockfish evaluations and candidate moves supplied at inference time.
- **Selection** — the model is shown *every legal move* and picks one, like a chess UI highlighting available squares. Legal moves come from `python-chess`, so no engine is needed at inference.

**Selection is the mode that matters.** Annotated mode scores 100% wins against Stockfish at max strength, but that result is an artifact: it is handed Stockfish's own top-3 candidates and learns only to copy candidate #1. Remove the engine and it collapses to baseline. Selection mode gives the model scaffolding that is free to compute and contains no answer, so any skill it shows is its own.

### Constrained move ranking

At inference the model does not generate a move and hope it parses. Instead every
legal move is enumerated and scored, and the argmax is played:

```
score(move) = mean log P(move tokens | prompt)
```

Illegal output becomes structurally impossible, and the model's full distribution
is used rather than a single sampled guess. The prompt dominates the sequence
(~250 tokens against ~3 for a move), so it is encoded once into a KV cache and
reused across candidates — single-token moves cost no extra forward pass at all.

### The metric that matters: match%, not legal%

Two separable problems:

| Problem | Fixed by |
|---|---|
| Producing a *legal* move | Format/scaffolding — free |
| Producing a *good* move | Training — the actual hard part |

In selection mode legal% reads ~100% for any model, trained or not, so it is a vanity metric. The real measure is **match%** — how often the model picks the move a 2000+ Elo human actually played. Random choice off the legal list scores ~4%.

Measure it directly, without playing games:

```bash
uv run python src/eval_match.py --adapter-path adapters/selection --n 200
```

Reports top1 / top3 / MRR against the human move, plus the random baseline and
the lift over it.

## Requirements

- macOS with Apple Silicon (M-series chip)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Stockfish](https://stockfishchess.org/) (for annotated mode): `brew install stockfish`

## Setup

```bash
# Install dependencies
uv sync
```

## Training Pipeline

### Step 1: Prepare Data

**Basic mode** — downloads ~500K games (2000+ Elo) from Lichess and converts to next-move prediction examples:

```bash
uv run python src/prepare_data.py
```

Options:
- `--max-games 100000` — number of games to download (default: 500K)
- `--min-elo 2200` — minimum player Elo (default: 2000)

**Selection mode** (recommended) — builds move-selection examples with all legal moves listed. No engine required, so it runs in minutes:

```bash
uv run python src/prepare_data_selection.py --max-games 100000 --positions-per-game 8
```

Options:
- `--label played` — (default) label with the move the 2000+ Elo player actually made. Free.
- `--label stockfish` — label with Stockfish's best move at `--depth`. Better labels, ~0.25s per position.
- `--positions-per-game 8` — positions sampled per game

**Annotated mode** — adds Stockfish evaluations and candidate move analysis to each position:

```bash
uv run python src/prepare_data_annotated.py

# Or reuse games from a previous download:
uv run python src/prepare_data_annotated.py --use-cached-games
```

Options:
- `--depth 16` — Stockfish search depth (default: 12, higher = slower but better annotations)
- `--candidates 5` — number of candidate moves per position (default: 3)
- `--positions-per-game 12` — positions to annotate per game (default: 8)
- `--max-games 100000` — number of games (default: 100K for annotated)

### Step 2: Train

```bash
# Selection training (recommended)
uv run python src/train.py --selection --small

# Basic training
uv run python src/train.py --small

# Annotated training
uv run python src/train.py --annotated
```

`--small` uses the subsampled dataset. On a 16 GB machine this is required for
basic and selection modes — the full sets (8.5 GB and 314 MB) will exhaust RAM.

Options:
- `--iters 5000` — training iterations (default: 5000)
- `--batch-size 4` — batch size (default: 4, lower if OOM)
- `--lora-rank 16` — LoRA rank (default: 16)
- `--learning-rate 1e-4` — learning rate (default: 1e-4)
- `--model Qwen/Qwen2.5-1.5B` — use a larger base model

### Step 3: Play

```bash
# Best model: constrained ranking + blunder filter
uv run python src/play.py --selection

# Show what the model was considering
uv run python src/play.py --selection --show-reasoning

# Pure model, no blunder filter
uv run python src/play.py --selection --no-safe

# Play as black
uv run python src/play.py --selection --play-as black
```

### Inspecting what the model believes

The model has **no chain-of-thought** — it was trained with loss masked to the
move token alone, so there is no verbal rationale to print and inventing one
would be fabrication. What it does have is a probability distribution over the
legal moves, and that is its actual opinion:

```bash
uv run python src/explain.py --moves "e4 e5 Nf3 Nc6 Bb5"
```

```
Black to move — 30 legal moves

  move      P(move)  material
  --------------------------------------------------------
  Nf6         11.1%       0  ███·····················  <- model's pick
  Nge7         8.3%       0  ██······················
  d5           8.2%    -100  ██······················
  Bb4          8.2%       0  ██······················
  d6           6.4%       0  ██······················

  Playing Nf6 — model and filter agree.
```

`material` is the static-exchange verdict: 0 means nothing hangs, −100 means the
move drops a pawn to the best reply. Here the model picks the Berlin Defence and
its other favourites are all book moves — but it also likes `d5`, which loses a
pawn. That failure mode is common: **its top pick hangs material in 40% of
positions.**

With `--show-reasoning`, `play.py` prints a short version of the same ranking.
After 1.e4 it produces:

```
model ranking: e5 (-0.50), Nf6 (-0.76), e6 (-0.93), c5 (-0.93)
Model plays: e5
```

All four are main-line replies to 1.e4 — the opening repertoire it picked up
from the Lichess games is recognisable.

In-game commands:
- Type moves in standard algebraic notation: `e4`, `Nf3`, `O-O`, `Qxd5`
- `board` — redraw the board
- `undo` — take back your last move
- `quit` — end the game

## Training Data Format

### Basic Format

Plain next-move prediction from game history:

```
1.e4 e5 2.Nf3 Nc6 3.Bb5 a6
```

### Annotated Format

Structured reasoning with Stockfish analysis:

```
[Position] 1.e4 e5 2.Nf3 Nc6
[Eval] +0.3
[Side] White to move
[Context] book
[Candidates] Bb5 (+0.3), Bc4 (+0.1), d4 (+0.1)
[Best] Bb5
```

The `[Context]` tag classifies the position based on evaluation change: `book`, `good`, `excellent`, `inaccuracy`, `mistake`, or `blunder`. This helps the model learn when moves matter most.

## Evaluation

Measure the model's Elo by playing automated games against Stockfish at various strength levels:

```bash
# Best model, with the blunder filter, against Stockfish's weakest setting
uv run python src/evaluate.py --safe --adapter-path adapters/selection_sf \
  --games-per-level 20 --elo-levels 0 --sf-skill 0 --sf-depth 1

# Pure model, no blunder filter - the honest measure of the network alone
uv run python src/evaluate.py --constrained --adapter-path adapters/selection_sf \
  --games-per-level 20 --elo-levels 0 --sf-skill 0 --sf-depth 1
```

**Always state the opponent configuration.** "Beats Stockfish" means nothing on
its own. Stockfish exposes several independent weakening knobs:

| Flag | Meaning |
|---|---|
| `--sf-skill 0..20` | `Skill Level`; at 0 the engine deliberately plays inferior moves |
| `--sf-depth N` | cap search depth; 1 is weak but still has a full static eval |
| `--sf-nodes N` | cap nodes per move |
| `--elo-levels 1320 …` | `UCI_Elo`, which **floors at 1320** — weaker settings need `--sf-skill` |

`--safe` and `--constrained` must be reported separately: `--safe` adds the SEE
blunder filter, which on current numbers contributes more than the network does.

For a faster read on move quality without playing full games:

```bash
uv run python src/eval_match.py --adapter-path adapters/selection_sf --n 150
```

Games are split evenly between White and Black, and each run outputs:
- Win/draw/loss record at each level
- Legal move percentage (how often the model generates valid moves)
- Estimated Elo rating using the performance rating formula
- Detailed per-game results saved to `results/`

## Project Structure

```
chess/
├── pyproject.toml                  # dependencies (managed by uv)
├── data/                           # training data (gitignored)
│   ├── games_checkpoint.jsonl      # download checkpoint (resumable)
│   ├── annotations_checkpoint.jsonl # annotation checkpoint (resumable)
│   ├── train.jsonl                 # basic training examples
│   ├── valid.jsonl                 # basic validation examples
│   ├── train_selection.jsonl       # selection training examples (754K)
│   ├── train_selection_small.jsonl # subsampled for 16GB training (300K)
│   ├── train_annotated.jsonl       # annotated training examples
│   └── valid_annotated.jsonl       # annotated validation examples
├── adapters/                       # LoRA weights (gitignored)
│   ├── basic/                      # basic model adapters
│   ├── selection/                  # selection model adapters
│   └── annotated/                  # annotated model adapters
├── models/                         # downloaded base model (gitignored)
├── results/                        # evaluation results (gitignored)
└── src/
    ├── utils.py                    # shared utilities (download, checkpointing, logging)
    ├── prepare_data.py             # basic data pipeline
    ├── prepare_data_selection.py   # move-selection data pipeline (no engine needed)
    ├── prepare_data_annotated.py   # Stockfish-annotated data pipeline
    ├── train.py                    # MLX LoRA fine-tuning
    ├── play.py                     # interactive chess interface
    ├── evaluate.py                 # automated Elo evaluation
    └── probe_prompts.py            # prompt-format / match% probe
```

## How It Works

1. **Data** — High-Elo games are streamed from the Lichess database and filtered for quality (both players 2000+, games with 10+ moves, decisive or drawn results).

2. **Annotation** — In annotated mode, Stockfish evaluates sampled positions at depth 12, providing the position evaluation, top candidate moves with scores, and a classification of each move's quality.

3. **Training** — LoRA (Low-Rank Adaptation) fine-tunes only a small set of adapter weights on top of the frozen base model. This keeps memory usage low (~2-4GB) and training fast on Apple Silicon via MLX.

4. **Inference** — During play, the annotated model receives the same structured format it was trained on. Stockfish provides the real-time position evaluation and candidates, and the model selects the best move — combining engine analysis with learned pattern recognition.

## Baseline Results

Before any fine-tuning, the base Qwen 2.5 0.5B was evaluated against Stockfish at various Elo levels (10 games each):

| Stockfish Elo | W | D | L | Win% | Legal Move % |
|---------------|---|---|---|------|--------------|
| 1320          | 0 | 0 | 10 | 0%  | 30.9%        |
| 1500          | 0 | 0 | 10 | 0%  | 22.7%        |
| 1800          | 0 | 0 | 10 | 0%  | 29.7%        |

**Effective Elo: ~0.** The base model has no chess knowledge — it generates legal moves less than 30% of the time (the rest are random fallbacks), and gets checkmated in ~40 moves every game. This establishes a clear floor: any improvement after fine-tuning is directly attributable to the training.

### The annotated model was reading the answer

Annotated mode looked spectacular and was measuring nothing:

| Setup | Legal % | Result |
|---|---|---|
| Annotated + Stockfish eval & candidates | 100% | **30W–0D–0L**, incl. 5–0 vs Stockfish 3190 (max) |
| Same model, Stockfish removed | 28–40% | **0W–0D–15L** |

Given Stockfish's top-3 candidates in the prompt, the model learned one rule — *output candidate #1* — and rode the engine to a fake 3190 Elo. Strip the engine and it is back at baseline. Any harness that hands the model ranked engine output is measuring the engine, not the model.

### Prompt engineering fixes legality, not skill

80 positions sampled from real games, base model, no fine-tuning. `match%` = picked the move a 2000+ Elo human actually played (random ≈ 2.9%):

| Prompt format | Legal % | Match % |
|---|---|---|
| instructed ("answer with one move from the list") | **100.0%** | 1.2% |
| few-shot (2 worked examples) | 95.0% | 1.2% |
| continuation (no scaffolding) | 25.0% | 1.2% |
| numbered list | 1.2% | 1.2% |
| bare `[Legal] … [Best]` | 0.0% | 0.0% |

A firm instruction takes legality from 0% to 100%. Match% stays at or *below* random for every format. Sample outputs make it plain — `Qf4`, `Qa5`, `Kd7`: all legal, all unrelated to the position. **Legality is a formatting problem and is free; move quality is a knowledge problem and only training buys it.**

Reproduce with `uv run python src/probe_prompts.py --positions 80`.

## Training Approaches

### Basic (raw moves)
Next-move prediction from 500K game transcripts. The model sees a move history and predicts the next move, generating notation from scratch. Teaches notation and common patterns, but every output can be illegal.

### Annotated (Stockfish-guided)
18.6K positions with Stockfish evaluation, top candidates, and a context tag. Kept for reference and as a cautionary result — see above. Not a real measure of chess ability.

### Selection (legal-move scaffolding)
The model receives all legal moves and picks one. Labels are the move actually played by a 2000+ Elo player, so generation costs no engine time — 754K examples in ~8 minutes versus ~64 hours for Stockfish labels at depth 12. Legality is guaranteed by construction; the training signal goes entirely into judgment.

All three are **independent models** — each is its own LoRA adapter over the same frozen Qwen 2.5 0.5B base.

## Progress

**Done**

- [x] Stream + filter 500K games (2000+ Elo) from the Lichess database
- [x] Checkpoint/resume for downloads — survived a mid-run connection drop at 180K games
- [x] Fix a 19 GB memory blowup by streaming conversion instead of buffering in RAM
- [x] Baseline the untrained model: ~28% legal, 0 wins, effective Elo ~0
- [x] Evaluation harness — plays Stockfish at a stated configuration, reports W/D/L and legal%
- [x] Train annotated model (18.6K examples, loss 1.82 → 0.85)
- [x] **Discover the annotated model was reading Stockfish's answer, not playing chess**
- [x] Prompt-format probe — proved prompting fixes legality but not skill
- [x] Selection mode: FEN board state, masked completion loss, length filtering
- [x] Constrained ranking — score every legal move, take argmax; illegal output impossible
- [x] `match%` evaluator against held-out human and engine moves
- [x] SEE blunder filter, reported separately from pure model play
- [x] Stacked training: human labels → Stockfish labels, run unattended
- [x] **First wins against Stockfish** — 3W/2D/15L vs Skill Level 0

**Next**

- [ ] **Close the gap to Skill Level 3+**, where the model currently scores 0%.
- [ ] **Investigate why stage 3 hurt.** More engine data raised top1 (20.0 → 21.3%)
      but dropped play from 20% to 7.5%. The proxy metric and the objective came
      apart; worth understanding before scaling data further.
- [ ] **Strengthen the model rather than the harness.** The ablation shows the SEE
      filter contributes most of the current wins (5% → 20%). That gap is the
      real work.
- [ ] **Search.** The model has none. Even 2-ply lookahead over its own top-k
      would likely help more than additional supervised data.
- [ ] **Instruct-tuned base** — `Qwen2.5-0.5B-Instruct` follows the selection
      format more readily than the raw base model.
- [ ] **Larger base model** — Qwen 2.5 1.5B, needs `--batch-size 2` on 16 GB.
- [ ] **Self-play** — generate games, keep the winning side's moves, retrain.

## Performance Notes

- **16GB Mac**: Qwen 2.5 0.5B with LoRA fits comfortably. Training peaks at ~8GB. The full 37.7M basic training set (8.5GB) is too large to load into memory — use `--small` for the subsampled 500K version.
- **Data download**: The Lichess database is large (~25GB compressed). The script streams and filters it without downloading the full file. Checkpoints every 5K games for resume support.
- **Annotation speed**: Stockfish annotation processes ~0.5 games/second at depth 12 (including shallow evals for context tags). 10K games takes ~6 hours.
- **Training speed**: ~1 iter/sec on M5, 1000 iterations takes ~17 minutes.
