# Novice

Fine-tune a small language model to play chess, using Apple Silicon (MLX) for fast local training and inference.

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
# Play against the basic model
uv run python src/play.py

# Play against the annotated model with reasoning visible
uv run python src/play.py --annotated --show-reasoning

# Play as black
uv run python src/play.py --annotated --play-as black
```

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
# Evaluate the selection model
uv run python src/evaluate.py --selection

# Evaluate basic model, custom settings
uv run python src/evaluate.py --games-per-level 40 --elo-levels 1320 1500 1800
```

Note: Stockfish's minimum `UCI_Elo` is **1320**, so that is the weakest opponent
available. Levels below it are rejected by the engine.

For a faster read on move quality without playing full games, use the prompt probe —
it reports match% against human moves across sampled positions:

```bash
uv run python src/probe_prompts.py --positions 80 --adapter-path adapters/selection
```

This plays 20 games per Elo level (half as white, half as black) and outputs:
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
- [x] Evaluation harness — plays Stockfish at fixed Elo, reports W/D/L, legal%, estimated Elo
- [x] Train annotated model (18.6K examples, loss 1.82 → 0.85)
- [x] **Discover the annotated model was reading Stockfish's answer, not playing chess**
- [x] Prompt-format probe — proved prompting fixes legality but not skill
- [x] Build selection-mode pipeline (data, training, inference) with no engine at inference
- [x] Generate 754K selection examples in ~8 minutes

**Next**

- [ ] Train and evaluate the selection model — the headline number is **match%**, not legal% or Elo
- [ ] Report Elo for selection mode against Stockfish 1320+
- [ ] **Stacked training** — basic first (notation and patterns), then selection on top (judgment). Like learning how the pieces move before learning strategy.
- [ ] **Scale selection data** — 754K examples exist; only 300K are used to stay inside 16 GB. Test whether the full set helps.
- [ ] **Stockfish-labelled selection data** — `--label stockfish` swaps human moves for engine best moves. Better labels, ~64h at depth 12 for 100K games; worth trying on a smaller slice.
- [ ] **Puzzle evaluation** — Lichess puzzle set for tactical sharpness, independent of full-game Elo
- [ ] **Larger base model** — Qwen 2.5 1.5B if 0.5B results justify it. Needs `--batch-size 2` on 16 GB.
- [ ] **Instruct-tuned base** — we use the raw base model; `Qwen2.5-0.5B-Instruct` follows the selection format far more readily
- [ ] **Self-play** — generate games, keep the winning side's moves, retrain

## Performance Notes

- **16GB Mac**: Qwen 2.5 0.5B with LoRA fits comfortably. Training peaks at ~8GB. The full 37.7M basic training set (8.5GB) is too large to load into memory — use `--small` for the subsampled 500K version.
- **Data download**: The Lichess database is large (~25GB compressed). The script streams and filters it without downloading the full file. Checkpoints every 5K games for resume support.
- **Annotation speed**: Stockfish annotation processes ~0.5 games/second at depth 12 (including shallow evals for context tags). 10K games takes ~6 hours.
- **Training speed**: ~1 iter/sec on M5, 1000 iterations takes ~17 minutes.
