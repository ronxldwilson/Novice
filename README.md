# Chess SLM

Fine-tune a small language model to play chess, using Apple Silicon (MLX) for fast local training and inference.

## Overview

This project trains [Qwen 2.5 0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) to play chess via LoRA fine-tuning on high-quality games from the [Lichess open database](https://database.lichess.org/). Two training modes are available:

- **Basic** — learns next-move prediction from raw game transcripts
- **Annotated** — learns structured reasoning using Stockfish evaluations, candidate moves, and position classification

The annotated mode produces a stronger player from fewer games because the model learns *why* moves are good, not just what was played.

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
# Basic training
uv run python src/train.py

# Annotated training (recommended)
uv run python src/train.py --annotated
```

Options:
- `--epochs 2` — number of training epochs (default: 1)
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

## Project Structure

```
chess/
├── pyproject.toml                  # dependencies (managed by uv)
├── data/                           # training data (gitignored)
│   ├── games.json                  # cached raw games
│   ├── train.jsonl                 # basic training examples
│   ├── valid.jsonl                 # basic validation examples
│   ├── train_annotated.jsonl       # annotated training examples
│   └── valid_annotated.jsonl       # annotated validation examples
├── adapters/                       # LoRA weights (gitignored)
│   ├── basic/                      # basic model adapters
│   └── annotated/                  # annotated model adapters
└── src/
    ├── prepare_data.py             # basic data pipeline
    ├── prepare_data_annotated.py   # Stockfish-annotated data pipeline
    ├── train.py                    # MLX LoRA fine-tuning
    └── play.py                     # interactive chess interface
```

## How It Works

1. **Data** — High-Elo games are streamed from the Lichess database and filtered for quality (both players 2000+, games with 10+ moves, decisive or drawn results).

2. **Annotation** — In annotated mode, Stockfish evaluates sampled positions at depth 12, providing the position evaluation, top candidate moves with scores, and a classification of each move's quality.

3. **Training** — LoRA (Low-Rank Adaptation) fine-tunes only a small set of adapter weights on top of the frozen base model. This keeps memory usage low (~2-4GB) and training fast on Apple Silicon via MLX.

4. **Inference** — During play, the annotated model receives the same structured format it was trained on. Stockfish provides the real-time position evaluation and candidates, and the model selects the best move — combining engine analysis with learned pattern recognition.

## Performance Notes

- **16GB Mac**: Qwen 2.5 0.5B with LoRA fits comfortably. Batch size 4 should work; reduce to 2 if you see memory pressure.
- **Data download**: The Lichess database is large (~25GB compressed). The script streams and filters it without downloading the full file.
- **Annotation speed**: Stockfish annotation processes ~5-20 games/second depending on depth. 100K games at depth 12 takes roughly 2-5 hours.
- **Expected strength**: A 0.5B model won't rival engines, but the annotated version should play coherent, legal chess in the 1200-1500 Elo range with good opening play.
