"""Fine-tune a small LM on chess data using MLX LoRA."""

import argparse
import subprocess
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "adapters"
MODEL = "Qwen/Qwen2.5-0.5B"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune chess SLM with MLX")
    parser.add_argument("--model", default=MODEL, help="HuggingFace model ID")
    parser.add_argument("--annotated", action="store_true", help="Use Stockfish-annotated data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    suffix = "_annotated" if args.annotated else ""
    train_path = DATA_DIR / f"train{suffix}.jsonl"
    valid_path = DATA_DIR / f"valid{suffix}.jsonl"

    if not train_path.exists() or not valid_path.exists():
        script = "prepare_data_annotated.py" if args.annotated else "prepare_data.py"
        print(f"Training data not found at {train_path}")
        print(f"Run: uv run python src/{script}")
        sys.exit(1)

    adapter_dir = OUTPUT_DIR / ("annotated" if args.annotated else "basic")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    data_dir = DATA_DIR / f"mlx{suffix}"
    data_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy(train_path, data_dir / "train.jsonl")
    shutil.copy(valid_path, data_dir / "valid.jsonl")

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--num-epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lora-rank", str(args.lora_rank),
        "--learning-rate", str(args.learning_rate),
        "--steps-per-eval", str(args.steps_per_eval),
        "--max-seq-length", str(args.max_seq_length),
        "--save-every", str(args.save_every),
    ]

    mode = "annotated (Stockfish-guided)" if args.annotated else "basic (raw moves)"
    print(f"Starting LoRA fine-tuning on {args.model}")
    print(f"Mode: {mode}")
    print(f"Training data: {train_path} ({sum(1 for _ in open(train_path))} examples)")
    print(f"Adapters will be saved to: {adapter_dir}")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)
    print(f"\nDone! Adapters saved to {adapter_dir}")


if __name__ == "__main__":
    main()
