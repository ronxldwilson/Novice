"""Fine-tune a small LM on chess data using MLX LoRA."""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "adapters"
MODEL = "Qwen/Qwen2.5-0.5B"
LOCAL_MODEL = Path(__file__).parent.parent / "models" / "Qwen2.5-0.5B"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune chess SLM with MLX")
    parser.add_argument("--model", default=MODEL, help="HuggingFace model ID")
    parser.add_argument("--annotated", action="store_true", help="Use Stockfish-annotated data")
    parser.add_argument("--small", action="store_true", help="Use subsampled basic data (500K examples)")
    parser.add_argument("--selection", action="store_true", help="Use move-selection data (legal moves format)")
    parser.add_argument("--iters", type=int, default=5000, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--mask-prompt", action="store_true",
                        help="Compute loss only on the completion (prompt/completion data)")
    parser.add_argument("--resume", default=None, help="Adapter file to resume from")
    args = parser.parse_args()

    if args.selection:
        suffix = "_selection"
    elif args.annotated:
        suffix = "_annotated"
    elif args.small:
        suffix = "_small"
    else:
        suffix = ""
    train_path = DATA_DIR / f"train{suffix}.jsonl"
    valid_path = DATA_DIR / f"valid{suffix}.jsonl"

    if not train_path.exists() or not valid_path.exists():
        scripts = {"_selection": "prepare_data_selection.py", "_annotated": "prepare_data_annotated.py"}
        script = scripts.get(suffix, "prepare_data.py")
        print(f"Training data not found at {train_path}")
        print(f"Run: uv run python src/{script}")
        sys.exit(1)

    if args.selection:
        adapter_name = "selection"
    elif args.annotated:
        adapter_name = "annotated"
    else:
        adapter_name = "basic"
    adapter_dir = OUTPUT_DIR / adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # mlx_lm expects train.jsonl and valid.jsonl in the data dir
    data_dir = DATA_DIR / f"mlx{suffix}"
    data_dir.mkdir(parents=True, exist_ok=True)

    import os
    for name in ["train.jsonl", "valid.jsonl"]:
        link = data_dir / name
        link.unlink(missing_ok=True)
        os.symlink((DATA_DIR / f"{name.split('.')[0]}{suffix}.jsonl").resolve(), link)

    # Write LoRA config
    lora_config = {
        "lora_parameters": {
            "rank": args.lora_rank,
            "alpha": args.lora_rank * 2,
            "dropout": 0.05,
            "scale": 1.0,
        },
    }
    config_path = adapter_dir / "lora_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(lora_config, f)

    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else args.model

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.lora_layers),
        "--learning-rate", str(args.learning_rate),
        "--steps-per-eval", str(args.steps_per_eval),
        "--max-seq-length", str(args.max_seq_length),
        "--save-every", str(args.save_every),
        "-c", str(config_path),
    ]
    if args.mask_prompt:
        cmd.append("--mask-prompt")
    if args.resume:
        cmd += ["--resume-adapter-file", args.resume]

    modes = {"_selection": "selection (legal moves)", "_annotated": "annotated (Stockfish-guided)"}
    mode = modes.get(suffix, "basic (raw moves)")
    train_size_mb = train_path.stat().st_size / (1024 * 1024)
    print(f"Starting LoRA fine-tuning on {model_path}")
    print(f"Mode: {mode}")
    print(f"Training data: {train_size_mb:.0f} MB")
    print(f"LoRA: rank={args.lora_rank}, layers={args.lora_layers}")
    print(f"Iters: {args.iters}, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"Adapters will be saved to: {adapter_dir}")
    print(f"Command: {' '.join(cmd)}\n", flush=True)

    subprocess.run(cmd, check=True)
    print(f"\nDone! Adapters saved to {adapter_dir}")


if __name__ == "__main__":
    main()
