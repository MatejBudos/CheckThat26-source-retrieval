"""
Runs the full augmented training pipeline:
  1. Augment train queries with LLM (Claude Haiku)
  2. Mine hard negatives using augmented queries
  3. Merge original + augmented hard negatives
  4. Fine-tune bi-encoder on mixed dataset

Usage:
    python scripts/run_augmented_training.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nERROR: step failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def merge_jsonl(src_a: Path, src_b: Path, dst: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  Merging {src_a.name} + {src_b.name} → {dst.name}")
    print(f"{'='*60}")
    with open(dst, "wb") as f:
        f.write(src_a.read_bytes())
        f.write(src_b.read_bytes())
    lines = sum(1 for _ in open(dst, encoding="utf-8"))
    print(f"  {lines} total training examples")


def main() -> None:
    # Step 1 — Augment train queries
    import os
    augment_cmd = [
        "scripts/augment_queries.py",
        "--splits", "train",
        "--tweets-per-call", "150",
        "--concurrency", "5",
    ]
    # Keys picked up from env automatically; fractions auto-distributed
    run(augment_cmd, "Step 1/4 — Augmenting train queries with LLM")

    # Step 2 — Mine hard negatives with augmented queries
    run(
        [
            "scripts/mine_hard_negatives.py",
            "--dense-model", "checkpoints/e5-finetuned/final",
            "--output", "data/hard_negatives_aug.jsonl",
            "--use-augmented",
        ],
        "Step 2/4 — Mining hard negatives (augmented queries)",
    )

    # Step 3 — Merge datasets
    merge_jsonl(
        ROOT / "data/hard_negatives.jsonl",
        ROOT / "data/hard_negatives_aug.jsonl",
        ROOT / "data/hard_negatives_mixed.jsonl",
    )

    # Step 4 — Train (continue from already fine-tuned checkpoint)
    run(
        [
            "scripts/train_biencoder.py",
            "--data", "data/hard_negatives_mixed.jsonl",
            "--output-dir", "checkpoints/e5-aug",
            "--model", "checkpoints/e5-finetuned/final",
            "--batch-size", "8",
            "--grad-accum", "8",
        ],
        "Step 4/4 — Fine-tuning bi-encoder on mixed dataset",
    )

    print("\n" + "="*60)
    print("  All steps completed!")
    print("  Next: update configs/default.yaml:")
    print('    dense_model: "checkpoints/e5-aug/final"')
    print("    use_augmented: true")
    print("    use_translations: false")
    print("  Then: python scripts/run_pipeline.py --no-reranker")
    print("="*60)


if __name__ == "__main__":
    main()
