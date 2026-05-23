"""
Fine-tunes a cross-encoder reranker on hard negative pairs (fast custom loop).

Speedups vs sentence_transformers.fit():
  - Pre-tokenize all pairs once into tensors (no per-step CPU tokenization)
  - Pure HuggingFace transformers (skip sentence_transformers training wrapper)
  - AMP (fp16) + fused AdamW + linear warmup
  - Shorter default max_length (96) and 1-epoch default

Reads data/hard_negatives.jsonl and creates pointwise pairs:
  - (query, positive) → label 1.0
  - (query, negative) → label 0.0

Output is saved in HuggingFace format and loadable by sentence_transformers
CrossEncoder for inference (src/reranker.py).

Usage:
    python scripts/train_reranker.py
    python scripts/train_reranker.py --max-samples 5000 --epochs 1
    python scripts/train_reranker.py --model BAAI/bge-reranker-v2-m3 --max-length 96
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# Flush stdout immediately so output appears in redirected files / background runs.
import functools
print = functools.partial(print, flush=True)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_pairs(data_path: Path, negs_per_query: int, seed: int, max_samples: int,
               extract_sentences: bool = False, extraction_top_k: int = 2):
    from src.text_utils import extract_key_sentences

    rng = random.Random(seed)
    lines = Path(data_path).read_text(encoding="utf-8").splitlines()
    pairs_per_line = 1 + negs_per_query
    if max_samples and len(lines) > max_samples // pairs_per_line:
        lines = rng.sample(lines, max_samples // pairs_per_line)

    def maybe_extract(query: str, doc: str) -> str:
        if extract_sentences:
            return extract_key_sentences(query, doc, top_k=extraction_top_k)
        return doc

    queries, docs, labels = [], [], []
    for line in lines:
        item = json.loads(line)
        # Strip e5 prefixes — cross-encoder doesn't use them.
        q = item["anchor"].removeprefix("query: ")
        pos = item["positive"].removeprefix("passage: ")
        negs = [n.removeprefix("passage: ") for n in item["negatives"]]

        queries.append(q); docs.append(maybe_extract(q, pos)); labels.append(1.0)
        for neg in rng.sample(negs, min(negs_per_query, len(negs))):
            queries.append(q); docs.append(maybe_extract(q, neg)); labels.append(0.0)

    return queries, docs, labels


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading reranker: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model.to(device)

    print(f"Loading training data: {args.data}")
    queries, docs, labels = build_pairs(
        Path(args.data), args.negs_per_query, args.seed, args.max_samples,
        extract_sentences=args.extract_sentences,
        extraction_top_k=args.extraction_top_k,
    )
    pos = sum(1 for l in labels if l == 1.0)
    print(f"  {pos} positive + {len(labels) - pos} negative = {len(labels)} pairs")

    print(f"Tokenizing {len(labels)} pairs (max_length={args.max_length})...")
    t_tok = time.time()
    enc = tokenizer(
        queries, docs,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    print(f"  tokenization done in {time.time() - t_tok:.1f}s")

    dataset = TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(labels, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    total_steps = len(loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, fused=(device.type == "cuda")
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    if args.compile:
        print("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nTraining: {args.epochs} epochs × {len(loader)} steps = {total_steps} steps"
        f" | batch={args.batch_size} | warmup={warmup_steps} | lr={args.lr} | amp={bool(scaler)}"
    )
    print(f"Output: {output_dir}\n")

    model.train()
    step = 0
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        for input_ids, attn, target in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=input_ids, attention_mask=attn).logits.squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids=input_ids, attention_mask=attn).logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            step += 1
            running_loss += loss.item()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                rate = elapsed / step
                eta = rate * (total_steps - step)
                print(
                    f"  step {step}/{total_steps} | loss={running_loss / args.log_every:.4f}"
                    f" | {rate:.2f}s/step | ETA {eta / 60:.1f} min"
                )
                running_loss = 0.0

    print(f"\nSaving to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Total training time: {(time.time() - t0) / 60:.1f} min")
    print(f'Use in pipeline with: reranker_model: "{output_dir}"')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--data", default="data/hard_negatives.jsonl")
    p.add_argument("--output-dir", default="checkpoints/reranker-finetuned")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128,
                   help="Tokens for [query, doc] pair. 128 covers tweet + title + most of abstract")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max training pairs (0=all). Full dataset ≈ 20–40 min on laptop GPU")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--negs-per-query", type=int, default=3,
                   help="Hard negatives per query (3 gives more signal per anchor)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model for ~10-20%% speedup (adds ~60s warmup on first step)")
    p.add_argument("--extract-sentences", dest="extract_sentences", action="store_true",
                   help="Enable BM25 sentence extraction before tokenization")
    p.add_argument("--extraction-top-k", type=int, default=10,
                   help="Number of abstract sentences to keep per passage")
    p.set_defaults(amp=True, extract_sentences=False)
    return p.parse_args()


if __name__ == "__main__":
    main()
