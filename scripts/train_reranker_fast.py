"""
Fast reranker fine-tune — designed for quick iteration on small data.

Differences vs train_reranker.py:
  - Default model: BAAI/bge-reranker-base (XLM-RoBERTa 278M, ~2x faster than bge-v2-m3)
  - Default 2 000 pairs, 1 neg/query, max_length=64, no sentence extraction
  - Expected runtime: 5–10 min on a laptop GPU

Usage:
    python scripts/train_reranker_fast.py
    python scripts/train_reranker_fast.py --max-samples 4000 --epochs 2
    python scripts/train_reranker_fast.py --model BAAI/bge-reranker-v2-m3 --max-samples 8000
"""

import argparse
import functools
import json
import math
import random
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_pairs(data_path: Path, negs_per_query: int, seed: int, max_samples: int):
    rng = random.Random(seed)
    lines = Path(data_path).read_text(encoding="utf-8").splitlines()

    pairs_per_line = 1 + negs_per_query
    if max_samples and len(lines) > max_samples // pairs_per_line:
        lines = rng.sample(lines, max_samples // pairs_per_line)

    queries, docs, labels = [], [], []
    for line in lines:
        item = json.loads(line)
        q = item["anchor"].removeprefix("query: ")
        pos = item["positive"].removeprefix("passage: ")
        negs = [n.removeprefix("passage: ") for n in item["negatives"]]

        queries.append(q)
        docs.append(pos)
        labels.append(1.0)

        for neg in rng.sample(negs, min(negs_per_query, len(negs))):
            queries.append(q)
            docs.append(neg)
            labels.append(0.0)

    return queries, docs, labels


class PairDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        self.attention_mask = [torch.tensor(x, dtype=torch.long) for x in attention_mask]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def collate_fn(batch):
    ids, masks, lbls = zip(*batch)
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return ids, masks, torch.stack(lbls)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.0f}M")

    print(f"Loading data: {args.data}")
    queries, docs, labels = build_pairs(
        Path(args.data), args.negs_per_query, args.seed, args.max_samples
    )
    n_pos = sum(1 for l in labels if l == 1.0)
    print(f"  {n_pos} pos + {len(labels) - n_pos} neg = {len(labels)} pairs")

    print(f"Tokenizing (max_length={args.max_length})...")
    enc = tokenizer(
        queries, docs,
        padding=False,
        truncation=True,
        max_length=args.max_length,
    )

    dataset = PairDataset(enc["input_ids"], enc["attention_mask"], labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    total_steps = len(loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
        fused=(device.type == "cuda"),
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nTraining: {args.epochs} epoch(s) × {len(loader)} steps = {total_steps} steps"
        f" | batch={args.batch_size} | lr={args.lr} | amp={scaler is not None}"
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

        print(f"  Epoch {epoch + 1}/{args.epochs} done ({(time.time() - t0) / 60:.1f} min elapsed)")

    elapsed_min = (time.time() - t0) / 60
    print(f"\nTotal training time: {elapsed_min:.1f} min")
    print(f"Saving to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Use in pipeline: reranker_model: "{output_dir}"')
    print(f"\nQuick eval command:")
    print(f"  python scripts/run_pipeline.py --config configs/default.yaml --sample 500 reranker_model={output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast reranker fine-tune for small-data iteration")
    p.add_argument("--model", default="BAAI/bge-reranker-base",
                   help="Base model. bge-reranker-base=278M (fast), bge-reranker-v2-m3=568M (better)")
    p.add_argument("--data", default="data/hard_negatives.jsonl")
    p.add_argument("--output-dir", default="checkpoints/reranker-fast")
    p.add_argument("--max-samples", type=int, default=2000,
                   help="Total training pairs (0=all). 2000 ~5 min, 8000 ~20 min")
    p.add_argument("--epochs", type=int, default=3,
                   help="More epochs compensate for fewer samples")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64,
                   help="Tokens per pair. 64 covers tweet + paper title well")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--negs-per-query", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    main()
