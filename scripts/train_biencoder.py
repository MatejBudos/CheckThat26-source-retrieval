"""
Fine-tunes the bi-encoder using MultipleNegativesRankingLoss + hard negatives.
Uses a plain PyTorch training loop for reliable GPU usage on Windows.

Usage:
    python scripts/train_biencoder.py
    python scripts/train_biencoder.py --epochs 3 --batch-size 8
    python scripts/train_biencoder.py --grad-accum 4 --batch-size 8
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


class TripletDataset(TorchDataset):
    def __init__(self, path: str):
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append((
                    ex["anchor"],
                    ex["positive"],
                    ex["negatives"],  # keep all; sample in __getitem__ each epoch
                ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        anchor, positive, negatives = self.examples[idx]
        return anchor, positive, random.choice(negatives)


def mnrl_loss(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
    """In-batch negatives loss: each positive is the diagonal, rest are negatives."""
    scores = embeddings_a @ embeddings_b.T  # (B, B)
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores * 20.0, labels)  # scale=20 is standard for E5


def encode_batch(
    model, tokenizer, texts: list[str], device: str, max_length: int
) -> torch.Tensor:
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(device)
    with torch.amp.autocast("cuda"):
        out = model(**enc)
    return F.normalize(out.last_hidden_state[:, 0], dim=-1)


def encode_combined(
    model, tokenizer,
    anchors: list[str], positives: list[str], negatives: list[str],
    device: str, anchor_max_length: int, passage_max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two forward passes: anchors at short max_length, passages (pos+neg) combined at full length.
    Tweets are ~60-80 tokens; padding them to 256 wastes 3x compute."""
    B = len(anchors)
    emb_a = encode_batch(model, tokenizer, list(anchors), device, anchor_max_length)
    all_passages = list(positives) + list(negatives)
    emb_pn = encode_batch(model, tokenizer, all_passages, device, passage_max_length)
    return emb_a, emb_pn[:B], emb_pn[B:]


def main():
    args = parse_args()
    random.seed(42)
    device = args.device

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading model: {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    # Recomputes activations during backward instead of storing them.
    # Saves ~3-4 GB of VRAM (24 layers × batch × seq tensors) at ~30% extra compute cost.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    torch.backends.cudnn.benchmark = True

    print(f"Loading training data: {args.data}")
    dataset = TripletDataset(args.data)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        num_workers=0, pin_memory=False,
    )
    print(f"  {len(dataset)} examples, {len(loader)} steps/epoch")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, fused=True
    )
    grad_accum = args.grad_accum
    effective_batch = args.batch_size * grad_accum
    total_optimizer_steps = (len(loader) // grad_accum) * args.epochs
    warmup_steps = int(total_optimizer_steps * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device != "cpu"))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nTraining: {args.epochs} epochs, batch={args.batch_size}, "
        f"grad_accum={grad_accum} (effective={effective_batch}), "
        f"max_len={args.max_length}, lr={args.lr}"
    )

    optimizer_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (anchors, positives, negatives) in enumerate(pbar):
            emb_a, emb_p, emb_n = encode_combined(
                model, tokenizer,
                list(anchors), list(positives), list(negatives),
                device, args.anchor_max_length, args.max_length,
            )

            emb_docs = torch.cat([emb_p, emb_n], dim=0)  # (2B, D)
            loss = mnrl_loss(emb_a, emb_docs) / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            epoch_loss += loss.item() * grad_accum
            pbar.set_postfix(
                loss=f"{loss.item() * grad_accum:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")

        ckpt = output_dir / f"epoch-{epoch}"
        # unwrap compiled model for saving
        raw_model = getattr(model, "_orig_mod", model)
        raw_model.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        print(f"  Saved checkpoint → {ckpt}")

    from sentence_transformers import SentenceTransformer, models
    word_embedding = models.Transformer(str(output_dir / f"epoch-{args.epochs}"))
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode="cls")
    st_model = SentenceTransformer(modules=[word_embedding, pooling])
    final_path = output_dir / "final"
    st_model.save(str(final_path))
    print(f"\nModel saved (sentence-transformers format) → {final_path}")
    print(f'Use in pipeline with: dense_model: "{final_path}"')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intfloat/multilingual-e5-base")
    p.add_argument("--data", default="data/hard_negatives.jsonl")
    p.add_argument("--output-dir", default="checkpoints/e5-finetuned")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--max-length", type=int, default=128,
                   help="Max token length for passages (positive/negative)")
    p.add_argument("--anchor-max-length", type=int, default=96,
                   help="Max token length for query tweets (much shorter than passages)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for ~20%% extra speedup (PyTorch 2.x, first epoch slower)")
    return p.parse_args()


if __name__ == "__main__":
    main()
