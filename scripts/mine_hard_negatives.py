"""
Mines hard negatives for bi-encoder fine-tuning.

For each train query:
  1. Retrieves top candidates via dense retrieval (uses cached embeddings)
  2. Removes the ground truth paper
  3. Takes top-k as hard negatives

Output: data/hard_negatives_v2.jsonl
  {"anchor": "query: ...", "positive": "passage: ...", "negatives": ["passage: ...", ...]}

Usage:
    # Mine with fine-tuned model (auto-detected from e5-large-finetuned/):
    python scripts/mine_hard_negatives.py

    # Explicit model path or HuggingFace ID:
    python scripts/mine_hard_negatives.py --dense-model e5-large-finetuned/final
    python scripts/mine_hard_negatives.py --langs en --top-k-neg 7
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    DATASET_NAME,
    load_collection,
    load_queries,
    prepare_collection,
    prepare_queries,
)
from src.retriever import DenseRetriever


def _find_st_model(base: Path, prefer: str = "final") -> Path | None:
    candidates = [p.parent.parent for p in base.rglob("1_Pooling/config.json")]
    if not candidates:
        return None
    preferred = [p for p in candidates if p.name == prefer]
    return preferred[0] if preferred else candidates[-1]


def _resolve_model(model_arg: str | None) -> str:
    """Auto-detect local fine-tuned model; fall back to HuggingFace base model."""
    if model_arg:
        return model_arg
    for base in (Path("checkpoints/e5-large-finetuned"), Path("e5-large-finetuned")):
        local = _find_st_model(base)
        if local:
            print(f"Auto-detected fine-tuned model: {local}")
            return str(local)
    print("Fine-tuned model not found — using base intfloat/multilingual-e5-large")
    return "intfloat/multilingual-e5-large"


def mine(args):
    model_name = _resolve_model(args.dense_model)
    p = Path(model_name)
    if p.exists():
        model_slug = f"{p.parent.name}_{p.name}"   # e.g. "e5-large-finetuned_final"
    else:
        model_slug = model_name.replace("/", "_")
    cache_path = f".cache/corpus_embeddings_{model_slug}.npy"

    print("Loading collection...")
    collection = load_dataset_collection()
    passage_texts, _, _, rerank_texts, collection_keys = prepare_collection(collection)

    print(f"Loading dense model: {model_name}")
    dense = DenseRetriever(model_name=model_name, device=args.device)
    dense.encode_corpus(passage_texts, batch_size=args.corpus_batch_size, cache_path=cache_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for lang in args.langs:
            print(f"\n=== {lang.upper()} / train ===")
            train_data, _ = load_queries(lang)

            translation_cache = None
            if args.use_augmented:
                tc = Path(f".cache/augmented/{lang}_train.json")
                if tc.exists():
                    translation_cache = str(tc)
                    print(f"  Using augmented queries from {tc}")
                else:
                    print(f"  WARNING: augmented cache not found: {tc}")
            elif args.use_translations and lang != "en":
                tc = Path(f".cache/translations/{lang}_train.json")
                if tc.exists():
                    translation_cache = str(tc)
                    print(f"  Using translations from {tc}")

            raw_queries, _, query_texts, indices, true_pubkeys, _ = prepare_queries(
                train_data, translation_cache
            )

            if true_pubkeys is None:
                print("  No ground truth — skipping")
                continue

            print(f"  Encoding {len(raw_queries)} queries...")
            query_embeddings = dense.encode_queries(query_texts, batch_size=args.query_batch_size)

            print("  Computing similarities...")
            sims = query_embeddings @ dense.corpus_embeddings.T  # (n_queries, n_corpus)

            print("  Mining hard negatives...")
            n_written = 0
            for i in tqdm(range(len(raw_queries)), desc=f"  {lang}"):
                true_key = true_pubkeys[i]
                true_idx = np.where(collection_keys == true_key)[0]
                if len(true_idx) == 0:
                    continue
                true_idx = true_idx[0]

                # Top-100 excluding ground truth
                row_sims = sims[i].copy()
                row_sims[true_idx] = -np.inf
                top_neg_indices = np.argsort(-row_sims)[: args.top_k_neg]

                anchor = f"query: {raw_queries[i]}"
                positive = passage_texts[true_idx]
                negatives = [passage_texts[j] for j in top_neg_indices]

                out_f.write(json.dumps({
                    "anchor": anchor,
                    "positive": positive,
                    "negatives": negatives,
                    "lang": lang,
                }, ensure_ascii=False) + "\n")
                n_written += 1

            print(f"  Written {n_written} examples for {lang}")
            total_written += n_written

    print(f"\nTotal: {total_written} training examples → {output_path}")


def load_dataset_collection():
    from datasets import load_dataset
    return load_dataset(
        "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims",
        "collection",
    )["collection"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["en", "de", "fr"])
    p.add_argument("--dense-model", default=None,
                   help="Model path or HuggingFace ID. Default: auto-detect e5-large-finetuned/")
    p.add_argument("--top-k-neg", type=int, default=7, help="Hard negatives per query")
    p.add_argument("--output", default="data/hard_negatives_v2.jsonl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--corpus-batch-size", type=int, default=64)
    p.add_argument("--query-batch-size", type=int, default=64)
    p.add_argument("--use-translations", action="store_true", default=True,
                   help="Use translated queries for DE/FR if available (default: True)")
    p.add_argument("--no-translations", dest="use_translations", action="store_false")
    p.add_argument("--use-augmented", action="store_true",
                   help="Use LLM-augmented queries (all langs) if available")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mine(args)
