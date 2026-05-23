"""
Export Stage 1 retrieval candidates from src2 for reranker training.

One JSONL row per query:
{
  "lang": "de",
  "split": "train",
  "index": "123",
  "query": "...",
  "true_pubkey": "...",
  "true_text": "passage: title. venue. abstract. authors",
  "candidates": [
    {"rank": 1, "pubkey": "...", "text": "passage: ..."},
    ...
  ]
}

Usage:
  python scripts/export_stage1_candidates_src2.py --config configs/src2_translated_hybrid.yaml --split train --top-n 50 --output exports/src2_stage1_train_top50.jsonl
  python scripts/export_stage1_candidates_src2.py --config configs/src2_translated_hybrid.yaml --split dev --top-n 50 --output exports/src2_stage1_dev_top50.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src2.data import load_rewrite_queries, load_split, prepare_queries
from src2.pipeline import Pipeline
from src2.retrievers import HybridRetriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Stage 1 candidates from src2")
    p.add_argument("--config", default="configs/src2_translated_hybrid.yaml")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--output", default="exports/src2_stage1_candidates.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = dict(config)
    config["use_reranker"] = False
    if args.langs:
        config["langs"] = args.langs

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading src2 pipeline...")
    pipeline = Pipeline(config)
    true_text_by_pubkey = {
        str(pubkey): text
        for pubkey, text in zip(pipeline.collection_keys.tolist(), pipeline.rerank_texts)
    }

    with output_path.open("w", encoding="utf-8") as out_f:
        for lang in config.get("langs", ["de", "fr", "en"]):
            print(f"\n=== {lang.upper()} / {args.split} ===")
            dense = pipeline._get_dense(lang)
            hybrid = HybridRetriever(
                bm25=pipeline.bm25,
                dense=dense,
                rrf_k=config.get("rrf_k", 60),
                bm25_weight=config.get("bm25_weight", 0.25),
                dense_weight=config.get("dense_weight", 1.0),
                bm25_rewrite_weight=config.get("bm25_rewrite_weight"),
                dense_weight_2=config.get("dense_weight_2"),
                dense_weight_3=config.get("dense_weight_3"),
            )

            data = load_split(lang, args.split)
            raw_queries, bm25_queries, dense_queries, original_dense_queries, indices, true_pubkeys = prepare_queries(
                data,
                lang=lang,
                split=args.split,
                use_translations=config.get("use_translations", True),
                translation_dir=config.get("translation_dir", ".cache/translations"),
                quote_extract=config.get("bm25_quote_extract", False),
                query_cleanup=config.get("query_cleanup", False),
                query_cleanup_langs=config.get("query_cleanup_langs"),
                bm25_concat_original=config.get("bm25_concat_original", False),
                bm25_concat_original_langs=config.get("bm25_concat_original_langs"),
                return_bm25_queries=True,
            )

            bm25_rewrite_queries = None
            dense_rewrite_queries = None
            if config.get("bm25_rewrite_view", False):
                bm25_rewrite_queries = load_rewrite_queries(
                    data,
                    lang=lang,
                    split=args.split,
                    rewrite_dir=config.get("rewrite_dir", ".cache/augmented"),
                )
            if config.get("dense_rewrite_view", False):
                dense_rewrite_langs = config.get("dense_rewrite_langs")
                if dense_rewrite_langs is None or lang in dense_rewrite_langs:
                    dense_rewrite_queries = load_rewrite_queries(
                        data,
                        lang=lang,
                        split=args.split,
                        rewrite_dir=config.get("rewrite_dir", ".cache/augmented"),
                    )

            query_embeddings = None
            if dense is not None:
                print(f"  Encoding {len(dense_queries)} queries...")
                query_embeddings = dense.encode_queries(
                    dense_queries,
                    batch_size=config.get("query_batch_size", 64),
                )

            query_embeddings_2 = None
            if dense is not None and config.get("multi_query", False):
                if any(a != b for a, b in zip(dense_queries, original_dense_queries)):
                    print("  Encoding original tweet queries for multi-query dense view...")
                    query_embeddings_2 = dense.encode_queries(
                        original_dense_queries,
                        batch_size=config.get("query_batch_size", 64),
                    )

            query_embeddings_3 = None
            if dense is not None and dense_rewrite_queries is not None:
                print("  Encoding rewrite queries for dense rewrite view...")
                rewrite_dense_queries = [f"query: {text}" for text in dense_rewrite_queries]
                query_embeddings_3 = dense.encode_queries(
                    rewrite_dense_queries,
                    batch_size=config.get("query_batch_size", 64),
                )

            print(f"  Stage 1 retrieval (top-{args.top_n})...")
            top_indices = hybrid.retrieve_batch(
                bm25_queries,
                query_embeddings,
                args.top_n,
                query_embeddings_2=query_embeddings_2,
                query_embeddings_3=query_embeddings_3,
                bm25_rewrite_queries=bm25_rewrite_queries,
                bm25_top_k=config.get("bm25_top_k_retrieve"),
                dense_top_k=config.get("dense_top_k_retrieve"),
            )

            for i, row in enumerate(top_indices):
                record = {
                    "lang": lang,
                    "split": args.split,
                    "index": indices[i],
                    "query": raw_queries[i],
                    "candidates": [
                        {
                            "rank": rank + 1,
                            "pubkey": str(pipeline.collection_keys[int(doc_idx)]),
                            "text": pipeline.rerank_texts[int(doc_idx)],
                        }
                        for rank, doc_idx in enumerate(row)
                    ],
                }
                if true_pubkeys is not None:
                    record["true_pubkey"] = str(true_pubkeys[i])
                    record["true_text"] = true_text_by_pubkey.get(str(true_pubkeys[i]), "")
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved Stage 1 export -> {output_path}")


if __name__ == "__main__":
    main()
