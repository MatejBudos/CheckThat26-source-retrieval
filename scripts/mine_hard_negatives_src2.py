"""
Mine hard negatives using the current src2 hybrid retrieval configuration.

This script uses the same query preparation and retrieval settings as `src2`,
but writes training triples for bi-encoder fine-tuning:

  {"anchor": "query: ...", "positive": "passage: ...", "negatives": ["passage: ...", ...]}

Unlike the older `scripts/mine_hard_negatives.py`, this script reads a src2 YAML
config and therefore respects settings such as:
  - dense_model / dense_model_by_lang
  - translation_dir
  - multi_query
  - bm25_rewrite_view / dense_rewrite_view
  - current BM25 and dense weights
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
    p = argparse.ArgumentParser(description="Mine hard negatives with src2 hybrid retrieval")
    p.add_argument("--config", default="configs/src2_translated_hybrid.yaml")
    p.add_argument("--split", default="train", choices=["train", "dev", "test"])
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--top-k-neg", type=int, default=7, help="Hard negatives per query")
    p.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="How many stage-1 candidates to retrieve before filtering ground truth. 0 => use max(config top_k_retrieve, top_k_neg + 5).",
    )
    p.add_argument(
        "--retrieval-source",
        choices=["fused", "union"],
        default="fused",
        help="Use fused hybrid ranking or candidate union when mining negatives.",
    )
    p.add_argument(
        "--anchor-query-mode",
        choices=["raw", "bm25"],
        default="raw",
        help="Which prepared query string to write into the JSONL anchor.",
    )
    p.add_argument("--output", default="data/hard_negatives_src2_hybrid.jsonl")
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
    collection_index_by_pubkey = {
        str(pubkey): idx for idx, pubkey in enumerate(pipeline.collection_keys.tolist())
    }
    top_n = args.top_n or max(int(config.get("top_k_retrieve", 20)), args.top_k_neg + 5)
    print(f"Mining with retrieval_source={args.retrieval_source}, top_n={top_n}, top_k_neg={args.top_k_neg}")

    total_written = 0
    total_short = 0

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

            if true_pubkeys is None:
                print("  No ground truth available for this split - skipping")
                continue

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

            retrieval_kwargs = dict(
                query_embeddings_2=query_embeddings_2,
                query_embeddings_3=query_embeddings_3,
                bm25_rewrite_queries=bm25_rewrite_queries,
                bm25_top_k=config.get("bm25_top_k_retrieve"),
                dense_top_k=config.get("dense_top_k_retrieve"),
            )

            print(f"  Stage 1 retrieval (top-{top_n})...")
            if args.retrieval_source == "union":
                top_indices = hybrid.candidate_union_batch(
                    bm25_queries,
                    query_embeddings,
                    top_k=top_n,
                    **retrieval_kwargs,
                )
            else:
                top_indices = hybrid.retrieve_batch(
                    bm25_queries,
                    query_embeddings,
                    top_n,
                    **retrieval_kwargs,
                )

            written = 0
            short = 0
            for i, row in enumerate(top_indices):
                true_pubkey = str(true_pubkeys[i])
                negatives: list[str] = []
                for doc_idx in row:
                    pubkey = str(pipeline.collection_keys[int(doc_idx)])
                    if pubkey == true_pubkey:
                        continue
                    negatives.append(pipeline.passage_texts[int(doc_idx)])
                    if len(negatives) >= args.top_k_neg:
                        break

                if not negatives:
                    continue

                if len(negatives) < args.top_k_neg:
                    short += 1

                anchor_text = raw_queries[i] if args.anchor_query_mode == "raw" else bm25_queries[i]
                true_idx = collection_index_by_pubkey.get(true_pubkey)
                if true_idx is None:
                    continue

                out_f.write(json.dumps({
                    "anchor": f"query: {anchor_text}",
                    "positive": pipeline.passage_texts[true_idx],
                    "negatives": negatives,
                    "lang": lang,
                    "split": args.split,
                    "index": int(indices[i]),
                    "query_mode": args.anchor_query_mode,
                    "retrieval_source": args.retrieval_source,
                }, ensure_ascii=False) + "\n")
                written += 1

            print(f"  Written {written} examples for {lang} (short<{args.top_k_neg}: {short})")
            total_written += written
            total_short += short

    print(f"\nTotal: {total_written} examples -> {output_path}")
    print(f"Examples with fewer than {args.top_k_neg} negatives: {total_short}")


if __name__ == "__main__":
    main()
