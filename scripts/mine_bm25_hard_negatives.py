"""
Mine BM25 hard negatives for bi-encoder training.

For each train query:
  1. Build BM25 index over the paper collection
  2. Retrieve top candidates with BM25
  3. Remove the ground-truth paper
  4. Keep top-k remaining papers as hard negatives

Output schema matches the dense hard-negative files:
  {"anchor": "query: ...", "positive": "passage: ...", "negatives": ["passage: ...", ...]}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src2.data import (
    load_collection,
    load_rewrite_queries,
    load_split,
    prepare_collection,
    prepare_queries,
)
from src2.retrievers import BM25Retriever, HybridRetriever


def _load_json_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine BM25 hard negatives")
    parser.add_argument("--langs", nargs="+", default=["en", "de", "fr"])
    parser.add_argument("--top-k-neg", type=int, default=5, help="BM25 hard negatives per query")
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=100,
        help="BM25 candidates to inspect before dropping the positive",
    )
    parser.add_argument("--output", default="data/hard_negatives_bm25_translated.jsonl")
    parser.add_argument("--translation-dir", default=".cache/translations")
    parser.add_argument("--rewrite-dir", default=".cache/augmented")
    parser.add_argument("--use-translations", action="store_true", default=True)
    parser.add_argument("--no-translations", dest="use_translations", action="store_false")
    parser.add_argument("--quote-extract", action="store_true", default=True)
    parser.add_argument("--no-quote-extract", dest="quote_extract", action="store_false")
    parser.add_argument("--bm25-rewrite-view", action="store_true", default=False)
    parser.add_argument("--bm25-tokenizer-name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--bm25-cache-path", default=".cache/src2_bm25_scibert_t4_v1_a1_au0")
    parser.add_argument("--bm25-title-boost", type=int, default=4)
    parser.add_argument("--bm25-venue-boost", type=int, default=1)
    parser.add_argument("--bm25-include-abstract", action="store_true", default=True)
    parser.add_argument("--no-bm25-include-abstract", dest="bm25_include_abstract", action="store_false")
    parser.add_argument("--bm25-include-authors", action="store_true", default=False)
    parser.add_argument("--bm25-numeric-boost", type=int, default=4)
    parser.add_argument("--rrf-k", type=int, default=10, help="Used only when --bm25-rewrite-view is enabled")
    return parser.parse_args()


def mine(args: argparse.Namespace) -> None:
    print("Loading collection...")
    collection = load_collection()
    passage_texts, bm25_texts, collection_keys = prepare_collection(
        collection,
        title_boost=args.bm25_title_boost,
        venue_boost=args.bm25_venue_boost,
        include_abstract=args.bm25_include_abstract,
        include_authors=args.bm25_include_authors,
    )

    bm25 = BM25Retriever(
        bm25_texts,
        tokenizer_name=args.bm25_tokenizer_name,
        cache_path=args.bm25_cache_path,
        numeric_boost=args.bm25_numeric_boost,
    )
    hybrid = None
    if args.bm25_rewrite_view:
        hybrid = HybridRetriever(
            bm25=bm25,
            dense=None,
            rrf_k=args.rrf_k,
            bm25_weight=1.0,
            dense_weight=1.0,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for lang in args.langs:
            print(f"\n=== {lang.upper()} / train ===")
            data = load_split(lang, "train")
            raw_queries, _, _, _, true_pubkeys = prepare_queries(
                data,
                lang=lang,
                split="train",
                use_translations=args.use_translations,
                translation_dir=args.translation_dir,
                quote_extract=args.quote_extract,
            )
            if true_pubkeys is None:
                print("  No ground truth - skipping")
                continue

            rewrite_queries = None
            if args.bm25_rewrite_view:
                rewrite_queries = load_rewrite_queries(
                    data,
                    lang=lang,
                    split="train",
                    rewrite_dir=args.rewrite_dir,
                )

            print("  Retrieving BM25 candidates...")
            if hybrid is not None:
                top_indices = hybrid.retrieve_batch(
                    raw_queries,
                    query_embeddings=None,
                    top_k=max(args.top_k_retrieve, args.top_k_neg + 1),
                    bm25_rewrite_queries=rewrite_queries,
                )
            else:
                top_indices = bm25.retrieve_batch(
                    raw_queries,
                    top_k=max(args.top_k_retrieve, args.top_k_neg + 1),
                )

            print("  Mining hard negatives...")
            written_lang = 0
            for i in tqdm(range(len(raw_queries)), desc=f"  {lang}"):
                true_key = true_pubkeys[i]
                true_idx_matches = np.where(collection_keys == true_key)[0]
                if len(true_idx_matches) == 0:
                    continue
                true_idx = int(true_idx_matches[0])

                neg_indices: list[int] = []
                seen: set[int] = set()
                for idx in top_indices[i]:
                    idx = int(idx)
                    if idx == true_idx or idx in seen:
                        continue
                    neg_indices.append(idx)
                    seen.add(idx)
                    if len(neg_indices) >= args.top_k_neg:
                        break

                if not neg_indices:
                    continue

                record = {
                    "index": int(data["index"][i]),
                    "anchor": f"query: {raw_queries[i]}",
                    "positive": passage_texts[true_idx],
                    "negatives": [passage_texts[j] for j in neg_indices],
                    "lang": lang,
                    "query_mode": "translated" if args.use_translations else "raw",
                    "bm25_source": "bm25+rewrite" if args.bm25_rewrite_view else "bm25",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_lang += 1

            print(f"  Written {written_lang} examples for {lang}")
            total_written += written_lang

    print(f"\nTotal: {total_written} training examples -> {output_path}")


if __name__ == "__main__":
    mine(parse_args())
