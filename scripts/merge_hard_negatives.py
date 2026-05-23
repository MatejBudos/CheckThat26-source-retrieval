"""
Merge dense-mined and BM25-mined hard negatives.

Typical use:
  python scripts/merge_hard_negatives.py ^
    --dense data/hard_negatives_e5_large_translated.jsonl ^
    --bm25 data/hard_negatives_bm25_translated.jsonl ^
    --bm25-per-query 1 ^
    --output data/hard_negatives_dense_plus_bm25_1.jsonl
"""

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_records(path: Path) -> list[dict]:
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


def _dedupe_negatives(negatives: list[str], positive: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = {positive}
    for neg in negatives:
        if neg not in seen:
            output.append(neg)
            seen.add(neg)
    return output


def _select_unique_bm25_negatives(
    dense_negatives: list[str],
    bm25_negatives: list[str],
    positive: str,
    bm25_per_query: int,
) -> tuple[list[str], int]:
    if bm25_per_query <= 0:
        return [], 0

    selected: list[str] = []
    seen: set[str] = set(dense_negatives)
    seen.add(positive)

    inspected = 0
    for neg in bm25_negatives:
        inspected += 1
        if neg in seen:
            continue
        selected.append(neg)
        seen.add(neg)
        if len(selected) >= bm25_per_query:
            break
    return selected, inspected


def _inject_missing_dense_indices(records: list[dict]) -> list[dict]:
    from src2.data import load_split

    if not records or all("index" in record for record in records):
        return records

    indices_by_lang: dict[str, list[int]] = {}
    for lang in ("en", "de", "fr"):
        split = load_split(lang, "train")
        indices_by_lang[lang] = [int(x) for x in split["index"]]

    lang_offsets = {lang: 0 for lang in indices_by_lang}
    enriched: list[dict] = []
    for record in records:
        if "index" in record:
            enriched.append(record)
            continue
        lang = record.get("lang")
        if lang not in indices_by_lang:
            raise KeyError(f"Cannot infer index for record with lang={lang!r}")
        pos = lang_offsets[lang]
        if pos >= len(indices_by_lang[lang]):
            raise IndexError(f"Ran out of train indices while inferring dense indices for lang={lang}")
        enriched_record = dict(record)
        enriched_record["index"] = indices_by_lang[lang][pos]
        lang_offsets[lang] += 1
        enriched.append(enriched_record)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge dense and BM25 hard negatives")
    parser.add_argument("--dense", required=True, help="Dense hard negatives JSONL/JSON")
    parser.add_argument("--bm25", required=True, help="BM25 hard negatives JSONL/JSON")
    parser.add_argument("--output", required=True, help="Merged JSONL path")
    parser.add_argument(
        "--bm25-per-query",
        type=int,
        default=1,
        help="How many BM25 negatives to append to each dense example",
    )
    parser.add_argument(
        "--match-on",
        choices=["lang_positive", "index_lang", "anchor", "anchor_positive"],
        default="lang_positive",
        help="How to align dense and BM25 examples",
    )
    parser.add_argument(
        "--target-total-negatives",
        type=int,
        default=None,
        help="Optional expected final negatives count; reported in summary only",
    )
    return parser.parse_args()


def _make_key(record: dict, match_on: str) -> tuple:
    if match_on == "lang_positive":
        if "lang" not in record or "positive" not in record:
            raise KeyError("lang_positive matching requires both 'lang' and 'positive' fields")
        return (record["lang"], record["positive"])
    if match_on == "index_lang":
        if "lang" not in record or "index" not in record:
            raise KeyError("index_lang matching requires both 'lang' and 'index' fields")
        return (record["lang"], int(record["index"]))
    if match_on == "anchor":
        return (record["anchor"],)
    return (record["anchor"], record["positive"])


def merge(args: argparse.Namespace) -> None:
    dense_records = _load_records(Path(args.dense))
    if (
        args.match_on == "index_lang"
        and dense_records
        and any("index" not in record for record in dense_records)
    ):
        dense_records = _inject_missing_dense_indices(dense_records)
    bm25_records = _load_records(Path(args.bm25))

    bm25_map = {_make_key(record, args.match_on): record for record in bm25_records}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched = 0
    exact_target_count = 0
    shortfall_count = 0
    bm25_inspected_total = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for dense_record in dense_records:
            try:
                key = _make_key(dense_record, args.match_on)
                bm25_record = bm25_map.get(key)
            except KeyError:
                bm25_record = None

            merged_record = dict(dense_record)
            dense_negs = list(dense_record.get("negatives", []))
            bm25_negs: list[str] = []
            bm25_inspected = 0
            if bm25_record is not None:
                bm25_negs, bm25_inspected = _select_unique_bm25_negatives(
                    dense_negs,
                    list(bm25_record.get("negatives", [])),
                    dense_record["positive"],
                    max(args.bm25_per_query, 0),
                )
                matched += 1
                bm25_inspected_total += bm25_inspected

            merged_record["negatives"] = _dedupe_negatives(
                dense_negs + bm25_negs,
                dense_record["positive"],
            )
            merged_record["bm25_added"] = len(bm25_negs)
            merged_record["dense_negatives_count"] = len(dense_negs)
            merged_record["final_negatives_count"] = len(merged_record["negatives"])
            merged_record["bm25_candidates_inspected"] = bm25_inspected
            merged_record["merged_from"] = {
                "dense": str(args.dense),
                "bm25": str(args.bm25),
            }
            if args.target_total_negatives is not None:
                merged_record["target_total_negatives"] = args.target_total_negatives
                if merged_record["final_negatives_count"] == args.target_total_negatives:
                    exact_target_count += 1
                else:
                    shortfall_count += 1
            out_f.write(json.dumps(merged_record, ensure_ascii=False) + "\n")

    print(f"Dense examples: {len(dense_records)}")
    print(f"BM25 examples: {len(bm25_records)}")
    print(f"Matched examples: {matched}")
    if matched:
        print(f"Avg BM25 candidates inspected per matched example: {bm25_inspected_total / matched:.2f}")
    if args.target_total_negatives is not None:
        print(f"Examples at target {args.target_total_negatives}: {exact_target_count}")
        print(f"Examples below target {args.target_total_negatives}: {shortfall_count}")
    print(f"Merged output: {output_path}")


if __name__ == "__main__":
    merge(parse_args())
