"""
Entry point for running the retrieval pipeline.

Examples:
    # Evaluate on dev set with default (hybrid + reranker) config
    python scripts/run_pipeline.py

    # BM25 baseline only
    python scripts/run_pipeline.py --config configs/bm25_only.yaml

    # Dense-only, single language, test split
    python scripts/run_pipeline.py --config configs/dense_only.yaml --split test --langs en

    # Override reranker at runtime
    python scripts/run_pipeline.py --no-reranker --top-k 50
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CheckThat! 2026 Task 1 pipeline")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--langs", nargs="+", default=None, help="Override langs (e.g. de fr en)")
    p.add_argument("--no-reranker", action="store_true", help="Skip re-ranking stage")
    p.add_argument("--no-bm25", action="store_true", help="Skip BM25 retrieval")
    p.add_argument("--top-k", type=int, default=None, help="Override top_k_retrieve")
    p.add_argument("--device", default=None, help="Override device (cuda / cpu)")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    p.add_argument("--sample", type=int, default=None, help="Evaluate on random N queries per lang (for quick testing)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for --sample (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config: dict = yaml.safe_load(f)

    if args.langs:
        config["langs"] = args.langs
    if args.no_reranker:
        config["use_reranker"] = False
    if args.no_bm25:
        config["use_bm25"] = False
    if args.top_k is not None:
        config["top_k_retrieve"] = args.top_k
    if args.device:
        config["device"] = args.device
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.sample:
        config["sample"] = args.sample
        config["sample_seed"] = args.seed

    pipeline = Pipeline(config)
    results = pipeline.run(split=args.split)

    if results:
        print("\n=== Summary ===")
        all_mrr5 = []
        all_s1_mrr5 = []
        for key, metrics in results.items():
            s1_mrr5  = metrics.get("stage1_mrr@5")
            s1_mrr1  = metrics.get("stage1_mrr@1")
            s1_rec20 = metrics.get("stage1_recall@20")
            s1_str = ""
            if s1_mrr5 is not None:
                s1_str += f"  S1_MRR@5={s1_mrr5:.4f}"
                all_s1_mrr5.append(s1_mrr5)
            if s1_mrr1 is not None:
                s1_str += f"  S1_MRR@1={s1_mrr1:.4f}"
            if s1_rec20 is not None:
                s1_str += f"  S1_Recall@20={s1_rec20:.4f}"
            print(f"  {key}: MRR@5={metrics['mrr@5']:.4f}  MRR@1={metrics['mrr@1']:.4f}{s1_str}")
            all_mrr5.append(metrics["mrr@5"])
        if len(all_mrr5) > 1:
            avg_str = f"  Avg_MRR@5={sum(all_mrr5)/len(all_mrr5):.4f}"
            if all_s1_mrr5:
                avg_str += f"  Avg_S1_MRR@5={sum(all_s1_mrr5)/len(all_s1_mrr5):.4f}"
            print(f"  {'---'}\n  AVERAGE:{avg_str}")


if __name__ == "__main__":
    main()
