import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src2.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal translated-only retrieval pipeline")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = Pipeline(config)
    results = pipeline.run(args.split)

    if results:
        print("\n=== Summary ===")
        all_mrr5 = []
        all_recall5 = []
        all_stage1_mrr5 = []
        all_stage1_recall10 = []
        for key, metrics in results.items():
            stage1 = ""
            if "stage1_mrr@5" in metrics:
                stage1 = (
                    f"  S1_MRR@5={metrics['stage1_mrr@5']:.4f}"
                    f"  S1_Recall@10={metrics.get('stage1_recall@10', 0.0):.4f}"
                    f"  S1_Recall@20={metrics.get('stage1_recall@20', 0.0):.4f}"
                )
                all_stage1_mrr5.append(metrics["stage1_mrr@5"])
                all_stage1_recall10.append(metrics.get("stage1_recall@10", 0.0))
            print(
                f"  {key}: MRR@1={metrics['mrr@1']:.4f}"
                f"  MRR@5={metrics['mrr@5']:.4f}"
                f"  Recall@5={metrics['recall@5']:.4f}"
                f"{stage1}"
            )
            all_mrr5.append(metrics["mrr@5"])
            all_recall5.append(metrics["recall@5"])
        if len(all_mrr5) > 1:
            avg_line = (
                f"  Avg_MRR@5={sum(all_mrr5)/len(all_mrr5):.4f}"
                f"  Avg_Recall@5={sum(all_recall5)/len(all_recall5):.4f}"
            )
            if all_stage1_mrr5:
                avg_line += (
                    f"  Avg_S1_MRR@5={sum(all_stage1_mrr5)/len(all_stage1_mrr5):.4f}"
                    f"  Avg_S1_Recall@10={sum(all_stage1_recall10)/len(all_stage1_recall10):.4f}"
                )
            print(f"  ---\n{avg_line}")


if __name__ == "__main__":
    main()
