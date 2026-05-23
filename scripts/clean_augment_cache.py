"""
Remove bad entries from .cache/augmented/ so augment_queries.py will re-process them.

Bad entries:
  - identical to the original tweet (batch API error fallback)
  - start with '[' (LLM refusal: "[Not a scientific query...]")
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_queries, load_test


def get_originals(lang: str, split: str) -> dict[str, str]:
    if split == "test":
        data = load_test(lang)
    else:
        train, dev = load_queries(lang)
        data = dev if split == "dev" else train
    return {str(i): t for i, t in zip(data["index"], data["text"])}


def clean_file(path: Path) -> None:
    lang, split = path.stem.split("_", 1)
    aug: dict = json.loads(path.read_text(encoding="utf-8"))

    try:
        originals = get_originals(lang, split)
    except Exception as e:
        print(f"  SKIP {path.name}: cannot load originals ({e})")
        return

    before = len(aug)
    bad_keys = [
        idx for idx, val in aug.items()
        if val.startswith("[") or originals.get(idx) == val
    ]
    for k in bad_keys:
        del aug[k]

    path.write_text(json.dumps(aug, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {path.name}: removed {len(bad_keys)}/{before} bad entries, {len(aug)} remain")


def main() -> None:
    cache_dir = Path(".cache/augmented")
    for f in sorted(cache_dir.glob("*.json")):
        clean_file(f)
    print("\nDone. Run augment_queries.py to re-process missing entries.")


if __name__ == "__main__":
    main()
