"""
Repair empty cached translations in `.cache/translations/{lang}_{split}.json`.

For each requested language/split:
- load the existing cache JSON
- find entries whose value is missing or empty/whitespace-only
- load the original query texts from the dataset
- re-translate only those empty entries
- write the repaired cache back in place (or to a separate output dir)

Examples:
    python scripts/repair_empty_translations.py
    python scripts/repair_empty_translations.py --langs de --splits dev train
    python scripts/repair_empty_translations.py --backend marian --device cuda
    python scripts/repair_empty_translations.py --cache-dir .cache/translations --output-dir .cache/translations_fixed
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.translate_queries import TranslationConfig, Translator
from src2.data import load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["de", "fr"])
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--backend", choices=["nllb", "marian"], default="nllb")
    parser.add_argument("--model-de", default=None, help="Optional DE->EN model override")
    parser.add_argument("--model-fr", default=None, help="Optional FR->EN model override")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", default=".cache/translations")
    parser.add_argument("--output-dir", default=None, help="Optional output dir. Default: overwrite cache-dir files in place.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-preclean", dest="source_preclean", action="store_true")
    parser.add_argument("--no-source-preclean", dest="source_preclean", action="store_false")
    parser.set_defaults(source_preclean=True)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> TranslationConfig:
    return TranslationConfig(
        backend=args.backend,
        batch_size=args.batch_size,
        device=args.device,
        num_beams=args.num_beams,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        source_preclean=args.source_preclean,
        cache_dir=Path(args.cache_dir),
        model_de=args.model_de,
        model_fr=args.model_fr,
    )


def find_empty_indices(cache: dict) -> list[str]:
    return [
        str(idx)
        for idx, text in cache.items()
        if text is None or not str(text).strip()
    ]


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    output_dir = Path(args.output_dir) if args.output_dir else cfg.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        if lang not in ("de", "fr"):
            raise ValueError(f"Only de/fr supported, got: {lang}")

        translator = Translator(lang, cfg)

        for split in args.splits:
            cache_file = cfg.cache_dir / f"{lang}_{split}.json"
            if not cache_file.exists():
                print(f"[skip] Missing cache file: {cache_file}")
                continue

            print(f"\n=== {lang.upper()} / {split} ===")
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            empty_indices = find_empty_indices(cache)
            if not empty_indices:
                print(f"  No empty translations in {cache_file}")
                if output_dir != cfg.cache_dir:
                    out_path = output_dir / cache_file.name
                    out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"  Copied unchanged cache -> {out_path}")
                continue

            data = load_split(lang, split)
            text_by_index = {str(idx): text for idx, text in zip(data["index"], data["text"])}
            missing_from_dataset = [idx for idx in empty_indices if idx not in text_by_index]
            if missing_from_dataset:
                print(f"  Warning: {len(missing_from_dataset)} empty indices not found in dataset: {missing_from_dataset[:10]}")

            repair_indices = [idx for idx in empty_indices if idx in text_by_index]
            repair_texts = [text_by_index[idx] for idx in repair_indices]
            print(f"  Empty cached translations: {len(empty_indices)}")
            print(f"  Re-translating:            {len(repair_indices)}")

            if repair_texts:
                _, translated = translator.translate_all(repair_texts)
                repaired = 0
                still_empty = 0
                for idx, text in zip(repair_indices, translated):
                    if text is not None and str(text).strip():
                        cache[idx] = text
                        repaired += 1
                    else:
                        still_empty += 1
                print(f"  Repaired:                 {repaired}")
                print(f"  Still empty after retry:  {still_empty}")

            out_path = output_dir / cache_file.name
            out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Saved repaired cache -> {out_path}")

            repaired_indices = [idx for idx in repair_indices if str(cache.get(idx, '')).strip()]
            for idx in repaired_indices[:3]:
                print(f"  [{idx}] {text_by_index[idx][:100]}")
                print(f"       -> {cache[idx][:100]}")


if __name__ == "__main__":
    main()
