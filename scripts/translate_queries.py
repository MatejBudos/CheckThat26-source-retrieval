"""
Translate DE/FR queries to English and cache them to `.cache/translations/{lang}_{split}.json`.

Default pipeline:
- pre-clean raw tweet input for DE/FR
- translate with NLLB
- use conservative beam search instead of greedy decoding

Examples:
    python scripts/translate_queries.py
    python scripts/translate_queries.py --langs de --splits dev --backend nllb --force
    python scripts/translate_queries.py --backend marian --no-source-preclean
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_queries, load_test
from src2.text_utils import preprocess_translation_input

MARIAN_MODELS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_SOURCE_LANG = {
    "de": "deu_Latn",
    "fr": "fra_Latn",
}
NLLB_TARGET_LANG = "eng_Latn"


@dataclass
class TranslationConfig:
    backend: str
    batch_size: int
    device: str
    num_beams: int
    max_input_length: int
    max_new_tokens: int
    source_preclean: bool
    cache_dir: Path
    model_de: str | None
    model_fr: str | None


class Translator:
    def __init__(self, lang: str, cfg: TranslationConfig):
        self.lang = lang
        self.cfg = cfg
        self.device = cfg.device
        self.backend = cfg.backend

        if self.backend == "marian":
            model_name = cfg.model_de if lang == "de" else cfg.model_fr
            model_name = model_name or MARIAN_MODELS[lang]
            print(f"  Loading Marian model: {model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self._forced_bos_token_id = None
        elif self.backend == "nllb":
            model_name = cfg.model_de if lang == "de" else cfg.model_fr
            model_name = model_name or NLLB_MODEL
            print(f"  Loading NLLB model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.tokenizer.src_lang = NLLB_SOURCE_LANG[lang]
            self._forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(NLLB_TARGET_LANG)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.model.eval()

    def preprocess(self, text: str) -> str:
        if not self.cfg.source_preclean:
            return text
        return preprocess_translation_input(text, self.lang)

    def translate_batch(self, texts: list[str]) -> list[str]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_input_length,
        ).to(self.device)
        gen_kwargs = {
            "num_beams": self.cfg.num_beams,
            "max_new_tokens": self.cfg.max_new_tokens,
            "early_stopping": True,
        }
        if self._forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = self._forced_bos_token_id
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def translate_all(self, texts: list[str]) -> tuple[list[str], list[str]]:
        cleaned = [self.preprocess(text) for text in texts]
        translated: list[str] = []
        desc = f"Translating {self.lang}->en ({self.backend})"
        for i in tqdm(range(0, len(cleaned), self.cfg.batch_size), desc=desc):
            batch = cleaned[i : i + self.cfg.batch_size]
            translated.extend(self.translate_batch(batch))
        return cleaned, translated


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Re-translate even if cache exists")
    parser.add_argument("--source-preclean", dest="source_preclean", action="store_true")
    parser.add_argument("--no-source-preclean", dest="source_preclean", action="store_false")
    parser.set_defaults(source_preclean=True)
    return parser.parse_args()


def load_split_data(lang: str, split: str):
    if split == "test":
        return load_test(lang)
    train_data, dev_data = load_queries(lang)
    return dev_data if split == "dev" else train_data


def main() -> None:
    args = parse_args()
    cfg = TranslationConfig(
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
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        assert lang in ("de", "fr"), f"Translation script only supports de/fr, got {lang}"
        translator = Translator(lang, cfg)

        for split in args.splits:
            cache_file = cfg.cache_dir / f"{lang}_{split}.json"
            if cache_file.exists() and not args.force:
                print(f"  Cache exists: {cache_file} - skipping (use --force to redo)")
                continue

            print(f"\n=== {lang.upper()} / {split} ===")
            try:
                data = load_split_data(lang, split)
            except KeyError:
                print(f"  Split '{split}' not available for {lang} - skipping")
                continue

            texts = list(data["text"])
            indices = list(data["index"])

            precleaned, translated = translator.translate_all(texts)
            output = {idx: tr for idx, tr in zip(indices, translated)}
            cache_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Saved {len(output)} translations -> {cache_file}")

            for i in range(min(3, len(texts))):
                print(f"  [{indices[i]}] raw:      {texts[i][:100]}")
                if cfg.source_preclean:
                    print(f"         cleaned:  {precleaned[i][:100]}")
                print(f"         english:  {translated[i][:100]}")


if __name__ == "__main__":
    main()
