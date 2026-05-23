import json
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset

from .text_utils import clean_tweet, extract_quoted_boost, normalize_de_query, preprocess_de_query_raw

DATASET_NAME = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"
VALID_LANGS = ("en", "de", "fr")


def load_collection() -> Dataset:
    return load_dataset(DATASET_NAME, "collection")["collection"]


def load_split(lang: str, split: str) -> Dataset:
    assert lang in VALID_LANGS, f"lang must be one of {VALID_LANGS}"
    assert split in ("train", "dev", "test"), "split must be train/dev/test"
    if split == "test":
        return load_dataset(DATASET_NAME, "test")[lang]
    return load_dataset(DATASET_NAME, lang)[split]


def _repeat(text: str, n: int) -> str:
    if n <= 0 or not text:
        return ""
    return " ".join([text] * n)


def make_passage_text(example: dict) -> str:
    return f'passage: {example["title"]}. {example["venue"]}. {example["abstract"]}. {example["authors"]}'


def make_rerank_text(example: dict, *, mode: str) -> str:
    fields = {
        "title": example.get("title", ""),
        "venue": example.get("venue", ""),
        "abstract": example.get("abstract", ""),
        "authors": example.get("authors", ""),
    }

    aliases = {
        "full": ["title", "venue", "abstract", "authors"],
        "title_only": ["title"],
        "abstract_only": ["abstract"],
        "venue_only": ["venue"],
        "authors_only": ["authors"],
    }

    if mode in aliases:
        field_names = aliases[mode]
    else:
        normalized = mode.replace("+", "_").replace(",", "_")
        field_names = [part.strip() for part in normalized.split("_") if part.strip()]
        invalid = [name for name in field_names if name not in fields]
        if not field_names or invalid:
            supported = ", ".join([*fields.keys(), *aliases.keys()])
            raise ValueError(
                f"Unsupported rerank_text_mode: {mode}. "
                f"Use a combination of title/venue/abstract/authors or one of: {supported}"
            )

    body = ". ".join(fields[name] for name in field_names if fields[name])
    return f"passage: {body}"


def make_bm25_text(
    example: dict,
    *,
    title_boost: int,
    venue_boost: int,
    include_abstract: bool,
    include_authors: bool,
) -> str:
    parts: list[str] = []
    title = _repeat(example.get("title", ""), title_boost)
    venue = _repeat(example.get("venue", ""), venue_boost)
    abstract = example.get("abstract", "") if include_abstract else ""
    authors = example.get("authors", "") if include_authors else ""
    for part in (title, venue, abstract, authors):
        if part:
            parts.append(part)
    return " ".join(parts)


def prepare_collection(
    collection: Dataset,
    *,
    title_boost: int,
    venue_boost: int,
    include_abstract: bool,
    include_authors: bool,
) -> tuple[list[str], list[str], list[str], np.ndarray]:
    passage_texts = [make_passage_text(ex) for ex in collection]
    rerank_texts = [make_rerank_text(ex, mode="full") for ex in collection]
    bm25_texts = [
        make_bm25_text(
            ex,
            title_boost=title_boost,
            venue_boost=venue_boost,
            include_abstract=include_abstract,
            include_authors=include_authors,
        )
        for ex in collection
    ]
    pubkeys = np.array(collection["pubkey"])
    return passage_texts, bm25_texts, rerank_texts, pubkeys


def load_translations(cache_dir: str | None, lang: str, split: str) -> dict | None:
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"{lang}_{split}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_cache(cache_dir: str | None, lang: str, split: str) -> dict | None:
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"{lang}_{split}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_queries(
    data: Dataset,
    *,
    lang: str,
    split: str,
    use_translations: bool,
    translation_dir: str | None,
    quote_extract: bool = False,
    query_cleanup: bool = False,
    query_cleanup_langs: list[str] | None = None,
    bm25_concat_original: bool = False,
    bm25_concat_original_langs: list[str] | None = None,
    return_bm25_queries: bool = False,
) -> tuple[list[str], list[str], list[str], list, np.ndarray | None] | tuple[list[str], list[str], list[str], list[str], list, np.ndarray | None]:
    translations = None
    if use_translations:
        translations = load_translations(translation_dir, lang, split)
        if translations:
            print(f"  Loaded {len(translations)} translations from {translation_dir}")

    indices = data["index"]
    raw_queries: list[str] = []
    bm25_queries: list[str] = []
    dense_queries: list[str] = []
    original_queries: list[str] = []
    cleanup_active = query_cleanup and (query_cleanup_langs is None or lang in query_cleanup_langs)
    concat_original_active = bm25_concat_original and (bm25_concat_original_langs is None or lang in bm25_concat_original_langs)
    for idx, original in zip(indices, data["text"]):
        original_raw = preprocess_de_query_raw(original) if cleanup_active and lang == "de" else original
        original_clean = clean_tweet(original_raw)
        if cleanup_active and lang == "de":
            original_clean = normalize_de_query(original_clean)

        text = original_raw
        if translations:
            translated = translations.get(str(idx))
            if translated is not None and translated.strip():
                text = translated
        if cleanup_active and lang == "de":
            text = preprocess_de_query_raw(text)
        cleaned = clean_tweet(text)
        if cleanup_active and lang == "de":
            cleaned = normalize_de_query(cleaned)
        if quote_extract:
            cleaned = extract_quoted_boost(cleaned)
            original_clean = extract_quoted_boost(original_clean)
        raw_queries.append(cleaned)
        bm25_query = cleaned
        if translations and concat_original_active and original_clean and original_clean != cleaned:
            bm25_query = f"{cleaned} {original_clean}"
        bm25_queries.append(bm25_query)
        dense_queries.append(f"query: {cleaned}")
        original_queries.append(f"query: {original_clean}")

    true_pubkeys = np.array(data["pubkey"]) if "pubkey" in data.column_names else None
    if return_bm25_queries:
        return raw_queries, bm25_queries, dense_queries, original_queries, indices, true_pubkeys
    return raw_queries, dense_queries, original_queries, indices, true_pubkeys


def load_rewrite_queries(
    data: Dataset,
    *,
    lang: str,
    split: str,
    rewrite_dir: str | None,
) -> list[str] | None:
    rewrites = load_optional_cache(rewrite_dir, lang, split)
    if not rewrites:
        return None

    indices = data["index"]
    output = []
    for idx, original in zip(indices, data["text"]):
        text = rewrites.get(str(idx), original)
        if not text or not str(text).strip():
            text = original
        output.append(clean_tweet(text))
    print(f"  Loaded {len(output)} rewrite queries from {rewrite_dir}")
    return output
