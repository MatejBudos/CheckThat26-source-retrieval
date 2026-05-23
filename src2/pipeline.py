import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    load_collection,
    load_rewrite_queries,
    load_split,
    make_rerank_text,
    prepare_collection,
    prepare_queries,
)
from .metrics import compute_metrics, compute_stage1_metrics
from .reranker import CrossEncoderReranker
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.langs = config.get("langs", ["de", "fr", "en"])
        self.top_k_retrieve = config.get("top_k_retrieve", 100)
        self.top_k_output = config.get("top_k_output", 5)
        self.output_dir = Path(config.get("output_dir", "predictions_src2"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.translation_dir = config.get("translation_dir", ".cache/translations")
        self.device = config.get("device", "cuda")
        self.default_dense_model = config.get("dense_model", "intfloat/multilingual-e5-large")
        self.dense_model_by_lang = config.get("dense_model_by_lang", {})

        use_bm25 = config.get("use_bm25", True)
        use_dense = config.get("use_dense", True)
        self.use_dense = use_dense

        print("Loading collection...")
        collection = load_collection()
        self.passage_texts, self.bm25_texts, _, self.collection_keys = prepare_collection(
            collection,
            title_boost=config.get("bm25_title_boost", 4),
            venue_boost=config.get("bm25_venue_boost", 1),
            include_abstract=config.get("bm25_include_abstract", True),
            include_authors=config.get("bm25_include_authors", False),
        )
        self.rerank_text_mode = config.get("rerank_text_mode", "full")
        self.rerank_texts = [make_rerank_text(ex, mode=self.rerank_text_mode) for ex in collection]
        print(f"Rerank text mode: {self.rerank_text_mode}")

        bm25 = None
        if use_bm25:
            cache_path = config.get("bm25_cache_path", ".cache/src2_bm25_scibert_t4_v1_a1_au0")
            bm25 = BM25Retriever(
                self.bm25_texts,
                tokenizer_name=config.get("bm25_tokenizer_name", "allenai/scibert_scivocab_uncased"),
                cache_path=cache_path,
                numeric_boost=config.get("bm25_numeric_boost", 1),
            )
        self.bm25 = bm25

        self.dense_by_model: dict[str, DenseRetriever] = {}
        if use_dense:
            model_names = {self.default_dense_model, *self.dense_model_by_lang.values()}
            for model_name in model_names:
                dense = DenseRetriever(
                    model_name,
                    device=self.device,
                    query_prompt_name=config.get("dense_query_prompt_name"),
                    query_instruction=config.get("dense_query_instruction"),
                )
                model_slug = model_name.replace("/", "_")
                cache_path = config.get("embeddings_cache_by_model", {}).get(
                    model_name,
                    f".cache/corpus_embeddings_{model_slug}.npy",
                )
                dense.encode_corpus(
                    self.passage_texts,
                    batch_size=config.get("corpus_batch_size", 64),
                    cache_path=cache_path,
                    chunk_tokens=config.get("dense_chunk_tokens", 0),
                    chunk_overlap=config.get("dense_chunk_overlap", 50),
                )
                self.dense_by_model[model_name] = dense

        self.reranker = None
        if config.get("use_reranker", False):
            self.reranker = CrossEncoderReranker(
                model_name=config.get("reranker_model", "BAAI/bge-reranker-v2-gemma"),
                device=self.device,
                max_length=config.get("rerank_max_length", 192),
                instruction=config.get("reranker_instruction"),
                prompt_name=config.get("reranker_prompt_name", "query"),
                query_prefix=config.get("reranker_query_prefix", ""),
            )

    def _get_dense(self, lang: str) -> DenseRetriever | None:
        if not self.use_dense:
            return None
        model_name = self.dense_model_by_lang.get(lang, self.default_dense_model)
        return self.dense_by_model[model_name]

    def _select_reranker_queries(
        self,
        lang: str,
        raw_queries: list[str],
        bm25_queries: list[str],
        original_dense_queries: list[str],
        rewritten_queries: list[str] | None,
    ) -> list[str]:
        mode = self.config.get("reranker_query_mode_by_lang", {}).get(lang, self.config.get("reranker_query_mode"))
        if mode is None:
            mode = "translated" if self.config.get("use_translations", True) and lang in {"de", "fr"} else "raw"

        if mode == "raw":
            return raw_queries
        if mode == "translated":
            return bm25_queries
        if mode == "original":
            return [q.removeprefix("query: ") for q in original_dense_queries]
        if mode == "rewritten":
            if rewritten_queries is None:
                raise ValueError(f"reranker_query_mode=rewritten requested for {lang}, but no rewritten queries were loaded")
            return rewritten_queries
        raise ValueError(
            "Unsupported reranker_query_mode: {mode}. Use raw, translated, original, or rewritten".format(mode=mode)
        )

    def _run_lang(self, lang: str, split: str) -> dict | None:
        print(f"\n=== {lang.upper()} / {split} ===")
        dense = self._get_dense(lang)
        hybrid = HybridRetriever(
            bm25=self.bm25,
            dense=dense,
            rrf_k=self.config.get("rrf_k", 60),
            bm25_weight=self.config.get("bm25_weight", 0.25),
            dense_weight=self.config.get("dense_weight", 1.0),
            bm25_rewrite_weight=self.config.get("bm25_rewrite_weight"),
            dense_weight_2=self.config.get("dense_weight_2"),
            dense_weight_3=self.config.get("dense_weight_3"),
        )
        data = load_split(lang, split)
        raw_queries, bm25_queries, dense_queries, original_dense_queries, indices, true_pubkeys = prepare_queries(
            data,
            lang=lang,
            split=split,
            use_translations=self.config.get("use_translations", True),
            translation_dir=self.translation_dir,
            quote_extract=self.config.get("bm25_quote_extract", False),
            query_cleanup=self.config.get("query_cleanup", False),
            query_cleanup_langs=self.config.get("query_cleanup_langs"),
            bm25_concat_original=self.config.get("bm25_concat_original", False),
            bm25_concat_original_langs=self.config.get("bm25_concat_original_langs"),
            return_bm25_queries=True,
        )
        bm25_rewrite_queries = None
        dense_rewrite_queries = None
        if self.config.get("bm25_rewrite_view", False):
            bm25_rewrite_queries = load_rewrite_queries(
                data,
                lang=lang,
                split=split,
                rewrite_dir=self.config.get("rewrite_dir", ".cache/augmented"),
            )
        if self.config.get("dense_rewrite_view", False):
            dense_rewrite_langs = self.config.get("dense_rewrite_langs")
            if dense_rewrite_langs is None or lang in dense_rewrite_langs:
                dense_rewrite_queries = load_rewrite_queries(
                    data,
                    lang=lang,
                    split=split,
                    rewrite_dir=self.config.get("rewrite_dir", ".cache/augmented"),
                )

        query_embeddings = None
        if dense is not None:
            print(f"  Encoding {len(dense_queries)} queries...")
            query_embeddings = dense.encode_queries(
                dense_queries,
                batch_size=self.config.get("query_batch_size", 64),
            )

        query_embeddings_2 = None
        if dense is not None and self.config.get("multi_query", False):
            if any(a != b for a, b in zip(dense_queries, original_dense_queries)):
                print("  Encoding original tweet queries for multi-query dense view...")
                query_embeddings_2 = dense.encode_queries(
                    original_dense_queries,
                    batch_size=self.config.get("query_batch_size", 64),
                )

        query_embeddings_3 = None
        if dense is not None and self.config.get("dense_rewrite_view", False) and dense_rewrite_queries is not None:
            print("  Encoding rewrite queries for dense rewrite view...")
            rewrite_dense_queries = [f"query: {text}" for text in dense_rewrite_queries]
            query_embeddings_3 = dense.encode_queries(
                rewrite_dense_queries,
                batch_size=self.config.get("query_batch_size", 64),
            )

        print("  Retrieving...")
        retrieval_kwargs = dict(
            query_embeddings_2=query_embeddings_2,
            query_embeddings_3=query_embeddings_3,
            bm25_rewrite_queries=bm25_rewrite_queries,
            bm25_top_k=self.config.get("bm25_top_k_retrieve"),
            dense_top_k=self.config.get("dense_top_k_retrieve"),
        )
        rerank_source = self.config.get("rerank_source", "fused")
        if self.reranker is not None and rerank_source == "union":
            top_indices = hybrid.candidate_union_batch(
                bm25_queries,
                query_embeddings,
                top_k=self.top_k_retrieve,
                **retrieval_kwargs,
            )
        else:
            top_indices = hybrid.retrieve_batch(
                bm25_queries,
                query_embeddings,
                self.top_k_retrieve,
                **retrieval_kwargs,
            )
        stage1_keys = self.collection_keys[top_indices]
        if self.reranker is not None:
            rerank_candidates = self.config.get("rerank_candidates", 20)
            reranker_queries = self._select_reranker_queries(
                lang,
                raw_queries,
                bm25_queries,
                original_dense_queries,
                dense_rewrite_queries,
            )
            print(
                f"  Reranking top-{rerank_candidates} candidates from {rerank_source} retrieval "
                f"with {self.config.get('reranker_model', 'BAAI/bge-reranker-v2-gemma')} "
                f"(query_mode={self.config.get('reranker_query_mode', 'auto')})..."
            )
            candidates_texts = [
                [self.rerank_texts[int(j)] for j in row[:rerank_candidates]]
                for row in top_indices
            ]
            local_indices_list = self.reranker.rerank_batch(
                reranker_queries,
                candidates_texts,
                top_k=self.top_k_output,
                batch_size=self.config.get("rerank_batch_size", 8),
            )
            final_keys = np.stack([
                self.collection_keys[top_indices[i][local_indices_list[i]]]
                for i in range(len(raw_queries))
            ])
        else:
            final_keys = np.stack([
                self.collection_keys[top_indices[i][:self.top_k_output]]
                for i in range(len(raw_queries))
            ])

        pd.DataFrame({"index": indices, "preds": final_keys.tolist()}).to_csv(
            self.output_dir / f"predictions_{lang}.tsv",
            index=False,
            sep="\t",
        )

        if true_pubkeys is None:
            return None

        metrics = compute_metrics(final_keys, true_pubkeys)
        metrics.update(compute_stage1_metrics(stage1_keys, true_pubkeys))
        print(f"  {lang}: {metrics}")
        return metrics

    def run(self, split: str = "dev") -> dict:
        results = {}
        for lang in self.langs:
            metrics = self._run_lang(lang, split)
            if metrics is not None:
                results[f"{lang}_{split}"] = metrics
        self._zip_predictions()
        return results

    def _zip_predictions(self) -> Path:
        zip_path = self.output_dir / "predictions.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for lang in self.langs:
                path = self.output_dir / f"predictions_{lang}.tsv"
                if path.exists():
                    zf.write(path, path.name)
        return zip_path
