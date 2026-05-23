import re
from pathlib import Path

import bm25s
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def resolve_model(name: str) -> str:
    """Return local path if it exists, otherwise use name as HF Hub model ID.

    Allows the config to specify HF Hub IDs (e.g. 'budos/checkthat26-e5-large')
    while still supporting local overrides (e.g. 'checkpoints/my-local-model').
    """
    if Path(name).exists():
        print(f"  [local]  {name}")
        return name
    print(f"  [HF Hub] {name}")
    return name

_PUNCT_RE = re.compile(r"[^\w\s%]")


def _subword_preprocess(texts: list[str], tokenizer, numeric_boost: int = 1) -> list[str]:
    out = []
    for text in texts:
        text = _PUNCT_RE.sub(" ", text.lower())
        tokens = tokenizer.tokenize(text)
        tokens = [tok for tok in tokens if tok]
        if numeric_boost > 1:
            boosted = []
            for tok in tokens:
                boosted.append(tok)
                if any(ch.isdigit() for ch in tok):
                    boosted.extend([tok] * (numeric_boost - 1))
            tokens = boosted
        out.append(" ".join(tokens))
    return out


def _load_spacy_nlp():
    try:
        import spacy
    except ImportError:
        raise ImportError("spacy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        raise OSError("spaCy model not found. Run: python -m spacy download en_core_web_sm")


def _spacy_preprocess(texts: list[str], nlp, numeric_boost: int = 1) -> list[str]:
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        tokens = []
        for token in doc:
            if token.is_stop:
                continue
            if not (token.is_alpha or token.like_num):
                continue
            lemma = token.lemma_.lower().strip()
            if not lemma:
                continue
            tokens.append(lemma)
            if numeric_boost > 1 and any(ch.isdigit() for ch in lemma):
                tokens.extend([lemma] * (numeric_boost - 1))
        out.append(" ".join(tokens))
    return out


class BM25Retriever:
    def __init__(
        self,
        corpus_texts: list[str],
        *,
        tokenizer_name: str | None = "allenai/scibert_scivocab_uncased",
        cache_path: str | None = None,
        numeric_boost: int = 1,
    ):
        self._use_spacy = tokenizer_name == "spacy"
        self._numeric_boost = numeric_boost

        if self._use_spacy:
            print("BM25 tokenizer: spacy (en_core_web_sm, lemmatization + stopword removal)")
            self._nlp = _load_spacy_nlp()
            self._tokenizer = None
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None
            self._nlp = None
            if self._tokenizer is not None:
                print(f"BM25 tokenizer: {tokenizer_name}")

        if cache_path and Path(cache_path).exists():
            print(f"Loading BM25 index from cache: {cache_path}")
            self.retriever = bm25s.BM25.load(cache_path, mmap=True)
            return

        if self._use_spacy:
            corpus_texts = _spacy_preprocess(corpus_texts, self._nlp, numeric_boost=self._numeric_boost)
        elif self._tokenizer is not None:
            corpus_texts = _subword_preprocess(corpus_texts, self._tokenizer, numeric_boost=self._numeric_boost)

        print("Building BM25 index...")
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=None, show_progress=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens, show_progress=False)

        if cache_path:
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            self.retriever.save(cache_path)
            print(f"BM25 index cached: {cache_path}")

    def retrieve_batch(self, queries: list[str], top_k: int) -> np.ndarray:
        if self._use_spacy:
            queries = _spacy_preprocess(queries, self._nlp, numeric_boost=self._numeric_boost)
        elif self._tokenizer is not None:
            queries = _subword_preprocess(queries, self._tokenizer, numeric_boost=self._numeric_boost)
        query_tokens = bm25s.tokenize(queries, stopwords=None, show_progress=False)
        top_indices, _ = self.retriever.retrieve(query_tokens, k=top_k, show_progress=False)
        return top_indices


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        device: str,
        *,
        query_prompt_name: str | None = None,
        query_instruction: str | None = None,
    ):
        print(f"Loading dense model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(resolve_model(model_name), device=device)
        self.corpus_embeddings: np.ndarray | None = None
        self.query_prompt_name = query_prompt_name
        self.query_instruction = query_instruction

    def _uses_qwen3_embedding_prompts(self) -> bool:
        return "qwen3-embedding" in self.model_name.lower()

    def _prepare_corpus_texts(self, texts: list[str]) -> list[str]:
        if not self._uses_qwen3_embedding_prompts():
            return texts
        return [text.removeprefix("passage: ").strip() for text in texts]

    def _prepare_query_texts(self, texts: list[str]) -> list[str]:
        if not self._uses_qwen3_embedding_prompts():
            return texts
        return [text.removeprefix("query: ").strip() for text in texts]

    def _chunk_texts(self, texts: list[str], chunk_tokens: int, overlap: int) -> tuple[list[str], list[int]]:
        tokenizer = self.model.tokenizer
        step = max(1, chunk_tokens - overlap)
        chunked: list[str] = []
        doc_chunk_counts: list[int] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= chunk_tokens:
                chunked.append(text)
                doc_chunk_counts.append(1)
            else:
                starts = list(range(0, len(ids), step))
                for s in starts:
                    chunk_text = tokenizer.decode(ids[s: s + chunk_tokens], skip_special_tokens=True)
                    chunked.append(chunk_text)
                doc_chunk_counts.append(len(starts))
        return chunked, doc_chunk_counts

    def encode_corpus(
        self,
        texts: list[str],
        *,
        batch_size: int,
        cache_path: str | None,
        chunk_tokens: int = 0,
        chunk_overlap: int = 50,
    ) -> None:
        if cache_path and Path(cache_path).exists():
            print(f"Loading corpus embeddings from cache: {cache_path}")
            self.corpus_embeddings = np.load(cache_path)
            return

        print("Encoding corpus...")
        texts = self._prepare_corpus_texts(texts)

        if chunk_tokens > 0:
            chunked_texts, doc_chunk_counts = self._chunk_texts(texts, chunk_tokens, chunk_overlap)
            print(f"  Chunked {len(texts)} docs → {len(chunked_texts)} chunks (max {chunk_tokens} tok, {chunk_overlap} overlap)")
            raw = self.model.encode(
                chunked_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            dim = raw.shape[1]
            embeddings = np.empty((len(texts), dim), dtype=np.float32)
            offset = 0
            for i, count in enumerate(doc_chunk_counts):
                embeddings[i] = raw[offset: offset + count].max(axis=0)
                offset += count
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.corpus_embeddings = embeddings / norms
        else:
            self.corpus_embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.corpus_embeddings)
            print(f"Corpus embeddings cached: {cache_path}")

    def encode_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        texts = self._prepare_query_texts(texts)
        kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": True,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.query_instruction:
            kwargs["prompt"] = self.query_instruction
        elif self.query_prompt_name:
            kwargs["prompt_name"] = self.query_prompt_name
        elif self._uses_qwen3_embedding_prompts():
            kwargs["prompt_name"] = "query"
        return self.model.encode(texts, **kwargs)

    def retrieve_batch(self, query_embeddings: np.ndarray, top_k: int) -> np.ndarray:
        assert self.corpus_embeddings is not None, "Call encode_corpus first"
        sims = query_embeddings @ self.corpus_embeddings.T
        return np.argsort(-sims, axis=1)[:, :top_k]


class HybridRetriever:
    def __init__(
        self,
        *,
        bm25: BM25Retriever | None,
        dense: DenseRetriever | None,
        rrf_k: int,
        bm25_weight: float,
        dense_weight: float,
        bm25_rewrite_weight: float | None = None,
        dense_weight_2: float | None = None,
        dense_weight_3: float | None = None,
    ):
        assert bm25 is not None or dense is not None
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25_rewrite_weight = bm25_weight if bm25_rewrite_weight is None else bm25_rewrite_weight
        self.dense_weight_2 = dense_weight if dense_weight_2 is None else dense_weight_2
        self.dense_weight_3 = dense_weight if dense_weight_3 is None else dense_weight_3

    def _rrf_fuse(self, ranked_lists: list[np.ndarray], weights: list[float], top_k: int, n_corpus: int) -> np.ndarray:
        rank_dicts = [{int(idx): rank for rank, idx in enumerate(lst)} for lst in ranked_lists]
        candidates: set[int] = set()
        for lst in ranked_lists:
            candidates.update(int(x) for x in lst)

        scores = {}
        for doc in candidates:
            score = 0.0
            for weight, rank_dict in zip(weights, rank_dicts):
                score += weight / (self.rrf_k + rank_dict.get(doc, n_corpus) + 1)
            scores[doc] = score
        fused = sorted(scores, key=scores.__getitem__, reverse=True)
        return np.array(fused[:top_k], dtype=np.int64)

    def _dedup_union(self, ranked_lists: list[np.ndarray], top_k: int) -> np.ndarray:
        seen: set[int] = set()
        merged: list[int] = []
        for lst in ranked_lists:
            for idx in lst:
                doc = int(idx)
                if doc not in seen:
                    seen.add(doc)
                    merged.append(doc)
                    if len(merged) >= top_k:
                        return np.array(merged, dtype=np.int64)
        return np.array(merged, dtype=np.int64)

    def _collect_ranked_lists(
        self,
        bm25_queries: list[str],
        query_embeddings: np.ndarray | None,
        *,
        query_embeddings_2: np.ndarray | None = None,
        query_embeddings_3: np.ndarray | None = None,
        bm25_rewrite_queries: list[str] | None = None,
        bm25_top_k: int,
        dense_top_k: int,
    ) -> tuple[list[list[np.ndarray]], list[list[float]], int]:
        ranked_lists_per_query: list[list[np.ndarray]] = [[] for _ in range(len(bm25_queries))]
        weights_per_query: list[list[float]] = [[] for _ in range(len(bm25_queries))]
        n_corpus = self.dense.corpus_embeddings.shape[0] if self.dense and self.dense.corpus_embeddings is not None else max(bm25_top_k, dense_top_k)

        if self.bm25 is not None:
            bm25_top = self.bm25.retrieve_batch(bm25_queries, top_k=bm25_top_k)
            for i in range(len(bm25_queries)):
                ranked_lists_per_query[i].append(bm25_top[i])
                weights_per_query[i].append(self.bm25_weight)

        if self.bm25 is not None and bm25_rewrite_queries is not None:
            bm25_rewrite_top = self.bm25.retrieve_batch(bm25_rewrite_queries, top_k=bm25_top_k)
            for i in range(len(bm25_queries)):
                ranked_lists_per_query[i].append(bm25_rewrite_top[i])
                weights_per_query[i].append(self.bm25_rewrite_weight)

        if self.dense is not None and query_embeddings is not None:
            dense_top = self.dense.retrieve_batch(query_embeddings, top_k=dense_top_k)
            for i in range(len(bm25_queries)):
                ranked_lists_per_query[i].append(dense_top[i])
                weights_per_query[i].append(self.dense_weight)

        if self.dense is not None and query_embeddings_2 is not None:
            dense_top_2 = self.dense.retrieve_batch(query_embeddings_2, top_k=dense_top_k)
            for i in range(len(bm25_queries)):
                ranked_lists_per_query[i].append(dense_top_2[i])
                weights_per_query[i].append(self.dense_weight_2)

        if self.dense is not None and query_embeddings_3 is not None:
            dense_top_3 = self.dense.retrieve_batch(query_embeddings_3, top_k=dense_top_k)
            for i in range(len(bm25_queries)):
                ranked_lists_per_query[i].append(dense_top_3[i])
                weights_per_query[i].append(self.dense_weight_3)

        return ranked_lists_per_query, weights_per_query, n_corpus

    def retrieve_batch(
        self,
        bm25_queries: list[str],
        query_embeddings: np.ndarray | None,
        top_k: int,
        query_embeddings_2: np.ndarray | None = None,
        query_embeddings_3: np.ndarray | None = None,
        bm25_rewrite_queries: list[str] | None = None,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
    ) -> np.ndarray:
        bm25_top_k = bm25_top_k or top_k
        dense_top_k = dense_top_k or top_k
        ranked_lists_per_query, weights_per_query, n_corpus = self._collect_ranked_lists(
            bm25_queries,
            query_embeddings,
            query_embeddings_2=query_embeddings_2,
            query_embeddings_3=query_embeddings_3,
            bm25_rewrite_queries=bm25_rewrite_queries,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
        )

        if all(len(parts) == 1 for parts in ranked_lists_per_query):
            return np.stack([parts[0] for parts in ranked_lists_per_query])

        return np.stack([
            self._rrf_fuse(ranked_lists_per_query[i], weights_per_query[i], top_k, n_corpus)
            for i in range(len(bm25_queries))
        ])

    def candidate_union_batch(
        self,
        bm25_queries: list[str],
        query_embeddings: np.ndarray | None,
        *,
        top_k: int,
        query_embeddings_2: np.ndarray | None = None,
        query_embeddings_3: np.ndarray | None = None,
        bm25_rewrite_queries: list[str] | None = None,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
    ) -> np.ndarray:
        bm25_top_k = bm25_top_k or top_k
        dense_top_k = dense_top_k or top_k
        ranked_lists_per_query, _, _ = self._collect_ranked_lists(
            bm25_queries,
            query_embeddings,
            query_embeddings_2=query_embeddings_2,
            query_embeddings_3=query_embeddings_3,
            bm25_rewrite_queries=bm25_rewrite_queries,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
        )
        return np.stack([
            self._dedup_union(ranked_lists_per_query[i], top_k)
            for i in range(len(bm25_queries))
        ])
