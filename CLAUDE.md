# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

CheckThat! Lab at CLEF 2026 — Task 1: **Source Retrieval for Scientific Web Claims**.

Given a social media post containing a scientific claim (implicit reference to a paper, no URL), retrieve the referenced paper from a pool of candidate papers. Evaluation metric: **MRR@5**.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (hybrid retrieval + reranker) on dev set
python scripts/run_pipeline.py

# Run with a specific config
python scripts/run_pipeline.py --config configs/bm25_only.yaml
python scripts/run_pipeline.py --config configs/dense_only.yaml

# Override options at runtime
python scripts/run_pipeline.py --split test --langs de fr en
python scripts/run_pipeline.py --no-reranker --top-k 50 --device cpu

# Run a single language
python scripts/run_pipeline.py --langs de
```

## Key Resources

- Dataset: `sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims` on HuggingFace
- Task site: `https://checkthat.gitlab.io/clef2026/task1/`
- Baseline notebook: `https://gitlab.com/checkthat_lab/clef2026-checkthat-lab/-/blob/main/task1/CT26_Task1_getting_started.ipynb`

## Planned Architecture (from `resources/task.md`)

Based on top-performing systems from the 2025 edition, the target pipeline is two-stage:

```
Tweet → [Query Augmentation] → [Stage 1: Hybrid Retrieval] → [Stage 2: Re-ranking] → Top-5 Papers
```

**Stage 1 — Hybrid Retrieval** (Retriever)
- BM25 sparse retrieval + dense bi-encoder retrieval (E5-large or GritLM-7B)
- Fusion via Reciprocal Rank Fusion (RRF)
- Corpus indexed on title + abstract from the candidate pool
- Bi-encoder fine-tuned with in-batch negatives + hard negative mining

**Stage 2 — Re-ranking** (Re-ranker)
- Scientific cross-encoder (SciBERT-based) re-ranks top-k from Stage 1
- Fine-tuned on positive/hard-negative pairs

**Query Augmentation** (style gap mitigation)
- Optional LLM rewrite (e.g. LLaMA 3.3 70B) to convert informal tweet language into formal scientific phrasing before retrieval

## Prior Art Baselines (MRR@5 on 2025 Task 4b)

| System | Stage | MRR@5 |
|---|---|---|
| BM25 baseline | — | 0.5025 |
| AIRwaves (E5-large bi-encoder) | Stage 1 | 0.6174 |
| AIRwaves (+ SciBERT re-ranker) | Stage 2 | 0.6828 |
