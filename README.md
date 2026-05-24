# CheckThat! 2026 Task 1 — Implicit Scientific Citation Retrieval

Submission for **CLEF 2026 CheckThat! Lab Task 1**: given a tweet in English, German, or French that implicitly references a scientific paper, retrieve the correct paper from a collection of 10,000 publications.

**Authors:** Matej Budoš, Jakub Vojtek

## Results

| Split | MRR@5 |
|-------|-------|
| Dev   | 0.650 |
| Test  | 0.574 |

## System Overview

Two-stage pipeline:

1. **Hybrid retrieval** — BM25 (GPT-2 BPE tokenizer) + fine-tuned multilingual E5-large, fused via weighted RRF. Multiple query views: translated query, original query, LLM rewrite (Claude Haiku).
2. **SciBERT reranker** — cross-encoder fine-tuned on Stage 1 errors, reranks top-5 candidates.

## Requirements

```bash
pip install -r requirements.txt
```

## Report

The paper is available in `report/latex/acl_latex.pdf`.
