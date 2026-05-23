import numpy as np


def mrr_at_k(predicted_keys: np.ndarray, true_keys: np.ndarray, k: int) -> float:
    top_k = predicted_keys[:, :k]
    matches = top_k == true_keys[:, None]
    found = matches.any(axis=1)
    positions = np.argmax(matches, axis=1)
    positions[~found] = -1
    rr = np.zeros(len(true_keys), dtype=float)
    rr[found] = 1.0 / (positions[found] + 1)
    return float(rr.mean())


def recall_at_k(predicted_keys: np.ndarray, true_keys: np.ndarray, k: int) -> float:
    return float((predicted_keys[:, :k] == true_keys[:, None]).any(axis=1).mean())


def compute_metrics(predicted_keys: np.ndarray, true_keys: np.ndarray) -> dict:
    n = predicted_keys.shape[1]
    return {
        "mrr@1": mrr_at_k(predicted_keys, true_keys, 1),
        "mrr@5": mrr_at_k(predicted_keys, true_keys, 5),
        "recall@5": recall_at_k(predicted_keys, true_keys, min(5, n)),
    }


def compute_stage1_metrics(stage1_keys: np.ndarray, true_keys: np.ndarray) -> dict:
    cutoffs = [k for k in (1, 5, 10, 20, 60, 100) if k <= stage1_keys.shape[1]]
    metrics = {}
    for k in cutoffs:
        metrics[f"stage1_mrr@{k}"] = mrr_at_k(stage1_keys, true_keys, k)
        metrics[f"stage1_recall@{k}"] = recall_at_k(stage1_keys, true_keys, k)
    return metrics
