"""Ranking metrics for retrieval evaluation.

A ranking is a list of relevance labels (1 = relevant, 0 = not) in the
order the matcher returned them for one query.
"""
from __future__ import annotations


def precision_at_k(ranking: list[int], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranking[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def recall_at_k(ranking: list[int], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    total_relevant = sum(ranking)
    if total_relevant == 0:
        return 0.0
    return sum(ranking[:k]) / total_relevant


def reciprocal_rank(ranking: list[int]) -> float:
    """1 / rank of the first relevant item (0 if none is relevant)."""
    for i, label in enumerate(ranking, start=1):
        if label == 1:
            return 1.0 / i
    return 0.0


def evaluate(rankings: list[list[int]], k: int = 3) -> dict[str, float]:
    """Macro-average precision@k, recall@k and MRR over all queries."""
    if not rankings:
        raise ValueError("no rankings to evaluate")
    n = len(rankings)
    return {
        f"precision@{k}": sum(precision_at_k(r, k) for r in rankings) / n,
        f"recall@{k}": sum(recall_at_k(r, k) for r in rankings) / n,
        "mrr": sum(reciprocal_rank(r) for r in rankings) / n,
    }
