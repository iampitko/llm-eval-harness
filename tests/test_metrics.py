import pytest

from metrics import evaluate, precision_at_k, recall_at_k, reciprocal_rank


def test_precision_at_k():
    assert precision_at_k([1, 0, 1], 3) == pytest.approx(2 / 3)
    assert precision_at_k([1, 1, 1], 3) == 1.0
    assert precision_at_k([0, 0, 0], 3) == 0.0
    assert precision_at_k([1, 0], 3) == pytest.approx(0.5)  # shorter than k


def test_recall_at_k():
    assert recall_at_k([1, 0, 0, 1], 3) == pytest.approx(0.5)
    assert recall_at_k([1, 1, 0, 0], 3) == 1.0
    assert recall_at_k([0, 0, 0], 3) == 0.0  # no relevant items at all


def test_reciprocal_rank():
    assert reciprocal_rank([1, 0, 0]) == 1.0
    assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_evaluate_macro_average():
    out = evaluate([[1, 0, 1, 0], [0, 1, 0, 0]], k=3)
    assert out["precision@3"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert out["recall@3"] == pytest.approx((1.0 + 1.0) / 2)
    assert out["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_evaluate_rejects_empty():
    with pytest.raises(ValueError):
        evaluate([], k=3)
