import torch
import pytest
from spd.evaluate import compute_metrics
from spd.categories import CATEGORIES, NUM_CATEGORIES


def _perfect_logits() -> tuple[torch.Tensor, torch.Tensor]:
    """logits that sigmoid to ~1 for positives and ~0 for negatives."""
    labels = torch.zeros(10, NUM_CATEGORIES)
    labels[:5, CATEGORIES.index("health")] = 1.0
    logits = torch.zeros(10, NUM_CATEGORIES)
    logits[:5, CATEGORIES.index("health")] = 10.0   # sigmoid ≈ 1.0
    logits[5:, CATEGORIES.index("health")] = -10.0  # sigmoid ≈ 0.0
    return logits, labels


def test_perfect_classifier_macro_f1():
    logits, labels = _perfect_logits()
    metrics = compute_metrics(logits, labels)
    assert metrics["macro_f1"] == pytest.approx(1.0, abs=0.01)


def test_returns_per_category_f1():
    logits, labels = _perfect_logits()
    metrics = compute_metrics(logits, labels)
    assert "health_f1" in metrics
    assert metrics["health_f1"] == pytest.approx(1.0, abs=0.01)


def test_all_categories_present_in_metrics():
    logits = torch.zeros(4, NUM_CATEGORIES)
    labels = torch.zeros(4, NUM_CATEGORIES)
    metrics = compute_metrics(logits, labels)
    for cat in CATEGORIES:
        assert f"{cat}_f1" in metrics


def test_random_predictions_lower_f1():
    torch.manual_seed(0)
    logits = torch.randn(50, NUM_CATEGORIES)
    labels = (torch.rand(50, NUM_CATEGORIES) > 0.7).float()
    metrics = compute_metrics(logits, labels)
    assert metrics["macro_f1"] < 0.9
