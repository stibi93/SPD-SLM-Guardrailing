"""Test-set evaluation with report generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from spd.categories import CATEGORIES
from spd.data import SPDDataset
from spd.dedup import validate_disjoint
from spd.evaluate import compute_metrics
from spd.model import SPDClassifier
from spd.reporting import (
    make_report_dir,
    per_category_good_bad,
    save_category_evaluation_bars,
    save_json,
    save_sample_level_chart,
    write_summary_markdown,
)


def load_thresholds(path: str) -> dict[str, float]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    default = cfg.get("defaults", {}).get("threshold", 0.5)
    return {cat: cfg.get("per_category", {}).get(cat, default) for cat in CATEGORIES}


def evaluate_test_set(
    model_path: str,
    test_data: str,
    train_data: str | None = None,
    model_name_or_path: str = "SZTAKI-HLT/hubert-base-cc",
    threshold: float = 0.5,
    report_dir: str | Path | None = None,
    training_history: list[dict] | None = None,
    dedup_stats: dict | None = None,
) -> dict:
    """
    Run test-set evaluation, optionally writing charts to report_dir.

    Returns metrics dict including per-category good/bad breakdown.
    """
    if train_data:
        validate_disjoint(train_data, test_data)

    out = Path(report_dir) if report_dir else make_report_dir()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    test_ds = SPDDataset(test_data, tokenizer)
    loader = DataLoader(test_ds, batch_size=8)

    model = SPDClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            all_logits.append(model(batch["input_ids"], batch["attention_mask"]))
            all_labels.append(batch["labels"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(logits, labels, threshold=threshold)

    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy().astype(int)
    y_pred = (probs > threshold).astype(int)

    exact_match = float(np.all(y_true == y_pred, axis=1).mean() * 100)
    metrics["exact_match_pct"] = round(exact_match, 2)
    metrics["n_samples"] = int(y_true.shape[0])

    good_bad = per_category_good_bad(y_true, y_pred)
    metrics["per_category_good_bad"] = good_bad

    if training_history:
        from spd.reporting import save_per_category_f1_curves, save_training_curves

        save_json(out / "training_history.json", training_history)
        save_training_curves(training_history, out)
        save_per_category_f1_curves(training_history, out)

    save_category_evaluation_bars(good_bad, out)
    macro_good_pct = float(np.mean([good_bad[c]["good_pct"] for c in CATEGORIES]))
    save_sample_level_chart(exact_match, macro_good_pct, out)
    save_json(out / "metrics.json", metrics)
    write_summary_markdown(out, metrics=metrics, good_bad=good_bad, dedup_stats=dedup_stats)

    print(f"Report saved to {out}")
    print(f"  macro-F1={metrics['macro_f1']:.4f}  exact_match={exact_match:.1f}%")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SPD model on test set and save report")
    parser.add_argument("--model-path", default="artifacts/checkpoints/best_model.pt")
    parser.add_argument("--test-data", default="data/processed/test.jsonl")
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument("--report-dir", default=None, help="Output dir (default: artifacts/reports/<timestamp>)")
    parser.add_argument("--history", default=None, help="training_history.json from a prior run")
    args = parser.parse_args()

    history = None
    if args.history:
        import json

        history = json.loads(Path(args.history).read_text())

    evaluate_test_set(
        model_path=args.model_path,
        test_data=args.test_data,
        train_data=args.train_data,
        report_dir=args.report_dir,
        training_history=history,
    )
