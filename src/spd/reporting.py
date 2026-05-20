"""Training and evaluation report generation (TensorBoard-style curves + bar charts)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from spd.categories import CATEGORIES

# TensorBoard-inspired palette
TB_BG = "#1e1e1e"
TB_GRID = "#333333"
TB_TEXT = "#cccccc"
COLOR_TRAIN = "#ff7043"
COLOR_DEV = "#42a5f5"
COLOR_GOOD = "#43a047"
COLOR_BAD = "#e53935"

CATEGORY_LABELS_HU: dict[str, str] = {
    "ethnicity": "Etnikai hovatartozás",
    "political_opinion": "Politikai vélemény",
    "religion_belief": "Vallás / világnézet",
    "trade_union": "Szakszervezeti tagság",
    "genetic": "Genetikai adat",
    "health": "Egészségügyi adat",
    "sex_life_orientation": "Szexualitás / életmód",
}


def _apply_tb_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TB_BG,
            "axes.facecolor": TB_BG,
            "axes.edgecolor": TB_GRID,
            "axes.labelcolor": TB_TEXT,
            "text.color": TB_TEXT,
            "xtick.color": TB_TEXT,
            "ytick.color": TB_TEXT,
            "grid.color": TB_GRID,
            "grid.alpha": 0.5,
            "legend.facecolor": TB_BG,
            "legend.edgecolor": TB_GRID,
            "font.size": 10,
        }
    )


def make_report_dir(base: str | Path = "artifacts/reports") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    latest = Path(base) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out.resolve(), target_is_directory=True)
    return out


def per_category_good_bad(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-category binary good/bad percentages (correct vs incorrect label)."""
    result: dict[str, dict[str, float]] = {}
    n = y_true.shape[0]
    for i, cat in enumerate(CATEGORIES):
        correct = int(np.sum(y_true[:, i] == y_pred[:, i]))
        wrong = n - correct
        result[cat] = {
            "good_pct": round(100.0 * correct / n, 2),
            "bad_pct": round(100.0 * wrong / n, 2),
            "good": correct,
            "bad": wrong,
            "total": n,
        }
    return result


def save_training_curves(history: list[dict[str, Any]], out_dir: Path) -> None:
    """TensorBoard-style loss and macro-F1 curves over epochs."""
    _apply_tb_style()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    dev_loss = [h.get("dev_loss", float("nan")) for h in history]
    macro_f1 = [h["dev_macro_f1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, train_loss, marker="o", color=COLOR_TRAIN, label="train loss", linewidth=2)
    if any(not np.isnan(v) for v in dev_loss):
        ax.plot(epochs, dev_loss, marker="s", color=COLOR_DEV, label="dev loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, macro_f1, marker="o", color=COLOR_DEV, label="dev macro-F1", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Validation Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.suptitle("Training progress", fontsize=13, color=TB_TEXT, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight", facecolor=TB_BG)
    plt.close(fig)


def save_per_category_f1_curves(history: list[dict[str, Any]], out_dir: Path) -> None:
    """Per-category dev F1 over epochs (TensorBoard scalars style)."""
    _apply_tb_style()
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))

    for i, cat in enumerate(CATEGORIES):
        series = [h["dev_per_category_f1"].get(cat, 0.0) for h in history]
        ax.plot(
            epochs,
            series,
            marker="o",
            label=CATEGORY_LABELS_HU.get(cat, cat),
            color=colors[i],
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("Per-category validation F1")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_category_f1_curves.png", dpi=150, bbox_inches="tight", facecolor=TB_BG)
    plt.close(fig)


def save_category_evaluation_bars(
    good_bad: dict[str, dict[str, float]],
    out_dir: Path,
    *,
    title: str = "Test set evaluation per category",
    filename: str = "test_category_good_bad.png",
    sort_by_bad: bool = True,
) -> None:
    """
    Stacked horizontal bar chart: jó (good) vs rossz (bad) percentages per category.
    Design inspired by intent evaluation reports.
    """
    _apply_tb_style()
    cats = list(CATEGORIES)
    if sort_by_bad:
        cats.sort(key=lambda c: good_bad[c]["bad_pct"], reverse=True)

    labels = [CATEGORY_LABELS_HU.get(c, c) for c in cats]
    good = [good_bad[c]["good_pct"] for c in cats]
    bad = [good_bad[c]["bad_pct"] for c in cats]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cats) * 0.55)))
    y = np.arange(len(cats))
    bar_h = 0.65

    bars_good = ax.barh(y, good, height=bar_h, color=COLOR_GOOD, label="jó")
    bars_bad = ax.barh(y, bad, left=good, height=bar_h, color=COLOR_BAD, label="rossz")

    for rect, val in zip(bars_good, good, strict=True):
        if val >= 8:
            ax.text(
                rect.get_width() / 2,
                rect.get_y() + rect.get_height() / 2,
                f"{val:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    for rect, val, offset in zip(bars_bad, bad, good, strict=True):
        if val >= 8:
            ax.text(
                offset + rect.get_width() / 2,
                rect.get_y() + rect.get_height() / 2,
                f"{val:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Százalék")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight", facecolor=TB_BG)
    plt.close(fig)


def save_sample_level_chart(
    exact_match_pct: float,
    any_label_match_pct: float,
    out_dir: Path,
) -> None:
    """Overall test sample accuracy summary bar chart."""
    _apply_tb_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Exact multi-label\nmatch", "Avg category\naccuracy"]
    values = [exact_match_pct, any_label_match_pct]
    colors = [COLOR_GOOD if v >= 80 else COLOR_DEV for v in values]
    bars = ax.bar(names, values, color=colors, width=0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Százalék")
    ax.set_title("Test set sample-level accuracy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "test_sample_accuracy.png", dpi=150, bbox_inches="tight", facecolor=TB_BG)
    plt.close(fig)


def write_summary_markdown(
    out_dir: Path,
    *,
    metrics: dict[str, Any],
    good_bad: dict[str, dict[str, float]],
    dedup_stats: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# SPD Model Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Aggregate metrics",
        "",
        f"- **Macro-F1:** {metrics.get('macro_f1', 0):.4f}",
        f"- **Micro-F1:** {metrics.get('micro_f1', 0):.4f}",
        f"- **Exact multi-label match:** {metrics.get('exact_match_pct', 0):.2f}%",
        f"- **Test samples:** {metrics.get('n_samples', 0)}",
        "",
        "## Per-category good / bad (test set)",
        "",
        "| Category | jó (%) | rossz (%) |",
        "|---|---:|---:|",
    ]
    for cat in CATEGORIES:
        gb = good_bad[cat]
        label = CATEGORY_LABELS_HU.get(cat, cat)
        lines.append(f"| {label} | {gb['good_pct']:.1f} | {gb['bad_pct']:.1f} |")

    if dedup_stats:
        lines.extend(
            [
                "",
                "## Dataset deduplication",
                "",
                f"- Train/test overlap removed: {dedup_stats.get('train_test_overlap_removed', 0)}",
                f"- Train internal dupes removed: {dedup_stats.get('train_internal_dupes_removed', 0)}",
                f"- Test internal dupes removed: {dedup_stats.get('test_internal_dupes_removed', 0)}",
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `training_curves.png` — loss & macro-F1 over epochs",
            "- `per_category_f1_curves.png` — per-category F1 over epochs",
            "- `test_category_good_bad.png` — stacked good/bad bars per category",
            "- `test_sample_accuracy.png` — sample-level accuracy summary",
            "- `metrics.json` — full numeric results",
            "- `training_history.json` — epoch-by-epoch training log",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
