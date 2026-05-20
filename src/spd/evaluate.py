import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from spd.categories import CATEGORIES


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy().astype(int)
    preds = (probs > threshold).astype(int)

    result: dict[str, float] = {}
    per_cat_f1s = []

    for i, cat in enumerate(CATEGORIES):
        result[f"{cat}_precision"] = float(
            precision_score(y_true[:, i], preds[:, i], zero_division=0)
        )
        result[f"{cat}_recall"] = float(
            recall_score(y_true[:, i], preds[:, i], zero_division=0)
        )
        cat_f1 = float(
            f1_score(y_true[:, i], preds[:, i], zero_division=0)
        )
        result[f"{cat}_f1"] = cat_f1

        if y_true[:, i].sum() > 0:
            per_cat_f1s.append(cat_f1)
            result[f"{cat}_roc_auc"] = float(
                roc_auc_score(y_true[:, i], probs[:, i])
            )

    result["macro_f1"] = float(np.mean(per_cat_f1s)) if per_cat_f1s else 0.0
    result["micro_f1"] = float(f1_score(y_true, preds, average="micro", zero_division=0))

    return result
