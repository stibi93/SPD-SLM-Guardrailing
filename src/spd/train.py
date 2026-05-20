import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from spd.categories import CATEGORIES, NUM_CATEGORIES
from spd.data import SPDDataset
from spd.evaluate import compute_metrics
from spd.model import SPDClassifier
from spd.reporting import make_report_dir, save_json, save_per_category_f1_curves, save_training_curves


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pos_weights(dataset: SPDDataset) -> torch.Tensor:
    all_labels = torch.stack([dataset[i]["labels"] for i in range(len(dataset))])
    pos = all_labels.sum(0).clamp(min=1)
    neg = len(dataset) - pos
    return (neg / pos).clamp(max=50.0)


def _eval_loader(
    model: SPDClassifier,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
) -> tuple[float, dict[str, float]]:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            total_loss += loss_fn(logits, batch["labels"]).item()
            all_logits.append(logits)
            all_labels.append(batch["labels"])
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, metrics


def train(config_path: str = "configs/train_config.yaml") -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _set_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    full_ds = SPDDataset(cfg["train_data"], tokenizer)

    dev_size = max(1, int(len(full_ds) * cfg["dev_split"]))
    train_size = len(full_ds) - dev_size
    train_ds, dev_ds = random_split(full_ds, [train_size, dev_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"])

    pos_weight = _pos_weights(full_ds)
    model = SPDClassifier(num_labels=NUM_CATEGORIES, dropout=cfg["dropout"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = str(output_dir / "best_model.pt")

    report_base = cfg.get("report_dir", "artifacts/reports")
    run_report_dir = make_report_dir(report_base)

    best_f1 = -1.0
    no_improve = 0
    patience = cfg["early_stopping_patience"]
    history: list[dict] = []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        dev_loss, metrics = _eval_loader(model, dev_loader, loss_fn)
        macro_f1 = metrics["macro_f1"]

        per_cat_f1 = {cat: metrics[f"{cat}_f1"] for cat in CATEGORIES}
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "dev_loss": round(dev_loss, 6),
                "dev_macro_f1": round(macro_f1, 6),
                "dev_micro_f1": round(metrics["micro_f1"], 6),
                "dev_per_category_f1": {k: round(v, 6) for k, v in per_cat_f1.items()},
            }
        )

        print(
            f"Epoch {epoch + 1}/{cfg['num_epochs']}  "
            f"loss={train_loss:.4f}  dev_loss={dev_loss:.4f}  "
            f"dev_macro_f1={macro_f1:.4f}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"Best dev macro-F1: {best_f1:.4f}  →  {best_model_path}")

    save_json(run_report_dir / "training_history.json", history)
    save_training_curves(history, run_report_dir)
    save_per_category_f1_curves(history, run_report_dir)

    test_data = cfg.get("test_data")
    if test_data and Path(test_data).exists():
        from spd.eval_report import evaluate_test_set

        dedup_stats = None
        dedup_path = Path(cfg["train_data"]).parent / "dedup_stats.json"
        if dedup_path.exists():
            import json

            dedup_stats = json.loads(dedup_path.read_text())

        evaluate_test_set(
            model_path=best_model_path,
            test_data=test_data,
            train_data=cfg["train_data"],
            model_name_or_path=cfg["model_name_or_path"],
            report_dir=run_report_dir,
            training_history=history,
            dedup_stats=dedup_stats,
        )

    return best_model_path


if __name__ == "__main__":
    train()
