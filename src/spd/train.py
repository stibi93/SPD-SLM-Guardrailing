import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from spd.categories import NUM_CATEGORIES
from spd.data import SPDDataset
from spd.evaluate import compute_metrics
from spd.model import SPDClassifier


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pos_weights(dataset: SPDDataset) -> torch.Tensor:
    all_labels = torch.stack([dataset[i]["labels"] for i in range(len(dataset))])
    pos = all_labels.sum(0).clamp(min=1)
    neg = len(dataset) - pos
    return (neg / pos).clamp(max=50.0)  # cap at 50 to avoid extreme values


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

    best_f1 = -1.0
    no_improve = 0
    patience = cfg["early_stopping_patience"]

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

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                all_logits.append(model(batch["input_ids"], batch["attention_mask"]))
                all_labels.append(batch["labels"])

        metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
        macro_f1 = metrics["macro_f1"]
        print(
            f"Epoch {epoch + 1}/{cfg['num_epochs']}  "
            f"loss={epoch_loss / len(train_loader):.4f}  "
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
    return best_model_path


if __name__ == "__main__":
    train()
