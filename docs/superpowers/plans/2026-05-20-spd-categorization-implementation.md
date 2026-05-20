# SPD Categorization Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Hungarian GDPR Article 9 multi-label classifier (huBERT-base + ONNX INT8) served as a FastAPI microservice for the OTP Bank chat guardrailing layer.

**Architecture:** Fine-tune `SZTAKI-HLT/hubert-base-cc` with an 8-output multi-label classification head; export to ONNX with INT8 dynamic quantization; serve via a stateless FastAPI service with per-category configurable thresholds. The chat backend calls `POST /v1/spd-check` synchronously before forwarding a user message to the guardrailing layer.

**Tech Stack:** Python 3.11, PyTorch 2.2+, HuggingFace Transformers 4.40+, ONNX Runtime 1.18+, FastAPI 0.110+, scikit-learn 1.4+, prometheus-client 0.20+, PyYAML 6+, pytest 8+, httpx 0.27+.

---

## File Map

```
slm-spd-guardrailing/
├── pyproject.toml                    # package definition + deps
├── .gitignore
├── configs/
│   ├── train_config.yaml             # all training hyperparameters
│   └── thresholds.yaml               # per-category decision thresholds
├── data/
│   ├── raw/          (.gitignored)   # organic JSONL from business team
│   ├── synthetic/    (.gitignored)   # Azure OpenAI generated samples
│   └── processed/    (.gitignored)   # train.jsonl / dev.jsonl / test.jsonl
├── artifacts/        (.gitignored)   # trained weights + exported model.onnx
├── src/spd/
│   ├── __init__.py
│   ├── categories.py                 # Article 9 enum + CATEGORIES list
│   ├── model.py                      # SPDClassifier (huBERT + head) + constants
│   ├── data.py                       # SPDDataset (JSONL → tokenized batches)
│   ├── train.py                      # training loop entry point
│   ├── evaluate.py                   # per-category + macro metrics
│   ├── export_onnx.py                # FP32 checkpoint → ONNX → INT8 quant
│   ├── inference.py                  # SPDPredictor (ONNX Runtime)
│   ├── service.py                    # FastAPI app + endpoints
│   └── synth.py                      # Azure OpenAI synthetic data generator
├── tests/
│   ├── __init__.py
│   ├── test_categories.py
│   ├── test_model.py
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_inference.py
│   └── test_service.py
└── scripts/
    ├── download_model.py             # pre-cache huBERT from HuggingFace Hub
    └── benchmark_latency.py          # measure p50/p95/p99 end-to-end latency
```

---

## ★ SMOKE TEST MILESTONE (Tasks 1–4)

After Task 4 you have: the project installed, the base model cached locally, and a verified forward pass on dummy Hungarian text. This is the agreed first working checkpoint.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `configs/train_config.yaml`
- Create: `configs/thresholds.yaml`
- Create: `src/spd/__init__.py`, `tests/__init__.py`
- Create: `data/raw/.gitkeep`, `data/synthetic/.gitkeep`, `data/processed/.gitkeep`, `artifacts/.gitkeep`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.build_meta"

[project]
name = "slm-spd-guardrailing"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "transformers>=4.40",
    "torch>=2.2",
    "onnxruntime>=1.18",
    "onnx>=1.16",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "scikit-learn>=1.4",
    "prometheus-client>=0.20",
    "numpy>=1.26",
    "openai>=1.30",
    "accelerate>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create .gitignore**

```
data/raw/
data/synthetic/
data/processed/
artifacts/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.pytest_cache/
```

- [ ] **Step 3: Create configs/train_config.yaml**

```yaml
model_name_or_path: SZTAKI-HLT/hubert-base-cc
train_data: data/processed/train.jsonl
output_dir: artifacts/checkpoints
seed: 42
learning_rate: 2.0e-5
weight_decay: 0.01
dropout: 0.1
num_epochs: 5
batch_size: 8
warmup_ratio: 0.1
dev_split: 0.1
early_stopping_patience: 1
```

- [ ] **Step 4: Create configs/thresholds.yaml**

```yaml
version: 1
defaults:
  threshold: 0.5
per_category:
  ethnicity: 0.5
  political_opinion: 0.5
  religion_belief: 0.5
  trade_union: 0.5
  genetic: 0.5
  biometric: 0.5
  health: 0.5
  sex_life_orientation: 0.5
decision_logic: any_over_threshold
```

- [ ] **Step 5: Create empty init files and data directories**

```bash
mkdir -p src/spd tests scripts data/raw data/synthetic data/processed artifacts
touch src/spd/__init__.py tests/__init__.py
touch data/raw/.gitkeep data/synthetic/.gitkeep data/processed/.gitkeep artifacts/.gitkeep
```

- [ ] **Step 6: Install the package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors; `pip show slm-spd-guardrailing` shows version 0.1.0.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore configs/ src/ tests/ scripts/ data/ artifacts/
git commit -m "chore: project scaffold, configs, dependencies"
```

---

### Task 2: Article 9 categories module

**Files:**
- Create: `src/spd/categories.py`
- Create: `tests/test_categories.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_categories.py
from spd.categories import Article9Category, CATEGORIES, NUM_CATEGORIES

def test_num_categories():
    assert NUM_CATEGORIES == 8

def test_categories_list_length():
    assert len(CATEGORIES) == 8

def test_expected_keys():
    expected = {
        "ethnicity", "political_opinion", "religion_belief", "trade_union",
        "genetic", "biometric", "health", "sex_life_orientation",
    }
    assert set(CATEGORIES) == expected

def test_enum_value_matches_key():
    assert Article9Category.HEALTH.value == "health"
    assert Article9Category.POLITICAL_OPINION.value == "political_opinion"

def test_categories_order_is_stable():
    # Order must be stable: it maps directly to model output indices.
    assert CATEGORIES[6] == "health"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_categories.py -v
```

Expected: `ModuleNotFoundError: No module named 'spd.categories'`

- [ ] **Step 3: Implement categories.py**

```python
# src/spd/categories.py
from enum import Enum


class Article9Category(str, Enum):
    ETHNICITY = "ethnicity"
    POLITICAL_OPINION = "political_opinion"
    RELIGION_BELIEF = "religion_belief"
    TRADE_UNION = "trade_union"
    GENETIC = "genetic"
    BIOMETRIC = "biometric"
    HEALTH = "health"
    SEX_LIFE_ORIENTATION = "sex_life_orientation"


CATEGORIES: list[str] = [c.value for c in Article9Category]
NUM_CATEGORIES: int = len(CATEGORIES)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_categories.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/spd/categories.py tests/test_categories.py
git commit -m "feat: Article 9 category enum and constants"
```

---

### Task 3: Model class and forward-pass smoke test

**Files:**
- Create: `src/spd/model.py`
- Create: `tests/test_model.py`

Note: the test in this task **downloads huBERT (~440 MB) on first run**. Run `scripts/download_model.py` first (Task 4) if you want the cache pre-warmed; otherwise pytest will download on first call to `from_pretrained`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
import pytest
from transformers import AutoTokenizer
from spd.model import SPDClassifier, MODEL_NAME, MAX_LENGTH
from spd.categories import NUM_CATEGORIES


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def model():
    return SPDClassifier()


def test_output_shape(tokenizer, model):
    enc = tokenizer(
        "Cukorbeteg vagyok.",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
    assert logits.shape == (1, NUM_CATEGORIES)


def test_output_dtype(tokenizer, model):
    enc = tokenizer("teszt", max_length=MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
    assert logits.dtype == torch.float32


def test_batch_processing(tokenizer, model):
    texts = ["Diabéteszes vagyok.", "Mennyi a számlaegyenlegem?"]
    enc = tokenizer(texts, max_length=MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
    assert logits.shape == (2, NUM_CATEGORIES)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'spd.model'`

- [ ] **Step 3: Implement model.py**

```python
# src/spd/model.py
import torch
import torch.nn as nn
from transformers import AutoModel
from spd.categories import NUM_CATEGORIES

MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"
MAX_LENGTH = 128


class SPDClassifier(nn.Module):
    def __init__(self, num_labels: int = NUM_CATEGORIES, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])  # [CLS] token
        return self.classifier(pooled)  # shape: (batch, num_labels)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_model.py -v
```

Expected: 3 passed. First run downloads ~440 MB to `~/.cache/huggingface/`.

- [ ] **Step 5: Commit**

```bash
git add src/spd/model.py tests/test_model.py
git commit -m "feat: SPDClassifier — huBERT encoder + 8-output multi-label head"
```

---

### Task 4: download_model.py script (cache pre-warmer)

**Files:**
- Create: `scripts/download_model.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python
# scripts/download_model.py
"""Pre-cache huBERT tokenizer and weights from HuggingFace Hub."""
from transformers import AutoTokenizer, AutoModel
from spd.model import MODEL_NAME


def main() -> None:
    print(f"Downloading {MODEL_NAME} ...")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModel.from_pretrained(MODEL_NAME)
    print("Done. Cached at ~/.cache/huggingface/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify the model is cached**

```bash
python scripts/download_model.py
```

Expected output:
```
Downloading SZTAKI-HLT/hubert-base-cc ...
Done. Cached at ~/.cache/huggingface/
```

- [ ] **Step 3: Verify smoke test — forward pass on dummy Hungarian text**

```bash
python -c "
import torch
from transformers import AutoTokenizer
from spd.model import SPDClassifier, MODEL_NAME, MAX_LENGTH
from spd.categories import CATEGORIES

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
m = SPDClassifier()
m.eval()
enc = tok('Cukorbeteg vagyok és hitelt szeretnék.', max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
with torch.no_grad():
    logits = m(enc['input_ids'], enc['attention_mask'])
    probs = torch.sigmoid(logits)[0]
for cat, p in zip(CATEGORIES, probs):
    print(f'  {cat}: {p:.4f}')
print('Smoke test PASSED — model loaded and forward pass OK.')
"
```

Expected: 8 probability scores printed (random, ~0.5 each since untrained), then `Smoke test PASSED`.

- [ ] **Step 4: Commit**

```bash
git add scripts/download_model.py
git commit -m "feat: download_model script + smoke test verified"
```

---

## ★ END OF SMOKE TEST MILESTONE

---

### Task 5: Data loading

**Files:**
- Create: `src/spd/data.py`
- Create: `tests/test_data.py`
- Create: `data/processed/smoke_test.jsonl` (tiny 4-example fixture for tests)

- [ ] **Step 1: Create the test fixture JSONL**

```bash
cat > data/processed/smoke_test.jsonl << 'EOF'
{"id": "001", "text": "Cukorbeteg vagyok.", "lang": "hu", "labels": {"ethnicity": 0, "political_opinion": 0, "religion_belief": 0, "trade_union": 0, "genetic": 0, "biometric": 0, "health": 1, "sex_life_orientation": 0}, "source": "organic", "annotator": "test", "notes": ""}
{"id": "002", "text": "Mennyi a számlaegyenlegem?", "lang": "hu", "labels": {"ethnicity": 0, "political_opinion": 0, "religion_belief": 0, "trade_union": 0, "genetic": 0, "biometric": 0, "health": 0, "sex_life_orientation": 0}, "source": "organic", "annotator": "test", "notes": ""}
{"id": "003", "text": "Református vagyok, az egyházamnak szeretnék adományozni.", "lang": "hu", "labels": {"ethnicity": 0, "political_opinion": 0, "religion_belief": 1, "trade_union": 0, "genetic": 0, "biometric": 0, "health": 0, "sex_life_orientation": 0}, "source": "organic", "annotator": "test", "notes": ""}
{"id": "004", "text": "Szakszervezeti tagdíjat szeretnék utalni.", "lang": "hu", "labels": {"ethnicity": 0, "political_opinion": 0, "religion_belief": 0, "trade_union": 1, "genetic": 0, "biometric": 0, "health": 0, "sex_life_orientation": 0}, "source": "organic", "annotator": "test", "notes": ""}
EOF
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_data.py
import pytest
import torch
from transformers import AutoTokenizer
from spd.data import SPDDataset
from spd.model import MODEL_NAME, MAX_LENGTH
from spd.categories import CATEGORIES, NUM_CATEGORIES

FIXTURE = "data/processed/smoke_test.jsonl"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def dataset(tokenizer):
    return SPDDataset(FIXTURE, tokenizer)


def test_dataset_length(dataset):
    assert len(dataset) == 4


def test_item_keys(dataset):
    item = dataset[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}


def test_input_ids_shape(dataset):
    item = dataset[0]
    assert item["input_ids"].shape == (MAX_LENGTH,)
    assert item["attention_mask"].shape == (MAX_LENGTH,)


def test_labels_shape(dataset):
    item = dataset[0]
    assert item["labels"].shape == (NUM_CATEGORIES,)
    assert item["labels"].dtype == torch.float32


def test_health_label_positive(dataset):
    # First example is health-positive
    assert dataset[0]["labels"][CATEGORIES.index("health")] == 1.0


def test_negative_example(dataset):
    # Second example has all zeros
    assert dataset[1]["labels"].sum() == 0.0
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_data.py -v
```

Expected: `ModuleNotFoundError: No module named 'spd.data'`

- [ ] **Step 4: Implement data.py**

```python
# src/spd/data.py
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from spd.categories import CATEGORIES
from spd.model import MAX_LENGTH


class SPDDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.examples: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [float(ex["labels"][cat]) for cat in CATEGORIES],
            dtype=torch.float32,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
```

- [ ] **Step 5: Run tests and verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/spd/data.py tests/test_data.py data/processed/smoke_test.jsonl
git commit -m "feat: SPDDataset — JSONL loader with tokenization"
```

---

### Task 6: Evaluation harness

**Files:**
- Create: `src/spd/evaluate.py`
- Create: `tests/test_evaluate.py`

(Training depends on evaluate.py for early stopping, so implement evaluation before training.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_evaluate.py -v
```

Expected: `ModuleNotFoundError: No module named 'spd.evaluate'`

- [ ] **Step 3: Implement evaluate.py**

```python
# src/spd/evaluate.py
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
    preds = (probs >= threshold).astype(int)

    result: dict[str, float] = {
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, preds, average="micro", zero_division=0)),
    }

    for i, cat in enumerate(CATEGORIES):
        result[f"{cat}_precision"] = float(
            precision_score(y_true[:, i], preds[:, i], zero_division=0)
        )
        result[f"{cat}_recall"] = float(
            recall_score(y_true[:, i], preds[:, i], zero_division=0)
        )
        result[f"{cat}_f1"] = float(
            f1_score(y_true[:, i], preds[:, i], zero_division=0)
        )
        if y_true[:, i].sum() > 0:
            result[f"{cat}_roc_auc"] = float(
                roc_auc_score(y_true[:, i], probs[:, i])
            )

    return result
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_evaluate.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/spd/evaluate.py tests/test_evaluate.py
git commit -m "feat: per-category + macro evaluation metrics"
```

---

### Task 7: Training loop

**Files:**
- Create: `src/spd/train.py`

No separate test file: the training loop is validated by running it end-to-end on the smoke_test.jsonl fixture and observing loss decrease. A full automated test would require GPU time or minutes of CPU time; that's not appropriate for the unit test suite.

- [ ] **Step 1: Implement train.py**

```python
# src/spd/train.py
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
```

- [ ] **Step 2: Run training on the smoke fixture to verify the loop works end-to-end**

```bash
python -c "
import yaml, pathlib
cfg = yaml.safe_load(open('configs/train_config.yaml'))
cfg['train_data'] = 'data/processed/smoke_test.jsonl'
cfg['num_epochs'] = 2
cfg['output_dir'] = 'artifacts/smoke_checkpoints'
pathlib.Path('/tmp/smoke_cfg.yaml').write_text(__import__('yaml').dump(cfg))
from spd.train import train
path = train('/tmp/smoke_cfg.yaml')
print('Training loop OK, checkpoint at:', path)
import pathlib; assert pathlib.Path(path).exists()
"
```

Expected: 2 epoch lines printed, `Training loop OK` at the end.

- [ ] **Step 3: Commit**

```bash
git add src/spd/train.py
git commit -m "feat: training loop with early stopping and pos_weight"
```

---

### Task 8: ONNX export and INT8 quantization

**Files:**
- Create: `src/spd/export_onnx.py`

- [ ] **Step 1: Implement export_onnx.py**

```python
# src/spd/export_onnx.py
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from spd.categories import NUM_CATEGORIES
from spd.model import MAX_LENGTH, MODEL_NAME, SPDClassifier


def export(
    model_path: str,
    output_dir: str,
    model_name_or_path: str = MODEL_NAME,
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = SPDClassifier(num_labels=NUM_CATEGORIES)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = tokenizer(
        "teszt szöveg",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    fp32_path = out / "model_fp32.onnx"
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )

    int8_path = out / "model.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    fp32_path.unlink()

    size_mb = int8_path.stat().st_size / 1e6
    print(f"INT8 model exported to {int8_path}  ({size_mb:.1f} MB)")
    return str(int8_path)


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/checkpoints/best_model.pt"
    export(model_path, "artifacts")
```

- [ ] **Step 2: Export the untrained (smoke) checkpoint to ONNX to verify the pipeline**

First save an untrained checkpoint so we can test export without real training:

```bash
python -c "
import torch
from spd.model import SPDClassifier
from pathlib import Path
Path('artifacts/smoke_checkpoints').mkdir(parents=True, exist_ok=True)
torch.save(SPDClassifier().state_dict(), 'artifacts/smoke_checkpoints/best_model.pt')
print('Untrained checkpoint saved.')
"
```

Then export it:

```bash
python -c "
from spd.export_onnx import export
path = export('artifacts/smoke_checkpoints/best_model.pt', 'artifacts/smoke_onnx')
print('ONNX export OK:', path)
import pathlib; assert pathlib.Path(path).exists()
"
```

Expected: `INT8 model exported to artifacts/smoke_onnx/model.onnx  (~30–80 MB)`

- [ ] **Step 3: Commit**

```bash
git add src/spd/export_onnx.py
git commit -m "feat: FP32 checkpoint → ONNX → INT8 dynamic quantization"
```

---

### Task 9: ONNX inference predictor

**Files:**
- Create: `src/spd/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
import pytest
from spd.inference import SPDPredictor
from spd.categories import CATEGORIES, NUM_CATEGORIES

SMOKE_ONNX = "artifacts/smoke_onnx/model.onnx"
TOKENIZER = "SZTAKI-HLT/hubert-base-cc"
THRESHOLDS = {cat: 0.5 for cat in CATEGORIES}


@pytest.fixture(scope="module")
def predictor():
    return SPDPredictor(SMOKE_ONNX, TOKENIZER, THRESHOLDS)


def test_predict_returns_required_keys(predictor):
    result = predictor.predict("Cukorbeteg vagyok.")
    assert set(result.keys()) == {"scores", "categories_triggered", "spd_detected", "latency_ms"}


def test_predict_scores_have_all_categories(predictor):
    result = predictor.predict("teszt")
    assert set(result["scores"].keys()) == set(CATEGORIES)


def test_scores_in_range(predictor):
    result = predictor.predict("Diabéteszes vagyok és hitelt kérek.")
    for cat, score in result["scores"].items():
        assert 0.0 <= score <= 1.0, f"{cat} score {score} out of range"


def test_latency_ms_is_positive(predictor):
    result = predictor.predict("teszt")
    assert result["latency_ms"] > 0


def test_categories_triggered_is_subset_of_categories(predictor):
    result = predictor.predict("teszt")
    assert set(result["categories_triggered"]).issubset(set(CATEGORIES))


def test_spd_detected_consistent_with_triggered(predictor):
    result = predictor.predict("Anyukámnak rákja van.")
    assert result["spd_detected"] == (len(result["categories_triggered"]) > 0)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_inference.py -v
```

Expected: `ModuleNotFoundError: No module named 'spd.inference'`

- [ ] **Step 3: Implement inference.py**

```python
# src/spd/inference.py
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from spd.categories import CATEGORIES
from spd.model import MAX_LENGTH


class SPDPredictor:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        thresholds: dict[str, float],
    ) -> None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.thresholds = thresholds

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()

        enc = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        logits: np.ndarray = self.session.run(
            ["logits"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0][0]

        scores = {
            cat: float(1.0 / (1.0 + np.exp(-logits[i])))
            for i, cat in enumerate(CATEGORIES)
        }
        triggered = [
            cat for cat, score in scores.items()
            if score >= self.thresholds.get(cat, 0.5)
        ]
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "scores": scores,
            "categories_triggered": triggered,
            "spd_detected": len(triggered) > 0,
            "latency_ms": round(latency_ms, 1),
        }
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_inference.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/spd/inference.py tests/test_inference.py
git commit -m "feat: SPDPredictor — ONNX Runtime CPU inference with configurable thresholds"
```

---

### Task 10: FastAPI service

**Files:**
- Create: `src/spd/service.py`
- Create: `tests/test_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_service.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from spd.categories import CATEGORIES


MOCK_PREDICT_RESULT = {
    "scores": {cat: 0.01 for cat in CATEGORIES} | {"health": 0.92},
    "categories_triggered": ["health"],
    "spd_detected": True,
    "latency_ms": 18.4,
}


@pytest.fixture(scope="module")
def client():
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = MOCK_PREDICT_RESULT

    with patch("spd.service.SPDPredictor") as MockCls, \
         patch("spd.service.load_thresholds", return_value={c: 0.5 for c in CATEGORIES}):
        MockCls.return_value = mock_predictor
        from spd.service import app
        return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_spd_check_200(client):
    r = client.post("/v1/spd-check", json={"text": "Cukorbeteg vagyok."})
    assert r.status_code == 200


def test_spd_check_response_keys(client):
    r = client.post("/v1/spd-check", json={"text": "teszt"})
    body = r.json()
    assert "request_id" in body
    assert "model_version" in body
    assert "scores" in body
    assert "spd_detected" in body
    assert "latency_ms" in body


def test_spd_check_scores_has_all_categories(client):
    r = client.post("/v1/spd-check", json={"text": "teszt"})
    assert set(r.json()["scores"].keys()) == set(CATEGORIES)


def test_spd_detected_true(client):
    r = client.post("/v1/spd-check", json={"text": "irrelevant — mock returns health=0.92"})
    assert r.json()["spd_detected"] is True
    assert "health" in r.json()["categories_triggered"]


def test_text_too_long_rejected(client):
    r = client.post("/v1/spd-check", json={"text": "x" * 9000})
    assert r.status_code == 422


def test_info_endpoint(client):
    r = client.get("/v1/info")
    assert r.status_code == 200
    assert "model_version" in r.json()
    assert "thresholds" in r.json()


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "spd_requests_total" in r.text
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'spd.service'`

- [ ] **Step 3: Implement service.py**

```python
# src/spd/service.py
import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from spd.categories import CATEGORIES
from spd.inference import SPDPredictor

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("spd_requests_total", "Total SPD check requests")
LATENCY_HIST = Histogram("spd_latency_seconds", "SPD check latency")
DETECTED_COUNT = Counter("spd_detected_total", "Requests where SPD was detected")

MODEL_VERSION = "spd-hubert-v1.0.0"

_predictor: SPDPredictor | None = None
_thresholds: dict[str, float] = {}


def load_thresholds(path: str) -> dict[str, float]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    default = cfg.get("defaults", {}).get("threshold", 0.5)
    return {cat: cfg.get("per_category", {}).get(cat, default) for cat in CATEGORIES}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _thresholds
    model_path = os.environ.get("MODEL_PATH", "artifacts/model.onnx")
    tokenizer_name = os.environ.get("TOKENIZER_NAME", "SZTAKI-HLT/hubert-base-cc")
    threshold_path = os.environ.get("THRESHOLD_PATH", "configs/thresholds.yaml")
    _thresholds = load_thresholds(threshold_path)
    _predictor = SPDPredictor(model_path, tokenizer_name, _thresholds)
    yield


app = FastAPI(lifespan=lifespan)


class CheckRequest(BaseModel):
    text: str = Field(..., max_length=8192)
    lang: str = "hu"
    request_id: str | None = None


class CheckResponse(BaseModel):
    request_id: str
    model_version: str
    lang_detected: str
    scores: dict[str, float]
    categories_triggered: list[str]
    spd_detected: bool
    latency_ms: float


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


@app.get("/v1/info")
def info() -> dict:
    return {"model_version": MODEL_VERSION, "thresholds": _thresholds}


@app.post("/v1/spd-check", response_model=CheckResponse)
def spd_check(req: CheckRequest) -> CheckResponse:
    REQUEST_COUNT.inc()
    request_id = req.request_id or str(uuid.uuid4())

    with LATENCY_HIST.time():
        result = _predictor.predict(req.text)

    if result["spd_detected"]:
        DETECTED_COUNT.inc()

    logger.info(
        str({
            "request_id": request_id,
            "model_version": MODEL_VERSION,
            "lang_detected": "hu",
            "scores": result["scores"],
            "categories_triggered": result["categories_triggered"],
            "spd_detected": result["spd_detected"],
            "latency_ms": result["latency_ms"],
            "text_sha256": hashlib.sha256(req.text.encode()).hexdigest(),
            "text_len": len(req.text),
        })
    )

    return CheckResponse(
        request_id=request_id,
        model_version=MODEL_VERSION,
        lang_detected="hu",
        **result,
    )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_service.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Run the full test suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass, no errors.

- [ ] **Step 6: Commit**

```bash
git add src/spd/service.py tests/test_service.py
git commit -m "feat: FastAPI service with /v1/spd-check, /healthz, /metrics, /v1/info"
```

---

### Task 11: Latency benchmark script

**Files:**
- Create: `scripts/benchmark_latency.py`

- [ ] **Step 1: Implement benchmark_latency.py**

```python
#!/usr/bin/env python
# scripts/benchmark_latency.py
"""Measure SPDPredictor p50/p95/p99 latency on representative Hungarian texts."""
import argparse
import statistics
import time

from spd.categories import CATEGORIES
from spd.inference import SPDPredictor

SAMPLE_TEXTS = [
    "Cukorbeteg vagyok és szeretnék hitelt felvenni.",
    "Mennyi a számlaegyenlegem?",
    "Református vagyok, adományozni szeretnék.",
    "Szakszervezeti tagdíjat fizetek minden hónapban.",
    "Mikor jár le a folyószámlahitelem?",
    "Anyukámnak rákdiagnózisa van.",
    "Hogyan tudom lezárni a bankszámlámat?",
    "Roma vagyok, kapok-e segélyt?",
    "Mikor kapom meg az utalásomat?",
    "HIV-pozitív vagyok, ez befolyásolja a hitelbírálatot?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="artifacts/model.onnx")
    parser.add_argument("--tokenizer", default="SZTAKI-HLT/hubert-base-cc")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    thresholds = {cat: 0.5 for cat in CATEGORIES}
    predictor = SPDPredictor(args.model_path, args.tokenizer, thresholds)

    print(f"Warming up ({args.warmup} reps)...")
    for i in range(args.warmup):
        predictor.predict(SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)])

    print(f"Benchmarking ({args.reps} reps)...")
    latencies: list[float] = []
    for i in range(args.reps):
        t0 = time.perf_counter()
        predictor.predict(SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)])
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)

    print(f"\nLatency over {args.reps} requests:")
    print(f"  mean = {mean:.1f} ms")
    print(f"  p50  = {p50:.1f} ms")
    print(f"  p95  = {p95:.1f} ms   ← target: ≤200 ms")
    print(f"  p99  = {p99:.1f} ms")
    gate = "PASS" if p95 <= 200.0 else "FAIL"
    print(f"\nLatency gate (p95 ≤ 200 ms): {gate}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against the smoke ONNX model (untrained) to confirm the script runs**

```bash
python scripts/benchmark_latency.py --model-path artifacts/smoke_onnx/model.onnx --reps 50
```

Expected: latency stats printed; p95 should be well under 200 ms.

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_latency.py
git commit -m "feat: latency benchmark script (p50/p95/p99 against target)"
```

---

### Task 12: Synthetic data generator

**Files:**
- Create: `src/spd/synth.py`

This task requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` environment variables. If those are not set, the script logs a clear error and exits.

- [ ] **Step 1: Implement synth.py**

```python
# src/spd/synth.py
"""
Generate synthetic Hungarian training examples per Article 9 category
using Azure OpenAI. Output is JSONL written to data/synthetic/.

Usage:
    export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
    export AZURE_OPENAI_API_KEY=<your-key>
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o   # or your deployment name
    python -m spd.synth --category health --count 100 --out data/synthetic/health.jsonl
"""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from openai import AzureOpenAI

from spd.categories import Article9Category, CATEGORIES

_SYSTEM = (
    "Te egy adatgeneráló asszisztens vagy. Kizárólag válaszolj egy JSON tömbbel, "
    "semmi más magyarázattal."
)

_PROMPTS: dict[str, str] = {
    "ethnicity": (
        "Generálj {n} rövid, természetes hangvételű magyar bankos chat üzenetet (1-3 mondat), "
        "amelyek tartalmazzák a feladó faji vagy etnikai hátterét. "
        "Légy változatos: közvetlen kijelentés, harmadik személyre utalás, implicit utalás. "
        "Válasz: JSON tömb stringekkel."
    ),
    "political_opinion": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek tartalmazzák a feladó "
        "politikai nézetét vagy párttagságát. JSON tömb."
    ),
    "religion_belief": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek tartalmazzák a feladó "
        "vallási vagy világnézeti hovatartozását (pl. katolikus, református, zsidó, ateista). "
        "JSON tömb."
    ),
    "trade_union": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek tartalmazzák a feladó "
        "szakszervezeti tagságát. JSON tömb."
    ),
    "genetic": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek genetikai adatot tartalmaznak "
        "(pl. örökletes betegség, DNS-teszt). JSON tömb."
    ),
    "biometric": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek biometrikus azonosítást "
        "tartalmaznak (pl. ujjlenyomat, arcfelismerés bankba belépéshez). JSON tömb."
    ),
    "health": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek egészségügyi adatot "
        "tartalmaznak (betegség, fogyatékosság, gyógyszer, kezelés). JSON tömb."
    ),
    "sex_life_orientation": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek tartalmazzák a feladó "
        "szexuális életét vagy orientációját. JSON tömb."
    ),
}


def _make_negative_prompt(n: int) -> str:
    return (
        f"Generálj {n} rövid, természetes hangvételű magyar bankos chat üzenetet (1-3 mondat), "
        "amelyek NEM tartalmaznak GDPR 9. cikk szerinti különleges kategóriájú adatot. "
        "Egyszerű banki kérdések, tranzakciók, számlaegyenleg, hitelkérelmek. JSON tömb."
    )


def generate(category: str, count: int, out_path: str) -> None:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    if not endpoint or not api_key:
        print(
            "ERROR: Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-02-01")

    is_negative = category == "negative"
    prompt = _make_negative_prompt(count) if is_negative else _PROMPTS[category].format(n=count)

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    texts: list[str] = json.loads(raw) if isinstance(json.loads(raw), list) else list(json.loads(raw).values())[0]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "a") as f:
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            labels = {cat: 0 for cat in CATEGORIES}
            if not is_negative:
                labels[category] = 1
            record = {
                "id": str(uuid.uuid4()),
                "text": text.strip(),
                "lang": "hu",
                "labels": labels,
                "source": "synthetic",
                "annotator": f"azure-{deployment}",
                "notes": "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True,
                        choices=CATEGORIES + ["negative"],
                        help="Article 9 category or 'negative'")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    generate(args.category, args.count, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable (no credentials needed for this check)**

```bash
python -c "import spd.synth; print('synth.py importable OK')"
```

Expected: `synth.py importable OK`

- [ ] **Step 3: Commit**

```bash
git add src/spd/synth.py
git commit -m "feat: Azure OpenAI synthetic Hungarian data generator per Article 9 category"
```

---

## Self-Review Against Spec

### Spec coverage check

| Spec section | Covered by task(s) |
|---|---|
| §3 functional requirements (8 categories, per-category scores, decision policy) | Task 2, Task 9, Task 10 |
| §3.4 non-functional: latency p95 ≤ 200 ms | Task 8 (INT8 quantization), Task 11 (benchmark) |
| §4 architecture (stateless HTTP service, ONNX) | Task 10 (service), Task 8 (export) |
| §5 label space | Task 2 (categories.py) |
| §6 data plan, labeling schema | Task 5 (SPDDataset), smoke_test.jsonl fixture, Task 12 (synth.py) |
| §7 model & training | Task 3 (model), Task 7 (train) |
| §8 evaluation (metrics, per-category) | Task 6 (evaluate) |
| §8.4 acceptance gate (macro-F1 ≥ 0.80, recall ≥ 0.70) | Task 6 (compute_metrics), benchmark: Task 11 |
| §9 API contract (POST /v1/spd-check, healthz, metrics, info) | Task 10 |
| §10 logging policy (no raw text, sha256, text_len) | Task 10 (service.py logger) |
| §11 threshold config (YAML, per-category) | Task 1 (thresholds.yaml), Task 10 (load_thresholds) |
| §12 repo layout | All tasks |
| §16 success criteria #5 (runnable scripts) | Tasks 4, 7, 11, 12 |

All sections covered. No gaps.

### Placeholder scan

None found. Every step has runnable code or commands with expected output.

### Type consistency

- `CATEGORIES: list[str]` defined in Task 2, consumed as-is in Tasks 5, 6, 9, 10, 12.
- `SPDClassifier.forward(input_ids, attention_mask) → Tensor` defined in Task 3, used in Tasks 7 and 8 correctly.
- `SPDPredictor.predict(text: str) → dict` with keys `scores, categories_triggered, spd_detected, latency_ms` defined in Task 9, destructured with `**result` in Task 10 — matches `CheckResponse` fields.
- `compute_metrics(logits: Tensor, labels: Tensor) → dict[str, float]` defined in Task 6, used in `train.py` Task 7 as `metrics["macro_f1"]` — key present.

No type mismatches found.
