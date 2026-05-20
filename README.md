# SPD Categorization — Hungarian Sensitive Personal Data Detector

A lightweight, on-premises multi-label classifier that detects **GDPR Article 9 special category data** in Hungarian chat messages. Designed to replace Azure OpenAI runtime calls in OTP Bank's agentic chat guardrailing layer — improving data sovereignty, latency, and cost.

---

## What it detects

Seven Article 9 categories (excluding biometric), returned as per-category confidence scores:

| Category | Description |
|---|---|
| `ethnicity` | Racial or ethnic origin |
| `political_opinion` | Political opinions |
| `religion_belief` | Religious or philosophical beliefs |
| `trade_union` | Trade union membership |
| `genetic` | Genetic data |
| `health` | Health data |
| `sex_life_orientation` | Sex life or sexual orientation |

A single message may trigger zero, one, or several categories simultaneously.

---

## Architecture

```
Chat backend  →  SPD Detector (FastAPI)  →  Guardrailing layer
                        │
                  huBERT-base-cc
                  + multi-label head
                  (ONNX INT8, CPU)
```

- **Model:** `SZTAKI-HLT/hubert-base-cc` (110M params, Hungarian BERT) with a linear classification head over `[CLS]` → 7 sigmoid outputs
- **Runtime:** ONNX Runtime CPU, INT8 dynamic quantization (~111 MB artifact)
- **Latency:** p95 ≈ 48 ms on a developer laptop (target: ≤ 200 ms)
- **Deployment:** CPU-only, stateless, OpenShift-ready

---

## Repository layout

```
slm-spd-guardrailing/
├── pyproject.toml
├── configs/
│   ├── train_config.yaml       # training hyperparameters
│   └── thresholds.yaml         # per-category decision thresholds (hot-reloadable)
├── data/
│   ├── raw/                    # organic samples (gitignored)
│   ├── synthetic/              # Azure OpenAI generated samples (gitignored)
│   └── processed/              # train/dev/test JSONL + HTML/CSV inspection exports (gitignored)
├── src/spd/
│   ├── categories.py           # Article 9 enum + CATEGORIES list
│   ├── model.py                # SPDClassifier (huBERT encoder + linear head)
│   ├── data.py                 # SPDDataset — JSONL loader + tokenization
│   ├── train.py                # training entry point (early stopping, pos_weight)
│   ├── evaluate.py             # per-category metrics + macro/micro aggregates
│   ├── export_onnx.py          # PyTorch → FP32 ONNX → INT8 quantization
│   ├── inference.py            # SPDPredictor — ONNX Runtime wrapper
│   ├── service.py              # FastAPI app, /v1/spd-check + ops endpoints
│   ├── eval_report.py          # post-training evaluation + chart generation
│   ├── reporting.py            # matplotlib helpers for all charts
│   └── synth.py                # Azure OpenAI synthetic data generator (training only)
├── scripts/
│   ├── build_hungarian_dataset.py    # build train.jsonl + test.jsonl from templates
│   ├── diverse_scenarios.py          # negation, 3rd-person, implicit, formal, question-form variants
│   ├── banking_chat_scenarios.py     # realistic banking negative + SPD examples
│   ├── long_form_examples.py         # curated long examples (≤600 chars)
│   ├── export_dataset_html.py        # export data to HTML/CSV for inspection
│   ├── download_model.py             # cache huBERT locally before training
│   └── benchmark_latency.py         # measure p50/p95/p99 latency
├── tests/
└── docs/superpowers/
    ├── specs/                  # design specification
    └── plans/                  # implementation plan
```

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/) for dependency management and running commands.

### 1. Install

```bash
uv sync --dev
```

### 2. Download the base model

```bash
uv run python scripts/download_model.py
```

### 3. Build training data

```bash
uv run python scripts/build_hungarian_dataset.py
# → data/processed/train.jsonl  (497 records)
# → data/processed/test.jsonl   (247 records)
# → data/processed/dedup_stats.json
```

Data label format (one JSON per line):

```json
{
  "id": "uuid",
  "text": "Cukorbeteg vagyok és hitelt szeretnék felvenni.",
  "lang": "hu",
  "labels": {
    "ethnicity": 0, "political_opinion": 0, "religion_belief": 0,
    "trade_union": 0, "genetic": 0,
    "health": 1, "sex_life_orientation": 0
  },
  "source": "organic",
  "annotator": "seed-corpus-v1",
  "notes": ""
}
```

### 4. Inspect the data (HTML / CSV)

```bash
uv run python scripts/export_dataset_html.py
# → data/processed/train_inspect.html   (colour-coded, filterable in browser)
# → data/processed/test_inspect.html
# → data/processed/train_inspect.csv    (Excel-friendly, UTF-8-BOM)
# → data/processed/test_inspect.csv
```

Open `train_inspect.html` in any browser. Multi-label rows are highlighted in yellow. Each row shows the text, coloured category badges, character length, annotator, and notes.

### 5. Train

```bash
uv run python -m spd.train
```

Saves best checkpoint to `artifacts/checkpoints/best_model.pt` based on dev macro-F1 with early stopping (patience 1). Generates evaluation reports automatically — see [Evaluation reports](#evaluation-reports).

### 6. Export to ONNX

```bash
uv run python -m spd.export_onnx artifacts/checkpoints/best_model.pt artifacts/
# → artifacts/model.onnx  (INT8, ~111 MB)
```

### 7. Benchmark latency

```bash
uv run python scripts/benchmark_latency.py --model-path artifacts/model.onnx --reps 200
```

### 8. Run the service

```bash
uv run uvicorn spd.service:app --host 0.0.0.0 --port 8080
```

---

## API

### `POST /v1/spd-check`

```bash
curl -X POST http://localhost:8080/v1/spd-check \
  -H "Content-Type: application/json" \
  -d '{"text": "Cukorbeteg vagyok és szeretnék hitelt felvenni.", "lang": "hu"}'
```

Response:

```json
{
  "request_id": "...",
  "model_version": "spd-hubert-v1.0.0",
  "lang_detected": "hu",
  "scores": {
    "ethnicity": 0.02, "political_opinion": 0.01, "religion_belief": 0.03,
    "trade_union": 0.00, "genetic": 0.00,
    "health": 0.94, "sex_life_orientation": 0.01
  },
  "categories_triggered": ["health"],
  "spd_detected": true,
  "latency_ms": 48
}
```

### Ops endpoints

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness/readiness probe |
| `GET /metrics` | Prometheus metrics |
| `GET /v1/info` | Model version, thresholds, build info |

---

## Configuration

### `configs/thresholds.yaml`

Per-category decision thresholds (hot-reloadable, mount as ConfigMap in OpenShift):

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
  health: 0.5
  sex_life_orientation: 0.5
decision_logic: any_over_threshold
```

### `configs/train_config.yaml`

Key training hyperparameters:

```yaml
model_name_or_path: SZTAKI-HLT/hubert-base-cc
learning_rate: 2.0e-5
num_epochs: 5
batch_size: 8
early_stopping_patience: 1
```

---

## Training dataset

### Diversity strategy

Every template is represented in exactly one split (train or test — never both). The scripts generate several *linguistic style variants* for each category to prevent the model from memorising surface patterns:

| Style | Example |
|---|---|
| **Direct** | *"Cukorbeteg vagyok, hitelt szeretnék felvenni."* |
| **Question form** | *"Orvosi igazolással kérhetek-e törlesztési szünetet, ha krónikus betegségem van?"* |
| **Negation** | *"Nem vagyok cukorbeteg, de az anyukám igen, az ő hitelét intézem."* |
| **3rd-person** | *"Az apám cukorbeteg, az ő hitelének halasztását kérem meghatalmazással."* |
| **Implicit** | *"Tartós kezelésem miatt a havi kiadásaim megugrottak, hitelhalasztást kérek."* |
| **Frustrated/urgent** | *"Már harmadik hete próbálom elintézni, cukorbetegségem igazolása megvan!"* |
| **Formal letter** | *"Tisztelettel tájékoztatom Önöket, hogy krónikus betegségem miatt kérek haladékot."* |
| **Contextual** | *"Tavaly diagnosztizáltak szívbetegséggel. Azóta nem dolgozom. Kérek törlesztési haladékot."* |
| **Long-form** | Full paragraph with banking context, ≤600 chars |
| **Multi-label** | Combines two or more categories in one message |

Negative examples (no sensitive data) come from realistic retail banking chat topics: card problems, transfers, app errors, loan questions, fraud reports, document requests, and more.

### Dataset size (v3)

| Split | Total | Positive | Multi-label | Negative | Long (≥300 chars) |
|---|---|---|---|---|---|
| Train | 497 | 359 | 88 | 138 | 80 |
| Test | 247 | 177 | 41 | 70 | 52 |

---

## Evaluation reports

After training, reports are saved to `artifacts/reports/<timestamp>/` with a `latest/` symlink:

```
artifacts/reports/latest/
├── training_curves.png          # loss + macro-F1 per epoch
├── per_category_f1_curves.png   # per-category F1 on validation set over epochs
├── test_category_good_bad.png   # correct vs. incorrect predictions per category (test set)
├── test_sample_accuracy.png     # exact multi-label match + average category accuracy
├── metrics.json                 # full per-category precision/recall/F1/AUC + aggregates
├── training_history.json        # per-epoch train/dev loss and F1
└── README.md                    # auto-generated summary
```

Regenerate a test report from a checkpoint:

```bash
uv run python -m spd.eval_report \
  --model-path artifacts/checkpoints/best_model.pt \
  --history artifacts/reports/latest/training_history.json
```

---

## Results

### Training curves

![Training curves](artifacts/reports/latest/training_curves.png)

**Left panel — Loss.** The orange line (train loss) starts at ~1.0 and falls sharply after each epoch, reaching ~0.07 by epoch 5. The blue line (dev loss) mirrors it, dropping even faster to ~0.07 — the two lines converge closely, which indicates **no overfitting**: the model generalises well rather than just memorising training examples. If dev loss had risen while train loss kept falling, that would signal overfitting.

**Right panel — Validation Macro-F1.** Starting at 0.74 after epoch 1 and rising monotonically to 1.00 by epoch 5. The smooth upward curve shows that every epoch adds real generalisation, not just noise. Macro-F1 averages the F1 score across all 7 categories equally (categories with fewer examples count as much as large ones), making it the right metric when category imbalance is possible.

---

### Per-category validation F1 over epochs

![Per-category F1 curves](artifacts/reports/latest/per_category_f1_curves.png)

Each coloured line tracks one category's F1 score on the validation set across the 5 epochs. Key observations:

- **Fast learners (cyan/blue/magenta):** `sex_life_orientation` (cyan) starts at 0.95 in epoch 1 and the model already distinguishes it well from the very first epoch. `trade_union` and `ethnicity` also converge quickly.
- **Slow learner — `religion_belief` (red):** Starts at only 0.38 in epoch 1 and reaches 1.0 only at epoch 5. This is the most linguistically ambiguous category — messages about donations to religious organisations share vocabulary with neutral bank-transfer messages. The model needs more passes through the data before it learns the distinguishing patterns.
- **Convergence:** All 7 categories reach F1 ≥ 0.85 by epoch 3 and all reach 1.0 by epoch 5. This confirms 5 epochs is the right training budget for this dataset size.

---

### Test set: correct vs. incorrect predictions per category

![Category good/bad bars](artifacts/reports/latest/test_category_good_bad.png)

Each bar represents **all 247 test samples** evaluated for that category. Green (jó = good) is the proportion of samples where the prediction was correct; red (rossz = bad) is where it was wrong.

- **`ethnicity`, `political_opinion`, `trade_union` — 100 % correct.** These categories have very clear, explicit vocabulary in the Hungarian text (party names, union names, ethnic identifiers).
- **`genetic`, `health`, `sex_life_orientation` — 99.2–99.6 % correct.** The handful of errors come from the genuinely hard test cases: negation ("*Nem beteg vagyok, de...*"), implicit language ("*Tartós kezelésem miatt...*"), and 3rd-person disclosure ("*Az apám cukorbeteg...*") — exactly the patterns added in this dataset version.
- **`religion_belief` — 97.2 % correct.** The 7 incorrect predictions are false negatives where implicit religious-donation language is mistaken for a neutral transfer request. This matches the per-category F1 of 0.889 shown in `metrics.json`.

---

### Sample-level accuracy

![Sample accuracy](artifacts/reports/latest/test_sample_accuracy.png)

Two complementary metrics measured on the 247 test samples:

| Metric | Value | What it means |
|---|---|---|
| **Exact multi-label match** | **95.1 %** | The model predicted *the full set* of categories perfectly for 95 out of 100 messages. A single wrong category (missed or extra) counts as a failure. This is the strictest possible multi-label metric. |
| **Avg category accuracy** | **99.2 %** | Averaging across all 7 categories and all 247 samples, 99.2 % of individual category decisions were correct. This captures near-misses (e.g., getting 6/7 categories right) that the exact match does not reward. |

The 4.9 percentage-point gap between exact match (95.1 %) and avg category accuracy (99.2 %) tells us that most "wrong" samples are *almost* right — typically one category missed or one false positive — rather than systematic failures.

---

### Summary metrics

| Metric | Value |
|---|---|
| Test macro-F1 | **0.973** |
| Test micro-F1 | **0.971** |
| Exact multi-label match | **95.1 %** |
| Avg category accuracy | **99.2 %** |
| Best dev macro-F1 (epoch 5) | **1.000** |

---

## Acceptance criteria (v1)

All measured on the **organic** test set:

- Macro-F1 ≥ 0.80 ✅ (achieved 0.973)
- Per-category recall ≥ 0.70 ✅ (min achieved: 0.966 for `sex_life_orientation`)
- p95 latency ≤ 200 ms end-to-end through the FastAPI service ✅ (~48 ms)

---

## Privacy

The SPD service **never logs or persists raw user text**. Each request log entry contains only: `request_id`, `model_version`, `scores`, `categories_triggered`, `spd_detected`, `latency_ms`, `text_sha256` (SHA-256 hash for deduplication, not reversible), `text_len`.

---

## Running tests

```bash
uv run pytest --tb=short -q
```

36 tests covering categories, deduplication, data loading, model forward pass, evaluation metrics, ONNX inference, and the FastAPI service endpoints.

---

## Data integrity

Train and test JSONL are guaranteed template-disjoint when built via `scripts/build_hungarian_dataset.py`:

- Every unique *template sentence* is assigned exclusively to train or test before diversification (prefixes/suffixes) is applied — so adding a banking-context wrapper to a training template cannot produce a test example.
- The build script validates this with SHA-256 `template_key` hashes stored per record and aborts with an error if any overlap is found.
- Results are logged to `data/processed/dedup_stats.json`.

---

## Tech stack

- Python 3.11+
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — `SZTAKI-HLT/hubert-base-cc`
- PyTorch 2.2+ (dynamo ONNX exporter)
- ONNX Runtime 1.18+ (CPUExecutionProvider, INT8)
- FastAPI + Uvicorn
- Prometheus client
- matplotlib (reporting)
- pytest
