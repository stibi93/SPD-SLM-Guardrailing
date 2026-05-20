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

A single message may trigger zero, one, or several categories.

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
│   └── processed/              # train/dev/test JSONL (gitignored)
├── src/spd/
│   ├── categories.py           # Article 9 enum + CATEGORIES list
│   ├── model.py                # SPDClassifier (huBERT encoder + linear head)
│   ├── data.py                 # SPDDataset — JSONL loader + tokenization
│   ├── train.py                # training entry point (early stopping, pos_weight)
│   ├── evaluate.py             # per-category metrics + macro/micro aggregates
│   ├── export_onnx.py          # PyTorch → FP32 ONNX → INT8 quantization
│   ├── inference.py            # SPDPredictor — ONNX Runtime wrapper
│   ├── service.py              # FastAPI app, /v1/spd-check + ops endpoints
│   └── synth.py                # Azure OpenAI synthetic data generator (training only)
├── scripts/
│   ├── download_model.py       # cache huBERT locally before training
│   └── benchmark_latency.py   # measure p50/p95/p99 latency
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

This creates `.venv/` and installs all runtime and development dependencies from `uv.lock`.

### 2. Download the base model

```bash
uv run python scripts/download_model.py
```

### 3. Prepare training data

Label data in JSONL format (one record per line):

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
  "annotator": "your-name",
  "notes": ""
}
```

Place train/dev/test splits at `data/processed/train.jsonl`, `dev.jsonl`, `test.jsonl`.

Optionally generate synthetic Hungarian examples via Azure OpenAI:

```bash
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
export AZURE_OPENAI_API_KEY=<key>
uv run python -m spd.synth --category health --count 100 --out data/synthetic/health.jsonl
```

### 4. Train

```bash
uv run python -m spd.train
# or with a custom config:
uv run python -m spd.train configs/train_config.yaml
```

Saves best checkpoint to `artifacts/checkpoints/best_model.pt` based on dev macro-F1 with early stopping (patience 1).

### 5. Export to ONNX

```bash
uv run python -m spd.export_onnx artifacts/checkpoints/best_model.pt artifacts/
# → artifacts/model.onnx  (INT8, ~111 MB)
```

### 6. Benchmark latency

```bash
uv run python scripts/benchmark_latency.py --model-path artifacts/model.onnx --reps 200
```

### 7. Run the service

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

## Acceptance criteria (v1)

All measured on the **organic** test set:

- Macro-F1 ≥ 0.80
- Per-category recall ≥ 0.70 (for categories with ≥ 30 positives in test set)
- p95 latency ≤ 200 ms end-to-end through the FastAPI service

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

## Evaluation reports

After training, reports are saved to `artifacts/reports/<timestamp>/` (symlink: `artifacts/reports/latest/`):

- **training_curves.png** — TensorBoard-style loss & macro-F1 over epochs
- **per_category_f1_curves.png** — per-category validation F1 over epochs
- **test_category_good_bad.png** — stacked **jó / rossz** bars per category (test set)
- **test_sample_accuracy.png** — exact multi-label match & average category accuracy
- **metrics.json**, **training_history.json**, **README.md**

Regenerate a test report from a checkpoint:

```bash
uv run python -m spd.eval_report \
  --model-path artifacts/checkpoints/best_model.pt \
  --history artifacts/reports/latest/training_history.json
```

Train and test JSONL are deduplicated with **zero text overlap** when built via `scripts/build_hungarian_dataset.py` (see `data/processed/dedup_stats.json`).

---

## Tech stack

- Python 3.11+
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — `SZTAKI-HLT/hubert-base-cc`
- PyTorch 2.2+ (dynamo ONNX exporter)
- ONNX Runtime 1.18+ (CPUExecutionProvider, INT8)
- FastAPI + Uvicorn
- Prometheus client
- pytest
