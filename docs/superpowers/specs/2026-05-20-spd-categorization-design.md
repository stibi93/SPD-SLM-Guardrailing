# SPD Categorization Model — Design Spec

- **Date:** 2026-05-20
- **Project:** `slm-spd-guardrailing`
- **Owner:** zandras.ai@hiflylabs.com
- **Status:** Approved for implementation
- **Scope:** v1 — Hungarian, user-input only

## 1. Problem

OTP Bank's agentic chat application needs a dedicated model that, for each incoming **user message**, classifies whether the text contains data falling under **GDPR Article 9 special categories** (ethnicity, political opinion, religion/belief, trade union, genetic, biometric, health, sex life/orientation). The model's output feeds the existing guardrailing layer, which decides how to handle SPD-positive messages.

The detector replaces the current Azure-OpenAI-based detection in the same slot, motivated by three roughly equal-weight constraints:

1. **Data sovereignty** — sending Article 9 content to Azure OpenAI is itself a compliance concern.
2. **Cost and latency** — per-message LLM calls are too expensive and too slow.
3. **Hungarian accuracy** — a Hungarian-tuned dedicated model should outperform a generic LLM at the same cost.

## 2. Non-goals (v1)

- English support. Multilingual is a v2 concern.
- Detection on agent outputs or tool/retrieval results. Only user input is in scope.
- Span-level identification of which phrase triggered which category. Multi-label is sufficient for the guardrailing layer's decision.
- Redaction or rewriting of input. The detector reports; the guardrailing layer acts.
- Java implementation. The artifact (ONNX model + thresholds.yaml) is designed to be portable; the Java port is a separate downstream effort.

## 3. Functional requirements

### 3.1 Inputs

- A single Hungarian text string per request. Expected length: typical chat message (1–500 chars); hard limit 8 KB.
- Optional language hint and request ID.

### 3.2 Outputs

Per-category multi-label scores plus an overall decision:

- 8 floating-point scores in `[0, 1]`, one per Article 9 category.
- A boolean `spd_detected` derived from per-category thresholds (config-driven).
- The list of categories that exceeded their threshold.
- Model version and observed latency.

### 3.3 Decision policy

- v1 decision logic: `spd_detected = ANY(score[c] ≥ threshold[c])`.
- Thresholds are per-category, sourced from a hot-reloadable YAML config.
- Default thresholds tuned on dev set to maximize macro-F1.

### 3.4 Non-functional requirements

- **Latency:** p95 ≤ 200 ms per request, including tokenization and decision logic.
- **Throughput:** must comfortably handle the chat backend's per-pod request rate (target: ≥50 RPS per pod). To be re-measured against production traffic shape.
- **Deployment:** OTP on-prem OpenShift, **CPU-only**.
- **Privacy:** the SPD service does not persist raw text. See §10.
- **Stateless:** no per-user state, no session affinity required.
- **Observability:** Prometheus metrics endpoint, structured JSON logs, OpenShift liveness/readiness probes.

## 4. Architecture

```
┌──────────────┐    user text     ┌──────────────────┐    decision       ┌─────────────────┐
│  Chat        │ ───────────────▶ │  SPD Detector    │ ─────────────────▶│  Guardrailing   │
│  Backend     │                  │  Service         │   (8 scores +     │  Layer          │
│  (existing)  │ ◀─────────────── │  (Python/FastAPI)│    block flag)    │  (existing)     │
└──────────────┘                  └──────────────────┘                   └─────────────────┘
                                          │
                                          ▼
                                  ┌──────────────────┐
                                  │  huBERT-base +   │
                                  │  multi-label head│
                                  │  (ONNX INT8)     │
                                  └──────────────────┘
```

The detector is a **stateless HTTP service** sitting synchronously in front of the existing guardrailing layer. It exposes a single classification endpoint plus standard ops endpoints. The model is shipped as an INT8-quantized ONNX artifact loaded once at process startup.

## 5. Label space (Article 9 categories)

| Key                    | Article 9 category                                 |
|------------------------|----------------------------------------------------|
| `ethnicity`            | Racial or ethnic origin                            |
| `political_opinion`    | Political opinions                                 |
| `religion_belief`      | Religious or philosophical beliefs                 |
| `trade_union`          | Trade union membership                             |
| `genetic`              | Genetic data                                       |
| `biometric`            | Biometric data (used for identification)           |
| `health`               | Health data                                        |
| `sex_life_orientation` | Sex life or sexual orientation                     |

Multi-label: a single message may trigger zero, one, or several categories.

## 6. Data plan

### 6.1 Sources, in order of trust

1. **Organic Hungarian samples** from the OTP business team. Gold standard. **Only source allowed in the test set.**
2. **Synthetic Hungarian samples** generated by Azure OpenAI (GPT-4o-class) using per-category prompted templates. Training and dev only; never test. Permitted because training-time use of Azure is separate from the runtime sovereignty concern.
3. **Weak-supervision seeds** — keyword/regex rules per category to pre-label any unlabeled chat logs that become available later. Pre-labels reviewed by humans before entering training.

### 6.2 Target dataset shape (v1)

- ~1,500–3,000 labeled examples total.
- Per-category positives: ≥100 each, where feasible. `genetic` and `biometric` are expected to be the hardest categories to source; the floor for these may be lower and is documented at training time.
- ~50% negatives drawn from realistic banking chat without Article 9 content.
- **Hold-out test set:** 200–400 organic examples, stratified across categories, never used in training or threshold tuning.

### 6.3 Labeling schema

JSONL, one row per example:

```json
{
  "id": "uuid",
  "text": "string",
  "lang": "hu",
  "labels": {
    "ethnicity": 0,
    "political_opinion": 0,
    "religion_belief": 0,
    "trade_union": 0,
    "genetic": 0,
    "biometric": 0,
    "health": 1,
    "sex_life_orientation": 0
  },
  "source": "organic | synthetic | weak",
  "annotator": "string",
  "notes": "string (optional)"
}
```

### 6.4 Annotation guidelines (separate document, owed before labeling at scale)

A short Hungarian-language annotator guide must define:

- What counts as direct vs. indirect disclosure for each category.
- Negation handling ("nem vagyok beteg" — still mentions health? Project decision: no).
- Third-party disclosure ("anyukám diabéteszes") — counts as health.
- Figurative or idiomatic uses that resemble Article 9 content but are not disclosures.
- Inter-annotator agreement target: Cohen's κ ≥ 0.7 per category on a 100-sample calibration round before full labeling starts.

## 7. Model and training

### 7.1 Model

- **Base:** `SZTAKI-HLT/hubert-base-cc`. BERT-base-uncased trained on Hungarian Common Crawl. 110M parameters. Permissively licensed.
- **Head:** Linear layer over `[CLS]` pooled output → 8 logits → sigmoid.
- **Tokenizer:** huBERT's bundled WordPiece tokenizer (uncased Hungarian).
- **Max sequence length:** 128 tokens. Chat messages are short; truncation at 128 keeps p95 latency well under budget.

### 7.2 Training

- Framework: HuggingFace Transformers `Trainer`.
- Precision: FP32 (CPU training; FP16/BF16 paths on this CPU are unreliable).
- Optimizer: AdamW. Learning rate 2e-5. Linear warmup over 10% of steps.
- Batch size: 8. Effective batch via gradient accumulation if needed.
- Epochs: up to 5, with early stopping on dev macro-F1 (patience 1).
- Loss: per-category `BCEWithLogitsLoss` with `pos_weight` set from training-set class frequencies to mitigate imbalance.
- Regularization: weight decay 0.01, dropout 0.1.
- Determinism: fixed seeds for `random`, `numpy`, `torch`, `transformers`; `torch.use_deterministic_algorithms(True)` where supported.
- All hyperparameters live in `configs/train_config.yaml`.

### 7.3 Training environment

- Local CPU on the developer laptop (Intel Core Ultra 7 255U, 16 GB RAM, no GPU) is sufficient for the planned dataset size. Expected wall-clock per 5-epoch run:
  - 500 samples: 3–5 min
  - 2,000 samples: 10–20 min
  - 10,000 samples: 1–2 hours
- Offloading to a rented cloud GPU (Colab / Modal / RunPod) is the contingency if the dataset grows past ~10k or a hyperparameter sweep is needed. Not required for v1.

## 8. Evaluation

### 8.1 Metrics

Reported per category and aggregated (macro and micro):

- Precision, recall, F1.
- ROC-AUC and PR-AUC.
- Confusion matrix per category.
- Calibration: reliability diagram and expected calibration error on the dev set, to validate that scores are usable as thresholds.

### 8.2 Error slicing

In addition to aggregate numbers, the eval harness reports metrics on slices:

- Code-mixed Hungarian/English (`"kérek időpontot a kardiológiához please"`).
- Common typos and slang.
- Indirect or third-party disclosures (`"anyukámnak rákja van"`, `"a férjem zsidó"`).
- Negations and counterfactuals (`"nem vagyok cukorbeteg"`).
- Figurative language and slur-adjacent uses (handled per annotation guidelines).

### 8.3 Stress test set

A separate hand-crafted set of 50–100 tricky Hungarian cases, scored independently. Tracked over time as a regression suite. Failures here block release even if aggregate metrics pass.

### 8.4 Acceptance gate for v1 production

All must hold on the **organic** test set:

- **Macro-F1 ≥ 0.80.**
- **Per-category recall ≥ 0.70** for every category that has ≥30 positives in the test set. Categories below that threshold are reported but do not block (and are flagged for more data collection).
- **p95 latency ≤ 200 ms** measured end-to-end through the FastAPI service on a representative CPU (developer laptop is an acceptable proxy for v1; re-measure on OpenShift before production rollout).

Failing the gate is not a bug — it's a signal to collect more data, revisit thresholds, or revisit the model choice.

## 9. API contract

### 9.1 `POST /v1/spd-check`

Request:

```json
{
  "text": "Cukorbeteg vagyok és szeretnék hitelt felvenni.",
  "lang": "hu",
  "request_id": "optional-string"
}
```

- `text` required, ≤ 8 KB.
- `lang` optional. v1 only `"hu"` is meaningful; `"auto"` and `"en"` accepted for forward compatibility but treated as `"hu"`.
- `request_id` optional; echoed back. If absent, the service generates one.

Response (`200 OK`):

```json
{
  "request_id": "...",
  "model_version": "spd-hubert-v1.0.0",
  "lang_detected": "hu",
  "scores": {
    "ethnicity": 0.02, "political_opinion": 0.01, "religion_belief": 0.03,
    "trade_union": 0.00, "genetic": 0.00, "biometric": 0.00,
    "health": 0.94, "sex_life_orientation": 0.01
  },
  "categories_triggered": ["health"],
  "spd_detected": true,
  "latency_ms": 28
}
```

Error responses use standard HTTP codes with a JSON body `{"error": "...", "request_id": "..."}`.

### 9.2 Ops endpoints

- `GET /healthz` — liveness/readiness for OpenShift probes.
- `GET /metrics` — Prometheus format.
- `GET /v1/info` — model version, build hash, loaded thresholds.

## 10. Logging and privacy policy

- The SPD service **does not persist user text** in any log, metric, or trace by default.
- Default structured log entry per request: `{request_id, timestamp, model_version, lang_detected, scores, categories_triggered, spd_detected, latency_ms, text_sha256, text_len}`. The SHA-256 is for correlating duplicate requests across services; it is not reversible to the original text.
- A debug log level may be enabled per-environment that adds `text_len_buckets` and tokenization timing — but **never the raw text**.
- If an audit trail of decisions is needed downstream, that responsibility belongs to the chat backend or the guardrailing layer, which already have legitimate reason to handle the text. The SPD service stays minimal.

## 11. Threshold configuration

`configs/thresholds.yaml` (mounted, hot-reloadable):

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

Thresholds are tuned post-training on the dev set. Production rollout includes the trained model's recommended thresholds.

## 12. Repository layout

```
slm-spd-guardrailing/
├── pyproject.toml
├── README.md
├── configs/
│   ├── train_config.yaml
│   └── thresholds.yaml
├── data/
│   ├── raw/                 # organic samples (gitignored)
│   ├── synthetic/           # generated samples (gitignored)
│   └── processed/           # train/dev/test JSONL (gitignored)
├── src/spd/
│   ├── __init__.py
│   ├── categories.py        # Article 9 enum + label-key mappings
│   ├── data.py              # dataset loader, tokenization
│   ├── model.py             # huBERT + multi-label classification head
│   ├── train.py             # training loop entry point
│   ├── evaluate.py          # metrics, slicing, calibration
│   ├── synth.py             # Azure OpenAI synthetic data generation
│   ├── export_onnx.py       # FP32 → ONNX → INT8 dynamic quantization
│   ├── inference.py         # ONNX Runtime loader + predict()
│   └── service.py           # FastAPI app, endpoints, threshold logic
├── tests/
│   ├── test_categories.py
│   ├── test_inference.py
│   └── test_service.py
├── scripts/
│   ├── download_model.py    # cache huBERT locally
│   └── benchmark_latency.py
└── docs/
    └── superpowers/specs/2026-05-20-spd-categorization-design.md
```

## 13. Deployment

- **Container image:** Python 3.11 slim base, ONNX Runtime CPU, FastAPI, Uvicorn. Model artifact (`model.onnx`) baked into the image and pinned to the same version tag as the image.
- **Runtime:** ONNX Runtime CPU execution provider with oneDNN; single Uvicorn worker per pod by default (revisit after load testing).
- **Probes:** `/healthz` for liveness and readiness.
- **Resources:** CPU request/limit and memory request/limit defined in the OpenShift manifest; informed by latency benchmarks.
- **Config injection:** `thresholds.yaml` mounted as a ConfigMap, hot-reloaded on file change.

## 14. Risks and mitigations

| Risk                                                                 | Mitigation                                                                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Cold-start with ~200 organic samples — fragile fine-tune.            | Combine organic + synthetic + weak supervision. Establish hold-out as organic-only. Iterate with the business team.              |
| Synthetic data introduces artifacts the model overfits to.           | Cap synthetic share of training data, hold synthetic out of test, monitor train-vs-test gap.                                     |
| `genetic` / `biometric` categories starved of positives.             | Explicitly request these from the business team. Document floor. Consider falling back to a higher decision threshold if needed. |
| Annotator disagreement on Article 9 edge cases.                      | Annotation guidelines + Cohen's κ calibration round before full labeling.                                                        |
| CPU latency on OpenShift differs from laptop benchmarks.             | Re-measure on a production-equivalent pod before sign-off. Build a load-test script as part of v1 deliverables.                  |
| Distribution shift after deployment (new slang, new banking topics). | v1.x active-learning loop: log hashed low-confidence requests, periodic re-label and retrain.                                    |
| Compliance with the privacy policy (no raw text logged).             | Code review of the logging path. Unit test asserts no log line contains the request text.                                        |

## 15. Roadmap (post-v1)

- **v1.x — Active learning.** Log hashed low-confidence predictions, re-label, retrain on a cadence.
- **v2 — English support.** Two candidate paths, chosen after a short spike:
  - Sibling English-specific encoder + language router upstream.
  - Migration to XLM-RoBERTa-base as a single multilingual model.
- **Java port.** Same ONNX artifact + thresholds.yaml; Java service uses ONNX Runtime Java with the identical REST contract from §9. No behavioral changes expected.

## 16. Success criteria for v1

The v1 effort is complete when all of the following hold:

1. A trained model artifact passes the acceptance gate in §8.4 on the organic test set.
2. The FastAPI service exposes the §9 API and ops endpoints, with the §10 logging policy enforced and tested.
3. The container image runs on an OpenShift environment matching production, with documented CPU/memory profile.
4. The service has been integrated end-to-end with the chat backend in a staging environment and exercised with a representative Hungarian conversation corpus.
5. The repository contains the spec (this document), an implementation plan, and runnable training/eval/export scripts.
