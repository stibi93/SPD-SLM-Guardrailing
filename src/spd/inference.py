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
