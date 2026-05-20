#!/usr/bin/env python
# scripts/benchmark_latency.py
"""Measure SPDPredictor p50/p95/p99 latency on representative Hungarian texts."""
import argparse
import statistics
import sys
import time

from spd.categories import CATEGORIES
from spd.inference import SPDPredictor

P95_TARGET_MS = 200.0

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

    if args.reps < 100:
        parser.error("--reps must be at least 100 for meaningful percentiles")

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
    qs = statistics.quantiles(latencies, n=100)
    p95 = qs[94]  # 95th percentile (0-indexed: index 94)
    p99 = qs[98]  # 99th percentile
    mean = statistics.mean(latencies)

    print(f"\nLatency over {args.reps} requests:")
    print(f"  mean = {mean:.1f} ms")
    print(f"  p50  = {p50:.1f} ms")
    print(f"  p95  = {p95:.1f} ms   ← target: ≤{P95_TARGET_MS:.0f} ms")
    print(f"  p99  = {p99:.1f} ms")
    gate = "PASS" if p95 <= P95_TARGET_MS else "FAIL"
    print(f"\nLatency gate (p95 ≤ {P95_TARGET_MS:.0f} ms): {gate}")
    if gate == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
