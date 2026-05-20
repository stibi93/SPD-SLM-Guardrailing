#!/usr/bin/env python3
"""
Export train.jsonl and test.jsonl to HTML and CSV for easy human inspection.

Outputs:
  data/processed/train_inspect.html
  data/processed/test_inspect.html
  data/processed/train_inspect.csv
  data/processed/test_inspect.csv

Usage:
  uv run python scripts/export_dataset_html.py
  uv run python scripts/export_dataset_html.py --data-dir data/processed
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

CATEGORIES = [
    "ethnicity",
    "political_opinion",
    "religion_belief",
    "trade_union",
    "genetic",
    "health",
    "sex_life_orientation",
]

_CATEGORY_COLORS: dict[str, str] = {
    "ethnicity":            "#f59e0b",
    "political_opinion":    "#ef4444",
    "religion_belief":      "#8b5cf6",
    "trade_union":          "#3b82f6",
    "genetic":              "#06b6d4",
    "health":               "#10b981",
    "sex_life_orientation": "#ec4899",
}

_HTML_HEAD = """<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; font-size: 14px; padding: 24px; background: #f8fafc; }}
  h1   {{ font-size: 20px; margin-bottom: 6px; }}
  p.meta {{ color: #64748b; font-size: 12px; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px;
           box-shadow: 0 1px 3px #0002; overflow: hidden; }}
  th   {{ background: #1e293b; color: #fff; text-align: left; padding: 10px 12px; font-size: 12px; }}
  td   {{ padding: 9px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f1f5f9; }}
  .neg  {{ color: #94a3b8; font-style: italic; }}
  .text {{ max-width: 540px; word-break: break-word; }}
  .badge {{ display: inline-block; border-radius: 4px; padding: 2px 7px; font-size: 11px;
             font-weight: 600; color: #fff; margin: 2px 2px 0 0; white-space: nowrap; }}
  .len  {{ font-size: 11px; color: #94a3b8; }}
  .row-multi {{ background: #fffbeb !important; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{subtitle}</p>
<table>
<thead>
<tr>
  <th>#</th>
  <th>Text</th>
  <th>Active categories</th>
  <th>Len</th>
  <th>Annotator</th>
  <th>Notes</th>
</tr>
</thead>
<tbody>
"""

_HTML_FOOT = "</tbody>\n</table>\n</body>\n</html>\n"


def _active_cats(labels: dict[str, int]) -> list[str]:
    return [c for c in CATEGORIES if labels.get(c, 0)]


def _badges(cats: list[str]) -> str:
    if not cats:
        return '<span class="neg">negative</span>'
    parts = []
    for cat in cats:
        color = _CATEGORY_COLORS.get(cat, "#64748b")
        parts.append(
            f'<span class="badge" style="background:{color}">{html.escape(cat)}</span>'
        )
    return "".join(parts)


def _row_class(cats: list[str]) -> str:
    return ' class="row-multi"' if len(cats) > 1 else ""


def build_html(records: list[dict], title: str, subtitle: str) -> str:
    body = _HTML_HEAD.format(title=html.escape(title), subtitle=html.escape(subtitle))
    for i, rec in enumerate(records, 1):
        cats = _active_cats(rec["labels"])
        text_esc = html.escape(rec["text"])
        length = len(rec["text"])
        annotator = html.escape(rec.get("annotator", ""))
        notes = html.escape(rec.get("notes", ""))
        body += (
            f'<tr{_row_class(cats)}>'
            f'<td>{i}</td>'
            f'<td class="text">{text_esc}</td>'
            f'<td>{_badges(cats)}</td>'
            f'<td class="len">{length}</td>'
            f'<td>{annotator}</td>'
            f'<td>{notes}</td>'
            f'</tr>\n'
        )
    return body + _HTML_FOOT


def build_csv(records: list[dict]) -> list[list[str]]:
    header = ["#", "text", "active_categories", "length", "annotator", "notes"] + CATEGORIES
    rows = [header]
    for i, rec in enumerate(records, 1):
        cats = _active_cats(rec["labels"])
        per_cat = [str(rec["labels"].get(c, 0)) for c in CATEGORIES]
        rows.append([
            str(i),
            rec["text"],
            "|".join(cats) if cats else "negative",
            str(len(rec["text"])),
            rec.get("annotator", ""),
            rec.get("notes", ""),
            *per_cat,
        ])
    return rows


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def export(data_dir: Path) -> None:
    for split in ("train", "test"):
        src = data_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"Skipping {src} (not found)")
            continue
        records = load_jsonl(src)
        cats_total = sum(
            1 for r in records if any(r["labels"].get(c, 0) for c in CATEGORIES)
        )
        neg_total = len(records) - cats_total
        multi_total = sum(
            1 for r in records if len(_active_cats(r["labels"])) > 1
        )
        subtitle = (
            f"{len(records)} records — "
            f"{cats_total} positive ({multi_total} multi-label), "
            f"{neg_total} negative"
        )
        title = f"SPD Dataset — {split.capitalize()} Split"

        html_path = data_dir / f"{split}_inspect.html"
        html_path.write_text(build_html(records, title, subtitle), encoding="utf-8")
        print(f"Wrote {html_path}")

        csv_path = data_dir / f"{split}_inspect.csv"
        rows = build_csv(records)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Wrote {csv_path} ({len(rows)-1} data rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()
    export(Path(args.data_dir))


if __name__ == "__main__":
    main()
