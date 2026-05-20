"""Text deduplication utilities for SPD JSONL datasets."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize text for deduplication (case, whitespace, unicode)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_key(text: str) -> str:
    """Stable hash key for a normalized text string."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_key(text: str, strip_phrases: tuple[str, ...] = ()) -> str:
    """
    Hash key for underlying message content after removing known synthetic wrappers.
    Used to detect train/test leakage when only prefixes/suffixes differ.
    """
    t = normalize_text(text)
    if not strip_phrases:
        return text_key(t)

    changed = True
    while changed:
        changed = False
        for phrase in sorted(strip_phrases, key=len, reverse=True):
            p = normalize_text(phrase)
            if p and t.startswith(p):
                t = t[len(p) :].lstrip(" ,.;:")
                changed = True
            if p and t.endswith(p):
                t = t[: -len(p)].rstrip(" ,.;:")
                changed = True
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def find_key_overlap(
    train_records: list[dict],
    test_records: list[dict],
    *,
    key_fn=text_key,
) -> list[str]:
    """Return sample normalized texts whose keys appear in both splits."""
    train_keys = {key_fn(r["text"]) for r in train_records}
    overlaps: list[str] = []
    for record in test_records:
        key = key_fn(record["text"])
        if key in train_keys:
            overlaps.append(normalize_text(record["text"])[:120])
    return overlaps


def dedupe_records(records: list[dict], *, key_fn=text_key) -> tuple[list[dict], int]:
    """Remove duplicate texts within a single record list. Returns (unique, removed_count)."""
    seen: set[str] = set()
    unique: list[dict] = []
    removed = 0
    for record in records:
        key = key_fn(record["text"])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, removed


def split_disjoint(
    train_records: list[dict],
    test_records: list[dict],
    *,
    key_fn=text_key,
    prefer: str = "test",
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """
    Ensure zero text overlap between train and test.

    When the same text appears in both splits, keep it in `prefer` split only.
    Also deduplicates within each split.
    """
    train_first = prefer != "test"
    first, second = (train_records, test_records) if train_first else (test_records, train_records)
    first_name, second_name = ("train", "test") if train_first else ("test", "train")

    first, dup_first = dedupe_records(first, key_fn=key_fn)
    reserved: set[str] = {key_fn(r["text"]) for r in first}

    second_unique: list[dict] = []
    overlap_removed = 0
    dup_second = 0
    seen_second: set[str] = set()
    for record in second:
        key = key_fn(record["text"])
        if key in reserved:
            overlap_removed += 1
            continue
        if key in seen_second:
            dup_second += 1
            continue
        seen_second.add(key)
        second_unique.append(record)

    if train_first:
        final_train, final_test = first, second_unique
    else:
        final_test, final_train = first, second_unique

    stats = {
        "train_internal_dupes_removed": dup_first if train_first else dup_second,
        "test_internal_dupes_removed": dup_second if train_first else dup_first,
        "train_test_overlap_removed": overlap_removed,
        "train_size": len(final_train),
        "test_size": len(final_test),
    }
    return final_train, final_test, stats


def find_template_overlap(
    train_records: list[dict],
    test_records: list[dict],
) -> list[str]:
    """Return records whose template_key appears in both splits."""
    train_keys = {
        r.get("template_key") or text_key(r["text"]) for r in train_records
    }
    overlaps: list[str] = []
    for record in test_records:
        key = record.get("template_key") or text_key(record["text"])
        if key in train_keys:
            overlaps.append(normalize_text(record["text"])[:120])
    return overlaps


def find_overlap(train_records: list[dict], test_records: list[dict]) -> list[str]:
    """Return normalized texts present in both splits (exact text key)."""
    return find_key_overlap(train_records, test_records, key_fn=text_key)


def validate_disjoint(
    train_path: str | Path,
    test_path: str | Path,
    *,
    strip_phrases: tuple[str, ...] = (),
) -> None:
    """Raise ValueError if train and test JSONL files share exact or content keys."""
    def _load(path: str | Path) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    train = _load(train_path)
    test = _load(test_path)
    overlap = find_overlap(train, test)
    if overlap:
        sample = overlap[0]
        raise ValueError(
            f"Train/test exact overlap detected ({len(overlap)} texts). Example: {sample!r}"
        )

    if strip_phrases:
        content_overlap = find_key_overlap(
            train, test, key_fn=lambda t: content_key(t, strip_phrases=strip_phrases)
        )
        if content_overlap:
            sample = content_overlap[0]
            raise ValueError(
                f"Train/test content overlap detected ({len(content_overlap)} texts). "
                f"Example: {sample!r}"
            )
