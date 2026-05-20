# tests/test_dedup.py
from spd.dedup import dedupe_records, find_overlap, find_template_overlap, normalize_text, split_disjoint, text_key


def _rec(text: str) -> dict:
    return {"text": text, "labels": {}}


def test_normalize_text_collapses_whitespace_and_case():
    assert normalize_text("  Hello   World  ") == "hello world"


def test_dedupe_records_removes_internal_dupes():
    records = [_rec("Azonos szöveg"), _rec("Azonos szöveg"), _rec("Másik")]
    unique, removed = dedupe_records(records)
    assert len(unique) == 2
    assert removed == 1


def test_split_disjoint_removes_train_test_overlap():
    train = [_rec("Train A"), _rec("Shared")]
    test = [_rec("Shared"), _rec("Test B")]
    train_out, test_out, stats = split_disjoint(train, test, prefer="test")
    assert stats["train_test_overlap_removed"] == 1
    assert len(test_out) == 2
    assert len(train_out) == 1
    assert not find_overlap(train_out, test_out)


def test_find_template_overlap_detects_shared_seed():
    train = [{"text": "Üdv! cukorbeteg vagyok.", "template_key": "abc", "labels": {}}]
    test = [{"text": "cukorbeteg vagyok.", "template_key": "abc", "labels": {}}]
    assert len(find_template_overlap(train, test)) == 1


def test_text_key_stable():
    assert text_key("Foo  Bar") == text_key("foo bar")
