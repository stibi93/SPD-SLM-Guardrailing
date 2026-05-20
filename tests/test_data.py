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
