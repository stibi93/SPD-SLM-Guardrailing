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
