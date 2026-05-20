#!/usr/bin/env python
# scripts/download_model.py
"""Pre-cache huBERT tokenizer and weights from HuggingFace Hub."""
from transformers import AutoTokenizer, AutoModel
from spd.model import MODEL_NAME


def main() -> None:
    print(f"Downloading {MODEL_NAME} ...")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModel.from_pretrained(MODEL_NAME)
    print("Done. Cached at ~/.cache/huggingface/")


if __name__ == "__main__":
    main()
