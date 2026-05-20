import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from spd.categories import CATEGORIES
from spd.model import MAX_LENGTH


class SPDDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.examples: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [float(ex["labels"][cat]) for cat in CATEGORIES],
            dtype=torch.float32,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
