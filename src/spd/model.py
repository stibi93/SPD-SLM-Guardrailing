import torch
import torch.nn as nn
from transformers import AutoModel
from spd.categories import NUM_CATEGORIES

MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"
MAX_LENGTH = 128


class SPDClassifier(nn.Module):
    def __init__(self, num_labels: int = NUM_CATEGORIES, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])  # [CLS] token
        return self.classifier(pooled)  # shape: (batch, num_labels)
