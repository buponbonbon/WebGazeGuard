from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from .text_preprocess import preprocess_text


class PhoBERTClassifier(nn.Module):

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # PhoBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Forward
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use CLS token representation (standard for PhoBERT classification)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_emb)
        return logits

    # Encode raw text → tensor
    def _encode_texts(self, texts: List[str], max_length: int = 256):
        # === NEW: call your preprocess ===
        texts = [
            preprocess_text(t, mode="phobert", lowercase=False)
            for t in texts
        ]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    # Predict API for web
    @torch.no_grad()
    def predict(
        self,
        texts: Union[str, List[str]],
        device: Union[str, torch.device] = "cpu",
    ):
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        self.to(device)

        input_ids, attention_mask = self._encode_texts(texts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = self.forward(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return preds.cpu().tolist(), probs.cpu().tolist()


    # Load checkpoint (production helper)
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 3,
        device: str = "cpu",
    ) -> "PhoBERTClassifier":
        model = cls(model_name=model_name, num_labels=num_labels)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
