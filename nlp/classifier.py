from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

try:
    from .text_preprocess import preprocess_text
except ImportError:
    from text_preprocess import preprocess_text


ID2EN = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
EN2VI = {"None": "None", "Mild": "Nhẹ", "Moderate": "Vừa", "Severe": "Nặng"}


@dataclass
class NLPFeatures:
    """Module2-friendly output."""
    discomfort_level: int
    severity_label: str
    confidence: float
    probabilities: Dict[str, float]
    original_text: str


class PhoBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_labels: int = 4,
        dropout: float = 0.3,
        max_length: int = 256,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)

    def _encode_texts(self, texts: List[str], max_length: Optional[int] = None):

        if max_length is None:
            max_length = self.max_length

        texts = [preprocess_text(t, mode="phobert", lowercase=False) for t in texts]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    @torch.no_grad()
    def predict(
        self,
        texts: Union[str, List[str]],
        device: Union[str, torch.device] = "cpu",
        max_length: Optional[int] = None,
    ):

        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        self.to(device)

        input_ids, attention_mask = self._encode_texts(texts, max_length=max_length)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = self.forward(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return preds.cpu().tolist(), probs.cpu().tolist()

    @torch.no_grad()
    def predict_features(
        self,
        text: str,
        device: Union[str, torch.device] = "cpu",
        max_length: Optional[int] = None,
    ) -> NLPFeatures:

        pred_ids, probs_list = self.predict(text, device=device, max_length=max_length)
        pred_id = int(pred_ids[0])  # 0..3
        probs = probs_list[0]

        # probs -> VN label dict
        prob_vn: Dict[str, float] = {}
        for i, p in enumerate(probs):
            en = ID2EN.get(i, str(i))
            vn = EN2VI.get(en, en)
            prob_vn[vn] = float(p)

        en_label = ID2EN.get(pred_id, str(pred_id))
        vn_label = EN2VI.get(en_label, en_label)
        confidence = float(probs[pred_id])

        return NLPFeatures(
            discomfort_level=pred_id + 1,
            severity_label=vn_label,
            confidence=confidence,
            probabilities=prob_vn,
            original_text=text,
        )

    @classmethod
    def load(
        cls,
        model_path: str,
        device: Union[str, torch.device] = "cpu",
        model_name: str = "vinai/phobert-base-v2",
        num_labels: int = 4,
        dropout: float = 0.3,
        max_length: int = 256,
        strict: bool = True,
        state_dict_key: str = "model_state_dict",
    ) -> "PhoBERTClassifier":

        model = cls(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            max_length=max_length,
        )

        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and state_dict_key in checkpoint:
            sd = checkpoint[state_dict_key]
        else:
            # fallback: treat file as state_dict-only
            sd = checkpoint

        model.load_state_dict(sd, strict=strict)
        model.to(device)
        model.eval()
        return model
