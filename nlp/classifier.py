# nlp/classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

try:
    # package import
    from .text_preprocess import preprocess_text
except ImportError:
    # script import
    from text_preprocess import preprocess_text


ID2EN = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
EN2VI = {"None": "None", "Mild": "Nhẹ", "Moderate": "Vừa", "Severe": "Nặng"}


@dataclass
class NLPFeatures:
    #Backward-compatible output
    discomfort_level: int
    severity_label: str
    confidence: float
    probabilities: Dict[str, float]
    original_text: str


def _as_device(device: Union[str, torch.device, None]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device


def _try_core_nlpfeatures() -> Optional[type]:

    try:
        from core.schemas import NLPFeatures as CoreNLPFeatures  # type: ignore
        return CoreNLPFeatures
    except Exception:
        return None


class PhoBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_labels: int = 4,
        dropout: float = 0.3,
        max_length: int = 256,
        tokenizer_name: Optional[str] = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.local_files_only = local_files_only


        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )


        tok_name = tokenizer_name or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, local_files_only=local_files_only)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)

    def _encode_texts(self, texts: List[str], max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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

        dev = _as_device(device)

        self.eval()
        self.to(dev)

        input_ids, attention_mask = self._encode_texts(texts, max_length=max_length)
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)

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
        discomfort_level_base: int = 0,
    ):
        # guard empty input to avoid tokenizer edge-cases
        if text is None or str(text).strip() == "":
            prob_vn = {"None": 1.0, "Nhẹ": 0.0, "Vừa": 0.0, "Nặng": 0.0}
            CoreNLP = _try_core_nlpfeatures()
            if CoreNLP is not None:
                try:
                    return CoreNLP(
                        discomfort_level=0,
                        severity_label="None",
                        confidence=1.0,
                        probabilities=prob_vn,
                        original_text=str(text) if text is not None else "",
                    )
                except Exception:
                    pass
            return NLPFeatures(
                discomfort_level=0,
                severity_label="None",
                confidence=1.0,
                probabilities=prob_vn,
                original_text=str(text) if text is not None else "",
            )

        pred_ids, probs_list = self.predict(text, device=device, max_length=max_length)
        pred_id = int(pred_ids[0])
        probs = probs_list[0]


        prob_vn: Dict[str, float] = {}
        for i, p in enumerate(probs):
            en = ID2EN.get(i, str(i))
            vn = EN2VI.get(en, en)
            prob_vn[vn] = float(p)

        en_label = ID2EN.get(pred_id, str(pred_id))
        vn_label = EN2VI.get(en_label, en_label)
        confidence = float(probs[pred_id])

        discomfort_level = int(pred_id + discomfort_level_base)
        # keep within expected range [0, 3]
        discomfort_level = max(0, min(3, discomfort_level))


        CoreNLP = _try_core_nlpfeatures()
        if CoreNLP is not None:

            try:
                return CoreNLP(
                    discomfort_level=discomfort_level,
                    severity_label=vn_label,
                    confidence=confidence,
                    probabilities=prob_vn,
                    original_text=text,
                )
            except Exception:
                pass


        return NLPFeatures(
            discomfort_level=discomfort_level,
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
        tokenizer_name: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "PhoBERTClassifier":
        model = cls(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            max_length=max_length,
            tokenizer_name=tokenizer_name,
            local_files_only=local_files_only,
        )

        dev = _as_device(device)
        checkpoint = torch.load(model_path, map_location=dev)

        if isinstance(checkpoint, dict) and state_dict_key in checkpoint:
            sd = checkpoint[state_dict_key]
        else:
            sd = checkpoint

        model.load_state_dict(sd, strict=strict)
        model.to(dev)
        model.eval()
        return model


_CACHED: dict = {}


def clear_classifier_cache() -> None:
    # manual cache reset (useful in dev)
    _CACHED.clear()

def get_classifier(
    model_path: str,
    *,
    device: Union[str, torch.device, None] = None,
    model_name: str = "vinai/phobert-base-v2",
    tokenizer_name: Optional[str] = None,
    local_files_only: bool = False,
) -> PhoBERTClassifier:

    dev = _as_device(device)
    # stable cache key across processes
    key = (os.path.abspath(model_path), str(dev), model_name, tokenizer_name, bool(local_files_only))
    if key in _CACHED:
        return _CACHED[key]
    clf = PhoBERTClassifier.load(
        model_path=model_path,
        device=str(dev),
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        local_files_only=local_files_only,
    )
    _CACHED[key] = clf
    return clf