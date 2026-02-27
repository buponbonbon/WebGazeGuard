from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple

from core.schemas import WindowFeatures, NLPFeatures, RiskOutput
from fusion.risk_engine import assess_risk
from fusion.coaching import generate_coaching

try:
    # cached loader for API usage
    from nlp.classifier import get_classifier
except Exception:
    get_classifier = None


class Orchestrator:
    def __init__(
        self,
        *,
        nlp_checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        local_files_only: bool = False,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.device = device
        self.local_files_only = local_files_only
        self.weights = weights
        self._nlp_ckpt = nlp_checkpoint_path
        self._nlp_model = None

    def _load_nlp_if_needed(self) -> None:
        # lazy load so CV-only still works
        if self._nlp_model is not None:
            return
        if not self._nlp_ckpt or get_classifier is None:
            return
        try:
            self._nlp_model = get_classifier(
                self._nlp_ckpt,
                device=self.device,
                local_files_only=self.local_files_only,
            )
        except Exception:
            self._nlp_model = None

    def analyze(
        self,
        window: WindowFeatures,
        *,
        text: Optional[str] = None,
    ) -> Tuple[RiskOutput, str]:
        # NLP is optional
        nlp_feat: Optional[NLPFeatures] = None

        if text is not None and str(text).strip() != "":
            self._load_nlp_if_needed()
            if self._nlp_model is not None:
                try:
                    nlp_feat = self._nlp_model.predict_features(text, device=self.device)
                except Exception:
                    nlp_feat = None

        risk = assess_risk(window, nlp=nlp_feat, weights=self.weights)
        msg = generate_coaching(risk)
        return risk, msg

    def analyze_to_dict(
        self,
        window: WindowFeatures,
        *,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        # for JSON responses
        risk, msg = self.analyze(window, text=text)
        out = asdict(risk)
        out["message"] = msg
        return out
