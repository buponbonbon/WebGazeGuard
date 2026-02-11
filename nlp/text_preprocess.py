
from __future__ import annotations
import re
from typing import List, Union, Literal

# Normalize whitespace and remove control characters
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\u0000-\u001F\u007F]")

# Remove punctuation/special characters for TF-IDF mode
_NON_ALNUM_SPACE = re.compile(r"[^\w\s]", flags=re.UNICODE)

Mode = Literal["phobert", "tfidf"]


def _normalize_common(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = _CTRL.sub(" ", text)
    text = text.strip()
    text = _WS.sub(" ", text)
    return text


# Minimal preprocessing for transformer (PhoBERT) or heavier cleaning for TF-IDF
def preprocess_text(
    text: str,
    *,
    mode: Mode = "phobert",
    lowercase: bool = False,
    return_tokens: bool = False,
) -> Union[str, List[str]]:
    text = _normalize_common(text)

    if lowercase:
        text = text.lower()

    if mode == "phobert":
        return text.split(" ") if return_tokens else text

    text = _NON_ALNUM_SPACE.sub(" ", text)
    text = _WS.sub(" ", text).strip()

    if return_tokens:
        return text.split(" ") if text else []
    return text
