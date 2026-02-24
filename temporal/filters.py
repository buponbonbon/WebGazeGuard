from __future__ import annotations

from typing import Optional

def ema(prev: Optional[float], x: Optional[float], alpha: float = 0.2) -> Optional[float]:
    # Simple exponential moving average for smoothing numeric signals.
    if x is None:
        return prev
    if prev is None:
        return x
    return alpha * x + (1 - alpha) * prev
