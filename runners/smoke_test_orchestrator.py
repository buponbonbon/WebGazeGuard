from __future__ import annotations

from core.schemas import WindowFeatures
from runners.orchestrator import Orchestrator


def main() -> None:
    # minimal smoke test to catch import/schema breakages
    win = WindowFeatures(
        window_start_ms=0,
        window_end_ms=3000,
        window_seconds=3.0,
        blink_rate_bpm=12.0,
        pitch_mean=5.0,
        yaw_mean=3.0,
        distance_cat_mode="normal"
    )

    orch = Orchestrator(nlp_checkpoint_path=None, device="cpu", local_files_only=True)
    risk, msg = orch.analyze(win, text="Tôi cảm thấy bình thường.")
    assert risk.risk_level in ("Low", "Medium", "High")
    assert isinstance(msg, str) and len(msg) > 0

    print("✅ smoke test passed")
    print("risk_level:", risk.risk_level)


if __name__ == "__main__":
    main()
