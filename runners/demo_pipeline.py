from __future__ import annotations

from core.schemas import WindowFeatures
from runners.orchestrator import Orchestrator


def main() -> None:
    # fake window for quick end-to-end demo (no webcam/video needed)
    win = WindowFeatures(
        window_start_ms=0,
        window_end_ms=3000,
        window_seconds=3.0,
        blink_rate_bpm=5.0,          # low blink -> higher risk
        pitch_mean=18.0,             # posture tilt
        yaw_mean=8.0,
        distance_cat_mode="too_close"
    )

    orch = Orchestrator(
        nlp_checkpoint_path=None,    # set path later when you have a .pt
        device="cpu",
        local_files_only=True
    )

    risk, msg = orch.analyze(win, text="Mắt tôi khô và hơi mỏi khi nhìn màn hình lâu.")
    print("=== RISK ===")
    print(risk)
    print("\n=== COACHING ===")
    print(msg)


if __name__ == "__main__":
    main()
