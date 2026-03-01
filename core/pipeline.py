from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .config import Config
from .schemas import CVFeatures, WindowFeatures
from vision.extractor import VisionExtractor
from temporal.aggregation import WindowAggregator
from vision.head_distance import DistanceCalib


@dataclass
class PipelineState:
    vision: VisionExtractor
    agg: WindowAggregator
    last_window: Optional[WindowFeatures] = None
    last_compute_ms: int = 0


def build_pipeline(
    cfg: Config,
    *,
    gaze_ckpt_path: Optional[str] = None,
    gaze_every_n_frames: int = 3,
    gaze_resize_hw: Optional[Tuple[int, int]] = (64, 64),
) -> PipelineState:
    """Build the realtime pipeline. Gaze is enabled only if gaze_ckpt_path is provided."""

    calib = DistanceCalib(
        Z0_cm=cfg.distance_Z0_cm,
        s0_px=cfg.distance_s0_px,
    )

    vision = VisionExtractor(
        ear_thresh=cfg.ear_thresh,
        ear_consec_frames=cfg.ear_consec_frames,
        distance_calib=calib,
        gaze_ckpt_path=gaze_ckpt_path,
        gaze_every_n_frames=gaze_every_n_frames,
        gaze_resize_hw=gaze_resize_hw or (64, 64),
    )

    agg = WindowAggregator(
        fps=cfg.fps_assumed,
        window_seconds=cfg.window_seconds,
    )

    return PipelineState(vision=vision, agg=agg)


def step(
    state: PipelineState,
    frame_bgr,
    timestamp_ms: int,
    *,
    compute_interval_ms: int = 250,
) -> Tuple[CVFeatures, Optional[WindowFeatures]]:

    cvf = state.vision.extract(frame_bgr, timestamp_ms=timestamp_ms)

    state.agg.update(cvf)

    win: Optional[WindowFeatures] = None
    if state.agg.ready():
        # throttle window compute to reduce CPU load in realtime
        if (timestamp_ms - state.last_compute_ms) >= int(compute_interval_ms):
            win = state.agg.compute()
            state.last_window = win
            state.last_compute_ms = int(timestamp_ms)
        else:
            win = state.last_window

    return cvf, win
