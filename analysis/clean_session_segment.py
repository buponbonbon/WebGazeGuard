

import math
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


def load_session(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "ts_ms" not in df.columns:
        raise ValueError("CSV must contain a 'ts_ms' column.")

    df = df.copy()
    df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce")

    if df["ts_ms"].isna().any():
        raise ValueError("Column 'ts_ms' contains invalid or missing values.")

    df = df.sort_values("ts_ms").reset_index(drop=True)

    # frame index of original file after sorting
    df["frame_idx_original"] = np.arange(len(df))

    # elapsed time from original first timestamp
    ts0 = float(df["ts_ms"].iloc[0])
    df["elapsed_s"] = (df["ts_ms"] - ts0) / 1000.0

    return df


def estimate_dt_ms(df: pd.DataFrame) -> float:
    diffs = df["ts_ms"].diff().dropna()
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]

    if len(diffs) == 0:
        raise ValueError("Cannot estimate frame interval from ts_ms.")

    return float(np.median(diffs))


def get_total_duration_seconds(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float((df["ts_ms"].iloc[-1] - df["ts_ms"].iloc[0]) / 1000.0)


def mark_usable_rows(
    df: pd.DataFrame,
    gap_factor: float = 3.0
) -> Tuple[pd.DataFrame, float, float]:
    """
    Mark usable rows by detecting large timestamp gaps.

    A row is considered near an unstable gap if:
    - the gap before it is too large
    - or the gap after it is too large

    Returns:
        out: dataframe with columns dt_ms, usable
        median_dt_ms: estimated nominal frame interval
        threshold_ms: large-gap threshold
    """
    out = df.copy()

    median_dt_ms = estimate_dt_ms(out)
    threshold_ms = gap_factor * median_dt_ms

    out["dt_ms"] = out["ts_ms"].diff()

    big_gap_prev = out["dt_ms"] > threshold_ms
    big_gap_next = big_gap_prev.shift(-1, fill_value=False)

    usable = ~(big_gap_prev | big_gap_next)

    if len(usable) > 0:
        usable.iloc[0] = True

    out["usable"] = usable.astype(bool)

    return out, median_dt_ms, threshold_ms


def split_contiguous_blocks(
    df: pd.DataFrame,
    median_dt_ms: float,
    gap_factor: float = 3.0
) -> List[pd.DataFrame]:
    threshold_ms = gap_factor * median_dt_ms
    usable_df = df[df["usable"]].copy()

    if usable_df.empty:
        return []

    dt = usable_df["ts_ms"].diff().fillna(median_dt_ms)
    block_id = (dt > threshold_ms).cumsum()

    blocks = []
    for _, g in usable_df.groupby(block_id):
        block = g.copy().reset_index(drop=True)
        blocks.append(block)

    return blocks


def concat_blocks_rebased(
    blocks: List[pd.DataFrame],
    target_seconds: Optional[float] = None
) -> pd.DataFrame:
    """
    Concatenate usable blocks into one continuous rebased timeline.
    Keeps measured values unchanged.
    """
    if not blocks:
        return pd.DataFrame()

    frames = []
    elapsed_offset = 0.0
    frame_idx = 0

    for block in blocks:
        block = block.copy()

        # local elapsed in this block
        block_elapsed = (block["ts_ms"] - block["ts_ms"].iloc[0]) / 1000.0

        # rebased elapsed after concatenation
        block["elapsed_s_rebased"] = block_elapsed + elapsed_offset

        # new clean frame index
        block["frame_idx"] = np.arange(frame_idx, frame_idx + len(block))

        frames.append(block)

        elapsed_offset = float(block["elapsed_s_rebased"].iloc[-1])
        frame_idx += len(block)

        if target_seconds is not None and elapsed_offset >= target_seconds:
            break

    out = pd.concat(frames, ignore_index=True)

    if target_seconds is not None:
        out = out[out["elapsed_s_rebased"] <= target_seconds].copy()

    return out.reset_index(drop=True)


def select_best_window_rebased(
    df_rebased: pd.DataFrame,
    target_seconds: float = 120.0,
    score_columns=("distance_cm", "yaw_deg", "pitch_deg", "strain_risk")
) -> pd.DataFrame:
    """
    Select the cleanest window of approximately target_seconds.
    Lower score = better window.
    """
    if df_rebased.empty:
        return df_rebased.copy()

    total_rebased_seconds = float(df_rebased["elapsed_s_rebased"].iloc[-1])
    if total_rebased_seconds <= target_seconds:
        out = df_rebased.copy()
        out["elapsed_s_rebased"] = out["elapsed_s_rebased"] - out["elapsed_s_rebased"].iloc[0]
        out["frame_idx"] = np.arange(len(out))
        return out.reset_index(drop=True)

    elapsed = df_rebased["elapsed_s_rebased"].to_numpy()
    n = len(df_rebased)

    best_window = None
    best_score = math.inf

    for start_idx in range(n):
        start_t = elapsed[start_idx]
        end_t = start_t + target_seconds

        end_idx = np.searchsorted(elapsed, end_t, side="right")
        window = df_rebased.iloc[start_idx:end_idx].copy()

        if window.empty:
            continue

        duration = float(window["elapsed_s_rebased"].iloc[-1] - window["elapsed_s_rebased"].iloc[0])

        # require window close enough to target length
        if duration < target_seconds * 0.97:
            continue

        nan_penalty = 0.0
        smooth_penalty = 0.0
        outlier_penalty = 0.0

        for col in score_columns:
            if col in window.columns:
                x = pd.to_numeric(window[col], errors="coerce")

                # penalize missing values
                nan_penalty += float(x.isna().mean()) * 1000.0

                # penalize abrupt jumps
                dx = x.diff().abs()
                smooth_penalty += float(dx.median(skipna=True) if not dx.dropna().empty else 0.0)

        # extra penalty for large pitch outliers
        if "pitch_deg" in window.columns:
            pitch_abs = pd.to_numeric(window["pitch_deg"], errors="coerce").abs()
            outlier_penalty += float((pitch_abs > 45).mean()) * 100.0

        total_score = nan_penalty + smooth_penalty + outlier_penalty

        if total_score < best_score:
            best_score = total_score
            best_window = window

    if best_window is None:
        best_window = df_rebased.copy()

    best_window = best_window.copy()
    best_window["elapsed_s_rebased"] = (
        best_window["elapsed_s_rebased"] - best_window["elapsed_s_rebased"].iloc[0]
    )
    best_window["frame_idx"] = np.arange(len(best_window))

    return best_window.reset_index(drop=True)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "frame_idx",
        "frame_idx_original",
        "elapsed_s",
        "elapsed_s_rebased",
        "ts_ms",
        "dt_ms",
        "usable",
        "blink_rate_per_min",
        "ear",
        "yaw_deg",
        "pitch_deg",
        "distance_cm",
        "strain_risk",
        "posture_flag",
    ]

    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def clean_and_select_segment(
    csv_path: str,
    output_csv: str,
    target_seconds: float = 120.0,
    gap_factor: float = 3.0
) -> pd.DataFrame:
    df = load_session(csv_path)

    total_seconds_raw = get_total_duration_seconds(df)

    marked, median_dt_ms, threshold_ms = mark_usable_rows(df, gap_factor=gap_factor)
    usable_rows = int(marked["usable"].sum())

    blocks = split_contiguous_blocks(marked, median_dt_ms, gap_factor=gap_factor)
    rebased = concat_blocks_rebased(blocks, target_seconds=None)

    usable_seconds = 0.0 if rebased.empty else float(rebased["elapsed_s_rebased"].iloc[-1])

    selected = select_best_window_rebased(
        rebased,
        target_seconds=target_seconds,
        score_columns=("distance_cm", "yaw_deg", "pitch_deg", "strain_risk"),
    )

    selected = reorder_columns(selected)
    selected.to_csv(output_csv, index=False)

    selected_seconds = 0.0 if selected.empty else float(selected["elapsed_s_rebased"].iloc[-1])

    print("=== CLEANING SUMMARY ===")
    print(f"Input file              : {csv_path}")
    print(f"Output file             : {output_csv}")
    print(f"Total rows (raw)        : {len(df)}")
    print(f"Total duration raw (s)  : {total_seconds_raw:.3f}")
    print(f"Estimated dt (ms)       : {median_dt_ms:.3f}")
    print(f"Gap threshold (ms)      : {threshold_ms:.3f}")
    print(f"Usable rows             : {usable_rows}")
    print(f"Number of usable blocks : {len(blocks)}")
    print(f"Usable duration (s)     : {usable_seconds:.3f}")
    print(f"Selected duration (s)   : {selected_seconds:.3f}")
    print(f"Selected rows           : {len(selected)}")

    return selected


# =========================
# COLAB USAGE EXAMPLE
# =========================
if __name__ == "__main__":
    # Example:
    # input_csv = "/content/session_normal.csv"
    # output_csv = "/content/session_normal_clean_2min.csv"

    input_csv = "/content/session_metrics.csv"
    output_csv = "/content/session_metrics_clean_selected.csv"

    clean_and_select_segment(
        csv_path=input_csv,
        output_csv=output_csv,
        target_seconds=120.0,  # 2 minutes
        gap_factor=3.0
    )