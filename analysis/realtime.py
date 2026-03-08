from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pct(a: np.ndarray, q: float) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def analyze(csv_path: Path) -> Dict[str, Any]:

    df = pd.read_csv(csv_path)

    # ---------- choose timestamp column ----------
    if "elapsed_s_rebased" in df.columns:
        ts = (
            pd.to_numeric(df["elapsed_s_rebased"], errors="coerce")
            .dropna()
            .astype(float)
            .to_numpy()
            * 1000.0
        )

    elif "ts_ms" in df.columns:
        ts = pd.to_numeric(df["ts_ms"], errors="coerce").dropna().astype(float).to_numpy()

    else:
        raise ValueError("CSV must contain 'elapsed_s_rebased' or 'ts_ms'.")

    if ts.size < 2:
        raise ValueError("Need at least 2 timestamps to compute FPS.")

    ts = np.sort(ts)
    dts = np.diff(ts)
    dts = dts[np.isfinite(dts) & (dts > 0)]

    duration_s = (ts[-1] - ts[0]) / 1000.0
    effective_fps = (ts.size - 1) / duration_s if duration_s > 0 else float("nan")

    inst_fps = 1000.0 / dts if dts.size else np.array([], dtype=float)

    summary: Dict[str, Any] = {
        "n_samples": int(df.shape[0]),
        "duration_s": float(duration_s),
        "effective_fps": float(effective_fps),
        "frame_interval_ms": {
            "mean": float(np.mean(dts)),
            "median": _pct(dts, 50),
            "p95": _pct(dts, 95),
            "p99": _pct(dts, 99),
            "min": float(np.min(dts)),
            "max": float(np.max(dts)),
        },
        "instantaneous_fps": {
            "mean": float(np.mean(inst_fps)),
            "median": _pct(inst_fps, 50),
            "p5": _pct(inst_fps, 5),
            "p95": _pct(inst_fps, 95),
            "min": float(np.min(inst_fps)),
            "max": float(np.max(inst_fps)),
        },
    }

    return summary


def save_outputs(summary: Dict[str, Any], csv_path: Path, outdir: Path):

    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "realtime_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    df = pd.read_csv(csv_path)

    if "elapsed_s_rebased" in df.columns:
        ts = (
            pd.to_numeric(df["elapsed_s_rebased"], errors="coerce")
            .dropna()
            .astype(float)
            .to_numpy()
            * 1000
        )
    else:
        ts = pd.to_numeric(df["ts_ms"], errors="coerce").dropna().astype(float).to_numpy()

    ts = np.sort(ts)
    dts = np.diff(ts)
    dts = dts[np.isfinite(dts) & (dts > 0)]

    inst_fps = 1000.0 / dts

    # ---------- frame interval histogram ----------
    plt.figure(figsize=(7, 4))
    plt.hist(dts, bins=40)
    plt.xlabel("Frame interval (ms)")
    plt.ylabel("Count")
    plt.title("Frame Interval Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "frame_interval_hist.png", dpi=200)
    plt.close()

    # ---------- fps histogram ----------
    plt.figure(figsize=(7, 4))
    plt.hist(inst_fps, bins=40)
    plt.xlabel("Instantaneous FPS")
    plt.ylabel("Count")
    plt.title("Instantaneous FPS Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "fps_hist.png", dpi=200)
    plt.close()

    # ---------- fps over time ----------
    if ts.size > 2:

        t0 = ts[0]
        mid_t = (ts[:-1] + ts[1:]) / 2
        mid_s = (mid_t - t0) / 1000

        bins = np.floor(mid_s).astype(int)

        fps_bin = {}
        for b, f in zip(bins, inst_fps):
            fps_bin.setdefault(int(b), []).append(float(f))

        xs = np.array(sorted(fps_bin.keys()), dtype=float)
        ys = np.array([np.mean(fps_bin[int(x)]) for x in xs], dtype=float)

        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys)
        plt.xlabel("Time (s)")
        plt.ylabel("FPS (1s mean)")
        plt.title("FPS Over Time")
        plt.tight_layout()
        plt.savefig(outdir / "fps_over_time.png", dpi=200)
        plt.close()

    # ---------- latex table ----------
    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Real-time performance summary.}\n"
        "\\label{tab:realtime}\n"
        "\\begin{tabular}{l r}\n"
        "\\hline\n"
        f"Samples & {summary['n_samples']} \\\\\n"
        f"Duration (s) & {summary['duration_s']:.2f} \\\\\n"
        f"Effective FPS & {summary['effective_fps']:.2f} \\\\\n"
        f"Median frame interval (ms) & {summary['frame_interval_ms']['median']:.2f} \\\\\n"
        f"P95 frame interval (ms) & {summary['frame_interval_ms']['p95']:.2f} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    (outdir / "latex_realtime_table.tex").write_text(latex)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="analysis_outputs/realtime")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    summary = analyze(csv_path)
    save_outputs(summary, csv_path, outdir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()