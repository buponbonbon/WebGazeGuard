#!/usr/bin/env python3

"""Real-time performance analysis for WebGazeGuard.

Reads a session_metrics CSV and computes:
- Effective FPS over the session
- Frame interval distribution (median/p95/p99)
- Instantaneous FPS distribution
Optionally, if a latency column exists (latency_ms/proc_ms/etc.), reports latency stats.

Outputs:
- realtime_summary.json
- frame_interval_hist.png
- fps_hist.png (if available)
- fps_over_time.png (if available)
- latex_realtime_table.tex

Usage:
    python analysis/realtime.py --csv path/to/session_metrics.csv --outdir analysis_out/realtime
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pct(a: np.ndarray, q: float) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def analyze(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    if "ts_ms" not in df.columns:
        raise ValueError("CSV must contain 'ts_ms' column (timestamp in ms).")

    ts = pd.to_numeric(df["ts_ms"], errors="coerce").dropna().astype(float).to_numpy()
    if ts.size < 2:
        raise ValueError("Need at least 2 timestamps to compute FPS/frame interval.")

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
            "mean": float(np.mean(dts)) if dts.size else float("nan"),
            "median": _pct(dts, 50),
            "p95": _pct(dts, 95),
            "p99": _pct(dts, 99),
            "min": float(np.min(dts)) if dts.size else float("nan"),
            "max": float(np.max(dts)) if dts.size else float("nan"),
        },
        "instantaneous_fps": {
            "mean": float(np.mean(inst_fps)) if inst_fps.size else float("nan"),
            "median": _pct(inst_fps, 50),
            "p5": _pct(inst_fps, 5),
            "p95": _pct(inst_fps, 95),
            "min": float(np.min(inst_fps)) if inst_fps.size else float("nan"),
            "max": float(np.max(inst_fps)) if inst_fps.size else float("nan"),
        },
        "notes": [],
    }

    # Optional latency column
    lat_col: Optional[str] = None
    for c in ("latency_ms", "proc_ms", "processing_ms", "lat_ms"):
        if c in df.columns:
            lat_col = c
            break

    if lat_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce").dropna().to_numpy(dtype=float)
        summary["latency_ms"] = {
            "mean": float(np.mean(lat)) if lat.size else float("nan"),
            "median": float(np.median(lat)) if lat.size else float("nan"),
            "p95": _pct(lat, 95),
            "p99": _pct(lat, 99),
            "min": float(np.min(lat)) if lat.size else float("nan"),
            "max": float(np.max(lat)) if lat.size else float("nan"),
        }
    else:
        summary["notes"].append(
            "No explicit latency column found (latency_ms/proc_ms/etc). Using frame interval as timing proxy."
        )

    return summary


def save_outputs(summary: Dict[str, Any], csv_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "realtime_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    df = pd.read_csv(csv_path)
    ts = pd.to_numeric(df["ts_ms"], errors="coerce").dropna().astype(float).to_numpy()
    ts = np.sort(ts)
    dts = np.diff(ts)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    inst_fps = 1000.0 / dts if dts.size else np.array([], dtype=float)

    # Frame interval histogram
    plt.figure(figsize=(7, 4))
    plt.hist(dts, bins=40)
    plt.xlabel("Frame interval (ms)")
    plt.ylabel("Count")
    plt.title("Frame Interval Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "frame_interval_hist.png", dpi=200)
    plt.close()

    # Instantaneous FPS histogram
    if inst_fps.size:
        plt.figure(figsize=(7, 4))
        plt.hist(inst_fps, bins=40)
        plt.xlabel("Instantaneous FPS (1000/dt)")
        plt.ylabel("Count")
        plt.title("Instantaneous FPS Distribution")
        plt.tight_layout()
        plt.savefig(outdir / "fps_hist.png", dpi=200)
        plt.close()

    # FPS over time (1s mean)
    if ts.size > 2 and inst_fps.size:
        t0 = ts[0]
        mid_t = (ts[:-1] + ts[1:]) / 2.0
        mid_s = (mid_t - t0) / 1000.0
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

    # LaTeX table snippet
    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Real-time performance summary (measured from session timestamp stream).}\n"
        "\\label{tab:realtime}\n"
        "\\begin{tabular}{l r}\n"
        "\\hline\n"
        f"Samples & {summary['n_samples']} \\\\n"
        f"Duration (s) & {summary['duration_s']:.2f} \\\\n"
        f"Effective FPS & {summary['effective_fps']:.2f} \\\\n"
        f"Median frame interval (ms) & {summary['frame_interval_ms']['median']:.2f} \\\\n"
        f"P95 frame interval (ms) & {summary['frame_interval_ms']['p95']:.2f} \\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    (outdir / "latex_realtime_table.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to session_metrics.csv")
    ap.add_argument("--outdir", default="analysis_out/realtime", help="Output directory")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    summary = analyze(csv_path)
    save_outputs(summary, csv_path, outdir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
