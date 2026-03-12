#!/usr/bin/env python3
"""
Robustness evaluation for WebGazeGuard CV sessions.

This script compares three practical webcam scenarios:
- normal usage
- head movement
- low light

It is intended for robustness analysis, not eye-strain classification.
It summarizes signal stability, valid-frame rate, risk distribution, and
scenario-wise variability using exported session CSV files.

Example:
python analysis/cv_eval.py \
  --normal data/session_normal_cleaned.csv \
  --head data/session_head_movement_cleaned.csv \
  --low data/session_low_light_cleaned.csv \
  --outdir analysis/results/cv_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


NUMERIC_CANDIDATES = [
    "distance_cm",
    "blink_rate_per_min",
    "ear",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "strain_risk",
    "elapsed_s_rebased",
    "ts_ms",
]

RISK_CANDIDATES = ["strain_risk", "risk_score"]
VALID_FACE_CANDIDATES = ["face_valid", "has_face", "valid_face", "face_detected"]
WARNING_CANDIDATES = ["warning", "alert", "message", "advice"]


def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col in NUMERIC_CANDIDATES or df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="ignore")
            df[col] = converted
    return df


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def valid_rate(df: pd.DataFrame) -> float | None:
    col = pick_first_existing(df, VALID_FACE_CANDIDATES)
    if col is None:
        return None
    s = df[col]
    if s.dtype == bool:
        return float(s.mean())
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().any():
        return float((s > 0).mean())
    return None


def warning_rate(df: pd.DataFrame) -> float | None:
    col = pick_first_existing(df, WARNING_CANDIDATES)
    if col is None:
        return None
    s = df[col]
    if s.dtype == bool:
        return float(s.mean())
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
        return float((s.fillna(0) != 0).mean())
    s = s.astype(str).str.strip().str.lower()
    empties = {"", "nan", "none", "null"}
    return float((~s.isin(empties)).mean())


def risk_zone_rates(df: pd.DataFrame) -> Dict[str, float] | None:
    col = pick_first_existing(df, RISK_CANDIDATES)
    if col is None:
        return None
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(x) == 0:
        return None
    return {
        "low_0_033": float((x < 0.33).mean()),
        "mid_033_066": float(((x >= 0.33) & (x < 0.66)).mean()),
        "high_066_100": float((x >= 0.66).mean()),
    }


def summarize_numeric(series: pd.Series) -> Dict[str, float] | None:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return None
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(x.median()),
        "p10": float(x.quantile(0.10)),
        "p90": float(x.quantile(0.90)),
    }


def scenario_summary(name: str, path: Path) -> Dict:
    df = safe_read_csv(path)

    summary: Dict[str, object] = {
        "scenario": name,
        "file": str(path),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "valid_face_rate": valid_rate(df),
        "warning_rate": warning_rate(df),
        "risk_zones": risk_zone_rates(df),
        "metrics": {},
    }

    for col in ["distance_cm", "blink_rate_per_min", "ear", "yaw_deg", "pitch_deg", "roll_deg", "strain_risk"]:
        if col in df.columns:
            stats = summarize_numeric(df[col])
            if stats is not None:
                summary["metrics"][col] = stats

    return summary


def pairwise_change(base: Dict, other: Dict, metric: str) -> Dict[str, float] | None:
    base_metrics = base.get("metrics", {})
    other_metrics = other.get("metrics", {})
    if metric not in base_metrics or metric not in other_metrics:
        return None
    b = base_metrics[metric]["mean"]
    o = other_metrics[metric]["mean"]
    return {
        "base_mean": float(b),
        "other_mean": float(o),
        "absolute_change": float(o - b),
        "relative_change_pct": float(((o - b) / b) * 100.0) if b not in (0, 0.0) else np.nan,
    }


def build_overview(normal: Dict, head: Dict, low: Dict) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for summary in [normal, head, low]:
        row: Dict[str, object] = {
            "scenario": summary["scenario"],
            "n_rows": summary["n_rows"],
            "valid_face_rate": summary["valid_face_rate"],
            "warning_rate": summary["warning_rate"],
        }
        zones = summary.get("risk_zones") or {}
        row["risk_low"] = zones.get("low_0_033")
        row["risk_mid"] = zones.get("mid_033_066")
        row["risk_high"] = zones.get("high_066_100")

        for metric in ["distance_cm", "blink_rate_per_min", "ear", "yaw_deg", "pitch_deg", "roll_deg", "strain_risk"]:
            row[f"{metric}_mean"] = summary.get("metrics", {}).get(metric, {}).get("mean")
            row[f"{metric}_std"] = summary.get("metrics", {}).get(metric, {}).get("std")
        rows.append(row)
    return pd.DataFrame(rows)


def write_text_report(outpath: Path, normal: Dict, head: Dict, low: Dict) -> None:
    lines: List[str] = []
    lines.append("WebGazeGuard CV Robustness Evaluation")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Protocol")
    lines.append("- normal: reference / inlier scenario")
    lines.append("- head_movement: robustness scenario")
    lines.append("- low_light: robustness scenario")
    lines.append("")
    lines.append("Note: this report is for robustness analysis, not CV-only eye-strain classification.")
    lines.append("")

    def add_scenario_block(summary: Dict) -> None:
        lines.append(f"[{summary['scenario']}]")
        lines.append(f"Rows: {summary['n_rows']}")
        if summary.get("valid_face_rate") is not None:
            lines.append(f"Valid-face rate: {summary['valid_face_rate']:.4f}")
        if summary.get("warning_rate") is not None:
            lines.append(f"Warning rate: {summary['warning_rate']:.4f}")
        zones = summary.get("risk_zones")
        if zones:
            lines.append(
                "Risk zones: "
                f"low={zones['low_0_033']:.4f}, "
                f"mid={zones['mid_033_066']:.4f}, "
                f"high={zones['high_066_100']:.4f}"
            )
        for metric, stats in summary.get("metrics", {}).items():
            lines.append(
                f"{metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}"
            )
        lines.append("")

    add_scenario_block(normal)
    add_scenario_block(head)
    add_scenario_block(low)

    lines.append("Pairwise comparison against normal")
    for scenario_name, summary in [("head_movement", head), ("low_light", low)]:
        lines.append(f"- {scenario_name}")
        for metric in ["distance_cm", "blink_rate_per_min", "yaw_deg", "pitch_deg", "strain_risk"]:
            comp = pairwise_change(normal, summary, metric)
            if comp is None:
                continue
            rel = comp["relative_change_pct"]
            rel_txt = "nan" if np.isnan(rel) else f"{rel:.2f}%"
            lines.append(
                f"  {metric}: base={comp['base_mean']:.4f}, "
                f"other={comp['other_mean']:.4f}, "
                f"delta={comp['absolute_change']:.4f}, rel={rel_txt}"
            )
        lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True, type=Path, help="Path to session_normal_cleaned.csv")
    parser.add_argument("--head", required=True, type=Path, help="Path to session_head_movement_cleaned.csv")
    parser.add_argument("--low", required=True, type=Path, help="Path to session_low_light_cleaned.csv")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    normal = scenario_summary("normal", args.normal)
    head = scenario_summary("head_movement", args.head)
    low = scenario_summary("low_light", args.low)

    results = {
        "protocol": {
            "normal": "reference_inlier",
            "head_movement": "robustness_scenario",
            "low_light": "robustness_scenario",
            "task_type": "cv_robustness_analysis",
        },
        "normal": normal,
        "head_movement": head,
        "low_light": low,
        "pairwise_vs_normal": {
            "head_movement": {
                metric: pairwise_change(normal, head, metric)
                for metric in ["distance_cm", "blink_rate_per_min", "ear", "yaw_deg", "pitch_deg", "roll_deg", "strain_risk"]
            },
            "low_light": {
                metric: pairwise_change(normal, low, metric)
                for metric in ["distance_cm", "blink_rate_per_min", "ear", "yaw_deg", "pitch_deg", "roll_deg", "strain_risk"]
            },
        },
    }

    overview = build_overview(normal, head, low)
    overview.to_csv(args.outdir / "cv_robustness_overview.csv", index=False)
    (args.outdir / "cv_robustness_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_text_report(args.outdir / "cv_robustness_summary.txt", normal, head, low)

    print(f"Saved to: {args.outdir}")


if __name__ == "__main__":
    main()
