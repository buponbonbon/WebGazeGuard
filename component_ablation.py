#!/usr/bin/env python3
"""
Component ablation analysis for WebGazeGuard session CSV files.

What this script does
---------------------
For each session CSV (normal / head_movement / low_light), it recomputes the
rule-based CV risk from the logged behavioral signals using the same scoring
logic as the current risk_engine:
- blink
- distance
- posture

Then it creates ablation variants:
- full_cv
- no_blink
- no_distance
- no_posture

For each variant, it reports mean/std/min/max risk score.

Expected columns
----------------
Blink:
- blink_rate_per_min   OR   blink_rate_bpm

Posture:
- pitch_deg (or pitch_mean)
- yaw_deg   (or yaw_mean)

Distance:
Preferred:
- distance_cat_mode / distance_cat / distance_category
    values like: too_close, normal, too_far
Fallback:
- distance_cm
    if category is missing, the script can infer categories from thresholds.

Example
-------
python analysis/component_ablation.py ^
  --normal data/session_normal_cleaned.csv ^
  --head data/session_head_movement_cleaned.csv ^
  --low data/session_low_light_cleaned.csv ^
  --outdir analysis/results/component_ablation

If your CSV only has distance_cm and not distance category:
python analysis/component_ablation.py ^
  --normal data/session_normal_cleaned.csv ^
  --head data/session_head_movement_cleaned.csv ^
  --low data/session_low_light_cleaned.csv ^
  --outdir analysis/results/component_ablation ^
  --infer-distance-category ^
  --too-close-cm 50 ^
  --too-far-cm 75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


def clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def lin_score(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clip01((x - lo) / (hi - lo))


def blink_risk(blink_rate_bpm: Optional[float]) -> float:
    if blink_rate_bpm is None or (isinstance(blink_rate_bpm, float) and np.isnan(blink_rate_bpm)):
        return 0.0
    br = float(blink_rate_bpm)
    low = 7.0
    ok = 15.0
    return clip01(1.0 - lin_score(br, low, ok))


def posture_risk(pitch_mean: Optional[float], yaw_mean: Optional[float]) -> Tuple[float, Optional[str]]:
    if (pitch_mean is None or pd.isna(pitch_mean)) and (yaw_mean is None or pd.isna(yaw_mean)):
        return 0.0, None

    p = abs(float(pitch_mean)) if pitch_mean is not None and not pd.isna(pitch_mean) else 0.0
    y = abs(float(yaw_mean)) if yaw_mean is not None and not pd.isna(yaw_mean) else 0.0

    pitch_s = lin_score(p, lo=10.0, hi=25.0)
    yaw_s = lin_score(y, lo=15.0, hi=35.0)
    score = clip01(0.6 * pitch_s + 0.4 * yaw_s)

    if score < 0.7:
        return score, None
    if p >= y:
        return score, "FORWARD_HEAD"
    return score, "TILT"


def distance_risk_from_category(cat: Optional[str]) -> Tuple[float, Optional[str]]:
    if not cat or str(cat).strip().lower() in {"", "nan", "none", "null"}:
        return 0.0, None
    c = str(cat).strip().lower()
    if c == "too_close":
        return 1.0, "too_close"
    if c == "too_far":
        return 0.4, "too_far"
    return 0.0, None


def infer_distance_category(distance_cm: Optional[float], too_close_cm: float, too_far_cm: float) -> Optional[str]:
    if distance_cm is None or (isinstance(distance_cm, float) and np.isnan(distance_cm)):
        return None
    d = float(distance_cm)
    if d < too_close_cm:
        return "too_close"
    if d > too_far_cm:
        return "too_far"
    return "normal"


def combine_scores(components: Dict[str, float], weights: Dict[str, float]) -> float:
    s = 0.0
    wsum = 0.0
    for k, v in components.items():
        w = float(weights.get(k, 0.0))
        s += w * float(v)
        wsum += w
    return float(s / wsum) if wsum > 0 else 0.0


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_numeric_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def safe_text_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    return df[col].astype(str)


def summarize_scores(s: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def compute_component_scores(
    df: pd.DataFrame,
    *,
    infer_distance: bool = False,
    too_close_cm: float = 50.0,
    too_far_cm: float = 75.0,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    weights = weights or {
        "blink": 0.30,
        "distance": 0.25,
        "posture": 0.25,
    }

    blink_col = pick_first_existing(df, ["blink_rate_per_min", "blink_rate_bpm"])
    pitch_col = pick_first_existing(df, ["pitch_deg", "pitch_mean"])
    yaw_col = pick_first_existing(df, ["yaw_deg", "yaw_mean"])
    dist_cat_col = pick_first_existing(df, ["distance_cat_mode", "distance_cat", "distance_category"])
    dist_cm_col = pick_first_existing(df, ["distance_cm"])

    blink_series = safe_numeric_series(df, blink_col)
    pitch_series = safe_numeric_series(df, pitch_col)
    yaw_series = safe_numeric_series(df, yaw_col)
    dist_cat_series = safe_text_series(df, dist_cat_col)
    dist_cm_series = safe_numeric_series(df, dist_cm_col)

    rows = []
    for idx in df.index:
        blink_s = blink_risk(blink_series.loc[idx])

        post_s, posture_flag = posture_risk(
            pitch_series.loc[idx] if pitch_col else None,
            yaw_series.loc[idx] if yaw_col else None,
        )

        if dist_cat_col:
            cat = dist_cat_series.loc[idx]
        elif infer_distance and dist_cm_col:
            cat = infer_distance_category(dist_cm_series.loc[idx], too_close_cm, too_far_cm)
        else:
            cat = None

        dist_s, dist_flag = distance_risk_from_category(cat)

        components = {
            "blink": blink_s,
            "distance": dist_s,
            "posture": post_s,
        }

        full_cv = combine_scores(components, weights)
        no_blink = combine_scores(
            {"distance": dist_s, "posture": post_s},
            {"distance": weights["distance"], "posture": weights["posture"]}
        )
        no_distance = combine_scores(
            {"blink": blink_s, "posture": post_s},
            {"blink": weights["blink"], "posture": weights["posture"]}
        )
        no_posture = combine_scores(
            {"blink": blink_s, "distance": dist_s},
            {"blink": weights["blink"], "distance": weights["distance"]}
        )

        rows.append({
            "blink_component": blink_s,
            "distance_component": dist_s,
            "posture_component": post_s,
            "distance_flag": dist_flag,
            "posture_flag": posture_flag,
            "risk_full_cv": full_cv,
            "risk_no_blink": no_blink,
            "risk_no_distance": no_distance,
            "risk_no_posture": no_posture,
        })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def scenario_summary(name: str, scored_df: pd.DataFrame) -> pd.DataFrame:
    variants = {
        "Full system": "risk_full_cv",
        "Without blink": "risk_no_blink",
        "Without distance": "risk_no_distance",
        "Without posture": "risk_no_posture",
    }
    rows = []
    for label, col in variants.items():
        stats = summarize_scores(scored_df[col])
        rows.append({
            "Scenario": name,
            "Configuration": label,
            "Mean Risk": stats["mean"],
            "Risk Std": stats["std"],
            "Min Risk": stats["min"],
            "Max Risk": stats["max"],
            "N": stats["n"],
        })
    return pd.DataFrame(rows)


def build_main_table(normal_summary: pd.DataFrame, head_summary: pd.DataFrame, low_summary: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([normal_summary, head_summary, low_summary], ignore_index=True)


def build_delta_table(main_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in main_table["Scenario"].unique():
        sub = main_table[main_table["Scenario"] == scenario].copy()
        full_row = sub[sub["Configuration"] == "Full system"].iloc[0]
        full_mean = float(full_row["Mean Risk"])
        for _, r in sub.iterrows():
            rows.append({
                "Scenario": scenario,
                "Configuration": r["Configuration"],
                "Mean Risk": r["Mean Risk"],
                "Delta vs Full": float(r["Mean Risk"] - full_mean),
            })
    return pd.DataFrame(rows)


def write_text_summary(outpath: Path, main_table: pd.DataFrame) -> None:
    lines = []
    lines.append("WebGazeGuard Component Ablation Summary")
    lines.append("=" * 42)
    lines.append("")
    for scenario in main_table["Scenario"].unique():
        lines.append(f"[{scenario}]")
        sub = main_table[main_table["Scenario"] == scenario]
        for _, r in sub.iterrows():
            lines.append(
                f"- {r['Configuration']}: "
                f"mean={r['Mean Risk']:.4f}, std={r['Risk Std']:.4f}, "
                f"min={r['Min Risk']:.4f}, max={r['Max Risk']:.4f}"
            )
        lines.append("")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True, type=Path, help="Path to normal session CSV")
    parser.add_argument("--head", required=True, type=Path, help="Path to head-movement session CSV")
    parser.add_argument("--low", required=True, type=Path, help="Path to low-light session CSV")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--infer-distance-category", action="store_true", help="Infer distance category from distance_cm if no category column exists")
    parser.add_argument("--too-close-cm", type=float, default=50.0, help="Threshold for too_close when inferring distance category")
    parser.add_argument("--too-far-cm", type=float, default=75.0, help="Threshold for too_far when inferring distance category")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    normal_df = pd.read_csv(args.normal)
    head_df = pd.read_csv(args.head)
    low_df = pd.read_csv(args.low)

    normal_scored = compute_component_scores(
        normal_df,
        infer_distance=args.infer_distance_category,
        too_close_cm=args.too_close_cm,
        too_far_cm=args.too_far_cm,
    )
    head_scored = compute_component_scores(
        head_df,
        infer_distance=args.infer_distance_category,
        too_close_cm=args.too_close_cm,
        too_far_cm=args.too_far_cm,
    )
    low_scored = compute_component_scores(
        low_df,
        infer_distance=args.infer_distance_category,
        too_close_cm=args.too_close_cm,
        too_far_cm=args.too_far_cm,
    )

    normal_scored.to_csv(args.outdir / "normal_scored.csv", index=False)
    head_scored.to_csv(args.outdir / "head_movement_scored.csv", index=False)
    low_scored.to_csv(args.outdir / "low_light_scored.csv", index=False)

    normal_summary = scenario_summary("Normal usage", normal_scored)
    head_summary = scenario_summary("Head movement", head_scored)
    low_summary = scenario_summary("Low light", low_scored)

    main_table = build_main_table(normal_summary, head_summary, low_summary)
    delta_table = build_delta_table(main_table)

    main_table.to_csv(args.outdir / "component_ablation_table.csv", index=False)
    delta_table.to_csv(args.outdir / "component_ablation_delta.csv", index=False)

    config = {
        "distance_inference": bool(args.infer_distance_category),
        "too_close_cm": float(args.too_close_cm),
        "too_far_cm": float(args.too_far_cm),
        "weights": {"blink": 0.30, "distance": 0.25, "posture": 0.25},
    }
    (args.outdir / "component_ablation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_text_summary(args.outdir / "component_ablation_summary.txt", main_table)

    print(f"Saved to: {args.outdir}")
    print("Generated:")
    print("- normal_scored.csv")
    print("- head_movement_scored.csv")
    print("- low_light_scored.csv")
    print("- component_ablation_table.csv")
    print("- component_ablation_delta.csv")
    print("- component_ablation_summary.txt")
    print("- component_ablation_config.json")


if __name__ == "__main__":
    main()
