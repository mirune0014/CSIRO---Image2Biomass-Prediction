"""
EDA script for Image2Biomass exp001.

- Resolves dataset directory via `--data-dir`, `CSIRO_BIOMASS_DATA`, or `./data`.
- Loads `train.csv` and derives simple features (day_of_year).
- Produces summary stats, correlations, and basic plots.
- Saves outputs under `project/experiments/exp001/output/`:
    - eda_summary.json
    - eda_report.md
    - class_distribution.png, state_distribution.png
    - histogram_target_<name>.png
    - scatter_target_vs_pre_ndvi.png, scatter_target_vs_height.png

Usage:
    python -m project.experiments.exp001.eda --data-dir <path/to/data>
    # or
    python project/experiments/exp001/eda.py --data-dir <path/to/data>
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

# Use non-interactive backend for headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


@dataclass
class EDAConfig:
    """Configuration for exp001 EDA run."""

    data_dir: Path
    output_dir: Path


def resolve_data_dir(cli_dir: Optional[str]) -> Path:
    """Resolve dataset directory in the following priority.

    1) `cli_dir` if provided
    2) environment variable `CSIRO_BIOMASS_DATA`
    3) repo-local `./data`
    """

    if cli_dir:
        p = Path(cli_dir).expanduser().resolve()
        return p
    env = os.environ.get("CSIRO_BIOMASS_DATA")
    if env:
        return Path(env).expanduser().resolve()
    # fallback: repo ./data relative to this file
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "data").resolve()


def ensure_paths(cfg: EDAConfig) -> None:
    """Validate presence of required files and create output dir."""

    train_csv = cfg.data_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"train.csv not found at {train_csv}. Set --data-dir or CSIRO_BIOMASS_DATA to folder containing train.csv, train/, test/."
        )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def load_train(data_dir: Path) -> pd.DataFrame:
    """Load train.csv and derive helper columns.

    Expects columns: sample_id, image_path, Sampling_Date, State, Species,
    Pre_GSHH_NDVI, Height_Ave_cm, target_name, target
    """

    df = pd.read_csv(data_dir / "train.csv")
    # Date parsing
    if "Sampling_Date" in df.columns:
        dt = pd.to_datetime(df["Sampling_Date"], errors="coerce", infer_datetime_format=True)
        df["day_of_year"] = dt.dt.dayofyear
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
    else:
        df["day_of_year"] = np.nan
        df["year"] = np.nan
        df["month"] = np.nan
    return df


def save_bar_series(series: pd.Series, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    series.sort_values(ascending=False).plot(kind="bar", color="#4C78A8")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_hist(series: pd.Series, title: str, path: Path, bins: int = 50) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(series.dropna().values, bins=bins, color="#72B7B2", edgecolor="black")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_scatter(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=6, alpha=0.5, color="#F58518")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_correlations(df: pd.DataFrame) -> Dict[str, float]:
    corrs: Dict[str, float] = {}
    def corr_safe(a: pd.Series, b: pd.Series) -> float:
        if a.dropna().empty or b.dropna().empty:
            return float("nan")
        try:
            return float(a.corr(b))
        except Exception:
            return float("nan")

    target = df.get("target")
    for col in ["day_of_year", "Pre_GSHH_NDVI", "Height_Ave_cm"]:
        if col in df.columns:
            corrs[f"target_vs_{col}"] = corr_safe(target, df[col])
    return corrs


def build_report_md(summary: Dict, corrs: Dict[str, float], df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append(f"# exp001 EDA Report ({datetime.utcnow().isoformat()}Z)")
    lines.append("")
    lines.append(f"- Rows: {len(df)}, Cols: {len(df.columns)}")
    lines.append(f"- Targets: {sorted(df['target_name'].unique().tolist()) if 'target_name' in df.columns else 'N/A'}")
    lines.append("")
    lines.append("## Missing Values")
    miss = df.isna().sum().sort_values(ascending=False)
    lines += [f"- {k}: {int(v)}" for k, v in miss.items() if v > 0][:50]
    lines.append("")
    lines.append("## Target Summary by target_name")
    if "target_name" in df.columns:
        g = df.groupby("target_name")["target"].agg(["count", "mean", "std", "min", "max"]).round(4)
        for name, row in g.iterrows():
            lines.append(
                f"- {name}: n={int(row['count'])}, mean={row['mean']}, std={row['std']}, min={row['min']}, max={row['max']}"
            )
    lines.append("")
    lines.append("## Pearson Correlations with target")
    for k, v in corrs.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def run(cfg: EDAConfig) -> None:
    ensure_paths(cfg)
    df = load_train(cfg.data_dir)

    # Summary
    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "data_dir": str(cfg.data_dir),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
    }

    # Plots
    if "target_name" in df.columns:
        save_bar_series(
            df["target_name"].value_counts(),
            title="Distribution of target_name",
            path=cfg.output_dir / "class_distribution.png",
        )
        # Per-target histograms
        for name, sub in df.groupby("target_name"):
            save_hist(sub["target"], f"Histogram of target ({name})", cfg.output_dir / f"histogram_target_{name}.png")

    if "State" in df.columns:
        save_bar_series(df["State"].value_counts(), "Distribution of State", cfg.output_dir / "state_distribution.png")

    # Scatter plots
    if "Pre_GSHH_NDVI" in df.columns:
        save_scatter(
            df["Pre_GSHH_NDVI"],
            df["target"],
            xlabel="Pre_GSHH_NDVI",
            ylabel="target",
            title="target vs Pre_GSHH_NDVI",
            path=cfg.output_dir / "scatter_target_vs_pre_ndvi.png",
        )
    if "Height_Ave_cm" in df.columns:
        save_scatter(
            df["Height_Ave_cm"],
            df["target"],
            xlabel="Height_Ave_cm",
            ylabel="target",
            title="target vs Height_Ave_cm",
            path=cfg.output_dir / "scatter_target_vs_height.png",
        )

    # Correlations
    corrs = compute_correlations(df)

    # Persist
    with open(cfg.output_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "correlations": corrs}, f, ensure_ascii=False, indent=2)

    report_md = build_report_md(summary, corrs, df)
    (cfg.output_dir / "eda_report.md").write_text(report_md, encoding="utf-8")

    print("EDA complete.")
    print(f"- Data dir: {cfg.data_dir}")
    print(f"- Output:   {cfg.output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp001 EDA")
    p.add_argument("--data-dir", type=str, default=None, help="Path to dataset folder (contains train.csv, train/, test/)")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for EDA artifacts (default: project/experiments/exp001/output)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "project" / "experiments" / "exp001" / "output")
    run(EDAConfig(data_dir=data_dir, output_dir=output_dir))

