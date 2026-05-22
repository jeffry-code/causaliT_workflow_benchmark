#!/usr/bin/env python
"""
plot_multidim_results.py  —  thesis-quality plots for the dimension-scaling study
==================================================================================
Post-processing script: reads CSVs produced by run_multidim_pipeline.py and
generates the figures and tables used in the thesis results chapter.

Run AFTER run_multidim_pipeline.py has completed for all dimensions.
Run from repo root:
  python optimization/plot_multidim_results.py

Figures produced (in optimization/multidim/results/thesis/):
  fidelity_vs_dimension.png          — R² and RMSE vs d
  normalized_regret_vs_dimension.png — mean regret vs d (CMA-ES vs Adam)
  boxplot_regret_by_dimension.png    — regret distribution per d
  convergence_mean_best_so_far.png   — best-so-far curves averaged over seeds

Create thesis-ready tables and plots from multidim experiment outputs.

Inputs:
- multidim/results/multidim_summary.csv (from run_multidim_pipeline.py)
- per-dimension optimization outputs under:
  <dim_dir>/results/styblinski_tang/{cma,autograd}/summary.csv
  <dim_dir>/results/styblinski_tang/{cma,autograd}/evaluations_seed_*.csv

Outputs (default: multidim/results/thesis):
- tables:
  - dimension_overview.csv
  - seed_level_results.csv
- figures:
  - fidelity_vs_dimension.png
  - normalized_regret_vs_dimension.png
  - boxplot_regret_by_dimension.png
  - convergence_mean_best_so_far.png
  - fidelity_vs_regret_scatter.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Return the known global minimum for supported benchmark functions.
def true_optimum(function_name: str, dim: int) -> Optional[float]:
    fn = (function_name or "").lower()
    if fn == "styblinski_tang":
        return -39.1661657037714 * dim
    return None


# Ground truth optimum for the benchmark functions.
# Used for computing normalized regret in result tables and plots.


# Extract the seed index from a filename like evaluations_seed_0.csv.
def parse_seed_from_name(path: Path) -> Optional[int]:
    stem = path.stem  # evaluations_seed_0
    if "seed_" not in stem:
        return None
    try:
        return int(stem.split("seed_")[-1])
    except ValueError:
        return None


# Load final best true objective values for each seed and compute normalized regret.
def load_seed_level(dim_row: pd.Series) -> pd.DataFrame:
    dim = int(dim_row["dim"])
    function_name = str(dim_row.get("function", ""))
    f_star = true_optimum(function_name, dim)
    base = Path(dim_row["dim_dir"]) / "results" / "styblinski_tang"

    rows = []
    for opt in ["cma", "autograd"]:
        summary_csv = base / opt / "summary.csv"
        if not summary_csv.exists():
            continue
        sdf = pd.read_csv(summary_csv)
        if "f_best_true_raw" not in sdf.columns:
            continue
        for _, r in sdf.iterrows():
            f_best = float(r["f_best_true_raw"])
            entry = {
                "dim": dim,
                "optimizer": opt,
                "seed": int(r.get("seed", len(rows))),
                "f_best_true_raw": f_best,
                "f_star": f_star,
            }
            if f_star is not None and abs(f_star) > 0:
                entry["norm_regret"] = (f_best - f_star) / abs(f_star)
            else:
                entry["norm_regret"] = np.nan
            rows.append(entry)
    return pd.DataFrame(rows)


# Load one row per seed across optimizers for a given dimension.
# This collects the final true objective from each seed and computes
# normalized regret relative to the known optima.


# Load per-seed best-so-far curves and interpolate them to a common x grid.
def load_convergence_curves(dim_row: pd.Series, optimizer: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (x_grid, y_mean) where x_grid is eval fraction in [0,1]
    and y_mean is mean best-so-far true objective across seeds.
    """
    dim = int(dim_row["dim"])
    base = Path(dim_row["dim_dir"]) / "results" / "styblinski_tang" / optimizer
    files = sorted(base.glob("evaluations_seed_*.csv"))
    if not files:
        return np.array([]), np.array([])

    curves = []
    max_len = 0
    for f in files:
        df = pd.read_csv(f)
        if "f_true_raw" not in df.columns:
            continue
        y = pd.to_numeric(df["f_true_raw"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(y) == 0:
            continue
        y_best = np.minimum.accumulate(y)
        curves.append(y_best)
        max_len = max(max_len, len(y_best))

    if not curves:
        return np.array([]), np.array([])

    # interpolate each curve to common 0..1 grid for averaging
    x_grid = np.linspace(0.0, 1.0, 200)
    y_stack = []
    for y in curves:
        x = np.linspace(0.0, 1.0, len(y))
        y_i = np.interp(x_grid, x, y)
        y_stack.append(y_i)

    y_arr = np.vstack(y_stack)
    return x_grid, y_arr


# Plot the surrogate fidelity metrics (RMSE/R2) as a function of dimension.
def plot_fidelity(summary: pd.DataFrame, out: Path) -> None:
    dfx = summary.sort_values("dim")
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(dfx["dim"], dfx["rmse_std"], marker="o", color="tab:blue", label="RMSE (std)")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("RMSE (std)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(dfx["dim"], dfx["r2_raw"], marker="s", color="tab:orange", label="R2 (raw)")
    ax2.set_ylabel("R2 (raw)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Surrogate fidelity vs dimension")

    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)


# Plot mean normalized regret per optimizer across dimensions.
def plot_regret_means(seed_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for opt in ["cma", "autograd"]:
        d = seed_df[seed_df["optimizer"] == opt]
        if d.empty:
            continue
        g = d.groupby("dim")["norm_regret"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["dim"], g["mean"], yerr=g["std"], marker="o", capsize=4, label=opt)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Normalized regret (lower is better)")
    ax.set_title("Optimization quality vs dimension")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)


def plot_regret_box(seed_df: pd.DataFrame, out: Path) -> None:
    dims = sorted(seed_df["dim"].unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 4.5), sharey=True)
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        d = seed_df[seed_df["dim"] == dim]
        groups = []
        labels = []
        for opt in ["cma", "autograd"]:
            vals = d.loc[d["optimizer"] == opt, "norm_regret"].dropna().to_numpy()
            if len(vals) > 0:
                groups.append(vals)
                labels.append(opt)
        if groups:
            ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.set_title(f"{dim}D")
        ax.set_xlabel("optimizer")
    axes[0].set_ylabel("Normalized regret")
    fig.suptitle("Seed variability by dimension")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)


# Plot mean convergence curves for each dimension, showing optimizer progress.
def plot_convergence(summary: pd.DataFrame, out: Path) -> None:
    dims = sorted(summary["dim"].unique())
    fig, axes = plt.subplots(2, int(np.ceil(len(dims) / 2)), figsize=(12, 7), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, dim in enumerate(dims):
        ax = axes[i]
        row = summary.loc[summary["dim"] == dim].iloc[0]
        f_star = true_optimum(str(row.get("function", "")), int(dim))

        for opt, color in [("cma", "tab:blue"), ("autograd", "tab:orange")]:
            x_grid, y_arr = load_convergence_curves(row, opt)
            if len(x_grid) == 0:
                continue
            y_mean = np.mean(y_arr, axis=0)
            y_std = np.std(y_arr, axis=0)

            if f_star is not None and abs(f_star) > 0:
                y_mean = (y_mean - f_star) / abs(f_star)
                y_std = y_std / abs(f_star)
                ylabel = "Normalized regret"
            else:
                ylabel = "Best-so-far true objective"

            ax.plot(x_grid, y_mean, color=color, label=opt)
            ax.fill_between(x_grid, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

        ax.set_title(f"{dim}D")
        ax.set_xlabel("Evaluation fraction")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Convergence across dimensions (mean +/- std over seeds)")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)


# Compare surrogate fidelity to optimization regret for each dimension.
def plot_fidelity_vs_regret(summary: pd.DataFrame, out: Path) -> None:
    # use cma/autograd mean normalized regrets from seed-level aggregation
    df = summary.copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt, color in [("cma", "tab:blue"), ("autograd", "tab:orange")]:
        col = f"{opt}_norm_regret_mean"
        if col not in df.columns:
            continue
        ax.scatter(df["rmse_std"], df[col], label=opt, color=color, s=70)
        for _, r in df.iterrows():
            ax.annotate(f"{int(r['dim'])}D", (r["rmse_std"], r[col]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Surrogate RMSE (std)")
    ax.set_ylabel("Mean normalized regret")
    ax.set_title("Fidelity vs optimization performance")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build thesis-ready multidim plots/tables.")
    ap.add_argument("--root", type=Path, default=Path("optimization/multidim"))
    ap.add_argument("--summary", type=Path, default=None, help="Path to multidim_summary.csv")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <root>/results/thesis)")
    args = ap.parse_args()

    summary_csv = args.summary or (args.root / "results" / "multidim_summary.csv")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    out_dir = args.out_dir or (args.root / "results" / "thesis")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_csv).sort_values("dim")

    # seed-level table
    seed_rows = []
    for _, row in summary.iterrows():
        sdf = load_seed_level(row)
        if not sdf.empty:
            seed_rows.append(sdf)
    seed_df = pd.concat(seed_rows, ignore_index=True) if seed_rows else pd.DataFrame()

    # enrich dimension summary with normalized-regret stats
    if not seed_df.empty:
        g = seed_df.groupby(["dim", "optimizer"]) ["norm_regret"].agg(["mean", "median", "std", "min", "max", "count"]).reset_index()
        for opt in ["cma", "autograd"]:
            go = g[g["optimizer"] == opt].copy()
            if go.empty:
                continue
            go = go.rename(columns={
                "mean": f"{opt}_norm_regret_mean",
                "median": f"{opt}_norm_regret_median",
                "std": f"{opt}_norm_regret_std",
                "min": f"{opt}_norm_regret_min",
                "max": f"{opt}_norm_regret_max",
                "count": f"{opt}_n",
            })
            keep = ["dim", f"{opt}_norm_regret_mean", f"{opt}_norm_regret_median", f"{opt}_norm_regret_std", f"{opt}_norm_regret_min", f"{opt}_norm_regret_max", f"{opt}_n"]
            summary = summary.merge(go[keep], on="dim", how="left")

    # Save tables
    summary.to_csv(out_dir / "dimension_overview.csv", index=False)
    if not seed_df.empty:
        seed_df.to_csv(out_dir / "seed_level_results.csv", index=False)

    # Plots
    plot_fidelity(summary, out_dir / "fidelity_vs_dimension.png")
    if not seed_df.empty:
        plot_regret_means(seed_df, out_dir / "normalized_regret_vs_dimension.png")
        plot_regret_box(seed_df, out_dir / "boxplot_regret_by_dimension.png")
    plot_convergence(summary, out_dir / "convergence_mean_best_so_far.png")
    if "cma_norm_regret_mean" in summary.columns or "autograd_norm_regret_mean" in summary.columns:
        plot_fidelity_vs_regret(summary, out_dir / "fidelity_vs_regret_scatter.png")

    print(f"Saved thesis tables/plots to: {out_dir}")


if __name__ == "__main__":
    main()
