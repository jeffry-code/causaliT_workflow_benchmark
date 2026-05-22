#!/usr/bin/env python
"""
Plot training and optimization performance:
 - Transformer training metrics (pass one or more metrics.csv paths)
 - Surrogate training curve (epoch,mse)
 - Optimization histories from run_with_model_surrogate (cma_es.csv, adam_fd.csv, torch_adam.csv)

Example:
  python optimization/plot_performance.py \
    --metrics experiments/example/k_0/logs/csv/version_*/metrics.csv \
    --surrogate-hist models/p_to_c.csv \
    --opt-logs optimization_logs/cma_es.csv optimization_logs/adam_fd.csv optimization_logs/torch_adam.csv \
    --out perf.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_many(patterns):
    files = []
    for p in patterns:
        # if a directory, pick the highest version_*/metrics.csv inside
        if Path(p).is_dir():
            cand = glob.glob(str(Path(p) / "version_*" / "metrics.csv"))
            if cand:
                files.append(sorted(cand, key=lambda s: int(Path(s).parent.name.split("_")[-1]))[-1])
        else:
            files.extend(glob.glob(p))
    return files


def plot_training(ax, metric_files):
    if not metric_files:
        ax.text(0.5, 0.5, "No metrics provided", ha="center")
        return
    plotted = False
    for path in metric_files:
        df = pd.read_csv(path)
        if "epoch" in df.columns:
            x = df["epoch"]
        elif "step" in df.columns:
            x = df["step"]
        else:
            x = range(len(df))
        # choose a metric column by priority
        candidates = [c for c in df.columns if c.startswith("val_")]
        y_col = None
        for col in ["val_loss", "val_mae", "val_mse"]:
            if col in df.columns:
                y_col = col; break
        if y_col is None and candidates:
            y_col = candidates[0]
        if y_col is None:
            for col in ["train_loss", "loss"]:
                if col in df.columns:
                    y_col = col; break
        if y_col is None:
            # fallback: first numeric column excluding step/epoch
            for col in df.columns:
                if col not in ("step", "epoch") and np.issubdtype(df[col].dtype, np.number):
                    y_col = col; break
        if y_col is None:
            continue
        y_series = pd.to_numeric(df[y_col], errors="coerce")
        mask = y_series.notna()
        if mask.any():
            x_plot = x[mask] if hasattr(x, "__len__") else np.arange(mask.sum())
            ax.plot(x_plot, y_series[mask], label=f"{Path(path).parent.name}:{y_col}")
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No numeric metrics found", ha="center")
    ax.set_xlabel("epoch/step")
    ax.set_ylabel("metric")
    ax.set_title("Transformer training")
    ax.legend(fontsize=7)


def plot_surrogate(ax, hist_file):
    if not hist_file:
        ax.text(0.5, 0.5, "No surrogate history", ha="center")
        return
    df = pd.read_csv(hist_file)
    ax.plot(df["epoch"], df["mse"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Surrogate training")


def plot_opt(ax, opt_file):
    df = pd.read_csv(opt_file)
    ax.plot(df["iter"], df["f_best"], label=Path(opt_file).stem)
    ax.set_xlabel("iteration")
    ax.set_ylabel("best f")
    ax.set_title(f"Opt progress: {Path(opt_file).stem}")
    ax.legend(fontsize=7)


def plot_overfit(metric_file: str, out_file: str):
    df = pd.read_csv(metric_file)
    if "epoch" in df.columns:
        x = df["epoch"]
        x_label = "epoch"
    elif "step" in df.columns:
        x = df["step"]
        x_label = "step"
    else:
        x = np.arange(len(df))
        x_label = "index"

    train_candidates = ["train_loss", "loss", "train_mse"]
    val_candidates = ["val_loss", "val_mse", "val_mae"]
    train_col = next((c for c in train_candidates if c in df.columns), None)
    val_col = next((c for c in val_candidates if c in df.columns), None)

    if train_col is None or val_col is None:
        raise ValueError(
            f"Need both train and validation columns. Found train={train_col}, val={val_col}. "
            f"Available columns: {list(df.columns)}"
        )

    def extract_series(df_: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
        tmp = df_[[x_col, y_col]].copy()
        tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
        tmp = tmp.dropna(subset=[x_col, y_col])
        if tmp.empty:
            return pd.Series(dtype=float)
        # For repeated x values, keep last logged metric value.
        tmp = tmp.groupby(x_col, as_index=False)[y_col].last()
        return tmp.set_index(x_col)[y_col].sort_index()

    x_col = "epoch" if "epoch" in df.columns else ("step" if "step" in df.columns else None)
    if x_col is None:
        raise ValueError("metrics.csv must contain either 'epoch' or 'step' column.")

    train_s = extract_series(df, x_col, train_col)
    val_s = extract_series(df, x_col, val_col)
    if train_s.empty or val_s.empty:
        raise ValueError(f"Could not extract non-empty train/val series from {metric_file}.")

    # Align on common x values for gap.
    common_x = train_s.index.intersection(val_s.index)
    if len(common_x) == 0:
        raise ValueError("No shared epoch/step points between train and val series.")
    gap_s = val_s.loc[common_x] - train_s.loc[common_x]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(train_s.index, train_s.values, label=train_col, color="C0")
    axes[0].plot(val_s.index, val_s.values, label=val_col, color="C1")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("metric")
    axes[0].set_title("Train vs Validation")
    axes[0].legend()

    axes[1].plot(gap_s.index, gap_s.values, color="C3")
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(f"{val_col} - {train_col}")
    axes[1].set_title("Generalization Gap")

    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    print(f"Saved overfit diagnostics to {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="*", default=[], help="Paths/patterns to transformer metrics.csv")
    ap.add_argument("--surrogate-hist", type=str, default=None, help="CSV with epoch,mse for surrogate")
    ap.add_argument("--opt-logs", nargs="*", default=[], help="Optimization logs (cma_es.csv etc.)")
    ap.add_argument("--overfit", action="store_true", help="Plot train/val + gap diagnostics from one metrics file.")
    ap.add_argument("--out", type=str, default="perf.png")
    args = ap.parse_args()

    metric_files = load_many(args.metrics)
    if args.overfit:
        if not metric_files:
            raise ValueError("No metrics file provided. Pass --metrics <path_or_dir>.")
        # Use latest selected metrics file (last in list) for overfit diagnostics.
        plot_overfit(metric_files[-1], args.out)
        return

    n_opt = max(1, len(args.opt_logs))
    n_cols = 2 + n_opt  # training, surrogate, each opt
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    axes = list(axes)
    plot_training(axes[0], metric_files)
    plot_surrogate(axes[1], args.surrogate_hist)
    if args.opt_logs:
        for i, path in enumerate(args.opt_logs):
            plot_opt(axes[2+i], path)
    else:
        axes[2].text(0.5, 0.5, "No optimization logs", ha="center")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved performance plot to {args.out}")


if __name__ == "__main__":
    main()
