#!/usr/bin/env python
"""
run_multidim_pipeline.py  —  Workflow I dimension-scaling pipeline
===================================================================
Orchestrates the full surrogate-optimization experiment across multiple
problem dimensionalities (d = 2, 10, 20, 40, 80) for the Styblinski-Tang
benchmark (Workflow I). Calls run_with_model_template.py and
evaluate_checkpoints_true_function.py as subprocesses, one per dimension.

INPUT  (--root, default: optimization/multidim):
  optimization/multidim/
    2dim/   10dim/   20dim/   40dim/   80dim/
      data/example/           ← dataset + normalisation maps
      experiments/example/    ← trained proT checkpoint

OUTPUT (written to <dim_dir>/results/styblinski_tang/):
  summary.csv, per-seed CSVs, trajectory CSVs, plots
  + combined multidim_summary.csv under optimization/multidim/results/

Run from repo root:
  python optimization/run_multidim_pipeline.py --dims 2 10 20 40 80 --seeds 10 --budget 1000 --optimizer both

Run multidimensional surrogate optimization/evaluation pipeline.

Expected per-dimension folder layout (relative to each <dim>dim folder):
- data/example/... (maps, meta.json, ds.npz)
- experiments/.../checkpoints/best_checkpoint.ckpt

This script will, for each dimension folder:
1) Run optimization/run_with_model_template.py (multiseed mode)
2) Run optimization/evaluate_checkpoints_true_function.py on best checkpoint
3) Collect metrics and optimization summaries
4) Write combined CSV + comparison plots to multidim/results

Example:
  python optimization/run_multidim_pipeline.py --dims 10 20 40 80 --seeds 10 --budget 1000 --optimizer both
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Parse a folder name like '80dim' and return the numeric dimension.
def parse_dim_from_name(name: str) -> Optional[int]:
    m = re.match(r"^(\d+)dim$", name.strip().lower())
    return int(m.group(1)) if m else None


# Find the requested dimension folders under the root path.
# If --dims is passed, only those folders are validated and returned.
def discover_dim_dirs(root: Path, dims: Optional[List[int]]) -> List[tuple[int, Path]]:
    if dims:
        out = []
        for d in dims:
            p = root / f"{d}dim"
            if not p.exists():
                raise FileNotFoundError(f"Missing dimension folder: {p}")
            out.append((int(d), p))
        return out

    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        d = parse_dim_from_name(p.name)
        if d is not None:
            out.append((d, p))
    if not out:
        raise FileNotFoundError(f"No '<N>dim' folders found in {root}")
    return sorted(out, key=lambda t: t[0])


# Determine which dimension folders should be processed.
# If the user passes --dims, only those folders are required. Otherwise,
# the script discovers all folders named `<N>dim` under the root directory.


def has_data_files(p: Path) -> bool:
    return (p / "input_vars_map.json").exists() and (p / "target_vars_map.json").exists() and (p / "meta.json").exists()


def find_data_dir(dim_dir: Path) -> Path:
    # Look for the expected data layout first, then fall back to a recursive search.
    direct = [dim_dir / "data" / "example", dim_dir / "data"]
    for c in direct:
        if has_data_files(c):
            return c

    # fallback: recursive search
    for c in dim_dir.rglob("meta.json"):
        parent = c.parent
        if has_data_files(parent):
            return parent

    raise FileNotFoundError(f"Could not find data dir with maps/meta under {dim_dir}")


# Locate the data directory for a dimension folder.
# The workflow expects `input_vars_map.json`, `target_vars_map.json`, and `meta.json`.
# It supports both a direct layout and a fallback recursive search.


def find_best_checkpoint(dim_dir: Path) -> Path:
    preferred = dim_dir / "experiments" / "example" / "k_0" / "checkpoints" / "best_checkpoint.ckpt"
    if preferred.exists():
        return preferred

    candidates = list(dim_dir.rglob("best_checkpoint.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best_checkpoint.ckpt found under {dim_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


# Prefer the standard best checkpoint path, but fall back to any found checkpoint
# under the dimension directory. This makes the pipeline robust across slightly
# different experiment layouts.


def find_summary_combined(dim_dir: Path) -> Optional[Path]:
    preferred = dim_dir / "results" / "styblinski_tang" / "summary_combined.csv"
    if preferred.exists():
        return preferred
    cands = list((dim_dir / "results").rglob("summary_combined.csv")) if (dim_dir / "results").exists() else []
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def run_cmd(cmd: List[str], cwd: Path) -> None:
    # Run a subprocess and echo stdout/stderr for debugging.
    print(f"\n[cwd={cwd}] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


# Extract per-optimizer mean/median/min/std of the true objective from a combined summary CSV.
def aggregate_opt_stats(summary_combined: Path) -> Dict[str, float]:
    df = pd.read_csv(summary_combined)
    out: Dict[str, float] = {}
    if "optimizer" not in df.columns or "f_best_true_raw" not in df.columns:
        return out
    for opt in sorted(df["optimizer"].dropna().unique()):
        vals = pd.to_numeric(df.loc[df["optimizer"] == opt, "f_best_true_raw"], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        out[f"{opt}_mean_true"] = float(vals.mean())
        out[f"{opt}_median_true"] = float(vals.median())
        out[f"{opt}_min_true"] = float(vals.min())
        out[f"{opt}_std_true"] = float(vals.std(ddof=0))
        out[f"{opt}_n"] = int(len(vals))
    return out


# Plot surrogate fidelity (RMSE/R²) and optimizer performance as a function of dimension.
def plot_dim_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dfx = df.sort_values("dim")

    # RMSE/R2 over dimensions
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(dfx["dim"], dfx["rmse_raw"], marker="o", label="RMSE(raw)", color="tab:blue")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("RMSE(raw)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(dfx["dim"], dfx["r2_raw"], marker="s", label="R2(raw)", color="tab:orange")
    ax2.set_ylabel("R2(raw)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("Checkpoint fidelity vs dimension")
    plt.tight_layout()
    plt.savefig(out_dir / "fidelity_vs_dimension.png", dpi=220)
    plt.close(fig)

    # Optimization performance by optimizer (median true best)
    opt_cols = [c for c in dfx.columns if c.endswith("_median_true")]
    if opt_cols:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for c in sorted(opt_cols):
            label = c.replace("_median_true", "")
            ax.plot(dfx["dim"], dfx[c], marker="o", label=f"{label} median best true f")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Objective (lower is better)")
        ax.set_title("Optimization quality vs dimension")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "optimization_vs_dimension.png", dpi=220)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run optimization + checkpoint fidelity eval across multidim folders.")
    ap.add_argument("--root", type=Path, default=Path("optimization/multidim"), help="Root folder containing 10dim, 20dim, ...")
    ap.add_argument("--dims", type=int, nargs="*", default=None, help="Optional subset of dimensions to run.")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")

    # optimization settings
    ap.add_argument("--optimizer", type=str, default="both", choices=["cma", "autograd", "both"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--skip-opt", action="store_true")

    # evaluation settings
    ap.add_argument("--eval-samples", type=int, default=5000)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--skip-eval", action="store_true")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_opt_script = repo_root / "optimization" / "run_with_model_template.py"
    eval_script = repo_root / "optimization" / "evaluate_checkpoints_true_function.py"

    root_abs = args.root.resolve()
    dim_dirs = discover_dim_dirs(root_abs, args.dims)
    rows: List[Dict[str, object]] = []

    for dim, dim_dir in dim_dirs:
        print(f"\n=== Dimension {dim} @ {dim_dir} ===")
        dim_dir = dim_dir.resolve()
        data_dir = find_data_dir(dim_dir).resolve()
        ckpt = find_best_checkpoint(dim_dir).resolve()

        if not args.skip_opt:
            cmd = [
                args.python,
                str(run_opt_script),
                "--mode", "multiseed",
                "--optimizer", args.optimizer,
                "--seeds", str(args.seeds),
                "--budget", str(args.budget),
                "--lr", str(args.lr),
            ]
            run_cmd(cmd, cwd=dim_dir)

        eval_csv = (dim_dir / "results" / "checkpoint_eval.csv").resolve()
        if not args.skip_eval:
            cmd = [
                args.python,
                str(eval_script),
                "--checkpoint", str(ckpt),
                "--data-dir", str(data_dir),
                "--n-samples", str(args.eval_samples),
                "--seed", str(args.eval_seed),
                "--out-csv", str(eval_csv),
            ]
            run_cmd(cmd, cwd=dim_dir)

        row: Dict[str, object] = {
            "dim": dim,
            "dim_dir": str(dim_dir),
            "data_dir": str(data_dir),
            "checkpoint": str(ckpt),
        }

        if eval_csv.exists():
            edf = pd.read_csv(eval_csv)
            if len(edf) > 0:
                best = edf.iloc[0]
                for col in ["rmse_raw", "r2_raw", "rmse_std", "r2_std", "function"]:
                    if col in edf.columns:
                        row[col] = best[col]

        sum_path = find_summary_combined(dim_dir)
        if sum_path is not None:
            row["summary_combined"] = str(sum_path)
            row.update(aggregate_opt_stats(sum_path))

        # optional Styblinski-Tang theoretical optimum
        if "function" in row and str(row["function"]).lower() == "styblinski_tang":
            row["true_global_optimum"] = -39.1661657037714 * dim

        rows.append(row)

    out_dir = args.root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows).sort_values("dim")
    out_csv = out_dir / "multidim_summary.csv"
    if out_csv.exists():
        prev_df = pd.read_csv(out_csv)
        if "dim" in prev_df.columns and "dim" in out_df.columns:
            processed_dims = set(int(v) for v in out_df["dim"].tolist())
            prev_keep = prev_df.loc[~prev_df["dim"].isin(processed_dims)].copy()
            out_df = pd.concat([prev_keep, out_df], ignore_index=True).sort_values("dim")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved combined summary: {out_csv}")

    if len(out_df) > 0:
        plot_dim_metrics(out_df, out_dir)
        print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
