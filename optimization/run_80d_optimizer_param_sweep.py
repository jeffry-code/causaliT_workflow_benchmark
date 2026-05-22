#!/usr/bin/env python
"""
run_80d_optimizer_param_sweep.py  —  hyperparameter sweep at 80D
=================================================================
Sweeps the primary hyperparameter of each optimizer at d=80 (400k samples):
  CMA-ES:  sigma0 in {0.2, 0.5, 0.8, 1.0, 1.2, 1.5}
  Adam:    learning rate in {0.01, 0.03, 0.05, 0.10, 0.12, 0.15}

Produces mean/std regret curves and boxplots showing sensitivity to each
hyperparameter. Key finding: CMA-ES is more sensitive but also more
responsive to tuning; best config is sigma0=1.0 (mean regret 0.305).

INPUT:  optimization/multidim/80dim/  (trained checkpoint + dataset)
OUTPUT: optimization/multidim/80dim/results/styblinski_tang/param_sweep/
  param_sweep_summary.csv, seed_level_results.csv
  param_sweep_mean_norm_regret.png, param_sweep_box_norm_regret.png

Run from repo root:
  python optimization/run_80d_optimizer_param_sweep.py

80D optimizer-parameter sweep for Styblinski-Tang surrogate optimization.

Sweeps:
- CMA-ES: sigma0 values
- Autograd: learning-rate values

Outputs:
- seed_level_results.csv
- param_sweep_summary.csv
- param_sweep_mean_norm_regret.png
- param_sweep_box_norm_regret.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYBLINSKI_TANG_OPT_PER_DIM = -39.1661657037714


# Execute a subprocess and echo output for each sweep configuration.
def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"\n[cwd={cwd}] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


def load_summary(path: Path) -> pd.DataFrame:
    # Load a seed-level summary CSV and verify the required true-objective column.
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    df = pd.read_csv(path)
    if "f_best_true_raw" not in df.columns:
        raise ValueError(f"'f_best_true_raw' missing in {path}")
    return df


# Plot mean normalized regret ± std for each hyperparameter value (sigma0 for CMA-ES, lr for Adam).
def plot_mean(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    cma = summary_df[summary_df["optimizer"] == "cma"].sort_values("param_value")
    if not cma.empty:
        axes[0].errorbar(
            cma["param_value"],
            cma["mean_norm_regret"],
            yerr=cma["std_norm_regret"],
            marker="o",
            capsize=4,
        )
    axes[0].set_title("CMA-ES sweep")
    axes[0].set_xlabel("sigma0")
    axes[0].set_ylabel("Normalized regret (lower is better)")
    axes[0].grid(alpha=0.25)

    ag = summary_df[summary_df["optimizer"] == "autograd"].sort_values("param_value")
    if not ag.empty:
        axes[1].errorbar(
            ag["param_value"],
            ag["mean_norm_regret"],
            yerr=ag["std_norm_regret"],
            marker="o",
            capsize=4,
            color="tab:orange",
        )
    axes[1].set_title("Autograd sweep")
    axes[1].set_xlabel("learning rate")
    axes[1].grid(alpha=0.25)

    fig.suptitle("80D optimizer parameter sweep (mean +/- std over seeds)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


# Plot mean normalized regret and standard deviation for each optimizer parameter.
# This visualizes how sensitive CMA-ES and Adam are to sigma0 and learning rate.


# Box plot showing the seed distribution of normalized regret for each hyperparameter value.
def plot_box(seed_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    cma = seed_df[seed_df["optimizer"] == "cma"].copy()
    cma_params = sorted(cma["param_value"].unique()) if not cma.empty else []
    cma_groups = [cma.loc[cma["param_value"] == p, "norm_regret"].to_numpy() for p in cma_params]
    if cma_groups:
        axes[0].boxplot(cma_groups, tick_labels=[str(p) for p in cma_params], showmeans=True)
    axes[0].set_title("CMA-ES")
    axes[0].set_xlabel("sigma0")
    axes[0].set_ylabel("Normalized regret")

    ag = seed_df[seed_df["optimizer"] == "autograd"].copy()
    ag_params = sorted(ag["param_value"].unique()) if not ag.empty else []
    ag_groups = [ag.loc[ag["param_value"] == p, "norm_regret"].to_numpy() for p in ag_params]
    if ag_groups:
        axes[1].boxplot(ag_groups, tick_labels=[str(p) for p in ag_params], showmeans=True)
    axes[1].set_title("Autograd")
    axes[1].set_xlabel("learning rate")

    fig.suptitle("80D parameter sweep: seed distribution")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 80D optimizer-parameter sweeps.")
    ap.add_argument("--dim-dir", type=Path, default=Path("optimization/multidim/80dim"))
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    # Mild extension beyond previous defaults to probe sensitivity at slightly larger steps.
    ap.add_argument("--cma-sigma0", type=float, nargs="+", default=[0.2, 0.5, 0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--autograd-lr", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.1, 0.12, 0.15])
    ap.add_argument("--skip-runs", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <dim-dir>/results/styblinski_tang/sweep_optimizer_params",
    )
    args = ap.parse_args()

    dim_dir = args.dim_dir.resolve()
    if not dim_dir.exists():
        raise FileNotFoundError(f"Missing dim dir: {dim_dir}")

    out_dir = (args.out_dir or (dim_dir / "results" / "styblinski_tang" / "sweep_optimizer_params")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    run_script = repo_root / "optimization" / "run_with_model_template.py"
    f_star = STYBLINSKI_TANG_OPT_PER_DIM * 80

    seed_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    # CMA sweep
    for sigma0 in sorted(set(float(v) for v in args.cma_sigma0)):
        run_out = out_dir / "cma" / f"sigma0_{sigma0:g}"
        run_out.mkdir(parents=True, exist_ok=True)
        if not args.skip_runs:
            cmd = [
                args.python,
                str(run_script),
                "--mode",
                "multiseed",
                "--optimizer",
                "cma",
                "--seeds",
                str(args.seeds),
                "--seed-start",
                str(args.seed_start),
                "--budget",
                str(args.budget),
                "--cma-sigma0",
                str(sigma0),
                "--out-dir",
                str(run_out),
            ]
            run_cmd(cmd, cwd=dim_dir)

        sdf = load_summary(run_out / "cma" / "summary.csv")
        vals = pd.to_numeric(sdf["f_best_true_raw"], errors="coerce").dropna().to_numpy(dtype=float)
        for _, r in sdf.iterrows():
            f_best = float(r["f_best_true_raw"])
            seed_rows.append(
                {
                    "optimizer": "cma",
                    "param_name": "sigma0",
                    "param_value": sigma0,
                    "seed": int(r.get("seed", -1)),
                    "budget": int(args.budget),
                    "f_best_true_raw": f_best,
                    "norm_regret": (f_best - f_star) / abs(f_star),
                }
            )
        summary_rows.append(
            {
                "optimizer": "cma",
                "param_name": "sigma0",
                "param_value": sigma0,
                "budget": int(args.budget),
                "seeds": int(len(vals)),
                "mean_true": float(np.mean(vals)),
                "median_true": float(np.median(vals)),
                "min_true": float(np.min(vals)),
                "std_true": float(np.std(vals)),
                "mean_norm_regret": float((np.mean(vals) - f_star) / abs(f_star)),
                "median_norm_regret": float((np.median(vals) - f_star) / abs(f_star)),
                "min_norm_regret": float((np.min(vals) - f_star) / abs(f_star)),
                "std_norm_regret": float(np.std((vals - f_star) / abs(f_star))),
            }
        )

    # Autograd sweep
    for lr in sorted(set(float(v) for v in args.autograd_lr)):
        run_out = out_dir / "autograd" / f"lr_{lr:g}"
        run_out.mkdir(parents=True, exist_ok=True)
        if not args.skip_runs:
            cmd = [
                args.python,
                str(run_script),
                "--mode",
                "multiseed",
                "--optimizer",
                "autograd",
                "--seeds",
                str(args.seeds),
                "--seed-start",
                str(args.seed_start),
                "--budget",
                str(args.budget),
                "--lr",
                str(lr),
                "--out-dir",
                str(run_out),
            ]
            run_cmd(cmd, cwd=dim_dir)

        sdf = load_summary(run_out / "autograd" / "summary.csv")
        vals = pd.to_numeric(sdf["f_best_true_raw"], errors="coerce").dropna().to_numpy(dtype=float)
        for _, r in sdf.iterrows():
            f_best = float(r["f_best_true_raw"])
            seed_rows.append(
                {
                    "optimizer": "autograd",
                    "param_name": "lr",
                    "param_value": lr,
                    "seed": int(r.get("seed", -1)),
                    "budget": int(args.budget),
                    "f_best_true_raw": f_best,
                    "norm_regret": (f_best - f_star) / abs(f_star),
                }
            )
        summary_rows.append(
            {
                "optimizer": "autograd",
                "param_name": "lr",
                "param_value": lr,
                "budget": int(args.budget),
                "seeds": int(len(vals)),
                "mean_true": float(np.mean(vals)),
                "median_true": float(np.median(vals)),
                "min_true": float(np.min(vals)),
                "std_true": float(np.std(vals)),
                "mean_norm_regret": float((np.mean(vals) - f_star) / abs(f_star)),
                "median_norm_regret": float((np.median(vals) - f_star) / abs(f_star)),
                "min_norm_regret": float((np.min(vals) - f_star) / abs(f_star)),
                "std_norm_regret": float(np.std((vals - f_star) / abs(f_star))),
            }
        )

    seed_df = pd.DataFrame(seed_rows).sort_values(["optimizer", "param_value", "seed"])
    summary_df = pd.DataFrame(summary_rows).sort_values(["optimizer", "param_value"])
    seed_df.to_csv(out_dir / "seed_level_results.csv", index=False)
    summary_df.to_csv(out_dir / "param_sweep_summary.csv", index=False)

    if not summary_df.empty:
        plot_mean(summary_df, out_dir / "param_sweep_mean_norm_regret.png")
    if not seed_df.empty:
        plot_box(seed_df, out_dir / "param_sweep_box_norm_regret.png")

    print(f"Saved parameter-sweep outputs to: {out_dir}")
    print(f" - {out_dir / 'param_sweep_summary.csv'}")
    print(f" - {out_dir / 'seed_level_results.csv'}")
    print(f" - {out_dir / 'param_sweep_mean_norm_regret.png'}")
    print(f" - {out_dir / 'param_sweep_box_norm_regret.png'}")


if __name__ == "__main__":
    main()
