#!/usr/bin/env python
"""
run_80d_budget_seed_sweep.py  —  budget and restart-count sweep at 80D
=======================================================================
Systematic study of the trade-off between per-run computational budget
(CMA-ES generations / Adam steps) and number of independent restarts (seeds)
at d=80 with 400k training samples.

Uses bootstrap subsampling to estimate "expected best-of-n-seeds" performance
for each (budget, n_seeds) combination without rerunning experiments.

INPUT:  optimization/multidim/80dim/  (trained checkpoint + dataset)
OUTPUT: optimization/multidim/80dim/results/styblinski_tang/sweep_budget_seed/
  budget_summary.csv, seed_level_results.csv
  budget_seed_heatmap_true.png, budget_curves_expected_best_norm_regret.png

Run from repo root (uses defaults):
  python optimization/run_80d_budget_seed_sweep.py

Budget/seed sweep for 80D Styblinski-Tang surrogate optimization.

What it does:
1) Runs run_with_model_template.py for multiple budgets.
2) For each budget, stores per-seed optimization outcomes.
3) Estimates "best-of-n-seeds" performance via bootstrap subsampling.
4) Writes compact CSVs + two thesis-focused plots.

Example:
  python optimization/run_80d_budget_seed_sweep.py ^
    --dim-dir multidim/80dim ^
    --budgets 200 500 1000 2000 ^
    --seed-counts 5 10 20 30 ^
    --max-seeds 30 ^
    --optimizer both
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


# Execute a subprocess for each budget configuration and echo its output.
def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"\n[cwd={cwd}] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


# Load the per-seed summary CSV produced by run_with_model_template.py for a given optimizer.
def load_summary(out_dir: Path, optimizer: str) -> pd.DataFrame:
    p = out_dir / optimizer / "summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary CSV: {p}")
    df = pd.read_csv(p)
    if "f_best_true_raw" not in df.columns:
        raise ValueError(f"'f_best_true_raw' missing in {p}")
    return df


# Simulate the expected best objective from a random subset of n seeds.
def bootstrap_best_of_n(vals: np.ndarray, n: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    m = len(vals)
    if n > m:
        raise ValueError(f"n={n} cannot exceed number of seeds m={m}")
    out = np.empty(resamples, dtype=float)
    idx = np.arange(m, dtype=int)
    for i in range(resamples):
        chosen = rng.choice(idx, size=n, replace=False)
        out[i] = float(np.min(vals[chosen]))
    return out


# Bootstrap the best objective value from a random subset of n seeds.
# This simulates the expected performance of a best-of-n restarts strategy
# without having to rerun the full optimization experiment for every n.


def plot_heatmap(seed_sweep: pd.DataFrame, out_path: Path) -> None:
    # Heatmap of expected best true objective for combinations of budget and seed count.
    opts = [o for o in ["cma", "autograd"] if o in seed_sweep["optimizer"].unique()]
    if not opts:
        return
    fig, axes = plt.subplots(1, len(opts), figsize=(6.5 * len(opts), 4.8), squeeze=False)
    axes = axes[0]

    for ax, opt in zip(axes, opts):
        d = seed_sweep[seed_sweep["optimizer"] == opt].copy()
        piv = d.pivot(index="seed_count", columns="budget", values="expected_best_true_mean")
        piv = piv.sort_index().reindex(sorted(piv.columns), axis=1)

        im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([str(int(v)) for v in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([str(int(v)) for v in piv.index])
        ax.set_xlabel("Budget")
        ax.set_ylabel("Number of seeds (n)")
        ax.set_title(f"{opt.upper()}: expected best true f")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Objective (lower is better)")

    fig.suptitle("80D sweep: budget x seeds")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


# Line plot of expected best normalized regret vs budget for each seed count n.
def plot_budget_curves(seed_sweep: pd.DataFrame, out_path: Path) -> None:
    opts = [o for o in ["cma", "autograd"] if o in seed_sweep["optimizer"].unique()]
    if not opts:
        return
    fig, axes = plt.subplots(1, len(opts), figsize=(6.5 * len(opts), 4.8), squeeze=False, sharey=True)
    axes = axes[0]

    for ax, opt in zip(axes, opts):
        d = seed_sweep[seed_sweep["optimizer"] == opt].copy()
        for n in sorted(d["seed_count"].unique()):
            dn = d[d["seed_count"] == n].sort_values("budget")
            ax.plot(
                dn["budget"],
                dn["expected_best_norm_regret_mean"],
                marker="o",
                label=f"n={int(n)}",
            )
            ax.fill_between(
                dn["budget"],
                dn["expected_best_norm_regret_q10"],
                dn["expected_best_norm_regret_q90"],
                alpha=0.15,
            )
        ax.set_xlabel("Budget")
        ax.set_title(f"{opt.upper()}: expected best normalized regret")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Normalized regret (lower is better)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Budget + seed sweep for 80D Styblinski-Tang.")
    ap.add_argument("--dim-dir", type=Path, default=Path("optimization/multidim/80dim"))
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--optimizer", choices=["cma", "autograd", "both"], default="both")
    ap.add_argument("--budgets", type=int, nargs="+", default=[200, 500, 1000, 2000])
    ap.add_argument("--seed-counts", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--max-seeds", type=int, default=None, help="If omitted, uses max(seed-counts).")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--resamples", type=int, default=400)
    ap.add_argument("--rng-seed", type=int, default=123)
    ap.add_argument("--skip-runs", action="store_true", help="Only re-aggregate existing run outputs.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: <dim-dir>/results/styblinski_tang/sweep_budget_seed")
    args = ap.parse_args()

    dim_dir = args.dim_dir.resolve()
    if not dim_dir.exists():
        raise FileNotFoundError(f"Missing dim dir: {dim_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    run_script = repo_root / "optimization" / "run_with_model_template.py"
    out_root = (args.out_dir or (dim_dir / "results" / "styblinski_tang" / "sweep_budget_seed")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    seed_counts = sorted(set(int(v) for v in args.seed_counts))
    max_seeds = int(args.max_seeds) if args.max_seeds is not None else max(seed_counts)
    if any(v <= 0 for v in seed_counts):
        raise ValueError("All seed counts must be > 0")
    if max_seeds < max(seed_counts):
        raise ValueError("--max-seeds must be >= max(--seed-counts)")

    dim = 80
    f_star = STYBLINSKI_TANG_OPT_PER_DIM * dim
    rng = np.random.default_rng(args.rng_seed)

    raw_rows: List[Dict[str, object]] = []
    budget_rows: List[Dict[str, object]] = []
    seed_rows: List[Dict[str, object]] = []

    for budget in sorted(set(int(b) for b in args.budgets)):
        run_out = out_root / f"budget_{budget}"
        run_out.mkdir(parents=True, exist_ok=True)

        if not args.skip_runs:
            cmd = [
                args.python,
                str(run_script),
                "--mode",
                "multiseed",
                "--optimizer",
                args.optimizer,
                "--seeds",
                str(max_seeds),
                "--seed-start",
                str(args.seed_start),
                "--budget",
                str(budget),
                "--lr",
                str(args.lr),
                "--out-dir",
                str(run_out),
            ]
            run_cmd(cmd, cwd=dim_dir)

        optimizers = ["cma", "autograd"] if args.optimizer == "both" else [args.optimizer]
        for opt in optimizers:
            sdf = load_summary(run_out, opt)
            vals = pd.to_numeric(sdf["f_best_true_raw"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            if len(vals) < max_seeds:
                print(f"[warn] budget={budget}, opt={opt}: only {len(vals)} seeds available, expected {max_seeds}.")

            for _, r in sdf.iterrows():
                raw_rows.append(
                    {
                        "budget": budget,
                        "optimizer": opt,
                        "seed": int(r.get("seed", -1)),
                        "f_best_true_raw": float(r["f_best_true_raw"]),
                        "f_star": f_star,
                        "norm_regret": (float(r["f_best_true_raw"]) - f_star) / abs(f_star),
                    }
                )

            budget_rows.append(
                {
                    "budget": budget,
                    "optimizer": opt,
                    "seeds_used": int(len(vals)),
                    "mean_true": float(np.mean(vals)),
                    "median_true": float(np.median(vals)),
                    "min_true": float(np.min(vals)),
                    "std_true": float(np.std(vals)),
                    "mean_norm_regret": float((np.mean(vals) - f_star) / abs(f_star)),
                    "median_norm_regret": float((np.median(vals) - f_star) / abs(f_star)),
                    "min_norm_regret": float((np.min(vals) - f_star) / abs(f_star)),
                }
            )

            for n in seed_counts:
                if n > len(vals):
                    continue
                best_samples = bootstrap_best_of_n(vals, n=n, resamples=args.resamples, rng=rng)
                reg_samples = (best_samples - f_star) / abs(f_star)
                seed_rows.append(
                    {
                        "budget": budget,
                        "optimizer": opt,
                        "seed_count": n,
                        "resamples": args.resamples,
                        "expected_best_true_mean": float(np.mean(best_samples)),
                        "expected_best_true_std": float(np.std(best_samples)),
                        "expected_best_true_q10": float(np.quantile(best_samples, 0.10)),
                        "expected_best_true_q90": float(np.quantile(best_samples, 0.90)),
                        "expected_best_norm_regret_mean": float(np.mean(reg_samples)),
                        "expected_best_norm_regret_std": float(np.std(reg_samples)),
                        "expected_best_norm_regret_q10": float(np.quantile(reg_samples, 0.10)),
                        "expected_best_norm_regret_q90": float(np.quantile(reg_samples, 0.90)),
                    }
                )

    raw_df = pd.DataFrame(raw_rows).sort_values(["optimizer", "budget", "seed"])
    budget_df = pd.DataFrame(budget_rows).sort_values(["optimizer", "budget"])
    seed_df = pd.DataFrame(seed_rows).sort_values(["optimizer", "seed_count", "budget"])

    raw_df.to_csv(out_root / "seed_level_results.csv", index=False)
    budget_df.to_csv(out_root / "budget_summary.csv", index=False)
    seed_df.to_csv(out_root / "seed_sweep_summary.csv", index=False)

    if not seed_df.empty:
        plot_heatmap(seed_df, out_root / "budget_seed_heatmap_true.png")
        plot_budget_curves(seed_df, out_root / "budget_curves_expected_best_norm_regret.png")

    print(f"Saved sweep outputs to: {out_root}")
    print(f" - {out_root / 'seed_level_results.csv'}")
    print(f" - {out_root / 'budget_summary.csv'}")
    print(f" - {out_root / 'seed_sweep_summary.csv'}")
    if not seed_df.empty:
        print(f" - {out_root / 'budget_seed_heatmap_true.png'}")
        print(f" - {out_root / 'budget_curves_expected_best_norm_regret.png'}")


if __name__ == "__main__":
    main()

