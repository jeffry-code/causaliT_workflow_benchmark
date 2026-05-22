"""
Generate all thesis-quality plots for the P->C->Y paraboloid example.

Produces (in results/):
  plot_model_vs_scm.png      -- ProT model vs SCM ground truth (2D slice + 3D surface)
  plot_surrogate_quality.png -- MLP P->C surrogate: predicted vs true C values
  plot_training_curves.png   -- ProT validation loss/R2 and surrogate MSE
  plot_optimization.png      -- Multi-seed results: boxplot + convergence

Run from repo root:
  python optimization/p_to_c_to_y/generate_thesis_plots.py
"""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from omegaconf.dictconfig import DictConfig
from torch.serialization import add_safe_globals

from proT.training.forecasters.transformer_forecaster import TransformerForecaster
from scm_ds.datasets import ds_scm_quad

_HERE    = Path(__file__).parent
CKPT     = _HERE / "experiments" / "example" / "k_0" / "checkpoints" / "best_checkpoint.ckpt"
SURR     = _HERE / "models" / "p_to_c.pt"
DATA_DIR = _HERE / "data" / "example"
OUT_DIR  = _HERE / "results"
TARGET   = 0.7
FONT     = {"fontsize": 11}


# ── helpers ─────────────────────────────────────────────────────────────────
def load_var_maps():
    with open(DATA_DIR / "input_vars_map.json") as f:
        iv_map = json.load(f)
    with open(DATA_DIR / "target_vars_map.json") as f:
        tv_map = json.load(f)
    return iv_map, tv_map


def load_prot():
    add_safe_globals([DictConfig])
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg  = ckpt["hyper_parameters"]
    try:
        cfg["model"]["kwargs"]["device"] = "cpu"
    except Exception:
        pass
    m = TransformerForecaster(cfg)
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()


def load_surrogate():
    class _MLP(torch.nn.Module):
        def __init__(self, i, o, h):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(i, h), torch.nn.ReLU(),
                torch.nn.Linear(h, h), torch.nn.ReLU(),
                torch.nn.Linear(h, o))
        def forward(self, x): return self.net(x)

    p = torch.load(SURR, map_location="cpu")
    h = p["model_state"]["net.0.weight"].shape[0]
    m = _MLP(len(p["p_names"]), len(p["c_names"]), h)
    m.load_state_dict(p["model_state"])
    return m.eval(), p["p_names"], p["c_names"]


def build_enc(iv_map, vals: dict) -> torch.Tensor:
    ordered = [v for v, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    feats = [float(vals.get(v, np.nan)) for v in ordered]
    ids   = [float(iv_map[v]) for v in ordered]
    arr   = np.stack([feats, ids], axis=-1)[None]
    return torch.tensor(arr, dtype=torch.float32)


def build_dec(tv_map) -> torch.Tensor:
    ordered = [v for v, _ in sorted(tv_map.items(), key=lambda kv: kv[1])]
    arr = np.stack([[0.0, float(tv_map[v])] for v in ordered], axis=0)[None]
    return torch.tensor(arr, dtype=torch.float32)


def scm_y(p1, p2):
    base  = ds_scm_quad.scm
    scm_i = base.do({"P1": float(p1), "P2": float(p2)})
    ctx   = scm_i.forward(context={}, eps_draws={v: np.zeros(1) for v in scm_i.specs})
    return float(ctx["Y"].reshape(-1)[0])


def model_y(model, iv_map, tv_map, surr, p_names, c_names, p1, p2):
    """ProT prediction using MLP-predicted C values."""
    P = np.array([p1, p2], dtype=np.float32)
    with torch.no_grad():
        c_pred = surr(torch.tensor(P).unsqueeze(0)).numpy()[0]
    vals = {"P1": p1, "P2": p2}
    vals.update({c_names[i]: float(c_pred[i]) for i in range(len(c_names))})
    enc = build_enc(iv_map, vals)
    dec = build_dec(tv_map)
    with torch.no_grad():
        out, *_ = model.forward(data_input=enc, data_trg=dec)
    return float(out.reshape(-1)[0])


# This helper evaluates the full Workflow II composed model at a given (P1,P2)
# point by first predicting children C from the P vector, then forwarding the
# combined representation through ProT to obtain Y_hat.


# ── Plot 1: Model vs SCM ─────────────────────────────────────────────────────
def plot_model_vs_scm(model, iv_map, tv_map, surr, p_names, c_names):
    print("Generating plot_model_vs_scm.png ...")
    N = 60
    p_grid = np.linspace(-1, 1, N)
    P1, P2 = np.meshgrid(p_grid, p_grid)

    Z_scm   = np.zeros((N, N))
    Z_model = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Z_scm[i, j]   = scm_y(P1[i, j], P2[i, j])
            Z_model[i, j] = model_y(model, iv_map, tv_map, surr, p_names, c_names,
                                    P1[i, j], P2[i, j])

    # 2D slice: P2 = 0
    p1_line = np.linspace(-1, 1, 120)
    y_scm_line   = np.array([scm_y(p, 0.0) for p in p1_line])
    y_model_line = np.array([model_y(model, iv_map, tv_map, surr, p_names, c_names, p, 0.0)
                              for p in p1_line])

    # dataset scatter (subsample)
    ds   = np.load(DATA_DIR / "ds.npz")
    X    = ds["x"][:, :, 0]
    Y_ds = ds["y"][:, 0, 0]
    with open(DATA_DIR / "input_vars_map.json") as f:
        ivm = json.load(f)
    P1_d = X[:, ivm["P1"] - 1]
    P2_d = X[:, ivm["P2"] - 1]
    mask = np.abs(P2_d) < 0.05
    rng  = np.random.default_rng(0)
    idx  = rng.choice(np.where(mask)[0], size=min(150, mask.sum()), replace=False)

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # dataset scatter for 3D (subsample all points, not just P2≈0 slice)
    rng3d = np.random.default_rng(1)
    idx3d = rng3d.choice(len(P1_d), size=min(400, len(P1_d)), replace=False)

    # left: 3D surface
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax3d.plot_surface(P1, P2, Z_scm,   alpha=0.50, color="steelblue",  zorder=1)
    ax3d.plot_surface(P1, P2, Z_model, alpha=0.40, color="darkorange", zorder=2)
    ax3d.scatter(P1_d[idx3d], P2_d[idx3d], Y_ds[idx3d],
                 s=4, color="C2", alpha=0.4, zorder=3)
    ax3d.set_xlabel("P1", labelpad=6)
    ax3d.set_ylabel("P2", labelpad=6)
    ax3d.set_zlabel("Y",  labelpad=6)
    ax3d.set_title("Y surface: SCM vs ProT model", **FONT)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color="steelblue",  lw=3, label="SCM (no noise)"),
                       Line2D([0], [0], color="darkorange", lw=3, label="ProT + surrogate"),
                       Line2D([0], [0], color="C2", lw=0, marker="o", markersize=4,
                              alpha=0.6, label="Dataset (noisy)")]
    ax3d.legend(handles=legend_elements, loc="upper left", fontsize=8)

    # right: 2D slice
    ax2d = fig.add_subplot(gs[1])
    ax2d.plot(p1_line, y_scm_line,   color="steelblue",  lw=2.0, label="SCM (no noise)")
    ax2d.plot(p1_line, y_model_line, color="darkorange",  lw=2.0, linestyle="--",
              label="ProT + surrogate")
    ax2d.scatter(P1_d[idx], Y_ds[idx], s=14, color="C2", alpha=0.55, zorder=2,
                 label="Dataset (noisy)")
    ax2d.axhline(TARGET, color="k", linestyle=":", lw=1.4, label=f"Target Y = {TARGET}")
    ax2d.set_xlabel("P1 (P2 fixed at 0)", **FONT)
    ax2d.set_ylabel("Y", **FONT)
    ax2d.set_title("2D slice: SCM vs ProT model", **FONT)
    ax2d.legend(fontsize=9)

    plt.suptitle("Paraboloid SCM  —  surrogate model quality", fontsize=12, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "plot_model_vs_scm.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Plot 2: Surrogate quality ────────────────────────────────────────────────
def plot_surrogate_quality(surr, p_names, c_names):
    print("Generating plot_surrogate_quality.png ...")
    ds = np.load(DATA_DIR / "ds.npz")
    X  = ds["x"][:, :, 0]
    with open(DATA_DIR / "input_vars_map.json") as f:
        ivm = json.load(f)

    p_cols = np.stack([X[:, ivm[n] - 1] for n in p_names], axis=1).astype(np.float32)
    c_cols = np.stack([X[:, ivm[n] - 1] for n in c_names], axis=1)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(p_cols), size=min(2000, len(p_cols)), replace=False)
    P_sub = torch.tensor(p_cols[idx])
    with torch.no_grad():
        C_pred = surr(P_sub).numpy()
    C_true = c_cols[idx]

    fig, axes = plt.subplots(1, len(c_names), figsize=(5 * len(c_names), 4.5))
    if len(c_names) == 1:
        axes = [axes]
    for k, (ax, cname) in enumerate(zip(axes, c_names)):
        ct, cp = C_true[:, k], C_pred[:, k]
        lim = [min(ct.min(), cp.min()) - 0.05, max(ct.max(), cp.max()) + 0.05]
        ax.scatter(ct, cp, s=8, alpha=0.35, color="steelblue")
        ax.plot(lim, lim, color="k", lw=1.2, linestyle="--", label="Ideal (y = x)")
        ss_res = np.sum((ct - cp) ** 2)
        ss_tot = np.sum((ct - ct.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax.set_xlabel(f"True {cname}", **FONT)
        ax.set_ylabel(f"Predicted {cname}", **FONT)
        ax.set_title(f"{cname} surrogate  —  R² = {r2:.4f}", **FONT)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=9)
        ax.set_aspect("equal")

    plt.suptitle("P→C surrogate (MLP): predicted vs true child variables", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "plot_surrogate_quality.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Plot 3: Training curves ──────────────────────────────────────────────────
def plot_training_curves():
    print("Generating plot_training_curves.png ...")
    # ProT: pick version with most rows
    files = glob.glob(str(_HERE / "experiments/example/k_0/logs/csv/version_*/metrics.csv"))
    best_file = max(files, key=lambda f: len(pd.read_csv(f)))
    prot_df = pd.read_csv(best_file)
    val_df  = prot_df[prot_df["val_loss"].notna()].copy()

    # Surrogate MLP
    surr_df = pd.read_csv(SURR.with_suffix(".csv"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: ProT validation loss
    axes[0].plot(val_df["epoch"], val_df["val_loss"], color="steelblue", lw=1.8)
    axes[0].set_xlabel("Epoch", **FONT)
    axes[0].set_ylabel("Validation loss (MSE)", **FONT)
    axes[0].set_title("ProT surrogate training", **FONT)

    # Panel 2: ProT validation R²
    axes[1].plot(val_df["epoch"], val_df["val_r2"], color="darkorange", lw=1.8)
    axes[1].axhline(1.0, color="k", lw=0.8, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epoch", **FONT)
    axes[1].set_ylabel("Validation R²", **FONT)
    axes[1].set_title("ProT surrogate — R²", **FONT)
    final_r2 = float(val_df["val_r2"].iloc[-1])
    axes[1].annotate(f"Final R² = {final_r2:.4f}",
                     xy=(val_df["epoch"].iloc[-1], final_r2),
                     xytext=(-60, -20), textcoords="offset points",
                     fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))

    # Panel 3: MLP surrogate MSE
    axes[2].plot(surr_df["epoch"], surr_df["mse"], color="seagreen", lw=1.8)
    axes[2].set_xlabel("Epoch", **FONT)
    axes[2].set_ylabel("Training MSE", **FONT)
    axes[2].set_title("P→C surrogate (MLP) training", **FONT)
    axes[2].set_yscale("log")

    plt.suptitle("Training curves — paraboloid P→C→Y example", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "plot_training_curves.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Plot 4: Optimization results ─────────────────────────────────────────────
def plot_optimization():
    print("Generating plot_optimization.png ...")
    csv_path = OUT_DIR / "multiseed_summary.csv"
    if not csv_path.exists():
        print("  multiseed_summary.csv not found — run run_multiseed_experiment.py first.")
        return

    df = pd.read_csv(csv_path)
    tasks = [("target", f"Target tracking  (Y → {TARGET})", TARGET),
             ("minimize", "Minimization  (Y → 0)", 0.0)]
    n_seeds = df[df["optimizer"] == "CMA-ES"]["seed"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(1)

    for ax, (task, title, ref) in zip(axes, tasks):
        sub = df[df["task"] == task]
        cma_vals  = sub[sub["optimizer"] == "CMA-ES"]["y_true"].values
        adam_vals = sub[sub["optimizer"] == "Adam"]["y_true"].values

        bp = ax.boxplot(
            [cma_vals, adam_vals],
            tick_labels=["CMA-ES", "Adam (autograd)"],
            showmeans=True, patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.55),
            medianprops=dict(color="navy", linewidth=2.0),
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="navy", markersize=7),
            whiskerprops=dict(linewidth=1.3),
            capprops=dict(linewidth=1.3),
        )
        bp["boxes"][1].set_facecolor("darkorange")
        for i, vals in enumerate([cma_vals, adam_vals], start=1):
            jitter = rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(i + jitter, vals, s=22, alpha=0.7, color="k", zorder=3)
        ref_label = f"Target Y = {ref}" if task == "target" else f"True minimum Y = {ref}"
        ax.axhline(ref, color="red", linestyle="--", lw=1.5, label=ref_label)
        ax.set_ylabel("True Y at best found P", **FONT)
        ax.set_title(f"{title}\n({n_seeds} seeds)", **FONT)
        ax.legend(fontsize=9)
        for i, (vals, col) in enumerate(zip([cma_vals, adam_vals], ["steelblue", "darkorange"]), 1):
            ax.text(i, max(vals) + 0.003 * (max(vals) - min(vals) + 0.01),
                    f"μ={vals.mean():.3f}\nσ={vals.std():.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=col)

    plt.suptitle("Paraboloid P→C→Y — surrogate optimization results", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "plot_optimization.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading models...")
    iv_map, tv_map = load_var_maps()
    model          = load_prot()
    surr, p_names, c_names = load_surrogate()

    plot_model_vs_scm(model, iv_map, tv_map, surr, p_names, c_names)
    plot_surrogate_quality(surr, p_names, c_names)
    plot_training_curves()
    plot_optimization()
    print("\nAll plots saved to", OUT_DIR)


if __name__ == "__main__":
    main()
