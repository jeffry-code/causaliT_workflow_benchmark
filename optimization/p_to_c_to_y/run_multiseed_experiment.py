"""
Multi-seed optimization experiment for the paraboloid P->C->Y example.

Runs CMA-ES and autograd Adam for N random seeds and produces:
  - results/multiseed_summary.csv       per-seed results
  - results/multiseed_boxplot.png       final objective comparison
  - results/multiseed_convergence.png   best-so-far curves per optimizer

Usage (from repo root):
  python optimization/p_to_c_to_y/run_multiseed_experiment.py
"""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from omegaconf.dictconfig import DictConfig
from torch.serialization import add_safe_globals

from proT.training.forecasters.transformer_forecaster import TransformerForecaster
from optimizers import cma_es

# ── paths (resolved relative to this file) ─────────────────────────────────
_HERE = Path(__file__).parent
CKPT_PATH  = _HERE / "experiments" / "example" / "k_0" / "checkpoints" / "best_checkpoint.ckpt"
SURR_PATH  = _HERE / "models" / "p_to_c.pt"
DATA_DIR   = _HERE / "data" / "example"
OUT_DIR    = _HERE / "results"

TARGET = 0.7
N_SEEDS = 20
ITERS_CMA  = 500   # CMA-ES generations
ITERS_ADAM = 500   # autograd Adam steps
LR_ADAM    = 0.05
BOUNDS     = np.array([-1.0, -1.0]), np.array([1.0, 1.0])


# ── surrogate: MLP  P -> C ─────────────────────────────────────────────────
# Two-layer MLP used as the P->C surrogate (same architecture as train_p_to_c_surrogate.py).
class _MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)


# Loads the trained MLP checkpoint and exposes numpy and torch prediction paths.
class SurrogateWrapper:
    def __init__(self, path: Path):
        payload = torch.load(path, map_location="cpu")
        self.p_names = payload["p_names"]
        self.c_names = payload["c_names"]
        hidden = payload["model_state"]["net.0.weight"].shape[0]
        self.model = _MLP(len(self.p_names), len(self.c_names), hidden)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    # Predict C values from P as a numpy dict — used by the gradient-free CMA-ES path.
    def predict_np(self, P_vec: np.ndarray) -> Dict[str, float]:
        with torch.no_grad():
            t = torch.tensor(P_vec, dtype=torch.float32).unsqueeze(0)
            out = self.model(t).numpy()[0]
        return {n: float(v) for n, v in zip(self.c_names, out)}

    # Predict C values from P as torch tensors — used by autograd Adam so gradients flow back through the MLP.
    def predict_torch(self, P_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model(P_t.unsqueeze(0)).squeeze(0)
        return {n: out[i] for i, n in enumerate(self.c_names)}


# ── ProT model ──────────────────────────────────────────────────────────────
# Load a ProT TransformerForecaster from a PyTorch Lightning checkpoint file.
def load_prot(ckpt_path: Path) -> TransformerForecaster:
    add_safe_globals([DictConfig])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["hyper_parameters"]
    try:
        cfg["model"]["kwargs"]["device"] = "cpu"
    except Exception:
        pass
    model = TransformerForecaster(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


# Load the variable ordering maps that define which positions in the input tensor correspond to P and C.
def load_var_maps(data_dir: Path):
    with open(data_dir / "input_vars_map.json") as f:
        iv_map = json.load(f)
    with open(data_dir / "target_vars_map.json") as f:
        tv_map = json.load(f)
    return iv_map, tv_map


# Build the two-channel ProT encoder tensor from a numpy P vector (gradient-free path).
# Channel 0: variable values (P direct, C from MLP surrogate). Channel 1: variable IDs.
def build_enc_np(iv_map, surr: SurrogateWrapper, P_vec: np.ndarray) -> torch.Tensor:
    ctrl  = {n: float(P_vec[i]) for i, n in enumerate(surr.p_names)}
    preds = surr.predict_np(P_vec)
    ordered = [v for v, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    vals   = [ctrl.get(v, preds.get(v, float("nan"))) for v in ordered]
    ids    = [float(iv_map[v]) for v in ordered]
    arr    = np.stack([vals, ids], axis=-1)[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


# Differentiable version of build_enc_np — uses torch tensors so gradients flow back to P through the MLP.
def build_enc_torch(iv_map, surr: SurrogateWrapper, P_t: torch.Tensor) -> torch.Tensor:
    ctrl  = {n: P_t[i] for i, n in enumerate(surr.p_names)}
    preds = surr.predict_torch(P_t)
    ordered = [v for v, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    vals, ids = [], []
    for v in ordered:
        val = ctrl[v] if v in ctrl else preds.get(v, torch.tensor(float("nan")))
        if not torch.is_tensor(val):
            val = torch.tensor(float(val))
        vals.append(val)
        ids.append(torch.tensor(float(iv_map[v])))
    return torch.stack([torch.stack(vals), torch.stack(ids)], dim=-1).unsqueeze(0)


# Build the ProT decoder input: zero placeholder values paired with target variable IDs.
def build_dec(tv_map) -> torch.Tensor:
    ordered = [v for v, _ in sorted(tv_map.items(), key=lambda kv: kv[1])]
    arr = np.stack([[0.0, float(tv_map[v])] for v in ordered], axis=0)[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


# ── true function (zero-noise SCM) ──────────────────────────────────────────
# Analytical ground truth used to evaluate the true Y at the optimizer's best-found P.
def true_y(P_vec: np.ndarray) -> float:
    """Y = P1^2 + P2^2 (paraboloid ground truth)."""
    return float(P_vec[0]**2 + P_vec[1]**2)


# ── per-seed runners ────────────────────────────────────────────────────────
# Run one CMA-ES trial from a random start for the given task (target tracking or minimization).
def run_cma(seed: int, model, iv_map, tv_map, surr, lb, ub, task: str = "target") -> dict:
    rng = np.random.default_rng(seed)
    x0  = rng.uniform(-1.0, 1.0, size=2)
    dec = build_dec(tv_map)

    def objective(P: np.ndarray) -> float:
        enc = build_enc_np(iv_map, surr, P)
        with torch.no_grad():
            out, *_ = model.forward(data_input=enc, data_trg=dec)
        y_hat = float(out.reshape(-1)[0])
        return (y_hat - TARGET) ** 2 if task == "target" else y_hat

    history = []
    best_f = float("inf")

    # Thin wrapper so CMA-ES can track best-so-far across its internal population evaluations.
    class WrappedObj:
        def __call__(self, P):
            nonlocal best_f
            f = objective(np.asarray(P))
            if f < best_f:
                best_f = f
            history.append(best_f)
            return f

    res = cma_es(WrappedObj(), x0=x0, sigma0=0.5, bounds=(lb, ub), max_iters=ITERS_CMA)
    x_best = res.x_best
    y_true  = true_y(x_best)
    return {
        "task": task, "seed": seed, "optimizer": "CMA-ES",
        "f_best": float(res.f_best),
        "y_true": y_true,
        "P1_best": float(x_best[0]), "P2_best": float(x_best[1]),
        "history": history,
    }


# Run one autograd Adam trial. P is a torch.nn.Parameter; gradients flow through ProT and the MLP surrogate.
def run_adam(seed: int, model, iv_map, tv_map, surr, lb, ub, task: str = "target") -> dict:
    rng  = np.random.default_rng(seed)
    x0   = rng.uniform(-1.0, 1.0, size=2)
    lb_t = torch.tensor(lb, dtype=torch.float32)
    ub_t = torch.tensor(ub, dtype=torch.float32)
    dec  = build_dec(tv_map)

    P = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32))
    opt = torch.optim.Adam([P], lr=LR_ADAM)
    best_f, best_x = float("inf"), x0.copy()
    history = []

    for _ in range(ITERS_ADAM):
        with torch.no_grad():
            P.data = torch.clamp(P.data, lb_t, ub_t)
        candidate = P.detach().numpy().copy()

        opt.zero_grad(set_to_none=True)
        enc  = build_enc_torch(iv_map, surr, P)
        out, *_ = model.forward(data_input=enc, data_trg=dec)
        y_hat = out.reshape(-1)[0]
        loss  = (y_hat - TARGET) ** 2 if task == "target" else y_hat
        f_val = float(loss.detach())

        # Backpropagate through ProT and the MLP surrogate to compute gradients w.r.t. P.
        loss.backward()
        opt.step()
        with torch.no_grad():
            P.data = torch.clamp(P.data, lb_t, ub_t)

        if f_val < best_f:
            best_f = f_val
            best_x = candidate
        history.append(best_f)

    y_true = true_y(best_x)
    return {
        "task": task, "seed": seed, "optimizer": "Adam",
        "f_best": best_f,
        "y_true": y_true,
        "P1_best": float(best_x[0]), "P2_best": float(best_x[1]),
        "history": history,
    }


# ── plotting ─────────────────────────────────────────────────────────────────
# Box plot comparing the true Y at best-found P across seeds for CMA-ES vs Adam.
def plot_boxplot(rows: list[dict], out: Path) -> None:
    cma_vals  = [r["y_true"] for r in rows if r["optimizer"] == "CMA-ES"]
    adam_vals = [r["y_true"] for r in rows if r["optimizer"] == "Adam"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([cma_vals, adam_vals], labels=["CMA-ES", "Adam"],
                    showmeans=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6),
                    medianprops=dict(color="navy", linewidth=2),
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="navy", markersize=7))
    bp["boxes"][1].set_facecolor("darkorange")
    for i, vals in enumerate([cma_vals, adam_vals], start=1):
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.05, 0.05, len(vals)),
                   vals, s=22, alpha=0.7, color="k", zorder=3)
    ax.axhline(TARGET, color="red", linestyle="--", linewidth=1.2, label=f"Target Y = {TARGET}")
    ax.set_ylabel("True Y at best found P")
    ax.set_title(f"Paraboloid P→C→Y: optimization results\n({N_SEEDS} seeds per optimizer)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"Saved: {out}")


# Convergence plot: mean and min-max band of best-so-far objective over iterations across seeds.
def plot_convergence(rows: list[dict], out: Path) -> None:
    cma_rows  = [r for r in rows if r["optimizer"] == "CMA-ES"]
    adam_rows = [r for r in rows if r["optimizer"] == "Adam"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, group, label, color, n_iters in [
        (axes[0], cma_rows,  "CMA-ES", "steelblue", ITERS_CMA),
        (axes[1], adam_rows, "Adam",   "darkorange", ITERS_ADAM),
    ]:
        histories = [r["history"] for r in group]
        max_len = max(len(h) for h in histories)
        # pad short histories with last value
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
        mean_h = padded.mean(axis=0)
        min_h  = padded.min(axis=0)
        max_h  = padded.max(axis=0)
        xs = np.arange(1, max_len + 1)
        ax.plot(xs, mean_h, color=color, linewidth=1.8, label="Mean")
        ax.fill_between(xs, min_h, max_h, color=color, alpha=0.2, label="Min–Max range")
        ax.set_xlabel("Iteration / Generation")
        ax.set_ylabel("Best objective (Y_hat − target)²")
        ax.set_title(f"{label} convergence ({N_SEEDS} seeds)")
        ax.legend(fontsize=9)
    plt.suptitle("Paraboloid P→C→Y: surrogate optimization convergence", fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────
# Entry point: runs N_SEEDS x 2 optimizers x 2 tasks and saves summary CSV and plots.
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    iv_map, tv_map = load_var_maps(DATA_DIR)
    surr  = SurrogateWrapper(SURR_PATH)
    model = load_prot(CKPT_PATH)
    lb, ub = BOUNDS

    print(f"Running {N_SEEDS} seeds x 2 optimizers x 2 tasks on paraboloid P->C->Y (target Y={TARGET})")
    rows = []
    for task in ["target", "minimize"]:
        print(f"\n  Task: {task}")
        for seed in range(N_SEEDS):
            r_cma  = run_cma(seed, model, iv_map, tv_map, surr, lb, ub, task=task)
            r_adam = run_adam(seed, model, iv_map, tv_map, surr, lb, ub, task=task)
            print(f"  seed {seed:2d}  CMA-ES f={r_cma['f_best']:.5f}  y_true={r_cma['y_true']:.4f} | "
                  f"Adam f={r_adam['f_best']:.5f}  y_true={r_adam['y_true']:.4f}")
            rows.append(r_cma)
            rows.append(r_adam)

    # Strip per-iteration history before saving — too large for tabular CSV format.
    csv_rows = [{k: v for k, v in r.items() if k != "history"} for r in rows]
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUT_DIR / "multiseed_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'multiseed_summary.csv'}")

    # summary stats
    for opt in ["CMA-ES", "Adam"]:
        sub = df[df["optimizer"] == opt]
        print(f"  {opt}: mean y_true={sub['y_true'].mean():.4f}, "
              f"std={sub['y_true'].std():.4f}, "
              f"mean f_best={sub['f_best'].mean():.5f}")

    # plots are generated by generate_thesis_plots.py


if __name__ == "__main__":
    main()
