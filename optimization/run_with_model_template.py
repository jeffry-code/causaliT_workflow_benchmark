"""
run_with_model_template.py  —  core single-dimension optimization driver (Workflow I)
======================================================================================
Loads a trained proT Transformer checkpoint, builds a surrogate predictor, and runs
CMA-ES and/or autograd-Adam to minimize the surrogate objective over normalized inputs.

This is the lowest-level optimization script; the pipeline / sweep scripts call it
as a subprocess. For typical use, call via run_multidim_pipeline.py.

Modes (--mode):
  single    — one run with surface/contour plots (good for quick inspection)
  autograd  — one run using torch autograd Adam only
  multiseed — N independent seeded runs; saves per-seed CSVs + trajectory CSVs

Key CLI arguments (all have defaults, override as needed):
  --mode         single | autograd | multiseed
  --optimizer    cma | autograd | both
  --budget       CMA-ES generations OR Adam steps per run
  --seeds        number of independent restarts (multiseed mode)
  --cma-sigma0   CMA-ES initial step size (0.5 default; 1.0 recommended at 80D)
  --lr           Adam learning rate (default 0.05)
  --out-dir      where to write result CSVs, plots, and trajectory files

Run from repo root:
  python optimization/run_with_model_template.py --mode multiseed --seeds 10 --budget 1000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from objectives import Objective
from optimizers import adam, cma_es
from predictors import ModelPredictor
from proT.training.forecasters.transformer_forecaster import TransformerForecaster


DATA_DIR = "data/example"
CKPT_PATH = "experiments/example/k_0/checkpoints/best_checkpoint.ckpt"
LOG_DIR = "optimization_logs"
RESULTS_DIR = "results/styblinski_tang"

with open(os.path.join(DATA_DIR, "input_vars_map.json"), "r") as f:
    iv_map = json.load(f)
with open(os.path.join(DATA_DIR, "target_vars_map.json"), "r") as f:
    tv_map = json.load(f)
with open(os.path.join(DATA_DIR, "meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

CONTROLLABLE = [k for k, _ in sorted(iv_map.items(), key=lambda kv: kv[1]) if k.startswith("P")]
DIM = len(CONTROLLABLE)
LB = np.array([-1.0] * DIM, dtype=float)
UB = np.array([1.0] * DIM, dtype=float)
X_LOWER = float(meta.get("x_lower", -5.0))
X_UPPER = float(meta.get("x_upper", 5.0))
Y_MEAN = float(meta.get("y_mean", 0.0))
Y_STD = float(meta.get("y_std", 1.0))
if Y_STD <= 0:
    Y_STD = 1.0

# `CONTROLLABLE` defines the active optimization inputs P1..Pd.
# These are the variables that the surrogate optimization loop will search over.
# The raw model is built from `input_vars_map.json`, which includes both
# controllable and non-controllable inputs, but only P variables are changed.


# Load checkpoint on CPU first to avoid device-mismatch issues.
# This forces the model config to CPU if possible, then the caller can move
# it to CUDA later if available.
def _load_checkpoint_cpu(ckpt_path: str) -> TransformerForecaster:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["hyper_parameters"]
    try:
        if "model" in config and "kwargs" in config["model"]:
            config["model"]["kwargs"]["device"] = "cpu"
    except Exception:
        pass
    model = TransformerForecaster(config)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


# Load the trained ProT model on CPU first, then move it to the selected device.
# This avoids GPU-specific checkpoint loading issues when the checkpoint was
# saved on a different machine or with a different device configuration.


_LIT_MODEL = _load_checkpoint_cpu(CKPT_PATH).eval()
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LIT_MODEL = _LIT_MODEL.to(_DEVICE)


# Map normalized search space z in [-1, 1]^d back to the original x range
# stored in meta.json. This is used only for true-function evaluation.
def z_to_x(z: np.ndarray) -> np.ndarray:
    l = np.array([X_LOWER] * len(z), dtype=float)
    u = np.array([X_UPPER] * len(z), dtype=float)
    return l + ((z + 1.0) / 2.0) * (u - l)


def styblinski_tang_true_x(x: np.ndarray) -> float:
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))


def y_denorm(y_std_value: float) -> float:
    return float(y_std_value * Y_STD + Y_MEAN)


def load_model(_: str) -> nn.Module:
    return _LIT_MODEL


def build_input(P: np.ndarray) -> torch.Tensor:
    ordered = [v for v, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    ctrl_vals = {var: P[i] for i, var in enumerate(CONTROLLABLE)}
    values = [float(ctrl_vals.get(v, float("nan"))) for v in ordered]
    var_ids = [float(iv_map[v]) for v in ordered]
    arr = np.stack([values, var_ids], axis=-1)[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


# Build the input tensor expected by ProT from a point P in the controllable
# search space. Missing values are filled with NaN and the second channel
# holds the variable ID encoding required by the model.


def build_target() -> torch.Tensor:
    ordered = [v for v, _ in sorted(tv_map.items(), key=lambda kv: kv[1])]
    values = [0.0 for _ in ordered]
    var_ids = [float(tv_map[v]) for v in ordered]
    arr = np.stack([values, var_ids], axis=-1)[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


# Convert a batch of P vectors into the model encoder format and execute
# a forward pass. This is the batch helper used by gradient-free optimizers.
def predict_model_raw_batch(P_batch: np.ndarray) -> np.ndarray:
    ordered = [v for v, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    idx_of = {v: i for i, v in enumerate(ordered)}
    n = int(P_batch.shape[0])
    data = np.empty((n, len(ordered), 2), dtype=np.float32)
    data[:, :, 0] = np.nan
    data[:, :, 1] = np.asarray([float(iv_map[v]) for v in ordered], dtype=np.float32)[None, :]
    for j, var in enumerate(CONTROLLABLE):
        data[:, idx_of[var], 0] = P_batch[:, j].astype(np.float32)
    x = torch.tensor(data, dtype=torch.float32, device=_DEVICE)
    y = build_target().to(_DEVICE).repeat(n, 1, 1)
    with torch.no_grad():
        forecast_out, *_ = _LIT_MODEL.forward(data_input=x, data_trg=y)
    y_std = forecast_out.reshape(n).detach().cpu().numpy()
    return y_std * Y_STD + Y_MEAN


# Batch prediction helper used by the optimizer.
# It evaluates the loaded ProT surrogate on a batch of P vectors and returns
# denormalized predicted target values.


def run_model(x: torch.Tensor) -> torch.Tensor:
    y = build_target().to(x.device)
    model = load_model(CKPT_PATH).eval().to(x.device)
    with torch.no_grad():
        forecast_out, *_ = model.forward(data_input=x, data_trg=y)
    return forecast_out.reshape(1, 1)


def build_input_torch(P: torch.Tensor) -> torch.Tensor:
    ordered = [v for v, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    ctrl_vals = {var: P[i] for i, var in enumerate(CONTROLLABLE)}
    values: list[torch.Tensor] = []
    var_ids: list[torch.Tensor] = []
    for var in ordered:
        v = ctrl_vals.get(var, torch.tensor(float("nan"), dtype=torch.float32, device=_DEVICE))
        if not torch.is_tensor(v):
            v = torch.tensor(float(v), dtype=torch.float32, device=_DEVICE)
        values.append(v)
        var_ids.append(torch.tensor(float(iv_map[var]), dtype=torch.float32, device=_DEVICE))
    arr = torch.stack([torch.stack(values), torch.stack(var_ids)], dim=-1).unsqueeze(0)
    return arr


def objective_torch(P_vec: torch.Tensor) -> torch.Tensor:
    X = build_input_torch(P_vec.to(_DEVICE))
    Y = build_target().to(_DEVICE)
    forecast_out, *_ = _LIT_MODEL.forward(data_input=X, data_trg=Y)
    return forecast_out.reshape(-1)[0]


# Autograd-based optimizer: treat the current point as a trainable torch
# parameter and update it with Adam while enforcing bound constraints.
def torch_adam_opt(x0: np.ndarray, lb: np.ndarray, ub: np.ndarray, lr: float = 0.05, iters: int = 200):
    P = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32, device=_DEVICE))
    opt = torch.optim.Adam([P], lr=lr)
    best_f = float("inf")
    best_x = None
    history = []
    lb_t = torch.tensor(lb, dtype=torch.float32, device=_DEVICE)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=_DEVICE)
    for _ in range(1, iters + 1):
        with torch.no_grad():
            P.data = torch.max(torch.min(P.data, ub_t), lb_t)
        x_before = P.detach().cpu().numpy().copy()
        opt.zero_grad(set_to_none=True)
        loss = objective_torch(P)
        loss.backward()
        opt.step()
        with torch.no_grad():
            P.data = torch.max(torch.min(P.data, ub_t), lb_t)
        # Use the loss value before clamping on the next iteration to compare
        # the current candidate point to the best-so-far solution.
        f = float(loss.item())
        if f < best_f:
            best_f = f
            best_x = x_before
        history.append((best_f, float(torch.linalg.norm(P.detach()).item())))
    return best_x, best_f, history


def torch_adam_opt_traced(
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    trace: "EvalTrace",
    lr: float = 0.05,
    iters: int = 200,
):
    P = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32, device=_DEVICE))
    opt = torch.optim.Adam([P], lr=lr)
    best_f = float("inf")
    best_x = None
    history = []
    lb_t = torch.tensor(lb, dtype=torch.float32, device=_DEVICE)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=_DEVICE)
    for _ in range(1, iters + 1):
        with torch.no_grad():
            P.data = torch.max(torch.min(P.data, ub_t), lb_t)
        x_before = P.detach().cpu().numpy().copy()
        opt.zero_grad(set_to_none=True)
        loss = objective_torch(P)
        f = float(loss.item())
        trace.add(np.asarray(x_before, dtype=float), f)
        loss.backward()
        opt.step()
        with torch.no_grad():
            P.data = torch.max(torch.min(P.data, ub_t), lb_t)
        if f < best_f:
            best_f = f
            best_x = x_before
        history.append((best_f, float(torch.linalg.norm(P.detach()).item())))
    return best_x, best_f, history


def write_history(path: Path, history: list[tuple[float, float]], label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("iter,f_best_std,norm_x\n")
        for i, (f_best, norm_x) in enumerate(history, start=1):
            f.write(f"{i},{f_best},{norm_x}\n")
    print(f"[{label}] history saved to {path}")


class EvalTrace:
    def __init__(self, seed: int):
        self.seed = seed
        self.eval_z: list[np.ndarray] = []
        self.eval_x: list[np.ndarray] = []
        self.eval_f_pred_std: list[float] = []
        self.best_path_x: list[np.ndarray] = []
        self.best_path_f_pred_std: list[float] = []
        self.best_f_pred_std = float("inf")

    def add(self, z: np.ndarray, f_pred_std: float) -> None:
        x = z_to_x(z)
        self.eval_z.append(z.copy())
        self.eval_x.append(x.copy())
        self.eval_f_pred_std.append(float(f_pred_std))
        if f_pred_std < self.best_f_pred_std:
            self.best_f_pred_std = float(f_pred_std)
            self.best_path_x.append(x.copy())
            self.best_path_f_pred_std.append(float(f_pred_std))

    def eval_df(self) -> pd.DataFrame:
        data = {
            "seed": self.seed,
            "eval_idx": np.arange(1, len(self.eval_f_pred_std) + 1),
            "f_pred_std": self.eval_f_pred_std,
            "f_pred_raw": [y_denorm(v) for v in self.eval_f_pred_std],
            "f_true_raw": [styblinski_tang_true_x(x) for x in self.eval_x],
        }
        if DIM >= 1:
            data["x_1"] = [float(x[0]) for x in self.eval_x]
        if DIM >= 2:
            data["x_2"] = [float(x[1]) for x in self.eval_x]
        return pd.DataFrame(data)

    def traj_df(self) -> pd.DataFrame:
        data = {
            "seed": self.seed,
            "step": np.arange(1, len(self.best_path_f_pred_std) + 1),
            "f_best_pred_std": self.best_path_f_pred_std,
            "f_best_pred_raw": [y_denorm(v) for v in self.best_path_f_pred_std],
        }
        if DIM >= 1:
            data["x_best_1"] = [float(x[0]) for x in self.best_path_x]
        if DIM >= 2:
            data["x_best_2"] = [float(x[1]) for x in self.best_path_x]
        return pd.DataFrame(data)


def optimize_one_seed(seed: int, optimizer: str, budget: int, lr: float, cma_sigma0: float) -> tuple[dict, EvalTrace]:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, size=DIM)
    bounds = (LB, UB)
    trace = EvalTrace(seed=seed)
    base_predictor = ModelPredictor(build_input=build_input, run_model=run_model)

    def traced_predict(P: np.ndarray) -> float:
        y_std = float(base_predictor(P))
        trace.add(np.asarray(P, dtype=float), y_std)
        return y_std

    objective = Objective(traced_predict, maximize=False)
    if optimizer == "cma":
        res = cma_es(objective, x0=x0, sigma0=float(cma_sigma0), bounds=bounds, max_iters=budget)
        z_best = res.x_best
        f_best_std = float(res.f_best)
        n_eval = int(res.n_eval)
    elif optimizer == "autograd":
        z_best, f_best_std, hist = torch_adam_opt_traced(x0=x0, lb=LB, ub=UB, trace=trace, lr=lr, iters=budget)
        n_eval = len(hist)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    x_best = z_to_x(z_best)
    row = {
        "seed": seed,
        "optimizer": optimizer,
        "budget": budget,
        "n_eval": n_eval,
        "z_best_1": float(z_best[0]),
        "z_best_2": float(z_best[1]) if DIM > 1 else float("nan"),
        "x_best_1": float(x_best[0]),
        "x_best_2": float(x_best[1]) if DIM > 1 else float("nan"),
        "z_best_json": json.dumps([float(v) for v in z_best.tolist()]),
        "x_best_json": json.dumps([float(v) for v in x_best.tolist()]),
        "f_best_pred_std": f_best_std,
        "f_best_pred_raw": y_denorm(f_best_std),
        "f_best_true_raw": float(styblinski_tang_true_x(x_best)),
    }
    return row, trace


def plot_boxplot(out_path: Path, summary_df: pd.DataFrame) -> None:
    vals = summary_df["f_best_true_raw"].to_numpy()
    plt.figure(figsize=(7, 5))
    plt.boxplot([vals], labels=["final best true f"], showmeans=True)
    plt.scatter(np.ones_like(vals), vals, alpha=0.7, s=18)
    plt.ylabel("Objective value (raw)")
    plt.title("Final best objective across seeds")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_boxplot_by_optimizer(out_path: Path, summary_df: pd.DataFrame) -> None:
    groups = []
    labels = []
    for opt in ["cma", "autograd"]:
        vals = summary_df.loc[summary_df["optimizer"] == opt, "f_best_true_raw"].to_numpy()
        if len(vals) > 0:
            groups.append(vals)
            labels.append(opt)
    if not groups:
        return
    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, labels=labels, showmeans=True)
    for i, vals in enumerate(groups, start=1):
        plt.scatter(np.full_like(vals, i, dtype=float), vals, alpha=0.65, s=18)
    plt.ylabel("Objective value (raw)")
    plt.title("Final best objective by optimizer")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_contour_2d(out_path: Path, summary_df: pd.DataFrame) -> None:
    if DIM != 2:
        return
    x1 = np.linspace(X_LOWER, X_UPPER, 260)
    x2 = np.linspace(X_LOWER, X_UPPER, 260)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 0.5 * ((X1**4 - 16.0 * X1**2 + 5.0 * X1) + (X2**4 - 16.0 * X2**2 + 5.0 * X2))
    plt.figure(figsize=(8, 7))
    cf = plt.contourf(X1, X2, Z, levels=70, cmap="viridis")
    plt.colorbar(cf, label="True Styblinski-Tang f(x)")
    plt.scatter(summary_df["x_best_1"], summary_df["x_best_2"], c=summary_df["f_best_true_raw"], cmap="plasma", s=35)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Best found points across seeds")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_contour_with_trajectory_2d(out_path: Path, opt_dir: Path, summary_df: pd.DataFrame) -> None:
    if DIM != 2 or summary_df.empty:
        return

    # use the best seed by true objective
    best_row = summary_df.loc[summary_df["f_best_true_raw"].idxmin()]
    best_seed = int(best_row["seed"])
    eval_path = opt_dir / f"evaluations_seed_{best_seed}.csv"
    traj_path = opt_dir / f"trajectory_seed_{best_seed}.csv"
    if not eval_path.exists() or not traj_path.exists():
        return

    eval_df = pd.read_csv(eval_path)
    traj_df = pd.read_csv(traj_path)
    if not {"x_1", "x_2", "f_true_raw"}.issubset(eval_df.columns):
        return
    if not {"x_best_1", "x_best_2"}.issubset(traj_df.columns):
        return

    x1 = np.linspace(X_LOWER, X_UPPER, 260)
    x2 = np.linspace(X_LOWER, X_UPPER, 260)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 0.5 * ((X1**4 - 16.0 * X1**2 + 5.0 * X1) + (X2**4 - 16.0 * X2**2 + 5.0 * X2))

    plt.figure(figsize=(8, 7))
    cf = plt.contourf(X1, X2, Z, levels=70, cmap="viridis")
    plt.colorbar(cf, label="True Styblinski-Tang f(x)")

    plt.scatter(eval_df["x_1"], eval_df["x_2"], c=eval_df["f_true_raw"], cmap="plasma", s=10, alpha=0.55, label="evaluations")
    plt.plot(traj_df["x_best_1"], traj_df["x_best_2"], color="white", linewidth=2.0, label="best-so-far trajectory")
    plt.scatter(traj_df["x_best_1"].iloc[0], traj_df["x_best_2"].iloc[0], marker="o", s=45, color="cyan", label="trajectory start")
    plt.scatter(traj_df["x_best_1"].iloc[-1], traj_df["x_best_2"].iloc[-1], marker="x", s=60, color="red", label="trajectory end")
    plt.scatter([-2.903534], [-2.903534], marker="+", s=80, color="lime", label="global minimum")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"2D contour with trajectory (best seed={best_seed})")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_surface_true_vs_model_2d(out_path: Path, num: int = 100) -> None:
    if DIM != 2:
        return

    z1 = np.linspace(-1.0, 1.0, num)
    z2 = np.linspace(-1.0, 1.0, num)
    Z1, Z2 = np.meshgrid(z1, z2)
    z_flat = np.stack([Z1.reshape(-1), Z2.reshape(-1)], axis=1)

    x_flat = np.array([z_to_x(z) for z in z_flat], dtype=float)
    X1 = x_flat[:, 0].reshape(num, num)
    X2 = x_flat[:, 1].reshape(num, num)
    Z_true = 0.5 * ((X1**4 - 16.0 * X1**2 + 5.0 * X1) + (X2**4 - 16.0 * X2**2 + 5.0 * X2))
    Z_model = predict_model_raw_batch(z_flat).reshape(num, num)

    zmin = float(min(np.nanmin(Z_true), np.nanmin(Z_model)))
    zmax = float(max(np.nanmax(Z_true), np.nanmax(Z_model)))

    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    s1 = ax1.plot_surface(X1, X2, Z_true, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax1.set_title("True surface")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("f(x)")
    ax1.set_zlim(zmin, zmax)
    fig.colorbar(s1, ax=ax1, shrink=0.6, pad=0.1)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    s2 = ax2.plot_surface(X1, X2, Z_model, cmap="plasma", linewidth=0, antialiased=True, alpha=0.95)
    ax2.set_title("Model surface")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("f_hat(x)")
    ax2.set_zlim(zmin, zmax)
    fig.colorbar(s2, ax=ax2, shrink=0.6, pad=0.1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_multiseed_experiment(
    optimizer: str,
    seeds: int,
    seed_start: int,
    budget: int,
    lr: float,
    out_dir: Path,
    cma_sigma0: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = out_dir / optimizer
    opt_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    eval_frames = []
    traj_frames = []
    for seed in range(seed_start, seed_start + seeds):
        row, trace = optimize_one_seed(seed=seed, optimizer=optimizer, budget=budget, lr=lr, cma_sigma0=cma_sigma0)
        summary_rows.append(row)
        eval_df = trace.eval_df()
        traj_df = trace.traj_df()
        eval_frames.append(eval_df)
        traj_frames.append(traj_df)
        eval_df.to_csv(opt_dir / f"evaluations_seed_{seed}.csv", index=False)
        traj_df.to_csv(opt_dir / f"trajectory_seed_{seed}.csv", index=False)
        print(
            f"[seed={seed}] best_pred_std={row['f_best_pred_std']:.6f}, "
            f"best_pred_raw={row['f_best_pred_raw']:.6f}, best_true_raw={row['f_best_true_raw']:.6f}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="seed").reset_index(drop=True)
    summary_df.to_csv(opt_dir / "summary.csv", index=False)
    pd.concat(eval_frames, ignore_index=True).to_csv(opt_dir / "evaluations_all.csv", index=False)
    pd.concat(traj_frames, ignore_index=True).to_csv(opt_dir / "trajectory_all.csv", index=False)
    plot_boxplot(out_path=opt_dir / "final_best_boxplot.png", summary_df=summary_df)
    plot_contour_2d(out_path=opt_dir / "best_points_contour.png", summary_df=summary_df)
    plot_contour_with_trajectory_2d(out_path=opt_dir / "contour_with_trajectory.png", opt_dir=opt_dir, summary_df=summary_df)
    plot_surface_true_vs_model_2d(out_path=opt_dir / "surface_true_vs_model.png")
    print(f"Saved surrogate experiment outputs to {opt_dir}")


def main_single():
    predictor = ModelPredictor(build_input=build_input, run_model=run_model)
    objective = Objective(predictor, maximize=False)
    x0 = np.zeros(DIM, dtype=float)
    bounds = (LB, UB)
    res = cma_es(objective, x0=x0, sigma0=1.0, bounds=bounds, max_iters=200)
    print("[CMA-ES] f_best_std:", res.f_best, "x*:", res.x_best)
    write_history(Path(LOG_DIR) / "cma_es.csv", res.history, "cma_es")
    res_adam = adam(objective, x0=x0, bounds=bounds, lr=0.05, iters=200)
    print("[Adam-FD] f_best_std:", res_adam.f_best, "x*:", res_adam.x_best)
    write_history(Path(LOG_DIR) / "adam_fd.csv", res_adam.history, "adam_fd")


def main_autograd():
    x0 = np.zeros(DIM, dtype=float)
    x_best, f_best, hist = torch_adam_opt(x0=x0, lb=LB, ub=UB, lr=0.05, iters=200)
    print("[Torch-Adam] best f_std:", f_best, "x*:", x_best)
    write_history(Path(LOG_DIR) / "torch_adam.csv", hist, "torch_adam")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "autograd", "multiseed"], default="single")
    parser.add_argument("--optimizer", choices=["cma", "autograd", "both"], default="cma")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--cma-sigma0", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path(RESULTS_DIR))
    args = parser.parse_args()

    if args.mode == "single":
        main_single()
    elif args.mode == "autograd":
        main_autograd()
    else:
        if args.optimizer == "both":
            run_multiseed_experiment("cma", args.seeds, args.seed_start, args.budget, args.lr, args.out_dir, args.cma_sigma0)
            run_multiseed_experiment("autograd", args.seeds, args.seed_start, args.budget, args.lr, args.out_dir, args.cma_sigma0)
            cma_summary = pd.read_csv(args.out_dir / "cma" / "summary.csv")
            aut_summary = pd.read_csv(args.out_dir / "autograd" / "summary.csv")
            combined = pd.concat([cma_summary, aut_summary], ignore_index=True)
            combined.to_csv(args.out_dir / "summary_combined.csv", index=False)
            plot_boxplot_by_optimizer(args.out_dir / "boxplot_cma_vs_autograd.png", combined)
            print(f"Saved combined comparison to {args.out_dir}")
        else:
            run_multiseed_experiment(
                optimizer=args.optimizer,
                seeds=args.seeds,
                seed_start=args.seed_start,
                budget=args.budget,
                lr=args.lr,
                out_dir=args.out_dir,
                cma_sigma0=args.cma_sigma0,
            )
