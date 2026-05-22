#!/usr/bin/env python
"""
evaluate_checkpoints_true_function.py  —  surrogate fidelity evaluation
========================================================================
Standalone utility called by the pipeline and sweep scripts to measure how
well a trained proT checkpoint approximates the true analytical function on
a held-out test set.

Steps:
  1. Loads the checkpoint and reads data/example/meta.json for bounds + function name.
  2. Samples n_samples points uniformly in normalized z-space [-1,1]^d.
  3. Maps z -> x using the stored min/max bounds.
  4. Evaluates the true function y(x) analytically (Styblinski-Tang).
  5. Computes model predictions ŷ(z) and de-standardises using stored μ/σ.
  6. Reports RMSE and R² and writes results to --out-csv / --out-plot.

This script is also callable standalone for manual fidelity checks:
  python optimization/evaluate_checkpoints_true_function.py \\
    --checkpoint optimization/multidim/80dim/experiments/example/k_0/checkpoints/best_checkpoint.ckpt \\
    --data-dir optimization/multidim/80dim/data/example \\
    --n-samples 5000 --out-csv results/fidelity.csv

Evaluate one or multiple model checkpoints against the true analytical function.

- Loads checkpoints
- Samples inputs in normalized z-space ([-1,1]^d)
- Maps z -> x using data/example/meta.json bounds
- Computes true y(x) from meta['function']
- Compares model predictions vs true targets with RMSE and R^2

Example:
  python optimization/evaluate_checkpoints_true_function.py ^
    --checkpoint experiments/example/k_0/checkpoints/best_checkpoint.ckpt ^
    --data-dir data/example ^
    --n-samples 5000 ^
    --seed 42 ^
    --out-csv results/checkpoint_eval.csv ^
    --out-plot results/checkpoint_eval.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from omegaconf.dictconfig import DictConfig
from torch.serialization import add_safe_globals

from proT.training.forecasters.transformer_forecaster import TransformerForecaster

DEFAULT_BEST_CKPT = Path('experiments/example/k_0/checkpoints/best_checkpoint.ckpt')


# Analytical Styblinski-Tang function: f(x) = 0.5 * sum(x_i^4 - 16x_i^2 + 5x_i), global min at x* ≈ -2.9035.
def styblinski_tang(x: np.ndarray) -> np.ndarray:
    # x shape: [N, d]
    return 0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)


def six_hump_camel(x: np.ndarray) -> np.ndarray:
    # expects x[:,0], x[:,1]
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (4 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


# Select the true analytical benchmark function by name.
def get_true_function(name: str):
    fn = (name or "").lower()
    if fn == "styblinski_tang":
        return styblinski_tang
    if fn == "rastrigin":
        return rastrigin
    if fn in {"six_hump_camel", "six-hump-camel", "sixhumpcamel"}:
        return six_hump_camel
    raise ValueError(f"Unsupported function '{name}'. Supported: styblinski_tang, rastrigin, six_hump_camel")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# Load variable ordering maps and dataset metadata (bounds, normalisation stats, function name).
def load_maps(data_dir: Path):
    with open(data_dir / "input_vars_map.json", "r", encoding="utf-8") as f:
        iv_map = json.load(f)
    with open(data_dir / "target_vars_map.json", "r", encoding="utf-8") as f:
        tv_map = json.load(f)
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    ordered_inputs = [k for k, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    return iv_map, tv_map, meta, ordered_inputs


# Resolve checkpoint paths from CLI args — supports file paths, globs, and directories.
def resolve_checkpoints(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.glob("*.ckpt")))
        else:
            matches = [Path(x) for x in glob.glob(p)]
            if matches:
                out.extend(sorted(matches))
            elif path.suffix == ".ckpt" and path.exists():
                out.append(path)
    # deduplicate keep order
    seen = set()
    uniq = []
    for p in out:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def build_encoder_batch(z_batch: np.ndarray, iv_map: Dict[str, int], ordered_inputs: List[str], device: str) -> torch.Tensor:
    # Build the encoder tensor for a batch of normalized z values.
    # The first channel is the value, the second is the variable ID.
    bsz, d = z_batch.shape
    if len(ordered_inputs) != d:
        raise ValueError(f"Dimension mismatch: input map has {len(ordered_inputs)} vars but z has dim {d}")
    ids = np.array([float(iv_map[v]) for v in ordered_inputs], dtype=np.float32)
    ids = np.repeat(ids[None, :], bsz, axis=0)
    enc = np.stack([z_batch.astype(np.float32), ids], axis=-1)
    return torch.tensor(enc, dtype=torch.float32, device=device)


def build_decoder_batch(tv_map: Dict[str, int], batch_size: int, device: str) -> torch.Tensor:
    # Build a decoder tensor of zero values and var IDs for the target side.
    ordered_t = [k for k, _ in sorted(tv_map.items(), key=lambda kv: kv[1])]
    vals = np.zeros((batch_size, len(ordered_t)), dtype=np.float32)
    ids = np.array([[float(tv_map[k]) for k in ordered_t]], dtype=np.float32)
    ids = np.repeat(ids, batch_size, axis=0)
    dec = np.stack([vals, ids], axis=-1)
    return torch.tensor(dec, dtype=torch.float32, device=device)


# Load a ProT checkpoint and move it to the target device in eval mode.
def load_model(ckpt_path: Path, device: str):
    add_safe_globals([DictConfig])
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["hyper_parameters"]
    if "model" in cfg and "kwargs" in cfg["model"]:
        cfg["model"]["kwargs"]["device"] = "cpu"
    model = TransformerForecaster(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval()


def predict_std(model, z: np.ndarray, iv_map, tv_map, ordered_inputs, device: str, batch_size: int = 512) -> np.ndarray:
    # Run the model in batches and return standardized predictions.
    preds = []
    with torch.no_grad():
        for i in range(0, len(z), batch_size):
            z_b = z[i : i + batch_size]
            x_b = build_encoder_batch(z_b, iv_map, ordered_inputs, device)
            y_b = build_decoder_batch(tv_map, len(z_b), device)
            out, *_ = model.forward(data_input=x_b, data_trg=y_b)
            out_flat = out.reshape(out.shape[0], -1)[:, 0]
            preds.append(out_flat.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Evaluate checkpoints against true analytical function (RMSE/R^2).")
    ap.add_argument("--checkpoint", nargs="*", default=None, help="Checkpoint path, glob, or directory. If omitted, uses default best checkpoint.")
    ap.add_argument("--data-dir", type=Path, default=Path("data/example"))
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument(
        "--use-dataset",
        action="store_true",
        help="If set, evaluate on rows from <data-dir>/ds.npz (or --dataset-npz) instead of random z-sampling.",
    )
    ap.add_argument(
        "--dataset-npz",
        type=Path,
        default=None,
        help="Optional explicit path to ds.npz used when --use-dataset is set.",
    )
    ap.add_argument("--out-csv", type=Path, default=Path("results/checkpoint_eval.csv"))
    ap.add_argument("--out-plot", type=Path, default=None, help="Optional bar plot for RMSE/R^2")
    ap.add_argument("--out-scatter", type=Path, default=None, help="Optional scatter true vs pred (best RMSE)")
    args = ap.parse_args()

    checkpoint_args = args.checkpoint if args.checkpoint else [str(DEFAULT_BEST_CKPT)]
    ckpts = resolve_checkpoints(checkpoint_args)
    if not ckpts:
        raise ValueError("No checkpoints found. Checked: " + ", ".join(checkpoint_args))

    iv_map, tv_map, meta, ordered_inputs = load_maps(args.data_dir)
    d = len(ordered_inputs)
    func_name = meta.get("function", "styblinski_tang")
    true_fn = get_true_function(func_name)

    x_lower = float(meta.get("x_lower", -5.0))
    x_upper = float(meta.get("x_upper", 5.0))
    y_mean = float(meta.get("y_mean", 0.0))
    y_std = float(meta.get("y_std", 1.0))
    if y_std <= 0:
        y_std = 1.0

    if args.use_dataset:
        ds_path = args.dataset_npz if args.dataset_npz is not None else (args.data_dir / "ds.npz")
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {ds_path}")
        ds = np.load(ds_path)
        if "x" not in ds:
            raise ValueError(f"Dataset file has no 'x' array: {ds_path}")
        x_arr = ds["x"]
        if x_arr.ndim != 3 or x_arr.shape[1] != d:
            raise ValueError(
                f"Unexpected x shape in {ds_path}: {x_arr.shape}. Expected [N, {d}, 2]."
            )
        z = x_arr[:, :, 0].astype(np.float32)
        x = x_lower + ((z + 1.0) / 2.0) * (x_upper - x_lower)
    else:
        rng = np.random.default_rng(args.seed)
        z = rng.uniform(-1.0, 1.0, size=(args.n_samples, d)).astype(np.float32)
        x = x_lower + ((z + 1.0) / 2.0) * (x_upper - x_lower)

    y_true_raw = true_fn(x)
    y_true_std = (y_true_raw - y_mean) / y_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    all_preds = {}
    for ckpt in ckpts:
        model = load_model(ckpt, device)
        y_pred_std = predict_std(model, z, iv_map, tv_map, ordered_inputs, device=device, batch_size=args.batch_size)
        y_pred_raw = y_pred_std * y_std + y_mean

        row = {
            "checkpoint": str(ckpt),
            "rmse_std": rmse(y_true_std, y_pred_std),
            "r2_std": r2_score(y_true_std, y_pred_std),
            "rmse_raw": rmse(y_true_raw, y_pred_raw),
            "r2_raw": r2_score(y_true_raw, y_pred_raw),
            "n_samples": int(len(z)),
            "seed": int(args.seed),
            "function": func_name,
            "dim": int(d),
            "mode": "dataset" if args.use_dataset else "random",
        }
        rows.append(row)
        all_preds[str(ckpt)] = (y_pred_raw, y_pred_std)
        print(
            f"[{ckpt.name}] RMSE(raw)={row['rmse_raw']:.6f}, R2(raw)={row['r2_raw']:.6f}, "
            f"RMSE(std)={row['rmse_std']:.6f}, R2(std)={row['r2_std']:.6f}"
        )

    df = pd.DataFrame(rows).sort_values("rmse_raw", ascending=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved checkpoint evaluation to {args.out_csv}")

    if args.out_plot is not None:
        labels = [Path(p).name for p in df["checkpoint"].tolist()]
        rmse_vals = df["rmse_raw"].to_numpy()
        r2_vals = df["r2_raw"].to_numpy()

        fig, ax1 = plt.subplots(figsize=(max(8, 0.8 * len(labels)), 4.5))
        x_idx = np.arange(len(labels))
        ax1.bar(x_idx - 0.2, rmse_vals, width=0.4, label="RMSE(raw)", color="tab:blue")
        ax1.set_ylabel("RMSE (raw)")
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(labels, rotation=30, ha="right")

        ax2 = ax1.twinx()
        ax2.bar(x_idx + 0.2, r2_vals, width=0.4, label="R2(raw)", color="tab:orange")
        ax2.set_ylabel("R2 (raw)")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
        ax1.set_title("Checkpoint comparison vs true function")
        plt.tight_layout()

        args.out_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_plot, dpi=220)
        print(f"Saved plot to {args.out_plot}")

    if args.out_scatter is not None:
        best_ckpt = str(df.iloc[0]["checkpoint"])
        y_pred_raw, _ = all_preds[best_ckpt]
        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(y_true_raw, y_pred_raw, s=8, alpha=0.5)
        lo = float(min(np.min(y_true_raw), np.min(y_pred_raw)))
        hi = float(max(np.max(y_true_raw), np.max(y_pred_raw)))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel("True y (raw)")
        plt.ylabel("Predicted y (raw)")
        plt.title(f"Best checkpoint: {Path(best_ckpt).name}")
        plt.tight_layout()

        args.out_scatter.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_scatter, dpi=220)
        print(f"Saved scatter to {args.out_scatter}")


if __name__ == "__main__":
    main()


