#!/usr/bin/env python
"""
Plot Y vs (P1, P2) for:
  - SCM (noise-free)
  - Trained model
  - Noisy dataset samples

Usage examples:

  # model only (easier to inspect surrogate surface)
  python optimization/plots.py \
    --checkpoint experiments/example/k_0/checkpoints/best_checkpoint.ckpt \
    --data-dir data/example \
    --mode surface \
    --show model \
    --out plot_model_only.png

  # SCM + model + data (full comparison)
  python optimization/plots.py \
    --checkpoint experiments/example/k_0/checkpoints/best_checkpoint.ckpt \
    --data-dir data/example \
    --mode surface \
    --show scm model data \
    --out plot_all.png

  # contour with SCM and model only
  python optimization/plots.py \
    --checkpoint experiments/example/k_0/checkpoints/best_checkpoint.ckpt \
    --data-dir data/example \
    --mode contour \
    --show scm model \
    --out plot_contour_compare.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torch.serialization import add_safe_globals

from proT.training.forecasters.transformer_forecaster import TransformerForecaster
from scm_ds.datasets import ds_scm_quad


# ---------- loaders ----------
def load_var_maps(data_dir: Path):
    with open(data_dir / "input_vars_map.json") as f:
        iv_map = json.load(f)
    with open(data_dir / "target_vars_map.json") as f:
        tv_map = json.load(f)
    return iv_map, tv_map


def load_model(ckpt_path: Path, device: str):
    add_safe_globals([DictConfig])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]
    if "model" in cfg and "kwargs" in cfg["model"]:
        cfg["model"]["kwargs"]["device"] = "cpu"
    model = TransformerForecaster(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# ---------- tensors ----------
def build_enc(iv_map, vals: dict) -> torch.Tensor:
    ordered = [var for var, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    feats = [float(vals.get(var, np.nan)) for var in ordered]
    ids = [float(iv_map[var]) for var in ordered]
    arr = np.stack([feats, ids], axis=-1)[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


def build_dec(tv_map) -> torch.Tensor:
    ordered = [var for var, _ in sorted(tv_map.items(), key=lambda kv: kv[1])]
    values = [0.0 for _ in ordered]
    ids = [float(tv_map[var]) for var in ordered]
    arr = np.stack([values, ids], axis=-1)[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


# ---------- evaluations ----------
def eval_scm(p1: float, p2: float):
    base = ds_scm_quad.scm
    assignments = {}
    for var in base.specs.keys():
        if var.startswith("P"):
            assignments[var] = 0.0
    assignments["P1"] = p1
    assignments["P2"] = p2
    scm_i = base.do(assignments)
    ctx = scm_i.forward(context={}, eps_draws={v: np.zeros(1) for v in scm_i.specs})
    return float(ctx["Y"].reshape(-1)[0])


def model_pred(model, iv_map, tv_map, device, p1: float, p2: float):
    vals = {var: 0.0 for var in iv_map.keys() if var.startswith("P")}
    vals["P1"] = float(p1)
    vals["P2"] = float(p2)
    x = build_enc(iv_map, vals).to(device)
    y = build_dec(tv_map).to(device)
    with torch.no_grad():
        out, *_ = model.forward(data_input=x, data_trg=y)
    return float(out.reshape(-1)[0])


def slice_data(data_dir: Path):
    data = np.load(data_dir / "ds.npz")
    X = data["x"]
    Y = data["y"][:, 0, 0]
    with open(data_dir / "input_vars_map.json") as f:
        iv_map = json.load(f)
    vals = X[:, :, 0]
    P1 = vals[:, iv_map["P1"] - 1]
    P2 = vals[:, iv_map["P2"] - 1]
    return P1, P2, Y


# ---------- main plotting ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--pmin", type=float, default=-1.0)
    ap.add_argument("--pmax", type=float, default=1.0)
    ap.add_argument("--num", type=int, default=80)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--mode", choices=["surface", "contour"], default="surface")
    ap.add_argument(
        "--show",
        nargs="+",
        choices=["scm", "model", "data"],
        default=["scm", "model", "data"],
        help="Select which sources to display.",
    )
    ap.add_argument("--mark-p", nargs=2, type=float, default=None, metavar=("P1", "P2"),
                    help="Mark optimized point in normalized plot coordinates (P1,P2).")
    ap.add_argument("--mark-x", nargs=2, type=float, default=None, metavar=("X1", "X2"),
                    help="Mark optimized point in physical coordinates (X1,X2), mapped to plot P-space.")
    ap.add_argument("--x-lower", nargs=2, type=float, default=[-3.0, -2.0], metavar=("L1", "L2"),
                    help="Lower physical bounds for x->p mapping (used with --mark-x).")
    ap.add_argument("--x-upper", nargs=2, type=float, default=[3.0, 2.0], metavar=("U1", "U2"),
                    help="Upper physical bounds for x->p mapping (used with --mark-x).")
    ap.add_argument("--mark-best-from", type=Path, default=None,
                    help="Path to multiseed summary.csv. Automatically marks best point.")
    ap.add_argument("--best-by", choices=["f_best_true", "f_best_pred"], default="f_best_true",
                    help="Column used to pick best row from --mark-best-from.")
    ap.add_argument("--zoom-center", nargs=2, type=float, default=None, metavar=("P1C", "P2C"),
                    help="Optional zoom center in plot coordinates (P-space).")
    ap.add_argument("--zoom-halfwidth", nargs=2, type=float, default=None, metavar=("W1", "W2"),
                    help="Half-width for zoom window around --zoom-center.")
    ap.add_argument("--zoom-out", type=Path, default=None,
                    help="Optional output path for zoomed plot. If omitted, uses <out>_zoom.")
    ap.add_argument("--zoom-zlim", nargs=2, type=float, default=None, metavar=("ZMIN", "ZMAX"),
                    help="Optional z-limits for zoomed 3D surface plot.")
    ap.add_argument("--out", type=Path, default=Path("plot_y.png"))
    args = ap.parse_args()

    show = set(args.show)
    if not show:
        raise ValueError("At least one source must be selected via --show.")
    manual_marks = int(args.mark_p is not None) + int(args.mark_x is not None)
    if manual_marks > 1:
        raise ValueError("Use either --mark-p or --mark-x, not both.")
    if manual_marks and args.mark_best_from is not None:
        raise ValueError("Use either manual mark (--mark-p/--mark-x) or --mark-best-from.")

    mark_p = None
    mark_label = None
    if args.mark_best_from is not None:
        if not args.mark_best_from.exists():
            raise FileNotFoundError(f"Summary file not found: {args.mark_best_from}")
        with args.mark_best_from.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Summary file is empty: {args.mark_best_from}")
        if args.best_by not in rows[0]:
            raise ValueError(f"Column '{args.best_by}' not found in {args.mark_best_from}")
        best_row = min(rows, key=lambda r: float(r[args.best_by]))
        if "z_best_1" not in best_row or "z_best_2" not in best_row:
            raise ValueError("summary.csv must contain z_best_1 and z_best_2 for automatic marking.")
        mark_p = (float(best_row["z_best_1"]), float(best_row["z_best_2"]))
        seed_txt = best_row.get("seed", "NA")
        mark_label = f"Best seed={seed_txt}"
    elif args.mark_p is not None:
        mark_p = (float(args.mark_p[0]), float(args.mark_p[1]))
        mark_label = "Marked point"
    elif args.mark_x is not None:
        x = np.array(args.mark_x, dtype=float)
        xl = np.array(args.x_lower, dtype=float)
        xu = np.array(args.x_upper, dtype=float)
        p = 2.0 * ((x - xl) / (xu - xl)) - 1.0
        mark_p = (float(p[0]), float(p[1]))
        mark_label = "Marked point"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    iv_map, tv_map = load_var_maps(args.data_dir)
    model = load_model(args.checkpoint, device) if "model" in show else None

    # Grid
    p1 = np.linspace(args.pmin, args.pmax, args.num)
    p2 = np.linspace(args.pmin, args.pmax, args.num)
    P1, P2 = np.meshgrid(p1, p2)
    Z_scm = np.zeros_like(P1) if "scm" in show else None
    Z_model = np.zeros_like(P1) if "model" in show else None
    for i in range(args.num):
        for j in range(args.num):
            if "scm" in show:
                Z_scm[i, j] = eval_scm(P1[i, j], P2[i, j])
            if "model" in show:
                Z_model[i, j] = model_pred(model, iv_map, tv_map, device, P1[i, j], P2[i, j])

    if "data" in show:
        P1_d, P2_d, Y_d = slice_data(args.data_dir)

    fig = plt.figure(figsize=(8, 6))

    # Y plot
    if args.mode == "surface":
        ax1 = fig.add_subplot(111, projection="3d")
        if "scm" in show:
            ax1.plot_surface(P1, P2, Z_scm, alpha=0.5, color="C0", label="SCM (no noise)")
        if "model" in show:
            ax1.plot_surface(P1, P2, Z_model, alpha=0.5, color="C1", label="Model")
        if "data" in show:
            ax1.scatter(P1_d, P2_d, Y_d, s=5, alpha=0.3, color="C2", label="Data")
        if mark_p is not None:
            mp1, mp2 = mark_p
            if "model" in show:
                my = model_pred(model, iv_map, tv_map, device, mp1, mp2)
                ax1.scatter([mp1], [mp2], [my], s=60, color="magenta", marker="X", label=f"{mark_label} (model)")
            elif "scm" in show:
                sy = eval_scm(mp1, mp2)
                ax1.scatter([mp1], [mp2], [sy], s=60, color="magenta", marker="X", label=mark_label)
        ax1.set_xlabel("P1"); ax1.set_ylabel("P2"); ax1.set_zlabel("Y")
        if args.target is not None:
            ax1.set_title(f"Y surfaces (target={args.target})")
        if len(show) > 1 or "data" in show:
            ax1.legend()
    else:
        ax1 = fig.add_subplot(111)
        if "scm" in show:
            cs1 = ax1.contourf(P1, P2, Z_scm, levels=30, cmap="Blues")
            fig.colorbar(cs1, ax=ax1, label="Y (SCM)")
        if "model" in show and "scm" in show:
            ax1.contour(P1, P2, Z_model, levels=15, colors="red", linewidths=0.7)
        elif "model" in show:
            cs2 = ax1.contourf(P1, P2, Z_model, levels=30, cmap="Reds")
            fig.colorbar(cs2, ax=ax1, label="Y (Model)")
        if "data" in show:
            ax1.scatter(P1_d, P2_d, s=5, alpha=0.3, color="k")
        if mark_p is not None:
            ax1.scatter([mark_p[0]], [mark_p[1]], s=70, c="magenta", marker="X", label=mark_label)
        ax1.set_xlabel("P1"); ax1.set_ylabel("P2")
        if args.target is not None and "scm" in show:
            ax1.contour(P1, P2, Z_scm, levels=[args.target], colors="k", linestyles="--", linewidths=1.0)
        ax1.set_title("Y (contours)")
        if mark_p is not None:
            ax1.legend(loc="best")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")

    # Optional zoomed version (saved in addition to the full plot)
    if args.zoom_center is not None:
        if args.zoom_halfwidth is None:
            raise ValueError("When using --zoom-center, also provide --zoom-halfwidth.")
        c1, c2 = float(args.zoom_center[0]), float(args.zoom_center[1])
        w1, w2 = float(args.zoom_halfwidth[0]), float(args.zoom_halfwidth[1])
        ax1.set_xlim(c1 - w1, c1 + w1)
        ax1.set_ylim(c2 - w2, c2 + w2)
        if args.mode == "surface" and args.zoom_zlim is not None:
            ax1.set_zlim(float(args.zoom_zlim[0]), float(args.zoom_zlim[1]))
        if args.zoom_out is None:
            zoom_out = args.out.with_name(f"{args.out.stem}_zoom{args.out.suffix}")
        else:
            zoom_out = args.zoom_out
        zoom_out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(zoom_out, dpi=220)
        print(f"Saved zoomed plot to {zoom_out}")


if __name__ == "__main__":
    main()
