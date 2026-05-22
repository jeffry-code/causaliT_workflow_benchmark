#!/usr/bin/env python
"""
test_trained_vs_scm.py  —  point-wise model vs SCM comparison (Workflow II)
============================================================================
Diagnostic script: evaluates the trained proT checkpoint and the composite
surrogate at specific P vectors and compares their outputs against the true
noise-free SCM output (Y = P1² + P2²).

Useful for:
  - Sanity-checking the trained model at known points
  - Verifying that the composite pipeline (P -> C_hat -> Y_hat) is wired correctly
  - Debugging discrepancies between model predictions and ground truth

Edit DEFAULT_TESTS below or pass --tests-json with a list of P vectors.

Run from repo root:
  python optimization/p_to_c_to_y/test_trained_vs_scm.py

Compare a trained proT checkpoint against the analytical P-only SCM for custom P vectors.

Edit `DEFAULT_TESTS` or pass --tests-json pointing to e.g.
[
  {"name": "baseline", "values": {"P1":0,"P2":0}},
  {"name": "custom1",  "values": {"P1":1,"P2":-1}}
]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.serialization import add_safe_globals
from omegaconf.dictconfig import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scm_ds.datasets import ds_scm_quad  # noqa: E402
from proT.training.forecasters.transformer_forecaster import TransformerForecaster  # noqa: E402

CONTROLLABLE = ["P1", "P2"]
TARGET = "Y"

DEFAULT_TESTS = [
    {"name": "zeros", "values": {p: 0.0 for p in CONTROLLABLE}},
    {"name": "mixed", "values": {"P1": 0.5, "P2": -0.25}},
]


def load_var_maps(data_dir: Path):
    with open(data_dir / "input_vars_map.json", "r") as f:
        iv_map = json.load(f)
    with open(data_dir / "target_vars_map.json", "r") as f:
        tv_map = json.load(f)
    return iv_map, tv_map


def _zero_noise(scm):
    return {var: np.zeros(1, dtype=float) for var in scm.specs.keys()}


def evaluate_scm(assignments: Dict[str, float]) -> float:
    base  = ds_scm_quad.scm
    scm_i = base.do({var: float(assignments[var]) for var in CONTROLLABLE})
    ctx   = scm_i.forward(context={}, eps_draws=_zero_noise(scm_i))
    return float(ctx[TARGET].reshape(-1)[0])


def build_encoder_tensor(iv_map: Dict[str, int], values: Dict[str, float]) -> torch.Tensor:
    ordered_vars = [var for var, _ in sorted(iv_map.items(), key=lambda kv: kv[1])]
    feats = [float(values.get(var, np.nan)) for var in ordered_vars]
    var_ids = [float(iv_map[var]) for var in ordered_vars]
    arr = np.stack([feats, var_ids], axis=-1)[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


def build_decoder_tensor(tv_map: Dict[str, int]) -> torch.Tensor:
    ordered_vars = [var for var, _ in sorted(tv_map.items(), key=lambda kv: kv[1])]
    values = [0.0 for _ in ordered_vars]
    var_ids = [float(tv_map[var]) for var in ordered_vars]
    arr = np.stack([values, var_ids], axis=-1)[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


def load_checkpoint(ckpt_path: Path, device: str | None):
    add_safe_globals([DictConfig])
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["hyper_parameters"]
    try:
        config["model"]["kwargs"]["device"] = "cpu"
    except Exception:
        pass
    model = TransformerForecaster(config)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device


def run_single_test(model, device, iv_map, tv_map, name: str, assignments: Dict[str, float]):
    missing = [p for p in CONTROLLABLE if p not in assignments]
    if missing:
        raise ValueError(f"Test '{name}' is missing values for: {missing}")
    y_true = evaluate_scm(assignments)

    x = build_encoder_tensor(iv_map, assignments).to(device)
    y = build_decoder_tensor(tv_map).to(device)
    with torch.no_grad():
        pred, *_ = model.forward(data_input=x, data_trg=y)
    y_hat = float(pred.reshape(-1)[0].item())
    abs_err = abs(y_hat - y_true)
    rel_err = abs_err / (abs(y_true) + 1e-8)
    return {
        "name": name,
        "y_true": y_true,
        "y_model": y_hat,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "assignments": assignments,
    }


def load_tests(json_path: str | None) -> List[Dict[str, Dict[str, float]]]:
    if not json_path:
        return DEFAULT_TESTS
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tests JSON must be a list of {'name': ..., 'values': {...}} objects.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Compare SCM outputs with a trained proT checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/example/k_0/checkpoints/best_checkpoint.ckpt"),
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/example"),
        help="Directory containing input/target *_vars_map.json.",
    )
    parser.add_argument(
        "--tests-json",
        type=str,
        default=None,
        help="Optional JSON file listing test cases (overrides DEFAULT_TESTS).",
    )
    parser.add_argument("--device", type=str, default=None, help="Force 'cpu' or 'cuda'. Defaults to auto-detection.")
    args = parser.parse_args()

    iv_map, tv_map = load_var_maps(args.data_dir)
    model, device = load_checkpoint(args.checkpoint, args.device)
    tests = load_tests(args.tests_json)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Using device: {device}")
    print(f"Evaluating {len(tests)} test cases\n")

    header = "{:<15} {:>12} {:>12} {:>12} {:>10}".format("Test", "Y_SCM", "Y_model", "AbsErr", "RelErr")
    print(header)
    print("-" * len(header))
    for spec in tests:
        name = spec.get("name", f"test_{len(spec)}")
        assignments = spec["values"]
        result = run_single_test(model, device, iv_map, tv_map, name, assignments)
        print(
            "{:<15} {:>12.6f} {:>12.6f} {:>12.6f} {:>10.4%}".format(
                result["name"], result["y_true"], result["y_model"], result["abs_err"], result["rel_err"]
            )
        )
        formatted = ", ".join(f"{var}={val:.6f}" for var, val in assignments.items())
        print(f"    inputs: {formatted}")
        print()
    print("Done.")


if __name__ == "__main__":
    main()
