#!/usr/bin/env python
"""
Train surrogate regressors mapping P -> C using either SCM-generated data
(data/<dataset>/ds.npz) or a custom industrial dataset (CSV, Parquet, etc.).

Usage examples:
  python optimization/train_p_to_c_surrogate.py \
      --mode scm --dataset example --output models/p_to_c.pt

  python optimization/train_p_to_c_surrogate.py \
      --mode industrial --data-file data/real_runs.csv \
      --p-cols P1 P2 P3 --c-cols C1 C2 \
      --output models/p_to_c_real.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def load_scm_data(dataset: str, data_root: Path) -> tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    ds_path = data_root / dataset / "ds.npz"
    meta = np.load(ds_path)
    X = meta["x"]  # shape (N, L, 2): value + variable id

    with open(data_root / dataset / "input_vars_map.json") as f:
        var_map = json.load(f)

    ordered = sorted(var_map.items(), key=lambda kv: kv[1])
    values = X[:, :, 0]  # actual numeric values, strip off var ids

    p_names = [name for name, _ in ordered if name.startswith("P")]
    c_names = [name for name, _ in ordered if name.startswith("C")]

    if not p_names or not c_names:
        raise ValueError("Could not infer controllable P* and child C* names from input_vars_map.json")

    P = np.stack([values[:, var_map[name]-1] for name in p_names], axis=1)
    C = np.stack([values[:, var_map[name]-1] for name in c_names], axis=1)
    return P, C, p_names, c_names

def load_industrial_csv(data_file: Path, p_cols: List[str], c_cols: List[str]) -> tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    import pandas as pd
    df = pd.read_csv(data_file)
    P = df[p_cols].to_numpy(dtype=np.float32)
    C = df[c_cols].to_numpy(dtype=np.float32)
    return P, C, p_cols, c_cols

class MultiOutputMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def train_model(P: np.ndarray, C: np.ndarray, epochs: int = 200, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiOutputMLP(P.shape[1], C.shape[1]).to(device)
    ds = TensorDataset(torch.tensor(P, dtype=torch.float32),
                       torch.tensor(C, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for p_batch, c_batch in loader:
            p_batch = p_batch.to(device)
            c_batch = c_batch.to(device)
            pred = model(p_batch)
            loss = loss_fn(pred, c_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * p_batch.size(0)
        avg = total / len(ds)
        history.append((epoch+1, avg))
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:04d}: mse={avg:.4f}")
    return model.cpu(), history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scm", "industrial"], default="scm")
    parser.add_argument("--dataset", type=str, default="example")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--data-file", type=Path, help="CSV/Parquet for industrial mode")
    parser.add_argument("--p-cols", nargs="+", default=None, help="Column names for Ps in industrial data")
    parser.add_argument("--c-cols", nargs="+", default=None, help="Column names for Cs in industrial data")
    parser.add_argument("--output", type=Path, default=Path("models/p_to_c.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "scm":
        P, C, p_names, c_names = load_scm_data(args.dataset, args.data_root)
    else:
        if not args.data_file or not args.p_cols or not args.c_cols:
            raise ValueError("Provide --data-file, --p-cols, --c-cols for industrial mode")
        P, C, p_names, c_names = load_industrial_csv(args.data_file, args.p_cols, args.c_cols)

    model, history = train_model(P, C, epochs=args.epochs, lr=args.lr)
    payload = {
        "model_state": model.state_dict(),
        "p_names": p_names,
        "c_names": c_names,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    # save training curve
    hist_path = args.output.with_suffix(".csv")
    with hist_path.open("w") as f:
        f.write("epoch,mse\n")
        for e, mse in history:
            f.write(f"{e},{mse}\n")
    print(f"Saved surrogate to {args.output}")

if __name__ == "__main__":
    main()



