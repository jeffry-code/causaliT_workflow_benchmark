"""
plot_input_distribution.py
--------------------------
Plots the distribution of found input values (x) across all dimensions
and seeds for CMA-ES and Adam, for each dimensionality.

Shows whether the optimizer converges to the global optimum (-2.903534)
or gets trapped at the local minimum (+2.903534).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("c:/Users/jeffr/Documents/Masterarbeit/master_thesis1/optimization/multidim")
DIMS = [2, 10, 20, 40, 80]
OPTIMIZERS = {"autograd": "Adam", "cma": "CMA-ES"}
TRUE_OPT = -2.903534
LOCAL_MIN = +2.903534

fig, axes = plt.subplots(len(DIMS), 2, figsize=(10, 14), sharey=False)
fig.suptitle("Distribution of Found Input Values vs. True Optimum", fontsize=13, fontweight="bold")

for row, d in enumerate(DIMS):
    for col, (opt_key, opt_name) in enumerate(OPTIMIZERS.items()):
        ax = axes[row, col]
        summary_path = BASE / f"{d}dim/results/styblinski_tang/{opt_key}/summary.csv"

        if not summary_path.exists():
            ax.set_visible(False)
            continue

        df = pd.read_csv(summary_path)
        all_x = []
        for _, row_data in df.iterrows():
            x_vec = json.loads(row_data["x_best_json"])
            # Flatten per-seed parameter vectors for the histogram.
            all_x.extend(x_vec)

        ax.hist(all_x, bins=40, color="steelblue" if opt_key == "autograd" else "darkorange",
                alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(TRUE_OPT, color="green", linewidth=1.8, linestyle="--", label="Global opt. (−2.90)")
        ax.axvline(LOCAL_MIN, color="red",   linewidth=1.8, linestyle="--", label="Local min. (+2.90)")
        ax.set_xlim(-5.5, 5.5)
        ax.set_title(f"{opt_name} — {d}D", fontsize=10)
        ax.set_xlabel("x value", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)

        if row == 0:
            ax.legend(fontsize=7, loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = BASE / "results/input_distribution.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight")
# Save a PNG variant too, since some thesis tools prefer raster images.
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
