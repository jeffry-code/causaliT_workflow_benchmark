# causaliT

Code repository for the master thesis:
**"Surrogate-Based Optimization in High-Dimensional Causal Process Models"**

The repository contains two studies:

1. **Styblinski-Tang benchmark** — a controlled high-dimensional surrogate optimization study using the Styblinski-Tang function across dimensions d ∈ {2, 10, 20, 40, 80}.
2. **P→C→Y causal workflow** — a proof-of-concept application of surrogate-based optimization on a structural causal model (SCM) with a paraboloid ground truth (Y = P1² + P2²).

Both studies use **ProT**, a Transformer-based regression surrogate trained with PyTorch Lightning, and **CMA-ES** (via `pycma`) as the primary optimizer.

> **Acknowledgement:** The ProT surrogate model and SCM framework are based on prior work by [@scipi1](https://github.com/scipi1). This work extends that foundation by implementing the surrogate-based optimization part.

---

## Repository structure

```
causaliT/
├── proT/                          # Transformer surrogate (ProT)
│   ├── core/                      # Model architecture (encoder-decoder Transformer)
│   ├── training/                  # Trainer, forecaster, callbacks, dataloader
│   └── config/                    # YAML configuration files
│
├── scm_ds/                        # Dataset generation
│   ├── scm.py                     # Structural causal model implementation
│   └── datasets.py                # Styblinski-Tang data generation and SCM datasets
│
├── optimization/                  # Styblinski-Tang optimization study (main work)
│   ├── objectives.py              # Objective wrapper (minimize / target)
│   ├── optimizers.py              # CMA-ES (pycma) and finite-difference Adam
│   ├── predictors.py              # ModelPredictor and SCMPredictor adapters
│   ├── run_with_model_template.py # Main optimization script
│   ├── run_multidim_pipeline.py   # Dimension scaling study (d=2,10,20,40,80)
│   ├── run_80d_budget_seed_sweep.py       # Budget and seed sweep at d=80
│   ├── run_80d_optimizer_param_sweep.py   # CMA-ES sigma0 and Adam lr sweep
│   ├── evaluate_checkpoints_true_function.py  # Surrogate fidelity (R², RMSE)
│   ├── plot_multidim_results.py   # Plotting for dimension scaling results
│   └── p_to_c_to_y/              # P->C->Y causal workflow (self-contained)
│       ├── data/example/          # Paraboloid SCM dataset (JSON metadata)
│       ├── experiments/example/   # ProT config
│       ├── train_p_to_c_surrogate.py      # Train MLP surrogate: P -> C
│       ├── run_multiseed_experiment.py    # Multi-seed optimization (CMA-ES + Adam, 20 seeds)
│       ├── test_trained_vs_scm.py # Compare model predictions to SCM ground truth
│       ├── plots.py               # Visualize Y surface (SCM vs model vs data)
│       └── plot_performance.py    # Training and optimization progress plots
│
├── data/                          # Styblinski-Tang dataset metadata (JSON)
├── setup.py                       # Package install (pip install -e .)
└── INSTALLATION.md                # Detailed setup and cluster instructions
```

---

## Installation

All scripts are run from the repository root. Install the package in editable mode once per environment:

```bash
pip install -e .
```

This makes `proT` and `scm_ds` importable from any script in the repository. See `INSTALLATION.md` for virtual environment setup, cluster configuration, and troubleshooting.

**Key dependencies:** `torch`, `pytorch-lightning`, `cma`, `numpy`, `pandas`, `matplotlib`, `omegaconf`

> **Note:** Trained checkpoints (`.ckpt`) and datasets (`.npz`) are not included in the repository — they are generated locally by running the training and pipeline scripts.

---

## Study 1: Styblinski-Tang benchmark

The Styblinski-Tang function is a separable, multimodal benchmark for evaluating surrogate fidelity and optimization performance across dimensions d ∈ {2, 10, 20, 40, 80}.

**Function:** f(x) = 0.5 · Σ (x_i⁴ − 16x_i² + 5x_i),  global minimum at x\* ≈ −2.9035 per dimension.

Inputs are normalized to [−1, 1]^d before being passed to the surrogate; surrogate outputs are standardized during training.

### Generating the dataset

```bash
# Generate Styblinski-Tang dataset (example: 80D, 400k samples)
python scm_ds/datasets.py --dim 80 --n 400000
```

This saves `ds.npz` and `meta.json` to `data/example/`. Repeat for each dimension folder under `optimization/multidim/`.

### Training the surrogate

```bash
# Train the surrogate — checkpoints are generated
# Use an appropriate config (see experiments/example/)
python proT/training/trainer.py
```

Surrogate fidelity on a held-out random test set can be evaluated with:

```bash
python optimization/evaluate_checkpoints_true_function.py
```

### Running the optimization experiments

All scripts are run from the repository root:

```bash
# Multi-seed experiment: CMA-ES or Adam, configurable budget
python optimization/run_with_model_template.py \
    --mode multiseed --optimizer cma --seeds 10 --budget 1000 --cma-sigma0 1.0

# Dimension scaling study (d = 2, 10, 20, 40, 80)
python optimization/run_multidim_pipeline.py

# Budget and seed sweep at d=80
python optimization/run_80d_budget_seed_sweep.py

# Optimizer hyperparameter sweep (CMA-ES sigma0, Adam learning rate)
python optimization/run_80d_optimizer_param_sweep.py
```

---

## Study 2: P→C→Y causal workflow

Applies the surrogate optimization framework to a structural causal model (SCM). The task is to find controllable inputs P such that the model-predicted outcome Y reaches a specified target value (target tracking) or is minimized. The paraboloid SCM serves as the ground truth: C1 = P1² + ε₁, C2 = P2² + ε₂, Y = C1 + C2 + ε₃.

**Two-surrogate pipeline:**

| Surrogate | Model | Task |
|-----------|-------|------|
| g_phi (MLP) | `models/p_to_c.pt` | P → Ĉ (predicts intermediate children) |
| f_theta (ProT) | `experiments/example/...` | (P, Ĉ) → Ŷ (predicts outcome) |

**Ablation variant provided:**
- **Point-wise comparison** (`test_trained_vs_scm.py`): evaluates the trained model at specific P vectors against the SCM ground truth

All scripts and JSON metadata for this study are self-contained in `optimization/p_to_c_to_y/`.
Checkpoints and datasets are generated locally by following the steps below.

### Workflow

```
P (controllable) --[MLP surrogate]--> C (children) --[ProT]--> Y (outcome)
                                                               ^
                                                       minimize (Y - Y*)^2
```

**Step 1 — Generate the SCM dataset:**

```bash
python optimization/p_to_c_to_y/scm_ds/datasets.py
```

This generates `data/example/ds.npz` (6000 samples from the paraboloid SCM) in `optimization/p_to_c_to_y/`.

**Step 2 — Train the ProT surrogate:**

```bash
# Train the surrogate — checkpoints are generated
python proT/training/trainer.py
```

Ensure the config used points to the correct data directory for this study.

**Step 3 — Train the P→C surrogate (MLP):**

```bash
python optimization/p_to_c_to_y/train_p_to_c_surrogate.py \
    --mode scm --dataset example \
    --data-root optimization/p_to_c_to_y/data \
    --output optimization/p_to_c_to_y/models/p_to_c.pt
```

**Step 4 — Run optimization:**

```bash
# Multi-seed experiment: CMA-ES + Adam, 20 seeds, target tracking + minimization
python optimization/p_to_c_to_y/run_multiseed_experiment.py

```

**Step 5 — Visualize:**

```bash
# Y surface: SCM ground truth vs trained model vs dataset
python optimization/p_to_c_to_y/plots.py \
    --checkpoint optimization/p_to_c_to_y/experiments/example/k_0/checkpoints/best_checkpoint.ckpt \
    --data-dir optimization/p_to_c_to_y/data/example \
    --mode surface --show scm model data --target 0.7 --out plot_y.png
```

---

## Cluster usage

All scripts run identically on a SLURM cluster. Ensure `pip install -e .` has been run inside the cluster virtual environment, then submit the training and optimization scripts via your cluster's job scheduler.
