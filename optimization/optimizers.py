
# =============================================================================
# optimizers.py  —  shared optimizer back-ends
# -----------------------------------------------------------------------------
# Provides two black-box optimizers used across all Workflow I experiments:
#
#   cma_es(objective, x0, sigma0, bounds, max_iters)
#       CMA-ES via the `pycma` library (gradient-free, population-based).
#       Robust to multimodal / noisy surrogates. Population size grows with d.
#       Returns an OptResult with x_best, f_best, n_eval, and history.
#
#   adam(objective, x0, bounds, lr, iters)
#       Adam with finite-difference gradients (gradient-free interface).
#       This is a separate optimizer implementation from the differentiable
#       autograd-based Adam loops that are used when the model supports
#       direct PyTorch gradients.
#
# Both optimizers are imported by run_with_model_template.py and the
# p_to_c_to_y scripts. Do not rename the public functions.
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any
import numpy as np

@dataclass
class OptResult:
    x_best: np.ndarray
    f_best: float
    n_eval: int
    history: list[tuple[float, float]]  # (f, ||x||) log


# Population-based gradient-free optimizer. Robust to multimodal/noisy surrogates; population size scales with d.
def cma_es(objective: Callable[[np.ndarray], float],
           x0: np.ndarray,
           sigma0: float,
           bounds: tuple[np.ndarray, np.ndarray] | None = None,
           max_iters: int = 200,
           tol: float = 1e-8) -> OptResult:
    """
    CMA-ES via `pycma`.
    This function is intentionally strict: if `pycma` is missing,
    it raises an error instead of silently falling back to another optimizer.
    """
    try:
        import cma
    except ImportError as exc:
        raise RuntimeError(
            "CMA-ES requested but dependency `pycma` is not installed. "
            "Install it with: pip install cma"
        ) from exc

    print(f"[cma_es] using pycma (sigma0={sigma0}, max_iters={max_iters})")

    try:
        opts = {"verb_disp": 0}
        if bounds is not None:
            opts["bounds"] = [bounds[0].tolist(), bounds[1].tolist()]
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
        history = []
        f_best = float("inf")
        x_best = x0.copy()
        n_eval = 0
        for _ in range(max_iters):
            xs = es.ask()
            fs = [objective(np.array(x, dtype=float)) for x in xs]
            n_eval += len(fs)
            es.tell(xs, fs)
            es.disp()
            # choose the best offspring from this generation
            idx = int(np.argmin(fs))
            f_iter = float(fs[idx])
            if f_iter < f_best:
                f_best = f_iter
                x_best = np.array(xs[idx], dtype=float)
            history.append((f_best, float(np.linalg.norm(x_best))))
            if es.stop():
                break
        return OptResult(x_best=x_best, f_best=f_best, n_eval=n_eval, history=history)
    except Exception as exc:
        raise RuntimeError(f"pycma failed during optimization: {exc}") from exc


# Gradient-free Adam via central finite differences. Use when autograd through the model is unavailable.
def adam(objective: Callable[[np.ndarray], float],
         x0: np.ndarray,
         bounds: tuple[np.ndarray, np.ndarray] | None = None,
         lr: float = 0.05,
         iters: int = 500,
         eps: float = 1e-8,
         beta1: float = 0.9,
         beta2: float = 0.999) -> OptResult:
    """
    Gradient-free Adam using finite differences (central) for small dimensions.
    If you have a differentiable torch model, prefer a true autograd loop.
    """
    # Central finite-difference gradient estimate for a black-box objective.
    # This is expensive in dimension d because it requires 2*d objective calls.
    def grad_fd(x: np.ndarray, h: float = 1e-4) -> np.ndarray:
        g = np.zeros_like(x)
        f0 = objective(x)
        for i in range(x.size):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (objective(xp) - objective(xm)) / (2*h)
        return g

    x = x0.copy().astype(float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    f_best = objective(x)
    x_best = x.copy()
    history = [(f_best, float(np.linalg.norm(x_best)))]
    for t in range(1, iters+1):
        g = grad_fd(x)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g*g)
        m_hat = m/(1-beta1**t)
        v_hat = v/(1-beta2**t)
        x = x - lr*m_hat/(np.sqrt(v_hat)+eps)
        if bounds is not None:
            lb, ub = bounds
            x = np.clip(x, lb, ub)
        # Evaluate the new candidate and update the best-so-far record.
        f = objective(x)
        if f < f_best:
            f_best = f
            x_best = x.copy()
        history.append((f_best, float(np.linalg.norm(x_best))))
    return OptResult(x_best=x_best, f_best=f_best, n_eval=len(history), history=history)
