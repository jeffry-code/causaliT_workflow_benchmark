
# =============================================================================
# predictors.py  —  surrogate predictor interfaces
# -----------------------------------------------------------------------------
# Defines the Predictor interface (P -> scalar) and concrete implementations:
#
#   SCMPredictor      wraps a ground-truth SCM callable (used for testing)
#   ModelPredictor    wraps a trained proT TransformerForecaster checkpoint
#                     — handles tokenisation, normalisation, and de-standardisation
#                       so callers just pass a raw P vector and get back Y
#
# ModelPredictor also supports autograd-Adam by exposing a differentiable
# torch forward path (predict_torch). This is what run_with_model_template.py
# uses when optimizer="autograd".
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@dataclass
class Bounds:
    lower: np.ndarray
    upper: np.ndarray

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(self.upper, np.maximum(self.lower, x))


class Predictor:
    """Abstract predictor interface mapping P -> scalar objective (higher is better by convention)."""
    def __call__(self, P: np.ndarray) -> float:
        raise NotImplementedError


class SCMPredictor(Predictor):
    """Wraps an SCM 'ground truth' callable f(P) -> Y (scalar)."""
    def __init__(self, fn: Callable[[np.ndarray], float]):
        self.fn = fn

    def __call__(self, P: np.ndarray) -> float:
        return float(self.fn(P))


class ModelPredictor(Predictor):
    """
    Wraps a proT Transformer checkpoint. Minimal version:
    - You must pass a callable builder that converts a parameter vector P -> model input tensor.
    - You must pass a callable runner that executes the model and returns scalar Y_hat.
    This avoids tight coupling to training specifics and lets you iterate fast.
    """
    def __init__(
        self,
        build_input: Callable[[np.ndarray], "torch.Tensor"],
        run_model: Callable[["torch.Tensor"], "torch.Tensor"],
        device: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for ModelPredictor.")
        import torch  # local import
        self.torch = torch
        self.build_input = build_input
        self.run_model = run_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, P: np.ndarray) -> float:
        x = self.build_input(P)  # shape (1, L_in, D)
        y_hat = self.run_model(x.to(self.device))  # expect shape (1, 1) or (1, 1, *)
        y_scalar = y_hat.reshape(-1)[0].detach().cpu().item()
        # return a plain Python float so this wrapper can be used by optimizer code
        return float(y_scalar)
