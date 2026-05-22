
# =============================================================================
# objectives.py  —  objective function wrappers
# -----------------------------------------------------------------------------
# Wraps a predictor (P -> scalar) into a minimization objective.
# Supports two modes:
#   - maximize:  returns -predictor(P)  (default: maximize surrogate output)
#   - target:    returns (predictor(P) - target)^2  (hit a specific value)
#
# Used by run_with_model_template.py when building the optimization objective
# from a trained surrogate. Import: from objectives import Objective
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class Objective:
    """Turn a predictor (higher is better) into a minimization objective if needed."""
    predictor: Callable[[np.ndarray], float]
    maximize: bool = True
    target: float | None = None  # if provided, aim to approach target instead of raw max

    def __call__(self, P: np.ndarray) -> float:
        val = self.predictor(P)
        if self.target is not None:
            # minimize squared error to a specific target value
            loss = (val - self.target) ** 2
            return float(loss)
        # default behavior: convert maximization into minimization by negating
        return float(-val if self.maximize else val)
