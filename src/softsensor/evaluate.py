"""Accuracy and uncertainty metrics in original engineering units, plus the CI quality gate."""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import torch


def predict(model, x: torch.Tensor, batch_size: int = 4096):
    model.eval()
    means, variances = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            m, v = model(x[i : i + batch_size])
            means.append(m)
            variances.append(v)
    return torch.cat(means).numpy(), torch.cat(variances).numpy()


def uq_metrics(y, mean, std, confidence: float = 0.95) -> dict:
    """Per-output metrics. y, mean, std are arrays of shape [N, n_out] in original units."""
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    lo, hi = mean - z * std, mean + z * std
    err = mean - y
    width = hi - lo
    span = y.max(axis=0) - y.min(axis=0)
    var = np.maximum(std**2, 1e-12)
    nll = 0.5 * (np.log(2 * math.pi * var) + err**2 / var)
    return {
        "rmse": np.sqrt(np.mean(err**2, axis=0)).tolist(),
        "mae": np.mean(np.abs(err), axis=0).tolist(),
        "picp": np.mean((y >= lo) & (y <= hi), axis=0).tolist(),  # coverage
        "piaw": np.mean(width, axis=0).tolist(),  # average interval width
        "pinaw": (np.mean(width, axis=0) / np.where(span == 0, 1, span)).tolist(),  # normalised width
        "nll": np.mean(nll, axis=0).tolist(),
        "confidence": confidence,
    }


@dataclass
class Gate:
    max_rmse: float
    picp_min: float = 0.90
    picp_max: float = 0.99

    def check(self, metrics: dict) -> list[str]:
        """Returns a list of failure messages; empty means the model passes."""
        failures = []
        for i, (rmse, picp) in enumerate(zip(metrics["rmse"], metrics["picp"], strict=True)):
            if rmse > self.max_rmse:
                failures.append(f"output {i}: RMSE {rmse:.4f} > {self.max_rmse}")
            if not self.picp_min <= picp <= self.picp_max:
                failures.append(f"output {i}: PICP {picp:.3f} outside [{self.picp_min}, {self.picp_max}]")
        return failures
