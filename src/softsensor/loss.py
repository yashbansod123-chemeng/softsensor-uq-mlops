"""MVEM loss: Gaussian NLL plus optional calibration, confidence and adaptive penalties.

With all lambdas at 0 (the CLI default) this reduces to the standard heteroscedastic
Gaussian negative log-likelihood.
"""

import torch
import torch.nn as nn


class MVEMLoss(nn.Module):
    def __init__(self, lambda_calibration=0.0, lambda_confidence=0.0, lambda_adaptive=0.0):
        super().__init__()
        self.lambda_calibration = lambda_calibration
        self.lambda_confidence = lambda_confidence
        self.lambda_adaptive = lambda_adaptive

    def forward(self, mean_pred, var_pred, targets):
        var_pred = torch.clamp(var_pred, min=1e-6, max=1e6)
        std_pred = torch.sqrt(var_pred)

        # Gaussian negative log-likelihood (constant term dropped)
        nll_loss = 0.5 * torch.mean(torch.log(var_pred) + (targets - mean_pred) ** 2 / var_pred)

        abs_error = torch.abs(targets - mean_pred)

        # Calibration loss: penalise deviation from Gaussian 1/2/3-sigma coverage.
        # NOTE: the indicator functions are piecewise constant, so this term contributes
        # no gradient; it is kept for parity with the research code and for reporting.
        cov_1 = torch.mean((abs_error <= std_pred).to(torch.float32))
        cov_2 = torch.mean((abs_error <= 2 * std_pred).to(torch.float32))
        cov_3 = torch.mean((abs_error <= 3 * std_pred).to(torch.float32))
        calibration_loss = (
            torch.abs(cov_1 - 0.68) + torch.abs(cov_2 - 0.95) + 0.5 * torch.abs(cov_3 - 0.997)
        )

        # Confidence penalty: underconfidence on small errors, overconfidence on large errors
        small_errors = (abs_error < 0.5 * std_pred).to(torch.float32)
        underconfidence_penalty = torch.mean(small_errors * std_pred)
        large_errors = (abs_error > 2 * std_pred).to(torch.float32)
        overconfidence_penalty = torch.mean(large_errors * torch.exp(-std_pred))
        confidence_penalty = overconfidence_penalty + 0.1 * underconfidence_penalty

        # Adaptive penalty: encourage positive correlation between |error| and predicted std
        error_norm = (abs_error - abs_error.mean()) / (abs_error.std() + 1e-8)
        std_norm = (std_pred - std_pred.mean()) / (std_pred.std() + 1e-8)
        adaptive_penalty = torch.exp(-torch.mean(error_norm * std_norm))

        return (
            nll_loss
            + self.lambda_calibration * calibration_loss
            + self.lambda_confidence * confidence_penalty
            + self.lambda_adaptive * adaptive_penalty
        )
