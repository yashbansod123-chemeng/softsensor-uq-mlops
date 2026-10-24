"""Three-phase MVEM training: mean warm-up -> variance (mean frozen) -> joint fine-tuning."""

import copy
import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss import MVEMLoss

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 30
    l2: float = 1e-4
    grad_clip: float = 1.0
    skip_variance: bool = False
    lambda_calibration: float = 0.0
    lambda_confidence: float = 0.0
    lambda_adaptive: float = 0.0
    seed: int = 0


def _l2_penalty(params):
    return sum(torch.norm(p, 2) for p in params)


def _batched_eval(fn, x, y, batch_size):
    """Mean of a per-batch loss over a full dataset, weighted by batch size."""
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb, yb = x[i : i + batch_size], y[i : i + batch_size]
            total += fn(xb, yb).item() * len(xb)
    return total / len(x)


def fit(model, params, batch_loss, x_tr, y_tr, x_mon, y_mon, cfg: TrainConfig, phase: str):
    """Mini-batch training with early stopping on the monitor set; restores the best weights."""
    params = list(params)
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(cfg.patience // 3, 1),
                              min_lr=cfg.lr * 1e-2)
    gen = torch.Generator().manual_seed(cfg.seed)
    best, best_state, best_epoch, history = math.inf, copy.deepcopy(model.state_dict()), -1, []

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(x_tr), cfg.batch_size):
            b = perm[i : i + cfg.batch_size]
            opt.zero_grad()
            loss = batch_loss(x_tr[b], y_tr[b]) + cfg.l2 * _l2_penalty(params)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

        model.eval()
        tr = _batched_eval(batch_loss, x_tr, y_tr, cfg.batch_size)
        mon = _batched_eval(batch_loss, x_mon, y_mon, cfg.batch_size)
        history.append({"phase": phase, "epoch": epoch, "train_loss": tr, "monitor_loss": mon,
                         "lr": opt.param_groups[0]["lr"]})

        if not math.isfinite(mon):  # explosion guard: roll back to the best weights
            log.warning("%s epoch %d: non-finite loss, restoring best weights", phase, epoch)
            model.load_state_dict(best_state)
            continue
        sched.step(mon)
        if mon < best:
            best, best_state, best_epoch = mon, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= cfg.patience:
            log.info("%s: early stop at epoch %d (best %d, loss %.4f)", phase, epoch, best_epoch, best)
            break
        if epoch % 25 == 0:
            log.info("%s epoch %4d  train %.4f  monitor %.4f", phase, epoch, tr, mon)

    model.load_state_dict(best_state)
    log.info("%s: done, best monitor loss %.4f at epoch %d", phase, best, best_epoch)
    return history


def train_mvem(model, data: dict, cfg: TrainConfig):
    """``data`` holds scaled float32 tensors: x_train/y_train, x_cal/y_cal, x_val/y_val."""
    torch.manual_seed(cfg.seed)
    mvem_loss = MVEMLoss(cfg.lambda_calibration, cfg.lambda_confidence, cfg.lambda_adaptive)
    mse = nn.MSELoss()
    history = []

    # Phase I: mean network only (MSE). Replaces loading a pre-trained mean model.
    history += fit(model.mean_network, model.mean_network.parameters(),
                   lambda xb, yb: mse(model.mean_network(xb), yb),
                   data["x_train"], data["y_train"], data["x_val"], data["y_val"], cfg, "mean")

    def nll(xb, yb):
        mean, var = model(xb)
        return mvem_loss(mean, var, yb)

    # Phase II: variance network only, mean frozen, fitted on held-out (calibration) residuals
    if not cfg.skip_variance:
        for p in model.mean_network.parameters():
            p.requires_grad_(False)
        var_params = [p for p in model.parameters() if p.requires_grad]
        history += fit(model, var_params, nll, data["x_cal"], data["y_cal"],
                       data["x_val"], data["y_val"], cfg, "variance")
        for p in model.mean_network.parameters():
            p.requires_grad_(True)

    # Phase III: joint fine-tuning on train + calibration
    x_joint = torch.cat([data["x_train"], data["x_cal"]])
    y_joint = torch.cat([data["y_train"], data["y_cal"]])
    history += fit(model, model.parameters(), nll, x_joint, y_joint,
                   data["x_val"], data["y_val"], cfg, "joint")
    return history
