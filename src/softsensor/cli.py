"""Command-line entry point: load data -> train MVEM -> evaluate -> quality gate -> save artefacts.

Exit codes: 0 = trained and passed the gate, 1 = error, 2 = trained but failed the gate.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .data import StandardScaler, load_dataset, split_indices
from .evaluate import Gate, predict, uq_metrics
from .models import ARCHITECTURES, build_mvem
from .train import TrainConfig, train_mvem

log = logging.getLogger("softsensor")
GATE_FAILED = 2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["ccpp", "synthetic"], default="ccpp")
    p.add_argument("--arch", choices=ARCHITECTURES, default="mean_and_hidden")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--activation", default="tanh")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--skip-variance", action="store_true", help="skip phase II (variance-only training)")
    p.add_argument("--lambda-calibration", type=float, default=0.0)
    p.add_argument("--lambda-confidence", type=float, default=0.0)
    p.add_argument("--lambda-adaptive", type=float, default=0.0)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--max-rmse", type=float, required=True, help="gate: max test RMSE, original units")
    p.add_argument("--picp-min", type=float, default=0.90)
    p.add_argument("--picp-max", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="runs/latest")
    return p.parse_args(argv)


def run(args) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.dataset, args.data_dir, seed=args.seed)
    log.info("dataset=%s  rows=%d  inputs=%s  outputs=%s",
             args.dataset, len(X), list(X.columns), list(y.columns))
    X_np, y_np = X.to_numpy(np.float32), y.to_numpy(np.float32)

    s = split_indices(len(X_np), seed=args.seed)
    x_scaler, y_scaler = StandardScaler.fit(X_np[s.train]), StandardScaler.fit(y_np[s.train])  # train only

    def tensors(idx):
        return (torch.tensor(x_scaler.transform(X_np[idx]), dtype=torch.float32),
                torch.tensor(y_scaler.transform(y_np[idx]), dtype=torch.float32))

    data = {}
    for name in ("train", "cal", "val", "test"):
        data[f"x_{name}"], data[f"y_{name}"] = tensors(getattr(s, name))
    log.info("split sizes: train=%d cal=%d val=%d test=%d", len(s.train), len(s.cal), len(s.val), len(s.test))

    model = build_mvem(args.arch, X_np.shape[1], y_np.shape[1], args.hidden, args.activation)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                      skip_variance=args.skip_variance, lambda_calibration=args.lambda_calibration,
                      lambda_confidence=args.lambda_confidence, lambda_adaptive=args.lambda_adaptive,
                      seed=args.seed)
    t0 = time.time()
    history = train_mvem(model, data, cfg)
    train_seconds = time.time() - t0

    # Evaluate every split in original units; the gate uses the untouched test split only
    metrics = {}
    for name in ("train", "val", "test"):
        mean_s, var_s = predict(model, data[f"x_{name}"])
        metrics[name] = uq_metrics(y_np[getattr(s, name)], y_scaler.inverse(mean_s),
                                   np.sqrt(var_s), args.confidence)

    gate = Gate(args.max_rmse, args.picp_min, args.picp_max)
    failures = gate.check(metrics["test"])

    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "loss_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    report = {
        "version": __version__,
        "git_sha": os.environ.get("GIT_SHA", "unknown"),
        "dataset": args.dataset,
        "inputs": list(X.columns),
        "outputs": list(y.columns),
        "architecture": args.arch,
        "train_config": asdict(cfg),
        "hidden": args.hidden,
        "scalers": {"x": x_scaler.to_dict(), "y": y_scaler.to_dict()},
        "train_seconds": round(train_seconds, 1),
        "metrics": metrics,
        "gate": {**asdict(gate), "passed": not failures, "failures": failures},
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    t = metrics["test"]
    log.info("TEST  RMSE=%s  PICP=%s  PINAW=%s", _fmt(t["rmse"]), _fmt(t["picp"]), _fmt(t["pinaw"]))
    if failures:
        for msg in failures:
            log.error("GATE FAILED: %s", msg)
        return GATE_FAILED
    log.info("GATE PASSED. Artefacts in %s", out)
    return 0


def _fmt(values):
    return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        sys.exit(run(parse_args(argv)))
    except Exception:
        log.exception("pipeline error")
        sys.exit(1)


if __name__ == "__main__":
    main()
