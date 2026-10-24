import json

import numpy as np
import pytest
import torch

from softsensor.cli import GATE_FAILED, parse_args, run
from softsensor.data import StandardScaler, make_synthetic, split_indices
from softsensor.evaluate import Gate, uq_metrics
from softsensor.loss import MVEMLoss
from softsensor.models import ARCHITECTURES, build_mvem


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_architectures_shapes_and_positive_variance(arch):
    torch.manual_seed(0)
    model = build_mvem(arch, n_in=4, n_out=2, hidden=[16, 8])
    mean, var = model(torch.randn(32, 4))
    assert mean.shape == var.shape == (32, 2)
    assert torch.all(var > 0) and torch.all(torch.isfinite(var))


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        build_mvem("nope", 2, 1)


def test_nll_minimised_by_true_variance():
    torch.manual_seed(0)
    y = torch.randn(20000, 1) * 2.0
    loss = MVEMLoss()
    mean = torch.zeros_like(y)
    losses = {s: loss(mean, torch.full_like(y, s**2), y).item() for s in (1.0, 2.0, 4.0)}
    assert losses[2.0] < losses[1.0] and losses[2.0] < losses[4.0]


def test_loss_with_penalties_is_finite_and_differentiable():
    mean = torch.randn(64, 1, requires_grad=True)
    var = torch.rand(64, 1, requires_grad=True)
    loss = MVEMLoss(0.2, 0.5, 0.3)(mean, var + 0.1, torch.randn(64, 1))
    loss.backward()
    assert torch.isfinite(loss) and mean.grad is not None and var.grad is not None


def test_splits_are_disjoint_and_complete():
    s = split_indices(1000, seed=1)
    all_idx = np.concatenate([s.train, s.cal, s.val, s.test])
    assert len(all_idx) == len(set(all_idx)) == 1000


def test_scaler_round_trip_and_variance_transform():
    a = np.random.default_rng(0).normal(5, 3, size=(500, 2))
    sc = StandardScaler.fit(a)
    np.testing.assert_allclose(sc.inverse(sc.transform(a)), a, rtol=1e-6)
    np.testing.assert_allclose(sc.inverse_variance(np.ones(2)), sc.factor**2)


def test_metrics_on_perfectly_calibrated_gaussian():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=(50000, 1))
    m = uq_metrics(y, np.zeros_like(y), np.ones_like(y), confidence=0.95)
    assert m["picp"][0] == pytest.approx(0.95, abs=0.01)
    assert m["rmse"][0] == pytest.approx(1.0, abs=0.02)


def test_gate_pass_and_fail():
    ok = {"rmse": [1.0], "picp": [0.95]}
    assert Gate(max_rmse=2.0).check(ok) == []
    assert len(Gate(max_rmse=0.5).check(ok)) == 1
    assert len(Gate(max_rmse=2.0).check({"rmse": [1.0], "picp": [0.70]})) == 1


def test_synthetic_is_heteroscedastic():
    X, y = make_synthetic(n=5000)
    resid = y["y"] - np.sin(X["x1"]) - 0.5 * X["x2"]
    assert resid[X["x1"].abs() > 2].std() > 2 * resid[X["x1"].abs() < 0.5].std()


def _cli(tmp_path, *extra):
    return parse_args(["--dataset", "synthetic", "--epochs", "15", "--patience", "5",
                       "--hidden", "16", "16", "--out", str(tmp_path), *extra])


def test_end_to_end_pipeline_writes_artefacts(tmp_path):
    code = run(_cli(tmp_path, "--max-rmse", "10"))
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert (tmp_path / "model.pt").exists() and (tmp_path / "loss_history.csv").exists()
    assert set(report["metrics"]) == {"train", "val", "test"}
    assert code in (0, GATE_FAILED) and report["gate"]["passed"] == (code == 0)


def test_gate_blocks_bad_model(tmp_path):
    assert run(_cli(tmp_path, "--max-rmse", "1e-6")) == GATE_FAILED
