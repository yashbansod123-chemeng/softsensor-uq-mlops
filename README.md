# softsensor-uq-mlops

![ci](https://github.com/yashbansod123-chemeng/softsensor-uq-mlops/actions/workflows/ci.yml/badge.svg)

An uncertainty-aware soft sensor built on **mean-variance estimation (MVEM)**: the network predicts both
a value and its variance, so every prediction carries a 95% interval. The project is packaged as a
reproducible MLOps pipeline: a Bash entry point, a Docker image, and a GitHub Actions workflow that
tests, trains, quality-gates and publishes on every push.

The demo uses the public [UCI Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294)
dataset (9,568 rows; ambient temperature, exhaust vacuum, pressure and humidity → net electrical
output in MW), a reasonable stand-in for an industrial soft-sensing problem. A synthetic
heteroscedastic dataset is included for fast tests.

## Pipeline

```
push ─► lint (ruff) ─► unit + pipeline tests (pytest) ─► shell lint (shellcheck)
     ─► docker build ─► train in container ─► quality gate ─► upload model + metrics
     ─► (main only) push image to GHCR, tagged with commit SHA
```

The gate runs on a held-out test split that is never used for training or early stopping. A model is
rejected if **test RMSE exceeds a threshold** or if **95% interval coverage (PICP) falls outside
[0.90, 0.99]**. The gate checks calibration, not only accuracy: a model with good RMSE but intervals
that cover 70% of outcomes would mislead anyone sizing decisions to its confidence, and a model whose
intervals cover 100% is likely too wide to be useful.

## Method

Training runs in three phases:

1. **Mean**: the mean network is trained alone with MSE.
2. **Variance**: the mean network is frozen and the variance network is fitted with Gaussian NLL on a
   separate calibration split, so it learns from out-of-sample residuals rather than training residuals.
3. **Joint**: both networks are fine-tuned together on train + calibration data.

Variance architectures (`--arch`):

| arch | variance network input |
|---|---|
| `original` | inputs only (independent mean and variance networks) |
| `mean_informed` | inputs + predicted mean |
| `mean_and_hidden` | inputs + predicted mean + mean network's last hidden layer |
| `residual` | as above, through residual blocks |
| `attention` | as above, with feature-wise self-attention and an error-aware scaling term |

The loss is Gaussian NLL, with optional calibration, confidence and adaptive penalty terms
(`--lambda-*`, all 0 by default).

Data splits: 60% train, 15% calibration, 10% validation (early stopping), 15% test (gate only).
Scalers are fitted on the training split only.

## Run it

Locally:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
bash scripts/run_pipeline.sh -d ccpp -a mean_and_hidden -e 200
```

In Docker:
```bash
docker build -t softsensor .
mkdir -p runs && chmod 777 runs
docker run --rm -v "$PWD/runs:/app/runs" softsensor -d ccpp -a attention -e 200
```

`run_pipeline.sh` options: `-d` dataset, `-a` architecture, `-e` epochs, `-r` max RMSE for the gate,
`-o` output root. Anything after `--` is passed to the Python CLI (see `python -m softsensor.cli -h`).

Each run writes `model.pt`, `metrics.json` (config, scalers, per-split metrics, gate result, git SHA),
`loss_history.csv` and `pipeline.log` to `runs/<dataset>-<arch>-<sha>/`.

Exit codes: `0` passed, `2` rejected by the gate, `1` error.

## Results

| arch | test RMSE (MW) | PICP (95%) | PINAW |
|---|---|---|---|
| _fill in from the first CI run_ | | | |

## Layout

```
src/softsensor/   data.py  models.py  loss.py  train.py  evaluate.py  cli.py
tests/            test_softsensor.py
scripts/          run_pipeline.sh
Dockerfile  .github/workflows/ci.yml  pyproject.toml
```
