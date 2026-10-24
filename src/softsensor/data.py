"""Datasets, splitting and scaling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CCPP_UCI_ID = 294  # UCI Combined Cycle Power Plant
CCPP_INPUTS = ["AT", "V", "AP", "RH"]
CCPP_OUTPUTS = ["PE"]


def load_ccpp(cache_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """UCI CCPP: ambient temperature, exhaust vacuum, pressure, humidity -> net output (MW)."""
    cache = Path(cache_dir) / "ccpp.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=CCPP_UCI_ID)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df[CCPP_INPUTS], df[CCPP_OUTPUTS]


def make_synthetic(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Heteroscedastic toy problem: noise std grows with |x1|, so a good MVEM must learn it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 2))
    noise_std = 0.1 + 0.4 * np.abs(x[:, 0])
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + rng.normal(0, noise_std)
    return pd.DataFrame(x, columns=["x1", "x2"]), pd.DataFrame({"y": y})


def load_dataset(name: str, cache_dir: str | Path = "data", seed: int = 0):
    if name == "ccpp":
        return load_ccpp(cache_dir)
    if name == "synthetic":
        return make_synthetic(seed=seed)
    raise ValueError(f"Unknown dataset '{name}'. Choose 'ccpp' or 'synthetic'.")


@dataclass
class Splits:
    """train: mean network + joint phase; cal: variance phase; val: early stopping; test: final gate."""

    train: np.ndarray
    cal: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_indices(n: int, fractions=(0.6, 0.15, 0.1, 0.15), seed: int = 0) -> Splits:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1")
    idx = np.random.default_rng(seed).permutation(n)
    cuts = np.cumsum([int(round(f * n)) for f in fractions[:-1]])
    return Splits(*np.split(idx, cuts))


@dataclass
class StandardScaler:
    shift: np.ndarray
    factor: np.ndarray

    @classmethod
    def fit(cls, a: np.ndarray) -> "StandardScaler":
        std = a.std(axis=0)
        return cls(shift=a.mean(axis=0), factor=np.where(std == 0, 1.0, std))

    def transform(self, a):
        return (a - self.shift) / self.factor

    def inverse(self, a):
        return a * self.factor + self.shift

    def inverse_variance(self, var):
        """If y = (y_orig - shift) / factor then var_orig = var * factor^2."""
        return var * self.factor**2

    def to_dict(self):
        return {"shift": self.shift.tolist(), "factor": self.factor.tolist()}
