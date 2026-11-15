"""Konfigurowalne rozklady HDRF (wersja robocza, PRZED eksperymentem E0).

Parametr `max_samples` oznacza tutaj *bootstrap ratio*, takze powyzej 1.0.
Nie jest to to samo ograniczenie co max_samples w sklearn.RandomForest.
"""

from dataclasses import dataclass
from typing import Literal
from collections.abc import Mapping

import numpy as np


HYPERPARAMETERS = (
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "criterion",
    "max_features",
    "max_samples",
)


@dataclass(frozen=True)
class SamplingConfig:
    """Parametry rozkladow i ROBOCZYCH wartosci centralnych (do kalibracji w E0)."""

    depth_unlimited_prob: float = 0.20
    depth_offsets: tuple[int, int] = (-2, 4)  # wlacznie z gorna granica
    split_bounds: tuple[int, int] = (2, 20)
    leaf_bounds: tuple[int, int] = (1, 10)
    classification_criteria: tuple[str, ...] = ("gini", "entropy")
    regression_criteria: tuple[str, ...] = ("squared_error", "friedman_mse")
    features_default_prob: float = 0.50
    features_bounds: tuple[float, float] = (0.50, 1.00)
    bootstrap_bounds: tuple[float, float] = (0.70, 1.00)
    fixed_params_mode: Literal["center", "default"] = "center"

    # Punkty centralne na potrzeby porownan. Jeszcze NIE sa medianami z E0.
    center_depth_offset: int = 2
    center_split: int = 6
    center_leaf: int = 3
    center_classification_features: str | float = "sqrt"
    center_regression_features: str | float = 1.0
    center_bootstrap: float = 0.85

    def __post_init__(self):
        if self.fixed_params_mode not in ("center", "default"):
            raise ValueError(
                "fixed_params_mode musi byc 'center' albo 'default'"
            )
        for name in ("depth_unlimited_prob", "features_default_prob"):
            prob = getattr(self, name)
            if not 0 <= prob <= 1:
                raise ValueError(f"{name} musi byc w [0, 1]")
        if not (isinstance(self.depth_offsets, tuple) and
                len(self.depth_offsets) == 2 and
                all(isinstance(v, int) for v in self.depth_offsets) and
                self.depth_offsets[0] <= self.depth_offsets[1]):
            raise ValueError("depth_offsets musi byc para rosnacych liczb calkowitych")
        for name, minimum in (("split_bounds", 2), ("leaf_bounds", 1)):
            low, high = getattr(self, name)
            if not (isinstance(low, int) and isinstance(high, int)
                    and minimum <= low <= high):
                raise ValueError(f"{name} musi miec calkowite granice >= {minimum}")
        low, high = self.features_bounds
        if not 0 < low <= high <= 1:
            raise ValueError("features_bounds musi byc w przedziale (0, 1]")
        low, high = self.bootstrap_bounds
        if not 0 < low <= high:
            raise ValueError("bootstrap_bounds musi byc dodatnie i rosnace")
        if not self.classification_criteria or not self.regression_criteria:
            raise ValueError("Listy criterion nie moga byc puste")
        if self.center_split < 2 or self.center_leaf < 1 or self.center_bootstrap <= 0:
            raise ValueError("Niepoprawne centralne hiperparametry")


def center_params(
    task: str,
    n_samples: int,
    n_features: int,
    config: SamplingConfig | None = None,
) -> dict:
    """Robocze punkty centralne; osobne od rozkladow do pozniejszej kalibracji."""
    if config is None:
        config = SamplingConfig()
    if task not in ("classification", "regression"):
        raise ValueError("task musi byc 'classification' albo 'regression'")
    if n_samples < 2 or n_features < 1:
        raise ValueError("Niepoprawny rozmiar zbioru")
    anchor = int(np.ceil(np.log2(n_samples)))
    return {
        "max_depth": max(1, anchor + config.center_depth_offset),
        "min_samples_split": config.center_split,
        "min_samples_leaf": config.center_leaf,
        "criterion": "gini" if task == "classification" else "squared_error",
        "max_features": (
            config.center_classification_features if task == "classification"
            else config.center_regression_features
        ),
        "max_samples": config.center_bootstrap,
    }


def default_params(
    task: str,
    n_samples: int,
    n_features: int
) -> dict:
    """Domyslne hiperparametry RF ze scikit-learn."""

    if task not in ("classification", "regression"):
        raise ValueError(
            "task musi byc 'classification' albo 'regression'"
        )

    if n_samples < 2 or n_features < 1:
        raise ValueError("Niepoprawny rozmiar zbioru")

    return {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": (
            "gini"
            if task == "classification"
            else "squared_error"
        ),
        "max_features": (
            "sqrt"
            if task == "classification"
            else 1.0
        ),
        "max_samples": 1.0,
    }


def _log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    if low == high:
        return low
    return int(np.clip(np.rint(np.exp(rng.uniform(np.log(low), np.log(high)))), low, high))


def sample_params(
    rng: np.random.Generator | Mapping[str, np.random.Generator],
    task: str,
    n_samples: int,
    n_features: int,
    randomize: tuple[str, ...],
    config: SamplingConfig | None = None,
) -> dict:
    """Jedna konfiguracja drzewa; RNG per parametr dla kontrolowanych ablacji.

    Dla kompatybilnosci przyjmujemy takze pojedynczy Generator (stare API).
    """

    if config is None:
        config = SamplingConfig()

    if config.fixed_params_mode == "default":
        params = default_params(
            task, n_samples, n_features
        )
    else:
        params = center_params(
            task, n_samples, n_features, config
        )

    enabled = set(randomize)

    unknown = enabled.difference(HYPERPARAMETERS)
    if unknown:
        raise ValueError(f"Nieznane hiperparametry: {sorted(unknown)}")

    def source(name):
        return rng[name] if isinstance(rng, Mapping) else rng

    anchor = int(np.ceil(np.log2(n_samples)))

    if "max_depth" in enabled:
        r = source("max_depth")
        if r.random() < config.depth_unlimited_prob:
            params["max_depth"] = None
        else:
            low = max(2, anchor + config.depth_offsets[0])
            high = max(low, anchor + config.depth_offsets[1])
            params["max_depth"] = int(r.integers(low, high + 1))

    if "min_samples_split" in enabled:
        params["min_samples_split"] = _log_uniform_int(
            source("min_samples_split"), *config.split_bounds
        )
    if "min_samples_leaf" in enabled:
        params["min_samples_leaf"] = _log_uniform_int(
            source("min_samples_leaf"), *config.leaf_bounds
        )
    if "criterion" in enabled:
        choices = (config.classification_criteria if task == "classification"
                   else config.regression_criteria)
        params["criterion"] = str(source("criterion").choice(choices))
    if "max_features" in enabled:
        r = source("max_features")
        if r.random() >= config.features_default_prob:
            low, high = config.features_bounds
            params["max_features"] = float(r.uniform(low, high))
    if "max_samples" in enabled:
        low, high = config.bootstrap_bounds
        params["max_samples"] = float(source("max_samples").uniform(low, high))
    return params
