"""Rozklady losowania hiperparametrow drzew HDRF (wersja robocza, PRZED E0).

Losowane grupy (`randomize`):

- ``complexity``: tryb ograniczenia zlozonosci drzewa, losowany per drzewo.
  * tryb ``depth`` (prawd. ``depth_mode_prob``): max_depth na {d_min, ..., D_full},
    d_min = ceil(depth_low_coef * log2 n); d = D_full oznacza pelne drzewo (None).
    ``depth_distribution``: 'uniform' (jednostajnie) albo 'log_full'
    (log-jednostajnie z grubym ogonem przy D_full: D_full + 1 - d ~ log-jedn.).
    min_samples_leaf = 1.
  * tryb ``leaf``: min_samples_leaf log-jednostajne na {1, ..., leaf_max};
    leaf = 1 oznacza pelne drzewo. max_depth = None.
  D_full to szacowana glebokosc pelnego drzewa (mediana z kilku drzew pilotazowych,
  liczona w forest.fit). min_samples_split jest stale rowne 2.
- ``criterion``: jednostajnie z listy kryteriow.
- ``splitter``: z prawd. ``splitter_random_prob`` 'random' (losowy prog dla kazdej
  z K cech, jak w Extra Trees), w przeciwnym razie 'best'. Domyslnie 0, czyli
  zawsze 'best' (zachowanie jak przed dodaniem grupy).
- ``max_features`` (K, liczba cech w wezle): mieszanka F4 + F1. Z prawd.
  ``features_default_prob`` K = K0, w przeciwnym razie
  ln K ~ U[ln(K0/a), ln(K0*b)], K zaokraglone i obciete do [1, p].
  K0 = sqrt(p) albo p (klasyfikacja, ``classification_k0``), p albo p/3
  (regresja, ``regression_k0``). Przy K0 = p gorna czesc rozkladu jest obcinana
  do p, wiec K lezy w [p/a, p] z duza masa w p.
- ``max_samples`` (BR, bootstrap ratio): jednostajne na ``bootstrap_bounds``;
  bootstrap ze zwracaniem o rozmiarze BR * n (BR moze byc > 1).

Parametry nielosowane maja wartosci domyslne scikit-learn albo wartosci podane
jawnie w ``fixed`` (np. srodek z hdrf.center.empirical_center).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np


HYPERPARAMETERS = ("complexity", "criterion", "max_features", "max_samples", "splitter")

# Klucze, ktore mozna ustawic jawnie dla parametrow nielosowanych.
FIXED_KEYS = ("max_depth", "min_samples_leaf", "max_leaf_nodes", "criterion", "max_features",
              "max_samples", "splitter")


@dataclass(frozen=True)
class SamplingConfig:
    """Stale rozkladow; wartosci sa robocze i do kalibracji w E0."""

    # complexity
    depth_mode_prob: float = 0.5
    depth_low_coef: float = 1.0
    depth_distribution: Literal["uniform", "log_full"] = "uniform"
    leaf_max: int = 10
    n_pilot_trees: int = 5
    # criterion
    classification_criteria: tuple[str, ...] = ("gini", "entropy")
    regression_criteria: tuple[str, ...] = ("squared_error", "friedman_mse")
    # splitter
    splitter_random_prob: float = 0.0
    # max_features
    features_default_prob: float = 0.3
    features_down: float = 2.0
    features_up: float = 2.0
    regression_k0: Literal["p", "p/3"] = "p"
    classification_k0: Literal["sqrt", "p"] = "sqrt"
    # max_samples
    bootstrap_bounds: tuple[float, float] = (0.7, 1.0)

    def __post_init__(self):
        for name in ("depth_mode_prob", "features_default_prob", "splitter_random_prob"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} musi byc w [0, 1]")
        if self.depth_low_coef <= 0:
            raise ValueError("depth_low_coef musi byc dodatnie")
        if self.depth_distribution not in ("uniform", "log_full"):
            raise ValueError("depth_distribution musi byc 'uniform' albo 'log_full'")
        if not isinstance(self.leaf_max, int) or self.leaf_max < 1:
            raise ValueError("leaf_max musi byc calkowite >= 1")
        if not isinstance(self.n_pilot_trees, int) or self.n_pilot_trees < 1:
            raise ValueError("n_pilot_trees musi byc calkowite >= 1")
        if self.features_down < 1 or self.features_up < 1:
            raise ValueError("features_down i features_up musza byc >= 1")
        if self.classification_k0 not in ("sqrt", "p"):
            raise ValueError("classification_k0 musi byc 'sqrt' albo 'p'")
        if self.regression_k0 not in ("p", "p/3"):
            raise ValueError("regression_k0 musi byc 'p' albo 'p/3'")
        low, high = self.bootstrap_bounds
        if not 0 < low <= high:
            raise ValueError("bootstrap_bounds musi byc dodatnie i rosnace")
        if not self.classification_criteria or not self.regression_criteria:
            raise ValueError("Listy criterion nie moga byc puste")


def _check_task(task, n_samples, n_features):
    if task not in ("classification", "regression"):
        raise ValueError("task musi byc 'classification' albo 'regression'")
    if n_samples < 2 or n_features < 1:
        raise ValueError("Niepoprawny rozmiar zbioru")


def sklearn_max_features(task: str, n_features: int) -> int:
    """K w domyslnym RF scikit-learn: floor(sqrt(p)) albo p."""
    if task == "classification":
        return max(1, int(np.sqrt(n_features)))
    return n_features


def k0(task: str, n_features: int, config: SamplingConfig) -> float:
    """Wartosc odniesienia K0 (rzeczywista, srodek rozkladu w skali log)."""
    if task == "classification":
        return float(np.sqrt(n_features)) if config.classification_k0 == "sqrt" else float(n_features)
    return float(n_features) if config.regression_k0 == "p" else n_features / 3.0


def default_params(task: str, n_samples: int, n_features: int) -> dict:
    """Domyslne hiperparametry RF ze scikit-learn (K jako liczba cech)."""
    _check_task(task, n_samples, n_features)
    return {
        "mode": None,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_leaf_nodes": None,
        "criterion": "gini" if task == "classification" else "squared_error",
        "splitter": "best",
        "max_features": sklearn_max_features(task, n_features),
        "max_samples": 1.0,
    }


def depth_bounds(n_samples: int, d_full: int, config: SamplingConfig) -> tuple[int, int]:
    """Zakres {d_min, ..., d_max} trybu depth; d_max = D_full oznacza pelne drzewo."""
    d_full = max(1, int(d_full))
    d_min = int(np.ceil(config.depth_low_coef * np.log2(n_samples)))
    d_min = min(max(1, d_min), d_full)
    return d_min, d_full


def log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    """Calkowita log-jednostajna na {low, ..., high}: P(k) ~ ln((k+1)/k)."""
    if low == high:
        return int(low)
    value = np.exp(rng.uniform(np.log(low), np.log(high + 1)))
    return int(min(high, np.floor(value)))


def sample_features(rng: np.random.Generator, task: str, n_features: int,
                    config: SamplingConfig) -> int:
    base = k0(task, n_features, config)
    if rng.random() < config.features_default_prob:
        value = base
    else:
        value = np.exp(rng.uniform(np.log(base / config.features_down),
                                   np.log(base * config.features_up)))
    return int(np.clip(np.rint(value), 1, n_features))


def sample_params(
    rng: np.random.Generator | Mapping[str, np.random.Generator],
    task: str,
    n_samples: int,
    n_features: int,
    randomize: tuple[str, ...],
    config: SamplingConfig | None = None,
    d_full: int | None = None,
    fixed: Mapping | None = None,
) -> dict:
    """Konfiguracja jednego drzewa. RNG per grupa (slownik) dla kontrolowanych ablacji."""
    config = SamplingConfig() if config is None else config
    params = default_params(task, n_samples, n_features)
    if fixed:
        unknown = set(fixed).difference(FIXED_KEYS)
        if unknown:
            raise ValueError(f"Nieznane klucze fixed_params: {sorted(unknown)}")
        params.update(fixed)

    enabled = set(randomize)
    unknown = enabled.difference(HYPERPARAMETERS)
    if unknown:
        raise ValueError(f"Nieznane hiperparametry: {sorted(unknown)}")

    def source(name):
        return rng[name] if isinstance(rng, Mapping) else rng

    if "complexity" in enabled:
        if d_full is None:
            raise ValueError("Losowanie complexity wymaga d_full")
        r = source("complexity")
        if r.random() < config.depth_mode_prob:
            low, high = depth_bounds(n_samples, d_full, config)
            if config.depth_distribution == "uniform":
                depth = int(r.integers(low, high + 1))
            else:
                depth = high + 1 - log_uniform_int(r, 1, high - low + 1)
            params.update(mode="depth", min_samples_leaf=1,
                          max_depth=None if depth >= high else depth)
        else:
            params.update(mode="leaf", max_depth=None,
                          min_samples_leaf=log_uniform_int(r, 1, config.leaf_max))
    if "criterion" in enabled:
        choices = (config.classification_criteria if task == "classification"
                   else config.regression_criteria)
        params["criterion"] = str(source("criterion").choice(choices))
    if "splitter" in enabled:
        random = source("splitter").random() < config.splitter_random_prob
        params["splitter"] = "random" if random else "best"
    if "max_features" in enabled:
        params["max_features"] = sample_features(
            source("max_features"), task, n_features, config)
    if "max_samples" in enabled:
        params["max_samples"] = float(source("max_samples").uniform(*config.bootstrap_bounds))
    return params


def tree_kwargs(params: Mapping) -> dict:
    """Argumenty dla DecisionTree* (bez max_samples i trybu)."""
    return {k: v for k, v in params.items() if k not in ("max_samples", "mode")}
