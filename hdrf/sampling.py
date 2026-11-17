"""Rozklady losowania hiperparametrow drzew HDRF (wariant A_p z pilotazy E0).

Losowane grupy (`randomize`), kazda z osobnego strumienia losowego:

- ``complexity``: tryb ograniczenia zlozonosci drzewa, losowany per drzewo.
  * tryb ``depth`` (prawd. ``depth_mode_prob``): max_depth na {d_min, ..., D_full},
    d_min = ceil(depth_low_coef * log2 n); d = D_full oznacza pelne drzewo (None).
    ``depth_distribution``: 'log_full' (log-jednostajnie z grubym ogonem przy D_full:
    D_full + 1 - d ~ log-jedn.) albo 'uniform'. min_samples_leaf = 1.
  * tryb ``leaf``: min_samples_leaf log-jednostajne na {1, ..., leaf_max};
    leaf = 1 oznacza pelne drzewo. max_depth = None.
  D_full to szacowana glebokosc pelnego drzewa (mediana z kilku drzew pilotazowych,
  liczona w forest.fit). min_samples_split jest stale rowne 2.
- ``criterion``: jednostajnie z listy kryteriow.
- ``splitter``: z prawd. ``splitter_random_prob`` 'random' (losowy prog dla kazdej
  z K cech, jak w Extra Trees), w przeciwnym razie 'best'.
- ``max_features`` (K, liczba cech w wezle), w obu zadaniach K in [p/a, p]:
  z prawd. w (``features_full_prob``) K = p, w przeciwnym razie
  ln K ~ U[ln(p/a), ln p) (a = ``features_down``); K zaokraglone do [1, p].
- ``max_samples`` (BR, bootstrap ratio): jednostajne na ``bootstrap_bounds``;
  bootstrap ze zwracaniem o rozmiarze BR * n (BR moze byc > 1).

Eksperymentalnie (``signal_rule``): leaf_max (i ewentualnie depth_mode_prob) zalezne
od sygnalu w danych, mierzonego wynikiem OOB lasu pilotazowego (forest.estimate_signal).

Parametry nielosowane maja wartosci domyslne scikit-learn albo wartosci podane
jawnie w ``fixed`` (np. srodek z hdrf.center.empirical_center).
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np


HYPERPARAMETERS = ("complexity", "criterion", "max_features", "max_samples", "splitter")

# Klucze, ktore mozna ustawic jawnie dla parametrow nielosowanych.
FIXED_KEYS = ("max_depth", "min_samples_leaf", "max_leaf_nodes", "criterion", "max_features",
              "max_samples", "splitter")


@dataclass(frozen=True)
class SamplingConfig:
    """Stale rozkladow (wariant A_p); do kalibracji w E0."""

    # complexity
    depth_mode_prob: float = 0.5
    depth_low_coef: float = 1.0
    depth_distribution: Literal["uniform", "log_full"] = "log_full"
    leaf_max: int = 5
    n_pilot_trees: int = 5
    # criterion
    classification_criteria: tuple[str, ...] = ("gini", "entropy")
    regression_criteria: tuple[str, ...] = ("squared_error", "friedman_mse")
    # splitter
    splitter_random_prob: float = 0.5
    # max_features: K in [p/a, p], P(K = p) = w
    features_down: float = 2.0  # a
    features_full_prob: float = 0.65  # w
    # max_samples
    bootstrap_bounds: tuple[float, float] = (0.7, 1.0)
    # Regula zalezna od sygnalu (eksperymentalna): s = OOB accuracy (klas.) / OOB R^2 (regr.)
    # lasu pilotazowego z domyslnymi hiperparametrami RF. s >= prog -> drzewa pelniejsze.
    # 'none': stale jak wyzej; 'leaf': tylko leaf_max; 'leaf_depth': leaf_max i depth_mode_prob.
    signal_rule: Literal["none", "leaf", "leaf_depth"] = "none"
    signal_threshold: float = 0.78
    signal_leaf_max: tuple[int, int] = (12, 2)  # (s < prog, s >= prog)
    signal_depth_prob: tuple[float, float] = (0.3, 0.7)  # (s < prog, s >= prog)
    n_signal_trees: int = 100

    def __post_init__(self):
        if self.signal_rule not in ("none", "leaf", "leaf_depth"):
            raise ValueError("signal_rule musi byc 'none', 'leaf' albo 'leaf_depth'")
        if not isinstance(self.n_signal_trees, int) or self.n_signal_trees < 1:
            raise ValueError("n_signal_trees musi byc calkowite >= 1")
        if any(not isinstance(v, int) or v < 1 for v in self.signal_leaf_max):
            raise ValueError("signal_leaf_max musi zawierac calkowite >= 1")
        if any(not 0 <= v <= 1 for v in self.signal_depth_prob):
            raise ValueError("signal_depth_prob musi byc w [0, 1]")
        for name in ("depth_mode_prob", "features_full_prob", "splitter_random_prob"):
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
        if self.features_down < 1:
            raise ValueError("features_down musi byc >= 1")
        low, high = self.bootstrap_bounds
        if not 0 < low <= high:
            raise ValueError("bootstrap_bounds musi byc dodatnie i rosnace")
        if not self.classification_criteria or not self.regression_criteria:
            raise ValueError("Listy criterion nie moga byc puste")


def apply_signal_rule(config: SamplingConfig, signal: float) -> SamplingConfig:
    """Stale zlozonosci po regule sygnalu (dla signal_rule='none' bez zmian)."""
    if config.signal_rule == "none":
        return config
    high = int(signal >= config.signal_threshold)
    changes = {"leaf_max": config.signal_leaf_max[high]}
    if config.signal_rule == "leaf_depth":
        changes["depth_mode_prob"] = config.signal_depth_prob[high]
    return replace(config, **changes)


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


def sample_features(rng: np.random.Generator, n_features: int, config: SamplingConfig) -> int:
    """K in [p/a, p]: z prawd. w K = p, w przeciwnym razie log-jednostajnie na [p/a, p)."""
    if rng.random() < config.features_full_prob:
        return n_features
    value = np.exp(rng.uniform(np.log(n_features / config.features_down), np.log(n_features)))
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
        params["max_features"] = sample_features(source("max_features"), n_features, config)
    if "max_samples" in enabled:
        params["max_samples"] = float(source("max_samples").uniform(*config.bootstrap_bounds))
    return params


def tree_kwargs(params: Mapping) -> dict:
    """Argumenty dla DecisionTree* (bez max_samples i trybu)."""
    return {k: v for k, v in params.items() if k not in ("max_samples", "mode")}
