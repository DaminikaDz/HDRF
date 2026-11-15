"""Empiryczny srodek rozkladow HDRF (model "RF ze srodka").

- max_features i max_samples: mediany brzegowe z samplera (Monte Carlo).
- criterion: wartosc domyslna (gini / squared_error).
- zlozonosc: mediana liczby lisci M drzew HDRF (z wytrenowanego HDRF albo z lasu
  pilotazowego), a nastepnie jedna wartosc max_depth albo min_samples_leaf, dla
  ktorej drzewa ze srodkowymi K i BR maja srednio liczbe lisci najblizsza M
  (w skali logarytmicznej).
"""

import numpy as np
from joblib import Parallel, delayed

from .forest import HDRFClassifier, HDRFRegressor, _fit_tree
from .sampling import (
    SamplingConfig, default_params, depth_bounds, sample_params,
)


def marginal_medians(task, n_samples, n_features, config, n_draws=20000, seed=0):
    """Mediany K i BR z n_draws losowan samplera."""
    rng = np.random.default_rng(seed)
    draws = [
        sample_params(rng, task, n_samples, n_features, ("max_features", "max_samples"),
                      config=config)
        for _ in range(n_draws)
    ]
    k = int(np.rint(np.median([d["max_features"] for d in draws])))
    br = float(np.median([d["max_samples"] for d in draws]))
    return {"max_features": max(1, k), "max_samples": br}


def _mean_leaves(tree_cls, X, y, params, n_trees, rng):
    n = X.shape[0]
    jobs = []
    for _ in range(n_trees):
        info = {"seed": int(rng.integers(0, 2**31 - 1)),
                "size": max(1, int(round(params["max_samples"] * n))), "train_size": n}
        jobs.append((info, int(rng.integers(0, 2**31 - 1))))
    trees = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fit_tree)(tree_cls, X, y, params, info, seed) for info, seed in jobs
    )
    return float(np.mean([t.get_n_leaves() for t in trees]))


def empirical_center(
    X, y, task, config=None, reference=None, limit="leaf",
    n_reference_trees=100, n_calibration_trees=4, random_state=0,
):
    """Parametry RF ze srodka dla danej czesci treningowej.

    reference: wytrenowany HDRF (complexity losowane) na tych samych danych; jesli
    None, uczymy pilotazowy HDRF z n_reference_trees drzewami.
    limit: 'leaf' (srodek przez min_samples_leaf) albo 'depth' (przez max_depth).
    Zwraca (fixed_params, info).
    """
    config = SamplingConfig() if config is None else config
    if limit not in ("leaf", "depth"):
        raise ValueError("limit musi byc 'leaf' albo 'depth'")
    model_cls = HDRFClassifier if task == "classification" else HDRFRegressor
    if reference is None:
        reference = model_cls(n_estimators=n_reference_trees, random_state=random_state,
                              sampling_config=config, n_jobs=-1).fit(X, y)
    if reference.d_full_ is None:
        raise ValueError("reference musi losowac complexity")

    X = np.asarray(X, dtype=np.float32)
    if task == "classification":
        y = np.searchsorted(reference.classes_, y).astype(np.float64)
    n_samples, n_features = X.shape
    target = float(np.median([t.get_n_leaves() for t in reference.estimators_]))

    center = default_params(task, n_samples, n_features)
    center.update(marginal_medians(task, n_samples, n_features, config, seed=random_state))
    tree_cls = reference._tree_cls

    if limit == "leaf":
        candidates = list(range(1, config.leaf_max + 1))
        key = "min_samples_leaf"
    else:
        low, high = depth_bounds(n_samples, reference.d_full_, config)
        candidates = list(range(low, high + 1))
        key = "max_depth"

    rng = np.random.default_rng(random_state)
    leaves = {}
    for value in candidates:
        params = dict(center, **{key: value})
        if key == "max_depth" and value >= candidates[-1]:
            params["max_depth"] = None
        leaves[value] = _mean_leaves(tree_cls, X, y, params, n_calibration_trees, rng)
    best = min(candidates, key=lambda v: abs(np.log(leaves[v]) - np.log(target)))
    center[key] = None if (key == "max_depth" and best >= candidates[-1]) else best
    center.pop("mode")
    center.pop("min_samples_split")
    info = {"target_leaves": target, "limit": limit, "leaves_by_value": leaves,
            "d_full": reference.d_full_}
    return center, info
