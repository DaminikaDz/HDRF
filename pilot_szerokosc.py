"""Pilotaz: czy szersze rozklady (wiecej zroznicowania drzew) pomagaja HDRF.

Dla kazdego wariantu rozkladow uczymy HDRF i jego wlasny RF ze srodka (leaf),
wiec roznica HDRF - RF ze srodka mierzy efekt samego zroznicowania.
Diagnostyka na foldzie testowym (w duchu sily i korelacji Breimana):
  tree_score - srednia jakosc pojedynczego drzewa (accuracy / RMSE),
  tree_corr  - srednia korelacja bledow miedzy parami drzew
               (regresja: reszty y - f_t; klasyfikacja: 1 - p_t(prawdziwa klasa)).

    python pilot_szerokosc.py [--trees 500] [--folds 10]
"""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig, empirical_center
from pilot_e0 import score

TASK_IDS = (361055, 361077)
BASE_SEED = 42

BASE = SamplingConfig()
VARIANTS = {
    "bazowy": BASE,
    "K szerzej (a=b=3)": replace(BASE, features_down=3.0, features_up=3.0),
    "lisc szerzej (do 20)": replace(BASE, leaf_max=20),
    "BR szerzej [0.5, 1]": replace(BASE, bootstrap_bounds=(0.5, 1.0)),
    "glebokosc nizej (c=0.75)": replace(BASE, depth_low_coef=0.75),
    "wszystko szerzej": replace(BASE, features_down=3.0, features_up=3.0, leaf_max=20,
                                bootstrap_bounds=(0.5, 1.0), depth_low_coef=0.75),
    "bardzo szeroko": replace(BASE, features_down=4.0, features_up=4.0, features_default_prob=0.0,
                              leaf_max=30, bootstrap_bounds=(0.4, 1.2), depth_low_coef=0.6),
}


def tree_diagnostics(model, X_test, y_test, classification):
    trees = model.estimators_
    if classification:
        codes = np.searchsorted(model.classes_, y_test)
        errors, scores = [], []
        for t in trees:
            proba = np.zeros((len(y_test), len(model.classes_)))
            proba[:, t.classes_.astype(int)] = t.predict_proba(X_test.astype(np.float32))
            errors.append(1.0 - proba[np.arange(len(y_test)), codes])
            scores.append(np.mean(proba.argmax(axis=1) == codes))
    else:
        errors, scores = [], []
        for t in trees:
            resid = y_test - t.predict(X_test.astype(np.float32))
            errors.append(resid)
            scores.append(np.sqrt(np.mean(resid ** 2)))
    corr = np.corrcoef(np.vstack(errors))
    off_diag = corr[~np.eye(len(trees), dtype=bool)]
    return {"tree_score": float(np.mean(scores)), "tree_corr": float(np.nanmean(off_diag))}


def run_task(task_id, n_trees, n_folds, out_dir):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    RF = RandomForestClassifier if clf else RandomForestRegressor
    rows, path = [], out_dir / f"{task_id}_folds.csv"
    print(f"\n=== {task.name} ({task.task_type}), B={n_trees} ===", flush=True)

    for fold in range(n_folds):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold
        center_cache = {}

        def record(variant, kind, model, seconds, extra=None):
            rows.append({"task_id": task_id, "dataset": task.name, "fold": fold,
                         "variant": variant, "kind": kind, "fit_seconds": seconds,
                         "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                         **score(model, task.task_type, X_te, y_te, labels),
                         **tree_diagnostics(model, X_te, y_te, clf), **(extra or {})})

        start = perf_counter()
        rf = RF(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        record("-", "RF domyslny", rf, perf_counter() - start)

        for name, config in VARIANTS.items():
            start = perf_counter()
            hdrf = Model(n_estimators=n_trees, random_state=seed, sampling_config=config,
                         n_jobs=-1).fit(X_tr, y_tr)
            record(name, "HDRF", hdrf, perf_counter() - start)
            start = perf_counter()
            center, _ = empirical_center(X_tr, y_tr, task.task_type, config=config,
                                         reference=hdrf, limit="leaf", random_state=seed)
            key = str(sorted(center.items()))
            if key not in center_cache:  # ten sam srodek -> ten sam las (to samo ziarno)
                center_cache[key] = Model(n_estimators=n_trees, randomize=(), fixed_params=center,
                                          random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
            record(name, "RF ze srodka", center_cache[key], perf_counter() - start,
                   {"center": str({k: center[k] for k in ("min_samples_leaf", "max_features", "max_samples")})})

        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  fold {fold + 1}/{n_folds} gotowy", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/pilot_szerokosc"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task_id in TASK_IDS:
        run_task(task_id, args.trees, args.folds, args.out)


if __name__ == "__main__":
    main()
