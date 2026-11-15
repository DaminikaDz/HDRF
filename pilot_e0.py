"""Pilotaz rozkladow v3 na credit i Ailerons (10 foldow OpenML).

Pytania: (1) jak wplywa losowanie kazdej grupy hiperparametrow osobno (reszta
domyslna), (2) pelny HDRF, (3) tryby zlozonosci (tylko depth / tylko leaf / oba),
(4) RF ze srodka (empiryczny, przez leaf albo depth), (5) regresja: K0 = p vs p/3.
To nie jest E0: tylko dwa zbiory, stale rozkladow robocze.

    python pilot_e0.py [--trees 500] [--folds 10]
"""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig, empirical_center

TASK_IDS = (361055, 361077)
BASE_SEED = 42


def variants(task_type):
    """(nazwa, randomize, config) dla modeli HDRF; kolejnosc = kolejnosc uczenia."""
    base = SamplingConfig()
    out = [
        ("tylko complexity", ("complexity",), base),
        ("tylko criterion", ("criterion",), base),
        ("tylko max_features", ("max_features",), base),
        ("tylko max_samples", ("max_samples",), base),
        ("HDRF pelny", None, base),
        ("HDRF pelny, tylko tryb depth", None, replace(base, depth_mode_prob=1.0)),
        ("HDRF pelny, tylko tryb leaf", None, replace(base, depth_mode_prob=0.0)),
    ]
    if task_type == "regression":
        p3 = replace(base, regression_k0="p/3")
        out += [("tylko max_features, K0=p/3", ("max_features",), p3),
                ("HDRF pelny, K0=p/3", None, p3)]
    return out


def score(model, task_type, X_test, y_test, labels):
    pred = model.predict(X_test)
    if task_type == "classification":
        return {"accuracy": accuracy_score(y_test, pred),
                "log_loss": log_loss(y_test, model.predict_proba(X_test), labels=labels)}
    return {"rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": r2_score(y_test, pred)}


def run_task(task_id, n_trees, n_folds, out_dir):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    RF = RandomForestClassifier if clf else RandomForestRegressor
    rows = []
    path = out_dir / f"{task_id}_folds.csv"
    print(f"\n=== {task.name} ({task.task_type}), B={n_trees} ===", flush=True)

    for fold in range(n_folds):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold
        fitted = {}

        def record(name, model, seconds, extra=None):
            row = {"task_id": task_id, "dataset": task.name, "fold": fold, "model": name,
                   "fit_seconds": seconds, **score(model, task.task_type, X_te, y_te, labels)}
            if extra:
                row.update(extra)
            rows.append(row)

        start = perf_counter()
        rf = RF(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        record("RF domyslny", rf, perf_counter() - start,
               {"median_leaves": np.median([t.get_n_leaves() for t in rf.estimators_])})

        for name, randomize, config in variants(task.task_type):
            kwargs = {} if randomize is None else {"randomize": randomize}
            start = perf_counter()
            model = Model(n_estimators=n_trees, random_state=seed, sampling_config=config,
                          n_jobs=-1, **kwargs).fit(X_tr, y_tr)
            leaves = np.median([t.get_n_leaves() for t in model.estimators_])
            record(name, model, perf_counter() - start, {"median_leaves": leaves})
            fitted[name] = model

        for limit in ("leaf", "depth"):
            start = perf_counter()
            center, info = empirical_center(X_tr, y_tr, task.task_type,
                                            reference=fitted["HDRF pelny"], limit=limit,
                                            random_state=seed)
            model = Model(n_estimators=n_trees, randomize=(), fixed_params=center,
                          random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
            leaves = np.median([t.get_n_leaves() for t in model.estimators_])
            value = center["min_samples_leaf"] if limit == "leaf" else center["max_depth"]
            record(f"RF ze srodka ({limit})", model, perf_counter() - start,
                   {"median_leaves": leaves, "center_value": value})

        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  fold {fold + 1}/{n_folds} gotowy", flush=True)

    summarize(pd.DataFrame(rows), task.task_type, out_dir / f"{task_id}_summary.csv")


def summarize(df, task_type, path):
    metric, better = ("accuracy", 1) if task_type == "classification" else ("rmse", -1)
    base = df[df.model == "RF domyslny"].set_index("fold")[metric]
    out = []
    for name, g in df.groupby("model", sort=False):
        s = g.set_index("fold")[metric]
        diff = s - base
        rel = diff / base if task_type == "regression" else diff
        out.append({
            "model": name, f"{metric}_mean": s.mean(),
            "diff_vs_RF_mean": rel.mean(),
            "wins_vs_RF": int((better * diff > 0).sum()), "folds": len(s),
            "median_leaves": g["median_leaves"].median() if "median_leaves" in g else np.nan,
            "fit_seconds": g.fit_seconds.mean(),
        })
    table = pd.DataFrame(out)
    table.to_csv(path, index=False)
    label = "roznica accuracy" if task_type == "classification" else "wzgl. zmiana RMSE"
    print(f"\n{label} vs RF domyslny (srednia po foldach), wygrane foldy:")
    for r in table.itertuples():
        print(f"  {r.model:<30} {getattr(r, metric + '_mean'):.6g}  "
              f"{r.diff_vs_RF_mean:+.4%}  {r.wins_vs_RF}/{r.folds}  liscie={r.median_leaves:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/pilot_e0"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task_id in TASK_IDS:
        run_task(task_id, args.trees, args.folds, args.out)


if __name__ == "__main__":
    main()
