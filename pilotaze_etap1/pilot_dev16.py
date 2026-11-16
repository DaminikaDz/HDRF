"""Pilotaz rozkladow bazowych na 16 zbiorach deweloperskich (10 foldow OpenML).

Modele: RF domyslny (scikit-learn), HDRF pelny (rozklady bazowe), RF ze srodka
(empiryczny, przez min_samples_leaf). Wyniki credit i Ailerons sa brane z
results/pilot_e0 (ten sam kod i ziarna). Podsumowanie po kazdym zbiorze.

    python pilot_dev16.py [--trees 500] [--folds 10]
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor, empirical_center
from pilot_e0 import score
from pilot_szerokosc import tree_diagnostics

BASE_SEED = 42
# Od najtanszych (czas RF z raportu jakosci M2).
TASK_IDS = (361070, 361278, 361275, 361065, 361082, 361085, 361097, 361074,
            361282, 361279, 361079, 361068, 361274, 361292)
PREVIOUS = {361055: "results/pilot_e0/361055_folds.csv", 361077: "results/pilot_e0/361077_folds.csv"}
MODELS = ("RF domyslny", "HDRF pelny", "RF ze srodka (leaf)")


def run_task(task_id, n_trees, n_folds, out_dir):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    RF = RandomForestClassifier if clf else RandomForestRegressor
    rows = []
    print(f"\n=== {task.name} ({task.task_type}) X={task.X.shape}, B={n_trees} ===", flush=True)
    start_task = perf_counter()
    for fold in range(n_folds):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold

        def record(name, model, seconds, extra=None):
            rows.append({"task_id": task_id, "dataset": task.name, "task_type": task.task_type,
                         "fold": fold, "model": name, "fit_seconds": seconds,
                         "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                         **score(model, task.task_type, X_te, y_te, labels),
                         **tree_diagnostics(model, X_te, y_te, clf), **(extra or {})})

        t0 = perf_counter()
        rf = RF(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        record("RF domyslny", rf, perf_counter() - t0)
        t0 = perf_counter()
        hdrf = Model(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        record("HDRF pelny", hdrf, perf_counter() - t0, {"d_full": hdrf.d_full_})
        t0 = perf_counter()
        center, _ = empirical_center(X_tr, y_tr, task.task_type, reference=hdrf,
                                     limit="leaf", random_state=seed)
        rfc = Model(n_estimators=n_trees, randomize=(), fixed_params=center,
                    random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        record("RF ze srodka (leaf)", rfc, perf_counter() - t0,
               {"center_leaf": center["min_samples_leaf"], "center_K": center["max_features"]})
        pd.DataFrame(rows).to_csv(out_dir / f"{task_id}_folds.csv", index=False)
    print(f"  gotowe w {perf_counter() - start_task:.0f}s", flush=True)


def summarize(out_dir):
    frames = [pd.read_csv(p) for p in sorted(out_dir.glob("*_folds.csv"))]
    for task_id, path in PREVIOUS.items():
        if Path(path).exists():
            prev = pd.read_csv(path)
            prev = prev[prev.model.isin(MODELS)].copy()
            prev["task_type"] = "classification" if "accuracy" in prev else "regression"
            frames.append(prev)
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for (task_id, name), g in df.groupby(["task_id", "dataset"], sort=False):
        clf = g.task_type.iloc[0] == "classification"
        metric = "accuracy" if clf else "rmse"
        piv = g.pivot(index="fold", columns="model", values=metric)
        row = {"dataset": name, "task": "klas." if clf else "regr.", "metric": metric}
        for m in MODELS:
            row[m] = piv[m].mean()
        for a, b, label in (("HDRF pelny", "RF domyslny", "HDRF-RF"),
                            ("HDRF pelny", "RF ze srodka (leaf)", "HDRF-srodek"),
                            ("RF ze srodka (leaf)", "RF domyslny", "srodek-RF")):
            diff = piv[a] - piv[b]
            if not clf:
                diff = -diff / piv[b]  # dodatnie = lepiej (spadek RMSE)
            row[label] = diff.mean()
            row[label + " wygr."] = int((diff > 0).sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    for task in ("klas.", "regr."):
        sub = table[table.task == task]
        if len(sub) >= 5:
            for label in ("HDRF-RF", "HDRF-srodek", "srodek-RF"):
                p = wilcoxon(sub[label]).pvalue
                print(f"{task} {label}: mediana {sub[label].median():+.4%}, "
                      f"dodatnie {int((sub[label] > 0).sum())}/{len(sub)}, Wilcoxon p={p:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/pilot_dev16"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for task_id in TASK_IDS:
            run_task(task_id, args.trees, args.folds, args.out)
            summarize(args.out)
    summarize(args.out)


if __name__ == "__main__":
    main()
