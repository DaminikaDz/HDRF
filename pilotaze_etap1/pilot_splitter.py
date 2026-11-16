"""Pilotaz: losowy splitter per drzewo (wariant A) i Extra Trees, 16 zbiorow dev.

Wariant A = HDRF pelniejszy (depth log_full, leaf_max 5) + splitter 'random'
z prawd. 0.5. Do tego jego RF ze srodka (max_leaf_nodes, splitter 'best')
i Extra Trees domyslne ze scikit-learn (bez bootstrapu).
RF domyslny, HDRF bazowy i pelniejszy sa brane z poprzednich pilotazy.

    python pilot_splitter.py [--trees 500] [--folds 10] [--summary-only]
"""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor, empirical_center
from pilot_dev16 import BASE_SEED, PREVIOUS
from pilot_e0 import score
from pilot_pelniejsze import ALL_TASKS, CONFIG as FULLER
from pilot_szerokosc import tree_diagnostics

CONFIG = replace(FULLER, splitter_random_prob=0.5)
NEW = ("HDRF A (splitter)", "RF ze srodka A", "Extra Trees")
OLD = ("RF domyslny", "HDRF pelny", "HDRF pelniejszy")


def run_task(task_id, n_trees, n_folds, out_dir):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    ET = ExtraTreesClassifier if clf else ExtraTreesRegressor
    rows = []
    print(f"\n=== {task.name} ({task.task_type}) ===", flush=True)
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
        hdrf = Model(n_estimators=n_trees, random_state=seed, sampling_config=CONFIG,
                     n_jobs=-1).fit(X_tr, y_tr)
        record(NEW[0], hdrf, perf_counter() - t0, {"d_full": hdrf.d_full_})
        t0 = perf_counter()
        center, _ = empirical_center(X_tr, y_tr, task.task_type, config=CONFIG,
                                     reference=hdrf, limit="leaves", random_state=seed)
        rfc = Model(n_estimators=n_trees, randomize=(), fixed_params=center,
                    random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        record(NEW[1], rfc, perf_counter() - t0, {"center_leaves": center["max_leaf_nodes"]})
        t0 = perf_counter()
        et = ET(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        record(NEW[2], et, perf_counter() - t0)
        pd.DataFrame(rows).to_csv(out_dir / f"{task_id}_folds.csv", index=False)
    print(f"  gotowe w {perf_counter() - start_task:.0f}s", flush=True)


def load_all(out_dir):
    frames = [pd.read_csv(p) for p in out_dir.glob("*_folds.csv")]
    old = [pd.read_csv(p) for p in Path("results/pilot_dev16").glob("*_folds.csv")]
    old += [pd.read_csv(p) for p in Path("results/pilot_pelniejsze").glob("*_folds.csv")]
    for path in PREVIOUS.values():
        prev = pd.read_csv(path)
        prev["task_type"] = "classification" if "accuracy" in prev else "regression"
        old.append(prev)
    old = pd.concat(old, ignore_index=True)
    return pd.concat(frames + [old[old.model.isin(OLD)]], ignore_index=True)


def summarize(out_dir):
    df = load_all(out_dir)
    pairs = (("HDRF A (splitter)", "RF domyslny", "A-RF"),
             ("HDRF A (splitter)", "HDRF pelniejszy", "A-pelniejszy"),
             ("HDRF A (splitter)", "RF ze srodka A", "A-srodek"),
             ("HDRF A (splitter)", "Extra Trees", "A-ET"),
             ("Extra Trees", "RF domyslny", "ET-RF"))
    rows = []
    for (task_id, name), g in df.groupby(["task_id", "dataset"], sort=False):
        if not set(NEW) <= set(g.model) or g[g.model == NEW[0]].fold.nunique() < 10:
            continue
        clf = g.task_type.iloc[0] == "classification"
        row = {"dataset": name, "task": "klas." if clf else "regr."}
        for metric in (("accuracy", "log_loss") if clf else ("rmse",)):
            piv = g.pivot_table(index="fold", columns="model", values=metric)
            for m in OLD + NEW:
                row[f"{metric}: {m}"] = piv[m].mean() if m in piv else np.nan
            for a, b, label in pairs:
                if b not in piv:
                    continue
                diff = piv[a] - piv[b]
                if metric != "accuracy":
                    diff = -diff / piv[b]  # dodatnie = lepiej
                row[f"{metric}: {label}"] = diff.mean()
                row[f"{metric}: {label} wygr."] = int((diff > 0).sum())
        for col in ("tree_corr", "amb_share", "fit_seconds"):
            means = g.groupby("model")[col].mean()
            for m in OLD + NEW:
                row[f"{col}: {m}"] = means.get(m, np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    for task, metrics in (("klas.", ("accuracy", "log_loss")), ("regr.", ("rmse",))):
        sub = table[table.task == task]
        if sub.empty:
            continue
        for metric in metrics:
            print(f"\n{task} {metric} ({len(sub)} zbiorow)")
            for *_, label in pairs:
                x = sub[f"{metric}: {label}"].dropna()
                if len(x) >= 5:
                    print(f"  {label}: mediana {x.median():+.4%}, dodatnie "
                          f"{int((x > 0).sum())}/{len(x)}, Wilcoxon p={wilcoxon(x).pvalue:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/pilot_splitter"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for task_id in ALL_TASKS:
            run_task(task_id, args.trees, args.folds, args.out)
            summarize(args.out)
    summarize(args.out)


if __name__ == "__main__":
    main()
