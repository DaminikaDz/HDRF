"""Pilotaz: K0 = p w klasyfikacji (jak w regresji) i splitter 'random' we wszystkich drzewach.

Klasyfikacja: wariant A z K0 = p (K w [p/2, p], duza masa w p) przy
splitter_random_prob 0.5 i 1.0. Regresja: wariant A przy splitter_random_prob 1.0
(wariant 0.5 jest w results/pilot_splitter). Pozostale modele (RF, ET, A, srodek A,
pelniejszy) sa brane z poprzednich pilotazy.

    python pilot_k_p.py [--trees 500] [--folds 10] [--summary-only]
"""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor
from pilot_dev16 import BASE_SEED, PREVIOUS
from pilot_e0 import score
from pilot_pelniejsze import ALL_TASKS
from pilot_splitter import CONFIG as A
from pilot_szerokosc import tree_diagnostics

VARIANTS = {
    "classification": {"HDRF A_p (pi=0.5)": replace(A, classification_k0="p"),
                       "HDRF A_p (pi=1)": replace(A, classification_k0="p",
                                                  splitter_random_prob=1.0)},
    "regression": {"HDRF A (pi=1)": replace(A, splitter_random_prob=1.0)},
}
OLD = ("RF domyslny", "HDRF pelniejszy", "HDRF A (splitter)", "RF ze srodka A", "Extra Trees")


def run_task(task_id, n_trees, n_folds, out_dir):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    rows = []
    print(f"\n=== {task.name} ({task.task_type}) ===", flush=True)
    start_task = perf_counter()
    for fold in range(n_folds):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold
        for name, config in VARIANTS[task.task_type].items():
            t0 = perf_counter()
            model = Model(n_estimators=n_trees, random_state=seed, sampling_config=config,
                          n_jobs=-1).fit(X_tr, y_tr)
            rows.append({"task_id": task_id, "dataset": task.name, "task_type": task.task_type,
                         "fold": fold, "model": name, "fit_seconds": perf_counter() - t0,
                         "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                         **score(model, task.task_type, X_te, y_te, labels),
                         **tree_diagnostics(model, X_te, y_te, clf)})
        pd.DataFrame(rows).to_csv(out_dir / f"{task_id}_folds.csv", index=False)
    print(f"  gotowe w {perf_counter() - start_task:.0f}s", flush=True)


def load_all(out_dir):
    frames = [pd.read_csv(p) for p in out_dir.glob("*_folds.csv")]
    old = []
    for folder in ("results/pilot_dev16", "results/pilot_pelniejsze", "results/pilot_splitter"):
        old += [pd.read_csv(p) for p in Path(folder).glob("*_folds.csv")]
    for path in PREVIOUS.values():
        prev = pd.read_csv(path)
        prev["task_type"] = "classification" if "accuracy" in prev else "regression"
        old.append(prev)
    old = pd.concat(old, ignore_index=True)
    return pd.concat(frames + [old[old.model.isin(OLD)]], ignore_index=True)


def summarize(out_dir):
    df = load_all(out_dir)
    rows, comparisons = [], {}
    for (task_id, name), g in df.groupby(["task_id", "dataset"], sort=False):
        task_type = g.task_type.iloc[0]
        new = list(VARIANTS[task_type])
        if not set(new) <= set(g.model) or g[g.model == new[0]].fold.nunique() < 10:
            continue
        clf = task_type == "classification"
        pairs = [(n, b) for n in new for b in OLD] + \
                ([("HDRF A_p (pi=0.5)", "HDRF A_p (pi=1)")] if clf else
                 [("HDRF A (splitter)", "HDRF A (pi=1)")])
        row = {"dataset": name, "task": "klas." if clf else "regr."}
        for metric in (("accuracy", "log_loss") if clf else ("rmse",)):
            piv = g.pivot_table(index="fold", columns="model", values=metric)
            for m in new + list(OLD):
                row[f"{metric}: {m}"] = piv[m].mean() if m in piv else np.nan
            for a, b in pairs:
                if b not in piv:
                    continue
                diff = piv[a] - piv[b]
                if metric != "accuracy":
                    diff = -diff / piv[b]  # dodatnie = lepiej
                label = f"{metric}: {a} - {b}"
                row[label] = diff.mean()
                row[label + " wygr."] = int((diff > 0).sum())
                comparisons.setdefault(row["task"], []).append(label)
        for col in ("tree_corr", "amb_share", "fit_seconds"):
            means = g.groupby("model")[col].mean()
            for m in new + list(OLD):
                row[f"{col}: {m}"] = means.get(m, np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    for task, labels in comparisons.items():
        sub = table[table.task == task]
        print(f"\n{task} ({len(sub)} zbiorow)")
        for label in dict.fromkeys(labels):
            x = sub[label].dropna()
            if len(x) >= 5:
                print(f"  {label}: mediana {x.median():+.4%}, dodatnie "
                      f"{int((x > 0).sum())}/{len(x)}, Wilcoxon p={wilcoxon(x).pvalue:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/pilot_k_p"))
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
