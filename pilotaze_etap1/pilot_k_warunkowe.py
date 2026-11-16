"""Pilotaz (tylko klasyfikacja): K0 zalezne od wylosowanego splittera.

Drzewa 'random' dostaja K0 = p, drzewa 'best' K0 = sqrt(p) (classification_k0 =
'by_splitter'); reszta jak w wariancie A (pi = 0.5). Porownanie z RF, A (K0 = sqrt p),
A_p (K0 = p) i Extra Trees z poprzednich pilotazy (te same foldy i ziarna).

    python pilot_k_warunkowe.py [--trees 500] [--folds 10] [--summary-only]
"""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data_loader import load_e0_task
from hdrf import HDRFClassifier
from pilot_dev16 import BASE_SEED, PREVIOUS
from pilot_e0 import score
from pilot_splitter import CONFIG as A
from pilot_szerokosc import tree_diagnostics

CONFIG = replace(A, classification_k0="by_splitter")
NEW = "HDRF A_war (K od splittera)"
OLD = ("RF domyslny", "HDRF A (splitter)", "HDRF A_p (pi=0.5)", "HDRF A_p (pi=1)", "Extra Trees")
# Od najtanszych.
TASKS = (361055, 361278, 361070, 361275, 361065, 361282, 361068, 361274)


def run_task(task_id, n_trees, n_folds, out_dir):
    task = load_e0_task(task_id)
    labels = np.unique(task.y)
    rows = []
    print(f"\n=== {task.name} ===", flush=True)
    start_task = perf_counter()
    for fold in range(n_folds):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        t0 = perf_counter()
        model = HDRFClassifier(n_estimators=n_trees, random_state=BASE_SEED + fold,
                               sampling_config=CONFIG, n_jobs=-1).fit(X_tr, y_tr)
        rows.append({"task_id": task_id, "dataset": task.name, "task_type": task.task_type,
                     "fold": fold, "model": NEW, "fit_seconds": perf_counter() - t0,
                     "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                     "median_K": np.median([t.max_features for t in model.estimators_]),
                     **score(model, task.task_type, X_te, y_te, labels),
                     **tree_diagnostics(model, X_te, y_te, True)})
        pd.DataFrame(rows).to_csv(out_dir / f"{task_id}_folds.csv", index=False)
    print(f"  gotowe w {perf_counter() - start_task:.0f}s", flush=True)


def load_all(out_dir):
    frames = [pd.read_csv(p) for p in out_dir.glob("*_folds.csv")]
    for folder in ("results/pilot_dev16", "results/pilot_splitter", "results/pilot_k_p"):
        frames += [pd.read_csv(p) for p in Path(folder).glob("*_folds.csv")]
    frames.append(pd.read_csv(PREVIOUS[361055]).assign(task_type="classification"))
    df = pd.concat(frames, ignore_index=True)
    return df[(df.task_type == "classification") & df.model.isin((NEW,) + OLD)]


def summarize(out_dir):
    df = load_all(out_dir)
    rows = []
    for name, g in df.groupby("dataset", sort=False):
        if g[g.model == NEW].fold.nunique() < 10:
            continue
        row = {"dataset": name}
        for metric in ("accuracy", "log_loss"):
            piv = g.pivot_table(index="fold", columns="model", values=metric)
            for m in (NEW,) + OLD:
                row[f"{metric}: {m}"] = piv[m].mean()
            for b in OLD:
                diff = piv[NEW] - piv[b]
                if metric == "log_loss":
                    diff = -diff / piv[b]  # dodatnie = lepiej
                row[f"{metric}: war - {b}"] = diff.mean()
        means = g.groupby("model")[["fit_seconds", "amb_share", "tree_corr"]].mean()
        for col in means:
            for m in (NEW,) + OLD:
                row[f"{col}: {m}"] = means[col].get(m, np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    if table.empty:
        return
    for metric in ("accuracy", "log_loss"):
        print(f"\n{metric} ({len(table)} zbiorow), dodatnie = A_war lepszy")
        cols = ["dataset"] + [f"{metric}: war - {b}" for b in OLD]
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        for b in OLD:
            x = table[f"{metric}: war - {b}"]
            p = f", Wilcoxon p={wilcoxon(x).pvalue:.3f}" if len(x) >= 5 else ""
            print(f"  war - {b}: mediana {x.median():+.4%}, dodatnie {int((x > 0).sum())}/{len(x)}{p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/pilot_k_warunkowe"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for task_id in TASKS:
            run_task(task_id, args.trees, args.folds, args.out)
            summarize(args.out)
    summarize(args.out)


if __name__ == "__main__":
    main()
