"""Pilotaz: HDRF z wieksza masa na pelniejszych drzewach, 16 zbiorow deweloperskich.

Wariant "pelniejszy": tryb depth log-jednostajny z grubym ogonem przy D_full
(depth_distribution='log_full'), tryb leaf do leaf_max = 5. RF ze srodka przez
max_leaf_nodes = mediana liczby lisci HDRF (dokladne dopasowanie).
RF domyslny i HDRF bazowy sa brane z results/pilot_dev16 i results/pilot_e0
(te same foldy i ziarna).

    python pilot_pelniejsze.py [--trees 500] [--folds 10] [--summary-only]
"""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig, empirical_center
from pilot_dev16 import BASE_SEED, PREVIOUS, TASK_IDS
from pilot_e0 import score
from pilot_szerokosc import tree_diagnostics

CONFIG = replace(SamplingConfig(), depth_distribution="log_full", leaf_max=5)
NEW = ("HDRF pelniejszy", "RF ze srodka (max_leaf_nodes)")
OLD = ("RF domyslny", "HDRF pelny")
ALL_TASKS = (361055, 361077) + TASK_IDS


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
        pd.DataFrame(rows).to_csv(out_dir / f"{task_id}_folds.csv", index=False)
    print(f"  gotowe w {perf_counter() - start_task:.0f}s", flush=True)


def load_all(out_dir):
    frames = [pd.read_csv(p) for p in out_dir.glob("*_folds.csv")]
    old = [pd.read_csv(p) for p in Path("results/pilot_dev16").glob("*_folds.csv")]
    for path in PREVIOUS.values():
        prev = pd.read_csv(path)
        prev["task_type"] = "classification" if "accuracy" in prev else "regression"
        old.append(prev)
    old = pd.concat(old, ignore_index=True)
    return pd.concat(frames + [old[old.model.isin(OLD)]], ignore_index=True)


def summarize(out_dir):
    df = load_all(out_dir)
    pairs = (("HDRF pelniejszy", "RF domyslny", "pelniejszy-RF"),
             ("HDRF pelniejszy", "HDRF pelny", "pelniejszy-bazowy"),
             ("HDRF pelniejszy", "RF ze srodka (max_leaf_nodes)", "pelniejszy-srodek"),
             ("RF ze srodka (max_leaf_nodes)", "RF domyslny", "srodek-RF"))
    rows = []
    for (task_id, name), g in df.groupby(["task_id", "dataset"], sort=False):
        if not set(NEW) <= set(g.model):
            continue
        clf = g.task_type.iloc[0] == "classification"
        metrics = ("accuracy", "log_loss") if clf else ("rmse",)
        row = {"dataset": name, "task": "klas." if clf else "regr."}
        for metric in metrics:
            piv = g.pivot(index="fold", columns="model", values=metric)
            for m in OLD + NEW:
                row[f"{metric}: {m}"] = piv[m].mean()
            for a, b, label in pairs:
                diff = piv[a] - piv[b]
                if metric != "accuracy":
                    diff = -diff / piv[b]  # dodatnie = lepiej
                row[f"{metric}: {label}"] = diff.mean()
                row[f"{metric}: {label} wygr."] = int((diff > 0).sum())
        leaves = g.groupby("model").median_leaves.median()
        row["liscie pelniejszy/RF"] = leaves["HDRF pelniejszy"] / leaves["RF domyslny"]
        row["liscie srodek/pelniejszy"] = leaves[NEW[1]] / leaves[NEW[0]]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    for task, metrics in (("klas.", ("accuracy", "log_loss")), ("regr.", ("rmse",))):
        sub = table[table.task == task]
        if sub.empty:
            continue
        for metric in metrics:
            cols = ["dataset"] + [f"{metric}: {m}" for m in OLD + NEW] + \
                   [f"{metric}: {label}" for *_, label in pairs]
            with pd.option_context("display.width", 250, "display.max_columns", 20):
                print(f"\n{task} {metric}")
                print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
            if len(sub) >= 5:
                for *_, label in pairs:
                    x = sub[f"{metric}: {label}"]
                    print(f"  {label}: mediana {x.median():+.4%}, dodatnie "
                          f"{int((x > 0).sum())}/{len(x)}, Wilcoxon p={wilcoxon(x).pvalue:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/pilot_pelniejsze"))
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
