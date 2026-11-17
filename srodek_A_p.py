"""RF ze srodka dla wariantu A_p w klasyfikacji (10 foldow, B = 500).

Srodek liczony jak w hdrf.center.empirical_center(limit='leaves'): K i BR to mediany
rozkladow A_p (Monte Carlo), criterion gini, splitter best, max_leaf_nodes = mediana
liczby lisci drzew HDRF A_p na tym samym foldzie. Mediana lisci jest brana z zapisanych
wynikow A_p (results/pilot_k_p), wiec srodek odpowiada dokladnie tamtym lasom.
W regresji A_p = wariant A, a jego srodek jest w results/pilot_splitter.

    python srodek_A_p.py [--trees 500] [--folds 10]
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from data_loader import load_e0_task
from hdrf import HDRFClassifier, SamplingConfig, default_params
from hdrf.center import marginal_medians
from hdrf.diagnostics import score, tree_diagnostics

BASE_SEED = 42
TASKS = (361055, 361278, 361070, 361275, 361065, 361282, 361068, 361274)
SOURCE = Path("results/pilot_k_p")
REFERENCE = "HDRF A_p (pi=0.5)"
NAME = "RF ze srodka A_p"


def run_task(task_id, n_trees, n_folds, out_dir):
    task = load_e0_task(task_id)
    labels = np.unique(task.y)
    ref = pd.read_csv(SOURCE / f"{task_id}_folds.csv")
    leaves = ref[ref.model == REFERENCE].set_index("fold").median_leaves
    rows = []
    print(f"\n=== {task.name} ===", flush=True)
    start = perf_counter()
    for fold in range(n_folds):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold
        n, p = X_tr.shape
        center = default_params("classification", n, p)
        center.update(marginal_medians("classification", n, p, SamplingConfig(), seed=seed))
        center["max_leaf_nodes"] = max(2, int(round(leaves[fold])))
        center.pop("mode")
        center.pop("min_samples_split")
        t0 = perf_counter()
        model = HDRFClassifier(n_estimators=n_trees, randomize=(), fixed_params=center,
                               random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        rows.append({"task_id": task_id, "dataset": task.name, "task_type": "classification",
                     "fold": fold, "model": NAME, "fit_seconds": perf_counter() - t0,
                     "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                     "center_leaves": center["max_leaf_nodes"], "center_K": center["max_features"],
                     "center_BR": center["max_samples"],
                     **score(model, "classification", X_te, y_te, labels),
                     **tree_diagnostics(model, X_te, y_te, True)})
        pd.DataFrame(rows).to_csv(out_dir / f"{task_id}_folds.csv", index=False)
    print(f"  gotowe w {perf_counter() - start:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/srodek_A_p"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task_id in TASKS:
        if (args.out / f"{task_id}_folds.csv").exists() and \
                len(pd.read_csv(args.out / f"{task_id}_folds.csv")) == args.folds:
            continue
        run_task(task_id, args.trees, args.folds, args.out)


if __name__ == "__main__":
    main()
