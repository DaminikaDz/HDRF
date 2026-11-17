"""E0: RF ze srodka z mieszanka splitterow 50/50 - rozklad zysku A_p na przesuniecie i heterogenicznosc.

Wszystkie drzewa maja konfiguracje srodka A_p (K i BR = mediany rozkladow A_p, kryterium domyslne,
max_leaf_nodes = mediana liczby lisci drzew A_p na tym samym foldzie, z results/regula_pelny),
a splitter jest losowany 50/50 jak w A_p. Ziarna 42 + fold, 10 foldow, B = 500.

Rozklad przy stalej mieszance 50/50:
  tylko splitter -> srodek 50/50 : efekt przesuniecia srednich (zlozonosc, BR)
  srodek 50/50  -> A_p          : efekt heterogenicznosci (rozrzut wokol srodka)
To samo dla drzew best: RF K=p (klas.) / RF domyslny (regr.) -> RF ze srodka -> A_p pi=0.

    python srodek_5050.py [--summary-only]
"""

import argparse
import glob
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig, default_params
from hdrf.center import marginal_medians
from hdrf.diagnostics import score, tree_diagnostics
from kalibracja_przesiew import BASE_SEED, TASKS
from regula_sygnalu import METRICS, RF_FULL

OUT = Path("results/srodek_5050")
N_TREES, N_FOLDS = 500, 10
NAME = "srodek 50/50"


def run_task(task_id):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    ref = pd.read_csv(f"results/regula_pelny/{task_id}_folds.csv")
    leaves = ref[ref.config == "A_p"].set_index("fold").median_leaves
    rows = []
    print(f"\n=== {task.name} ({task.task_type}) ===", flush=True)
    start = perf_counter()
    for fold in range(N_FOLDS):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold
        n, p = X_tr.shape
        center = default_params(task.task_type, n, p)
        center.update(marginal_medians(task.task_type, n, p, SamplingConfig(), seed=seed))
        center["max_leaf_nodes"] = max(2, int(round(leaves[fold])))
        for key in ("mode", "min_samples_split", "splitter"):
            center.pop(key)
        t0 = perf_counter()
        model = Model(n_estimators=N_TREES, randomize=("splitter",), fixed_params=center,
                      sampling_config=SamplingConfig(splitter_random_prob=0.5),
                      random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        rows.append({"task_id": task_id, "dataset": task.name, "task_type": task.task_type,
                     "fold": fold, "config": NAME, "fit_seconds": perf_counter() - t0,
                     "center_leaves": center["max_leaf_nodes"], "center_K": center["max_features"],
                     "center_BR": center["max_samples"],
                     "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                     **score(model, task.task_type, X_te, y_te, labels),
                     **tree_diagnostics(model, X_te, y_te, clf)})
        pd.DataFrame(rows).to_csv(OUT / f"{task_id}_partial.csv", index=False)
    (OUT / f"{task_id}_partial.csv").replace(OUT / f"{task_id}_folds.csv")
    print(f"  gotowe w {perf_counter() - start:.0f}s", flush=True)


def read(pattern):
    files = glob.glob(pattern)
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def load_all():
    new = read(f"{OUT}/*_folds.csv")
    types = new.groupby("dataset").task_type.first()
    parts = [new]
    hd = read("results/regula_pelny/*_folds.csv")
    parts.append(hd[hd.config == "A_p"])
    parts.append(read("results/splitter_tylko/*_folds.csv"))
    parts.append(read("results/interakcja_2x2/*_folds.csv"))
    old = []
    rf = pd.concat([read(f"{d}/*_folds.csv") for d in RF_FULL], ignore_index=True)
    old.append(rf[rf.model == "RF domyslny"].assign(config="RF domyslny"))
    cen = read("results/srodek_A_p/*_folds.csv")  # klasyfikacja, srodek A_p, best
    old.append(cen.assign(config="RF ze srodka"))
    sp = read("results/pilot_splitter/*_folds.csv")  # regresja, srodek A (= A_p), best
    old.append(sp[(sp.model == "RF ze srodka A") & sp.rmse.notna()].assign(config="RF ze srodka"))
    old = pd.concat(old, ignore_index=True).drop(columns=["model", "task_type"], errors="ignore")
    parts.append(old.assign(task_type=old.dataset.map(types)))
    df = pd.concat(parts, ignore_index=True)
    return df[df.dataset.isin(types.index)]


def test(d):
    d = d.dropna()
    p = wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else float("nan")
    return f"mediana {d.median():+.3f}, srednia {d.mean():+.3f}, >0 na {int((d > 0).sum())}/{len(d)}, p = {p:.3f}"


def summarize():
    df = load_all()
    lines = ["Rozklad zysku: przesuniecie srodka i heterogenicznosc (pelny protokol). "
             "Zysk nad RF domyslnym: accuracy w p.p., log-loss/RMSE w %."]
    for task_type, metrics in METRICS.items():
        sub = df[df.task_type == task_type]
        if sub.empty:
            continue
        base = "RF K=p" if task_type == "classification" else "RF domyslny"
        for metric, lower_better in metrics.items():
            piv = sub.pivot_table(index=["dataset", "fold"], columns="config", values=metric)
            diff = piv.sub(piv["RF domyslny"], axis=0)
            if lower_better:
                diff = -diff.div(piv["RF domyslny"], axis=0)
            g = diff.groupby(level="dataset").mean() * 100
            cells = {"best: stale": base, "best: srodek": "RF ze srodka", "best: A_p": "A_p pi=0",
                     "50/50: stale": "tylko splitter", "50/50: srodek": NAME, "50/50: A_p": "A_p"}
            t = pd.DataFrame({k: g[v] for k, v in cells.items() if v in g})
            for row in ("best", "50/50"):
                if {f"{row}: stale", f"{row}: srodek", f"{row}: A_p"} <= set(t):
                    t[f"{row}: przesuniecie"] = t[f"{row}: srodek"] - t[f"{row}: stale"]
                    t[f"{row}: heterogen."] = t[f"{row}: A_p"] - t[f"{row}: srodek"]
            lines.append(f"\n{task_type}, {metric} ({len(t)} zbiorow)")
            lines.append(t.round(3).to_string())
            for c in t:
                lines.append(f"  {c:22s}: {test(t[c])}")
    text = "\n".join(lines)
    (OUT / "podsumowanie.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for task_id in TASKS:
            if not (OUT / f"{task_id}_folds.csv").exists():
                run_task(task_id)
    summarize()


if __name__ == "__main__":
    main()
