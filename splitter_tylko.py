"""E0: izolacja efektu splittera. HDRF, w ktorym losujemy tylko splitter (50/50 best/random),
a reszta jest stala: K = p (obie klasy zadan), pelne drzewa, bootstrap BR = 1, kryterium domyslne.

Porownanie (te same ziarna 42 + fold, 10 foldow, B = 500) z A_p i regula leaf+depth
(results/regula_pelny), RF domyslnym (pilotaze) i Extra Trees (results/pilot_splitter).
Pytanie: czy losowanie pozostalych hiperparametrow cokolwiek daje poza mieszanka splitterow.

    python splitter_tylko.py [--summary-only]
"""

import argparse
import glob
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig
from hdrf.diagnostics import score, tree_diagnostics
from kalibracja_przesiew import BASE_SEED, TASKS
from regula_sygnalu import METRICS, RF_FULL

NAME = "tylko splitter"
OUT = Path("results/splitter_tylko")
N_TREES, N_FOLDS = 500, 10


def run_task(task_id):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    rows = []
    print(f"\n=== {task.name} ({task.task_type}) ===", flush=True)
    start = perf_counter()
    for fold in range(N_FOLDS):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold
        t0 = perf_counter()
        model = Model(n_estimators=N_TREES, randomize=("splitter",), random_state=seed,
                      sampling_config=SamplingConfig(splitter_random_prob=0.5),
                      fixed_params={"max_features": X_tr.shape[1]}, n_jobs=-1).fit(X_tr, y_tr)
        rows.append({"task_id": task_id, "dataset": task.name, "task_type": task.task_type,
                     "fold": fold, "config": NAME, "fit_seconds": perf_counter() - t0,
                     "share_random": np.mean([p["splitter"] == "random" for p in model.tree_params_]),
                     "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                     **score(model, task.task_type, X_te, y_te, labels),
                     **tree_diagnostics(model, X_te, y_te, clf)})
        pd.DataFrame(rows).to_csv(OUT / f"{task_id}_partial.csv", index=False)
    (OUT / f"{task_id}_partial.csv").replace(OUT / f"{task_id}_folds.csv")
    print(f"  gotowe w {perf_counter() - start:.0f}s", flush=True)


def load_all():
    new = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("*_folds.csv"))], ignore_index=True)
    types = new.groupby("dataset").task_type.first()
    hd = pd.concat([pd.read_csv(p) for p in glob.glob("results/regula_pelny/*_folds.csv")], ignore_index=True)
    hd = hd[hd.config.isin(("A_p", "regula leaf+depth"))]
    rf = pd.concat([pd.read_csv(p) for d in RF_FULL for p in glob.glob(f"{d}/*_folds.csv")], ignore_index=True)
    rf = rf[rf.model == "RF domyslny"].assign(config="RF domyslny")
    sp = pd.concat([pd.read_csv(p) for p in glob.glob("results/pilot_splitter/*_folds.csv")], ignore_index=True)
    et = sp[sp.model == "Extra Trees"].assign(config="Extra Trees")
    old = pd.concat([rf, et], ignore_index=True).drop(columns=["model", "task_type"], errors="ignore")
    old = old.assign(task_type=old.dataset.map(types))
    df = pd.concat([new, hd, old], ignore_index=True)
    return df[df.dataset.isin(types.index)]


def summarize():
    df = load_all()
    order = [NAME, "A_p", "regula leaf+depth", "Extra Trees"]
    pairs = [("A_p", NAME), ("regula leaf+depth", NAME), (NAME, "Extra Trees")]
    lines = ["Izolacja splittera (pelny protokol). Zysk nad RF: accuracy w p.p., "
             "log-loss/RMSE jako wzgledny spadek w %."]
    for task_type, metrics in METRICS.items():
        sub = df[df.task_type == task_type]
        if sub.empty:
            continue
        for metric, lower_better in metrics.items():
            piv = sub.pivot_table(index=["dataset", "fold"], columns="config", values=metric)
            diff = piv.sub(piv["RF domyslny"], axis=0)
            if lower_better:
                diff = -diff.div(piv["RF domyslny"], axis=0)
            g = diff.drop(columns="RF domyslny").groupby(level="dataset").mean() * 100
            g = g[[c for c in order if c in g]]
            lines.append(f"\n{task_type}, {metric} ({len(g)} zbiorow)")
            lines.append(g.round(3).to_string())
            for c in g:
                lines.append(f"  {c:18s} vs RF: mediana {g[c].median():+.3f}, wygrane {int((g[c] > 0).sum())}/{len(g)}")
            for a, b in pairs:
                if a in g and b in g:
                    d = (g[a] - g[b]).dropna()
                    p = wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else float("nan")
                    lines.append(f"  {a} - {b}: mediana {d.median():+.3f}, srednia {d.mean():+.3f}, "
                                 f"lepszy na {int((d > 0).sum())}/{len(d)}, Wilcoxon p = {p:.3f}")
        diag = sub.groupby("config")[["tree_corr", "amb_share", "median_leaves"]].mean()
        lines.append("  diagnostyka (srednie): \n" + diag.round(3).to_string())
    times = df[df.config.isin(order)].groupby("config").fit_seconds.sum() / 60
    lines.append("\nCzas uczenia (min, suma; A_p/regula moga byc zawyzone przez sen komputera): "
                 + ", ".join(f"{k} {v:.1f}" for k, v in times.items()))
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
