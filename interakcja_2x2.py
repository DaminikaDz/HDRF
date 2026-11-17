"""E0: uklad 2x2 - czy losowy splitter zwieksza wplyw losowania pozostalych hiperparametrow?

                       pozostale stale                 pozostale losowane (A_p)
  splitter best        RF K=p (klas.) / RF dom. (regr.)  A_p pi=0          <- liczone tutaj
  splitter 50/50       tylko splitter                    A_p

'RF K=p' liczymy jako HDRF bez losowania (K = p, pelne drzewa, BR = 1, best, kryterium domyslne),
zeby mial te same bootstrapy co 'tylko splitter'. W regresji RF domyslny ma juz K = p.
Pelny protokol: 10 foldow, B = 500, ziarna 42 + fold. Wyniki A_p, 'tylko splitter' i RF
wczytywane z dysku (results/regula_pelny, results/splitter_tylko, pilotaze).

    python interakcja_2x2.py [--summary-only]
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

OUT = Path("results/interakcja_2x2")
N_TREES, N_FOLDS = 500, 10
AP0, RFKP = "A_p pi=0", "RF K=p"


def models(task_type, p):
    out = {AP0: dict(sampling_config=SamplingConfig(splitter_random_prob=0.0))}
    if task_type == "classification":
        out[RFKP] = dict(randomize=(), fixed_params={"max_features": p})
    return out


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
        for name, kwargs in models(task.task_type, X_tr.shape[1]).items():
            t0 = perf_counter()
            model = Model(n_estimators=N_TREES, random_state=seed, n_jobs=-1, **kwargs).fit(X_tr, y_tr)
            rows.append({"task_id": task_id, "dataset": task.name, "task_type": task.task_type,
                         "fold": fold, "config": name, "fit_seconds": perf_counter() - t0,
                         "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                         **score(model, task.task_type, X_te, y_te, labels),
                         **tree_diagnostics(model, X_te, y_te, clf)})
        pd.DataFrame(rows).to_csv(OUT / f"{task_id}_partial.csv", index=False)
    (OUT / f"{task_id}_partial.csv").replace(OUT / f"{task_id}_folds.csv")
    print(f"  gotowe w {perf_counter() - start:.0f}s", flush=True)


def load_all():
    def read(pattern):
        return pd.concat([pd.read_csv(p) for p in glob.glob(pattern)], ignore_index=True)
    new = read(f"{OUT}/*_folds.csv")
    types = new.groupby("dataset").task_type.first()
    hd = read("results/regula_pelny/*_folds.csv")
    hd = hd[hd.config == "A_p"]
    sp = read("results/splitter_tylko/*_folds.csv")
    rf = pd.concat([read(f"{d}/*_folds.csv") for d in RF_FULL], ignore_index=True)
    rf = rf[rf.model == "RF domyslny"].assign(config="RF domyslny")
    rf = rf.drop(columns=["model", "task_type"], errors="ignore").assign(task_type=lambda r: r.dataset.map(types))
    df = pd.concat([new, hd, sp, rf], ignore_index=True)
    return df[df.dataset.isin(types.index)]


def test(d):
    d = d.dropna()
    p = wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else float("nan")
    return f"mediana {d.median():+.3f}, srednia {d.mean():+.3f}, >0 na {int((d > 0).sum())}/{len(d)}, p = {p:.3f}"


def summarize():
    df = load_all()
    lines = ["Uklad 2x2 (pelny protokol). Zysk nad RF domyslnym: accuracy w p.p., log-loss/RMSE w %."]
    for task_type, metrics in METRICS.items():
        sub = df[df.task_type == task_type]
        if sub.empty:
            continue
        base = RFKP if task_type == "classification" else "RF domyslny"
        for metric, lower_better in metrics.items():
            piv = sub.pivot_table(index=["dataset", "fold"], columns="config", values=metric)
            diff = piv.sub(piv["RF domyslny"], axis=0)
            if lower_better:
                diff = -diff.div(piv["RF domyslny"], axis=0)
            g = diff.groupby(level="dataset").mean() * 100
            cells = {"best, stale": base, "best, losowane": AP0, "50/50, stale": "tylko splitter",
                     "50/50, losowane": "A_p"}
            t = pd.DataFrame({k: g[v] for k, v in cells.items()})
            t["efekt HP | best"] = t["best, losowane"] - t["best, stale"]
            t["efekt HP | 50/50"] = t["50/50, losowane"] - t["50/50, stale"]
            t["interakcja"] = t["efekt HP | 50/50"] - t["efekt HP | best"]
            t["efekt splittera | stale"] = t["50/50, stale"] - t["best, stale"]
            t["efekt splittera | losowane"] = t["50/50, losowane"] - t["best, losowane"]
            lines.append(f"\n{task_type}, {metric} ({len(t)} zbiorow)")
            lines.append(t.round(3).to_string())
            for c in ("best, stale", "best, losowane", "50/50, stale", "50/50, losowane"):
                lines.append(f"  {c:16s} vs RF: mediana {t[c].median():+.3f}, wygrane {int((t[c] > 0).sum())}/{len(t)}")
            for c in ("efekt HP | best", "efekt HP | 50/50", "interakcja",
                      "efekt splittera | stale", "efekt splittera | losowane"):
                lines.append(f"  {c:28s}: {test(t[c])}")
            lines.append(f"  rozrzut efektu HP miedzy zbiorami (odch. std / sredni |efekt|): "
                         f"best {t['efekt HP | best'].std():.3f} / {t['efekt HP | best'].abs().mean():.3f}, "
                         f"50/50 {t['efekt HP | 50/50'].std():.3f} / {t['efekt HP | 50/50'].abs().mean():.3f}")
    times = df[df.config.isin((AP0, RFKP, "A_p", "tylko splitter"))].groupby("config").fit_seconds.sum() / 60
    lines.append("\nCzas uczenia (min, suma; stare przebiegi moga byc zawyzone przez sen komputera): "
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
