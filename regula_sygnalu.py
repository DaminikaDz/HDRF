"""E0: regula sygnalu (stale zlozonosci zalezne od wyniku OOB lasu pilotazowego).

Hipoteza z przesiewu zlozonosci: zbiory z silnym sygnalem (wysoki wynik RF) wola pelne
drzewa (maly leaf_max, wiecej trybu depth), zbiory zaszumione - wygladzone (duzy leaf_max).
Regula: s = OOB accuracy / OOB R^2 lasu pilotazowego (100 drzew, domyslne RF);
s >= 0,78 -> leaf_max = 2 (i P_depth = 0,7), s < 0,78 -> leaf_max = 12 (i P_depth = 0,3).

Warianty: 'regula leaf' (tylko leaf_max) i 'regula leaf+depth' (leaf_max i P_depth).
Ta sama para ziaren co A_p: rozni sie tylko rozklad zlozonosci (porownanie parami).

    python regula_sygnalu.py --protokol przesiew   # 5 foldow, B = 150; A_p i RF z kalibracja_zlozonosc
    python regula_sygnalu.py --protokol pelny      # 10 foldow, B = 500; A_p liczony ponownie, RF z pilotazy
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

RULES = {"regula leaf": SamplingConfig(signal_rule="leaf"),
         "regula leaf+depth": SamplingConfig(signal_rule="leaf_depth")}
PROTOCOLS = {
    "przesiew": dict(folds=5, trees=150, configs=RULES, out="results/regula_przesiew",
                     reference=("results/kalibracja_zlozonosc", "config")),
    "pelny": dict(folds=10, trees=500, configs={"A_p": SamplingConfig(), **RULES},
                  out="results/regula_pelny", reference=None),
}
RF_FULL = ("results/pilot_dev16", "results/pilot_e0")
METRICS = {"classification": {"accuracy": False, "log_loss": True}, "regression": {"rmse": True}}


def run_task(task_id, configs, n_trees, n_folds, out_dir, diagnostics):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    rows = []
    print(f"\n=== {task.name} ({task.task_type}) ===", flush=True)
    start = perf_counter()
    for fold in range(n_folds):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold
        for name, config in configs.items():
            t0 = perf_counter()
            model = Model(n_estimators=n_trees, random_state=seed, sampling_config=config,
                          n_jobs=-1).fit(X_tr, y_tr)
            eff = model.effective_config_
            row = {"task_id": task_id, "dataset": task.name, "task_type": task.task_type,
                   "fold": fold, "config": name, "fit_seconds": perf_counter() - t0,
                   "signal": model.signal_, "leaf_max": eff.leaf_max, "p_depth": eff.depth_mode_prob,
                   "median_leaves": np.median([t.get_n_leaves() for t in model.estimators_]),
                   **score(model, task.task_type, X_te, y_te, labels)}
            if diagnostics:
                row.update(tree_diagnostics(model, X_te, y_te, clf))
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"{task_id}_partial.csv", index=False)
        print(f"  fold {fold}: s = {rows[-1]['signal']:.3f}, leaf_max = {rows[-1]['leaf_max']}", flush=True)
    (out_dir / f"{task_id}_partial.csv").replace(out_dir / f"{task_id}_folds.csv")
    print(f"  gotowe w {perf_counter() - start:.0f}s", flush=True)


def load_all(protocol, out_dir):
    df = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("*_folds.csv"))], ignore_index=True)
    ref = PROTOCOLS[protocol]["reference"]
    if ref:  # A_p i RF z kalibracji zlozonosci (te same ziarna, foldy, B)
        old = pd.concat([pd.read_csv(p) for p in glob.glob(f"{ref[0]}/*_folds.csv")], ignore_index=True)
        old = old[old.config.isin(("A_p", "RF domyslny"))]
        old = old[old.dataset.isin(df.dataset.unique())]
        df = pd.concat([df, old], ignore_index=True)
    else:
        rf = pd.concat([pd.read_csv(p) for d in RF_FULL for p in glob.glob(f"{d}/*_folds.csv")],
                       ignore_index=True)
        rf = rf[(rf.model == "RF domyslny") & rf.dataset.isin(df.dataset.unique())]
        # pilot_e0 nie ma kolumny task_type: bierzemy ja z nowych wynikow.
        types = df.groupby("dataset").task_type.first()
        rf = rf.drop(columns=["model", "task_type"], errors="ignore").assign(
            config="RF domyslny", task_type=lambda r: r.dataset.map(types))
        df = pd.concat([df, rf], ignore_index=True)
    return df


def gains(sub, metric, lower_better):
    """Zysk nad RF na zbior: accuracy w p.p., straty jako wzgledny spadek w %."""
    piv = sub.pivot_table(index=["dataset", "fold"], columns="config", values=metric)
    diff = piv.sub(piv["RF domyslny"], axis=0)
    if lower_better:
        diff = -diff.div(piv["RF domyslny"], axis=0)
    return diff.drop(columns="RF domyslny").groupby(level="dataset").mean() * 100


def summarize(protocol, out_dir):
    df = load_all(protocol, out_dir)
    lines = [f"Regula sygnalu, protokol: {protocol}. Zysk nad RF: accuracy w p.p., "
             f"log-loss/RMSE jako wzgledny spadek w %. Roznica = regula - A_p."]
    rules = df[df.config.isin(RULES)]
    sig = rules.groupby("dataset").agg(s=("signal", "mean"), s_min=("signal", "min"),
                                       s_max=("signal", "max"), leaf_max=("leaf_max", "mean"))
    for task_type, metrics in METRICS.items():
        sub = df[df.task_type == task_type]
        if sub.empty:
            continue
        for metric, lower_better in metrics.items():
            g = gains(sub, metric, lower_better)
            names = [c for c in ("A_p", *RULES) if c in g]
            table = g[names].join(sig)
            for rule in RULES:
                if rule in g:
                    table[f"{rule} - A_p"] = g[rule] - g["A_p"]
            lines.append(f"\n{task_type}, {metric} ({len(g)} zbiorow)")
            lines.append(table.sort_values("s").round(3).to_string())
            for name in names:
                lines.append(f"  {name:18s} vs RF: mediana {g[name].median():+.3f}, "
                             f"wygrane {int((g[name] > 0).sum())}/{len(g)}")
            for rule in RULES:
                if rule not in g:
                    continue
                d = g[rule] - g["A_p"]
                p = wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else float("nan")
                lines.append(f"  {rule:18s} - A_p: mediana {d.median():+.3f}, srednia {d.mean():+.3f}, "
                             f"lepsza na {int((d > 0).sum())}/{len(d)}, Wilcoxon p = {p:.3f}")
    times = df[df.config != "RF domyslny"].groupby("config").fit_seconds.sum() / 60
    lines.append("\nCzas uczenia (min, suma): " + ", ".join(f"{k} {v:.1f}" for k, v in times.items()))
    text = "\n".join(lines)
    (out_dir / "podsumowanie.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protokol", choices=PROTOCOLS, required=True)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    spec = PROTOCOLS[args.protokol]
    out_dir = Path(spec["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for task_id in TASKS:
            if (out_dir / f"{task_id}_folds.csv").exists():
                continue  # wznowienie po przerwaniu
            run_task(task_id, spec["configs"], spec["trees"], spec["folds"], out_dir,
                     diagnostics=args.protokol == "pelny")
    summarize(args.protokol, out_dir)


if __name__ == "__main__":
    main()
