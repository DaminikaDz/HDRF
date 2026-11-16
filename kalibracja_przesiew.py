"""E0, kalibracja stalych: przesiew (ograniczony protokol).

Kalibrujemy 4 stale: pi (splitter_random_prob), a (features_down), w (features_full_prob)
i BR_low (bootstrap_bounds = (BR_low, 1.0)). Zlozonosc zamrozona (P(depth) = 0.5, c = 1,
leaf_max = 5, log_full).

Konfiguracje: A_p (obecne stale, znany wynik w pelnym protokole), 2 kontrole zlozonosci
(A_p z leaf_max = 3 i 10), N konfiguracji z LHS (Latin hypercube sampling) i RF domyslny
jako punkt odniesienia (nie bierze udzialu w rankingu).
Ograniczony protokol: 16 zbiorow dev, foldy 0-4, B = 150, wspolne ziarna dla wszystkich
konfiguracji (rozne konfiguracje widza te same foldy i bootstrapy). Wybor: srednia ranga.

    python kalibracja_przesiew.py [--configs 20] [--trees 150] [--folds 5] [--summary-only]
"""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig
from hdrf.diagnostics import score

BASE_SEED = 42
LHS_SEED = 2026
# Od najtanszych.
TASKS = (361055, 361077, 361070, 361278, 361275, 361065, 361082, 361085, 361097, 361074,
         361282, 361279, 361079, 361068, 361274, 361292)
RANGES = {"pi": (0.1, 0.9), "a": (1.5, 4.0), "w": (0.3, 0.9), "br_low": (0.5, 0.95)}
REFERENCE = "A_p"
RF = "RF domyslny"
# Wyniki pelnego protokolu (10 foldow, B = 500) z pilotazy etapu 1.
FULL = {"classification": ("results/pilot_k_p", "HDRF A_p (pi=0.5)"),
        "regression": ("results/pilot_splitter", "HDRF A (splitter)")}
FULL_RF = ("results/pilot_dev16", "results/pilot_e0")


def make_configs(n_lhs):
    base = SamplingConfig()
    configs = {REFERENCE: base,
               "A_p leaf_max=3": replace(base, leaf_max=3),
               "A_p leaf_max=10": replace(base, leaf_max=10)}
    lows, highs = zip(*RANGES.values())
    points = qmc.scale(qmc.LatinHypercube(d=len(RANGES), seed=LHS_SEED,
                                          optimization="random-cd").random(n_lhs), lows, highs)
    for i, (pi, a, w, br_low) in enumerate(points):
        configs[f"LHS{i:02d}"] = replace(base, splitter_random_prob=float(pi), features_down=float(a),
                                         features_full_prob=float(w),
                                         bootstrap_bounds=(float(br_low), 1.0))
    return configs


def constants(config):
    return {"pi": config.splitter_random_prob, "a": config.features_down,
            "w": config.features_full_prob, "br_low": config.bootstrap_bounds[0],
            "leaf_max": config.leaf_max}


def run_task(task_id, configs, n_trees, n_folds, out_dir):
    task = load_e0_task(task_id)
    clf = task.task_type == "classification"
    labels = np.unique(task.y) if clf else None
    Model = HDRFClassifier if clf else HDRFRegressor
    RFModel = RandomForestClassifier if clf else RandomForestRegressor
    rows = []
    print(f"\n=== {task.name} ({task.task_type}) ===", flush=True)
    start = perf_counter()
    for fold in range(n_folds):
        X_tr, X_te, y_tr, y_te = task.get_fold(fold)
        seed = BASE_SEED + fold
        base = {"task_id": task_id, "dataset": task.name, "task_type": task.task_type, "fold": fold}
        t0 = perf_counter()
        rf = RFModel(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        rows.append({**base, "config": RF, "fit_seconds": perf_counter() - t0,
                     **score(rf, task.task_type, X_te, y_te, labels)})
        for name, config in configs.items():
            t0 = perf_counter()
            model = Model(n_estimators=n_trees, random_state=seed, sampling_config=config,
                          n_jobs=-1).fit(X_tr, y_tr)
            rows.append({**base, "config": name, "fit_seconds": perf_counter() - t0,
                         **constants(config), **score(model, task.task_type, X_te, y_te, labels)})
        pd.DataFrame(rows).to_csv(out_dir / f"{task_id}_partial.csv", index=False)
    (out_dir / f"{task_id}_partial.csv").replace(out_dir / f"{task_id}_folds.csv")
    print(f"  gotowe w {perf_counter() - start:.0f}s", flush=True)


METRICS = {"classification": {"accuracy": False, "log_loss": True}, "regression": {"rmse": True}}


def full_protocol(task_type):
    """Zysk A_p wzgledem RF w pelnym protokole (B = 500), na wszystkich foldach i na `folds`."""
    folder, model = FULL[task_type]
    frames = [pd.read_csv(p) for p in Path(folder).glob("*_folds.csv")]
    for rf_folder in FULL_RF:
        frames += [pd.read_csv(p) for p in Path(rf_folder).glob("*_folds.csv")]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.model.isin((model, RF))]
    return df.assign(config=df.model.replace({model: REFERENCE}))


def gains_vs_rf(df, metric, lower_better):
    """Na zbior: sredni po foldach zysk wzgledem RF (accuracy: roznica, reszta: wzgledny spadek)."""
    piv = df.pivot_table(index=["dataset", "fold"], columns="config", values=metric)
    diff = piv.sub(piv[RF], axis=0)
    if lower_better:
        diff = -diff.div(piv[RF], axis=0)
    return diff.drop(columns=RF).groupby(level="dataset").mean()


def summarize(out_dir, n_folds):
    files = sorted(out_dir.glob("*_folds.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    lines, table = [], []
    for task_type, metrics in METRICS.items():
        sub = df[df.task_type == task_type]
        if sub.empty:
            continue
        n_sets = sub.dataset.nunique()
        consts = (sub[sub.config != RF].groupby("config")[list(RANGES) + ["leaf_max"]].first())
        out = consts.copy()
        for metric, lower_better in metrics.items():
            means = sub[sub.config != RF].pivot_table(index="dataset", columns="config", values=metric)
            out[f"rank {metric}"] = means.rank(axis=1, ascending=lower_better).mean()
            gains = gains_vs_rf(sub, metric, lower_better)
            out[f"zysk vs RF {metric} (mediana)"] = gains.median()
            out[f"wygr. z RF {metric}"] = (gains > 0).sum()
        rank_cols = [c for c in out if c.startswith("rank ")]
        out["rank srednia"] = out[rank_cols].mean(axis=1)
        out = out.sort_values("rank srednia")
        out.insert(0, "task_type", task_type)
        table.append(out)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            lines.append(f"\n{task_type} ({n_sets} zbiorow, {len(out)} konfiguracji); "
                         f"zysk: accuracy jako roznica (ulamek), reszta wzgledny spadek straty")
            lines.append(out.drop(columns="task_type").to_string(float_format=lambda v: f"{v:.4g}"))
        lhs = out[out.index.str.startswith("LHS") | (out.index == REFERENCE)]
        lines.append("  Spearman: stala vs srednia ranga (ujemny = wieksza wartosc lepsza)")
        for const in RANGES:
            rho, p = spearmanr(lhs[const], lhs["rank srednia"])
            lines.append(f"    {const}: rho = {rho:+.2f} (p = {p:.3f})")
        # A_p: ograniczony vs pelny protokol (czy przesiew mowi to samo).
        full = full_protocol(task_type)
        full = full[full.dataset.isin(sub.dataset.unique())]
        for metric, lower_better in metrics.items():
            limited = gains_vs_rf(sub, metric, lower_better)[REFERENCE]
            full_all = gains_vs_rf(full, metric, lower_better)[REFERENCE]
            full_5 = gains_vs_rf(full[full.fold < n_folds], metric, lower_better)[REFERENCE]
            lines.append(f"  A_p vs RF, {metric}: przesiew {limited.median():+.4%} ({int((limited > 0).sum())}/"
                         f"{len(limited)}), pelny 5 foldow {full_5.median():+.4%} "
                         f"({int((full_5 > 0).sum())}/{len(full_5)}), pelny 10 foldow "
                         f"{full_all.median():+.4%} ({int((full_all > 0).sum())}/{len(full_all)}), "
                         f"Spearman przesiew-pelny po zbiorach {spearmanr(limited, full_all.loc[limited.index])[0]:+.2f}")
    pd.concat(table).to_csv(out_dir / "summary.csv")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=int, default=20)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/kalibracja_przesiew"))
    parser.add_argument("--tasks", type=int, nargs="*", default=TASKS)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = make_configs(args.configs)
    pd.DataFrame({n: constants(c) for n, c in configs.items()}).T.to_csv(args.out / "configs.csv")
    if not args.summary_only:
        for task_id in args.tasks:
            if (args.out / f"{task_id}_folds.csv").exists():
                continue  # wznowienie po przerwaniu
            run_task(task_id, configs, args.trees, args.folds, args.out)
            summarize(args.out, args.folds)
    summarize(args.out, args.folds)


if __name__ == "__main__":
    main()
