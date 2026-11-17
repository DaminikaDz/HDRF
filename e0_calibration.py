"""Kontrolowana kalibracja max_features dla HDRF Mateusza (E0).

Ten plik jest DODATKOWYM skryptem. Nie modyfikuje hdrf/sampling.py,
hdrf/forest.py, hdrf/center.py ani zadnych istniejacych plikow projektu.

Punkt odniesienia to dwa wybrane warianty:
  - klasyfikacja: HDRF A_p (pi=0.5), K0=p;
  - regresja: HDRF A (splitter), K0=p.

Wszystkie wspolczynniki pozostaja takie jak u Mateusza:
  depth_distribution='log_full', leaf_max=5, depth_mode_prob=0.5,
  depth_low_coef=1, splitter_random_prob=0.5, features_default_prob=0.3,
  features_down=2, features_up=2, bootstrap_bounds=(0.7, 1.0),
  n_pilot_trees=5 oraz oryginalne rozklady criterion.

Skrypt porownuje tylko jeden wspolczynnik naraz (one-factor-at-a-time):
  q = features_default_prob: zmienia mase w K0; granice K pozostaja te same.
  a = features_down: zmienia dolna granice K0/a.
  b = features_up: zmienia gorna granice K0*b PRZED obcieciem do p.

Wazne dla K0=p: wieksze b nie zwieksza maksymalnego K powyzej p,
ale zwieksza mase prawdopodobienstwa w K=p. Testowane alternatywy
nie sa przyjetymi rozkladami. Bazowy wariant Mateusza jest zawsze obecny.
Nie laczymy niezaleznie najlepszych a, b i q bez dodatkowego sprawdzenia.

Przyklady w terminalu CMD/PowerShell, z katalogu glownego projektu:
  python e0_calibration.py --dry-run
  python e0_calibration.py --task 361055 --parameter q --trees 100 --folds 2
  python e0_calibration.py --task 361077 --parameter all --trees 100 --folds 2
  python e0_calibration.py --all --parameter q --trees 500 --folds 10

Ostatnia komenda jest BARDZO kosztowna. Domyslnie testujemy tylko dwa
zbiory i dwa foldy; takich wynikow NIE nazywamy koncowym E0.

Wyniki: results/e0_calibration/<identyfikator_eksperymentu>/
  protocol.json, folds.csv, datasets.csv, ranking.csv,
  statistics.csv, omnibus.csv
Mozna wznowic identyczne uruchomienie: wykonane modele sa pomijane.
Identyfikator uwzglednia liste zadan i wartosci kandydatow, wiec rozne
eksperymenty nie nadpisuja swoich wynikow.

Testy statystyczne wykonujemy na SREDNICH Z FOLDOW OSOBNO DLA ZBIORU,
nie na pojedynczych foldach. Porownania Wilcoxona z baza Mateusza maja
korekte Holma w obrebie kazdej rodziny (q, a lub b). Dla Friedmana
raportujemy p z permutacji oraz orientacyjne p asymptotyczne.
Przy mniej niz 5 kompletnych zbiorach nie raportujemy p-value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

from data_loader import load_e0_task
from fetch_e0 import TASKS
from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig


BASE = SamplingConfig(
    depth_distribution="log_full",
    leaf_max=5,
    splitter_random_prob=0.5,
)

# To tylko JAWNE KANDYDATY do eksperymentu, nie zmiany domyslnych granic HDRF.
# Jedna wartosc jest zmieniana naraz, zawsze wzgledem BASE Mateusza.
DEFAULT_Q = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0)
DEFAULT_A = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
DEFAULT_B = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
N_PERMUTATIONS = 10_000
MIN_DATASETS_FOR_TEST = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    selected = parser.add_mutually_exclusive_group()
    selected.add_argument("--all", action="store_true", help="wszystkie 16 zbiorow E0")
    selected.add_argument("--task", type=int, action="append", choices=sorted(TASKS),
                          help="jeden lub kilka ID OpenML (flage mozna powtarzac)")
    parser.add_argument("--parameter", choices=("q", "a", "b", "all"), default="q",
                        help="ktory wspolczynnik badamy; 'all' nadal oznacza osobne ablacje")
    parser.add_argument("--q-values", type=float, nargs="+", default=DEFAULT_Q)
    parser.add_argument("--a-values", type=float, nargs="+", default=DEFAULT_A)
    parser.add_argument("--b-values", type=float, nargs="+", default=DEFAULT_B)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--folds", type=int, default=2, help="pierwsze N z 10 foldow OpenML")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/e0_calibration"))
    parser.add_argument("--dry-run", action="store_true", help="pokaz plan bez trenowania")
    args = parser.parse_args()
    if args.trees < 1 or not 1 <= args.folds <= 10 or args.seed < 0:
        parser.error("wymagane: --trees >=1, --folds od 1 do 10, --seed >=0")
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs musi byc -1 albo dodatnia liczba calkowita")
    for value in args.q_values:
        if not 0 <= value <= 1:
            parser.error("kazde q musi nalezec do [0,1]")
    for name, values in (("a", args.a_values), ("b", args.b_values)):
        if any(value < 1 for value in values):
            parser.error(f"kazde {name} musi byc >=1 (jak w SamplingConfig)")
    return args


def candidates(args):
    """Baseline + tylko jedna zmiana parametru; bez gridu krzyzowego."""
    variants = {"base_Mateusz": {}}

    def add(letter, values, name, default):
        for value in values:
            if np.isclose(value, default):
                continue
            label = f"{letter}_{value:g}"
            variants[label] = {name: float(value)}

    if args.parameter in ("q", "all"):
        add("q", args.q_values, "features_default_prob", 0.3)
    if args.parameter in ("a", "all"):
        add("a", args.a_values, "features_down", 2.0)
    if args.parameter in ("b", "all"):
        add("b", args.b_values, "features_up", 2.0)
    return variants


def config_for_task(task_type, changes):
    # K0=p W OBU wybranych modelach. W klasyfikacji to dodatkowa zmiana
    # z HDRF A na A_p; w regresji p jest wartoscia domyslna samplera.
    config = (replace(BASE, classification_k0="p")
              if task_type == "classification" else BASE)
    return replace(config, **changes) if changes else config


def score(model, X_test, y_test, task_type, all_classes=None):
    pred = model.predict(X_test)
    if task_type == "regression":
        return {"accuracy": np.nan, "log_loss": np.nan,
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "r2": float(r2_score(y_test, pred))}

    # Pelny zestaw klas zapewnia zgodnosc kolumn predict_proba, nawet gdy
    # jakas rzadka klasa nie wystapila w konkretnym foldzie treningowym.
    p = model.predict_proba(X_test)
    full = np.zeros((len(y_test), len(all_classes)), dtype=np.float64)
    positions = np.searchsorted(all_classes, model.classes_)
    full[:, positions] = p
    return {"accuracy": float(accuracy_score(y_test, pred)),
            "log_loss": float(log_loss(y_test, np.clip(full, 1e-15, 1.0),
                                       labels=all_classes)),
            "rmse": np.nan, "r2": np.nan}


def atomic_csv(frame, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def permutation_friedman_p(ranks, task_type, family):
    """Monte Carlo: permutujemy etykiety wariantow w kazdym zbiorze osobno.

    Rangi z remisami zachowujemy bez zmian, a p-value wyznaczamy z rozkladu
    permutacyjnego. Jednostka replikacji jest caly zbior danych, NIE fold.
    """
    n_datasets, n_variants = ranks.shape
    center = (n_variants + 1) / 2.0
    observed = float(np.sum((ranks.mean(axis=0) - center) ** 2))
    seed = 2026 + sum(ord(char) for char in f"{task_type}_{family}")
    rng = np.random.default_rng(seed)
    exceeds = 0
    for _ in range(N_PERMUTATIONS):
        permuted = np.array([row[rng.permutation(n_variants)] for row in ranks])
        statistic = np.sum((permuted.mean(axis=0) - center) ** 2)
        exceeds += int(statistic >= observed - 1e-12)
    return (exceeds + 1) / (N_PERMUTATIONS + 1)


def holm_correction(p_values):
    """Skorygowane p-value Holma (bez dodatkowej zaleznosci statsmodels)."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    multiplicity = len(p_values) - np.arange(len(p_values))
    adjusted_sorted = np.maximum.accumulate(multiplicity * p_values[order])
    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    return adjusted


def statistical_tests(complete_rows, variants, output_dir):
    """Wilcoxon + Holm i Friedman, osobno dla q/a/b i klasyfikacji/regresji.

    Do testow wchodza tylko kompletne srednie po foldach. Kazda rodzina
    uzywa tych samych zbiorow dla wszystkich swoich wariantow.
    """
    result_columns = [
        "task_type", "parameter", "variant", "datasets", "metric", "mean_rank",
        "wins_vs_base_Mateusz", "median_gain_vs_base_Mateusz", "gain_unit",
        "wilcoxon_p", "wilcoxon_p_holm", "significant_holm_005",
    ]
    omnibus_columns = [
        "task_type", "parameter", "datasets", "variants", "friedman_chi2",
        "friedman_p_asymptotic", "friedman_p_permutation",
        "n_permutations", "comment",
    ]
    results = []
    omnibus = []

    if complete_rows:
        data = pd.DataFrame(complete_rows)
        for task_type, subset in data.groupby("task_type"):
            metric = "accuracy" if task_type == "classification" else "rmse"
            for family in ("q", "a", "b"):
                names = ["base_Mateusz"] + [
                    name for name in variants if name.startswith(f"{family}_")
                ]
                if len(names) < 2:
                    continue

                wide = subset.pivot(index="task_id", columns="variant", values="metric_mean")
                wide = wide.reindex(columns=names).dropna()
                n = len(wide)
                if n == 0:
                    continue

                rank_frame = wide.rank(axis=1, method="average",
                                       ascending=(task_type == "regression"))
                friedman_stat = np.nan
                friedman_asymptotic = np.nan
                friedman_permutation = np.nan
                if n >= MIN_DATASETS_FOR_TEST and len(names) >= 3:
                    values = [wide[name].to_numpy(dtype=float) for name in names]
                    # Gdy wszystkie warianty daja identyczne wyniki, nie ma
                    # roznic do testowania; przypisujemy p=1.
                    if np.allclose(wide.to_numpy(), wide[names[0]].to_numpy()[:, None],
                                   rtol=0, atol=1e-12):
                        friedman_stat, friedman_asymptotic, friedman_permutation = 0.0, 1.0, 1.0
                    else:
                        friedman = friedmanchisquare(*values)
                        friedman_stat = float(friedman.statistic)
                        friedman_asymptotic = float(friedman.pvalue)
                        friedman_permutation = permutation_friedman_p(
                            rank_frame.to_numpy(dtype=float), task_type, family
                        )

                omnibus.append({
                    "task_type": task_type, "parameter": family, "datasets": n,
                    "variants": len(names), "friedman_chi2": friedman_stat,
                    "friedman_p_asymptotic": friedman_asymptotic,
                    "friedman_p_permutation": friedman_permutation,
                    "n_permutations": (N_PERMUTATIONS if n >= MIN_DATASETS_FOR_TEST
                                       and len(names) >= 3 else 0),
                    "comment": ("eksploracyjny test; asymptotyczne p orientacyjne"
                                if n >= MIN_DATASETS_FOR_TEST and len(names) >= 3
                                else "za malo zbiorow lub wariantow dla testu Friedmana"),
                })

                family_rows = []
                for name in names[1:]:
                    if task_type == "classification":
                        gains = (wide[name] - wide["base_Mateusz"]) * 100.0
                        unit = "punkty procentowe accuracy"
                    else:
                        denominator = wide["base_Mateusz"].replace(0, np.nan)
                        gains = (wide["base_Mateusz"] - wide[name]) / denominator * 100.0
                        unit = "% spadku RMSE"
                    valid_gains = gains.dropna()
                    p = np.nan
                    if len(valid_gains) >= MIN_DATASETS_FOR_TEST:
                        values = valid_gains.to_numpy(dtype=float)
                        if np.allclose(values, 0.0, rtol=0, atol=1e-12):
                            p = 1.0
                        else:
                            p = float(wilcoxon(values, alternative="two-sided",
                                               zero_method="wilcox", method="auto").pvalue)
                    family_rows.append({
                        "task_type": task_type, "parameter": family, "variant": name,
                        "datasets": int(len(valid_gains)), "metric": metric,
                        "mean_rank": float(rank_frame[name].mean()),
                        "wins_vs_base_Mateusz": int((valid_gains > 0).sum()),
                        "median_gain_vs_base_Mateusz": float(valid_gains.median()),
                        "gain_unit": unit, "wilcoxon_p": p,
                        "wilcoxon_p_holm": np.nan,
                        "significant_holm_005": False,
                    })

                # Jedna korekta Holma na wszystkie porownania z baza
                # w danej rodzinie, a nie na kazdy test z osobna.
                tested_indices = [i for i, item in enumerate(family_rows)
                                  if np.isfinite(item["wilcoxon_p"])]
                if tested_indices:
                    p_values = [family_rows[i]["wilcoxon_p"] for i in tested_indices]
                    adjusted = holm_correction(p_values)
                    for i, corrected in zip(tested_indices, adjusted):
                        family_rows[i]["wilcoxon_p_holm"] = float(corrected)
                        family_rows[i]["significant_holm_005"] = bool(corrected <= 0.05)
                results.extend(family_rows)

    atomic_csv(pd.DataFrame(results, columns=result_columns), output_dir / "statistics.csv")
    atomic_csv(pd.DataFrame(omnibus, columns=omnibus_columns), output_dir / "omnibus.csv")
    if results:
        n_tested = sum(np.isfinite(r["wilcoxon_p_holm"]) for r in results)
        print(f"Testy statystyczne: {n_tested} porownan Wilcoxona z korekta Holma "
              f"(min. {MIN_DATASETS_FOR_TEST} kompletnych zbiorow).")
    else:
        print("Testy statystyczne: brak kompletnych porownan do obliczenia.")


def summarize(frame, variants, n_folds, output_dir):
    """Porownania wewnatrz zbioru; nie usredniamy bezposrednio RMSE roznych zbiorow."""
    if frame.empty:
        return

    dataset_rows = []
    ranking_rows = []
    complete_rows = []
    variant_names = ["RF_domyslny"] + list(variants)

    for (task_id, dataset, task_type), group in frame.groupby(
            ["task_id", "dataset", "task_type"], sort=False):
        metric = "accuracy" if task_type == "classification" else "rmse"
        by_name = {name: g.set_index("fold") for name, g in group.groupby("variant")}
        base = by_name.get("RF_domyslny")
        if base is None:
            continue

        # Rangi i wnioski miedzy zbiorami wyliczamy tylko dla zbiorow,
        # w ktorych WSZYSTKIE warianty maja te same kompletne foldy.
        expected = set(range(n_folds))
        complete = all(name in by_name and set(by_name[name].index) == expected
                       for name in variant_names)
        measured = {}
        for name in variant_names:
            part = by_name.get(name)
            if part is None:
                continue
            common = part.index.intersection(base.index)
            if len(common):
                mine = part.loc[common, metric].astype(float)
                rf = base.loc[common, metric].astype(float)
                if task_type == "classification":
                    gain = (mine - rf) * 100.0  # punkty procentowe accuracy
                else:
                    gain = (rf - mine) / rf * 100.0  # spadek RMSE w %
                mean_gain = float(gain.mean())
            else:
                mean_gain = np.nan
            metric_mean = float(part[metric].mean())
            dataset_rows.append({
                "task_id": task_id, "dataset": dataset, "task_type": task_type,
                "variant": name, "metric": metric,
                "metric_mean": metric_mean, "folds": part.index.nunique(),
                "complete_dataset": complete, "gain_vs_RF": mean_gain,
                "mean_fit_seconds": float(part["fit_seconds"].mean()),
                "median_tree_leaves": float(part["median_leaves"].median()),
            })
            if len(part) == n_folds and set(part.index) == expected:
                complete_rows.append({
                    "task_id": task_id, "task_type": task_type,
                    "variant": name, "metric_mean": metric_mean,
                })
            if complete:
                measured[name] = metric_mean

        if complete:
            values = pd.Series(measured)
            ranks = values.rank(ascending=(task_type == "regression"), method="average")
            for name in variant_names:
                ranking_rows.append({
                    "dataset": dataset, "task_type": task_type, "variant": name,
                    "rank": float(ranks[name]),
                    "gain_vs_RF": next(x["gain_vs_RF"] for x in dataset_rows[::-1]
                                       if x["dataset"] == dataset and x["variant"] == name),
                })

    atomic_csv(pd.DataFrame(dataset_rows), output_dir / "datasets.csv")
    if ranking_rows:
        r = pd.DataFrame(ranking_rows)
        stats = r.groupby(["task_type", "variant"], sort=False).agg(
            datasets=("dataset", "nunique"), mean_rank=("rank", "mean"),
            wins_vs_RF=("gain_vs_RF", lambda s: int((s > 0).sum())),
            mean_gain_vs_RF=("gain_vs_RF", "mean"),
            median_gain_vs_RF=("gain_vs_RF", "median"),
        ).reset_index()
        stats["gain_unit"] = np.where(stats.task_type == "classification",
                                      "punkty procentowe accuracy", "% spadku RMSE")
        atomic_csv(stats, output_dir / "ranking.csv")
        print("\nSrednie rangi (tylko kompletne zbiory):")
        print(stats[["task_type", "variant", "datasets", "mean_rank", "wins_vs_RF",
                     "median_gain_vs_RF"]].to_string(index=False))
    else:
        print("\nNa razie brak kompletnych zbiorow do rankingu.")

    statistical_tests(complete_rows, variants, output_dir)


def main():
    args = parse_args()
    tasks = list(TASKS) if args.all else list(dict.fromkeys(args.task or [361055, 361077]))
    variants = candidates(args)
    print("\nE0 max_features: jednoczynnikowa kalibracja HDRF")
    print("Wybrane modele: A_p (klasyfikacja) i A (regresja), pi=0.5")
    print("Zadania:", tasks)
    print(f"Drzewa={args.trees}, foldy={args.folds}, warianty={len(variants)} + RF")
    for name, change in variants.items():
        print(f"  {name}: {change or 'oryginalne wspolczynniki Mateusza'}")
    print("Laczna liczba dopasowan:", len(tasks) * args.folds * (len(variants) + 1))
    print("Uwaga: tylko konfiguracja 500 drzew / 10 foldow na wszystkich 16 "
          "zbiorach moze byc podstawa pelnej analizy E0.")
    if args.dry_run:
        return 0

    protocol = {
        "tasks": tasks, "parameter": args.parameter, "variants": variants,
        "trees": args.trees, "folds": args.folds, "seed": args.seed,
        "base_config": asdict(BASE),
        "classification_override": {"classification_k0": "p"},
        "regression_override": {"regression_k0": "p"},
        "note": "Kandydaci sa odchyleniami od Mateusza; kod HDRF pozostaje bez zmian.",
    }
    # JSON zamienia tuple na listy: normalizujemy przed porownaniem przy wznowieniu.
    protocol = json.loads(json.dumps(protocol, ensure_ascii=False))
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")
                                 ).hexdigest()[:10]
    run_name = (f"{args.parameter}_B{args.trees}_F{args.folds}_"
                f"seed{args.seed}_{fingerprint}")
    out = args.out / run_name
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "protocol.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != protocol:
            print(f"BLAD: w {out} jest inny protokol. Uzyj nowego --out.", file=sys.stderr)
            return 2
    else:
        manifest.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    folds_csv = out / "folds.csv"
    if folds_csv.exists():
        rows = pd.read_csv(folds_csv).to_dict("records")
    else:
        rows = []
    done = {(int(r["task_id"]), int(r["fold"]), str(r["variant"])) for r in rows}
    failures = []

    for task_id in tasks:
        try:
            task = load_e0_task(task_id)
        except Exception as exc:
            failures.append((task_id, repr(exc)))
            print(f"BRAK DANYCH {task_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        classification = task.task_type == "classification"
        model_cls = HDRFClassifier if classification else HDRFRegressor
        rf_cls = RandomForestClassifier if classification else RandomForestRegressor
        all_classes = np.unique(task.y) if classification else None
        print(f"\n{task.name} ({task.task_type}), X={task.X.shape}", flush=True)

        for fold in range(args.folds):
            X_train, X_test, y_train, y_test = task.get_fold(fold)
            seed = args.seed + fold
            options = [("RF_domyslny", None)] + list(variants.items())
            for name, changes in options:
                key = (task_id, fold, name)
                if key in done:
                    print(f"  fold={fold}, {name}: cached", flush=True)
                    continue
                if changes is None:
                    model = rf_cls(n_estimators=args.trees, random_state=seed,
                                   n_jobs=args.n_jobs)
                    config = None
                else:
                    config = config_for_task(task.task_type, changes)
                    model = model_cls(n_estimators=args.trees, random_state=seed,
                                      n_jobs=args.n_jobs, sampling_config=config)
                t0 = perf_counter()
                model.fit(X_train, y_train)
                seconds = perf_counter() - t0
                scores = score(model, X_test, y_test, task.task_type, all_classes)
                row = {
                    "task_id": task_id, "dataset": task.name, "task_type": task.task_type,
                    "fold": fold, "variant": name, "seed": seed,
                    "n_estimators": args.trees,
                    "features_default_prob": (config.features_default_prob if config else np.nan),
                    "features_down": (config.features_down if config else np.nan),
                    "features_up": (config.features_up if config else np.nan),
                    "fit_seconds": seconds,
                    "median_leaves": float(np.median(
                        [t.get_n_leaves() for t in model.estimators_])),
                    **scores,
                }
                rows.append(row)
                done.add(key)
                atomic_csv(pd.DataFrame(rows), folds_csv)
                metric = "accuracy" if classification else "rmse"
                print(f"  fold={fold}, {name}: {metric}={scores[metric]:.6g}; "
                      f"fit={seconds:.1f}s", flush=True)

        # W razie przerwania zachowujemy porownania z gotowych zbiorow.
        summarize(pd.DataFrame(rows), variants, args.folds, out)

    print("\nWyniki:", out.resolve())
    if failures:
        print("Nie wszystkie zbiory sa dostepne:", failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())