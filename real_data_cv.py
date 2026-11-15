"""Diagnostyczne 10-fold CV na oryginalnych podzialach OpenML (credit, Ailerons).

To nie jest kalibracja E0: rozklady sa wciaz robocze i B=100.
Uruchom z katalogu glownego projektu:
    python real_data_cv.py
    python real_data_cv.py --trees 500

Wyniki foldow sa zapisywane na biezaco do results/cv_smoke/.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error, r2_score

from data_loader import load_e0_task
from hdrf import HDRFClassifier, HDRFRegressor

TASK_IDS = (361055, 361077)
MODEL_NAMES = ("RF default", "Homogeneous center", "HDRF full")
BASE_SEED = 42


def make_models(task_type: str, n_trees: int, seed: int):
    """Nowe (niewytrenowane) modele dla jednego foldu."""
    if task_type == "classification":
        return {
            "RF default": RandomForestClassifier(
                n_estimators=n_trees, random_state=seed, n_jobs=1
            ),
            "Homogeneous center": HDRFClassifier(
                n_estimators=n_trees, randomize=(), random_state=seed
            ),
            "HDRF full": HDRFClassifier(
                n_estimators=n_trees, random_state=seed
            ),
        }
    if task_type == "regression":
        return {
            "RF default": RandomForestRegressor(
                n_estimators=n_trees, random_state=seed, n_jobs=1
            ),
            "Homogeneous center": HDRFRegressor(
                n_estimators=n_trees, randomize=(), random_state=seed
            ),
            "HDRF full": HDRFRegressor(
                n_estimators=n_trees, random_state=seed
            ),
        }
    raise ValueError(f"Nieznany typ zadania: {task_type}")


def evaluate_fold(model, task_type, X_train, X_test, y_train, y_test, labels):
    """Trening i miary na jednym foldzie; czas obejmuje samo fit()."""
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    prediction = model.predict(X_test)

    metrics = {"fit_seconds": fit_seconds}
    if task_type == "classification":
        metrics.update(
            accuracy=accuracy_score(y_test, prediction),
            f1_macro=f1_score(y_test, prediction, average="macro", zero_division=0),
            log_loss=log_loss(
                y_test, model.predict_proba(X_test), labels=labels
            ),
        )
    else:
        metrics.update(
            rmse=np.sqrt(mean_squared_error(y_test, prediction)),
            r2=r2_score(y_test, prediction),
        )
    return metrics


def evaluate_task(task_id: int, n_trees: int, out_dir: Path):
    task = load_e0_task(task_id)
    folds = sorted(task.folds, key=lambda f: (f.repeat, f.fold))
    if len(folds) != 10 or [f.fold for f in folds] != list(range(10)):
        raise ValueError("Oczekiwano dziesieciu foldow OpenML 0..9")

    print(f"\n=== {task.name} | task={task_id} | {task.task_type} ===", flush=True)
    print(f"X={task.X.shape}, folds={len(folds)}, B={n_trees}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / f"{task_id}_folds.csv"
    summary_path = out_dir / f"{task_id}_summary.csv"
    rows = []
    labels = np.unique(task.y) if task.task_type == "classification" else None

    for split in folds:
        # Tylko oryginalne indeksy OpenML, bez train_test_split / KFold.
        X_train, X_test, y_train, y_test = task.get_fold(
            fold=split.fold, repeat=split.repeat
        )
        seed = BASE_SEED + split.fold
        models = make_models(task.task_type, n_trees, seed)
        print(
            f"\nFold {split.fold + 1}/10: "
            f"train={len(y_train)}, test={len(y_test)}, seed={seed}",
            flush=True,
        )

        for name in MODEL_NAMES:
            metrics = evaluate_fold(
                models[name], task.task_type, X_train, X_test,
                y_train, y_test, labels,
            )
            row = {
                "task_id": task_id,
                "dataset": task.name,
                "task_type": task.task_type,
                "repeat": split.repeat,
                "fold": split.fold,
                "model": name,
                "seed": seed,
                "n_trees": n_trees,
                "n_train": len(y_train),
                "n_test": len(y_test),
                **metrics,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(fold_path, index=False)

            primary = (f"accuracy={metrics['accuracy']:.4f}"
                       if task.task_type == "classification"
                       else f"RMSE={metrics['rmse']:.8f}")
            print(
                f"  {name:<19} {primary} "
                f"fit={metrics['fit_seconds']:.2f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    columns = (["accuracy", "f1_macro", "log_loss"]
               if task.task_type == "classification" else ["rmse", "r2"])
    columns += ["fit_seconds"]

    summaries = []
    print("\n--- SREDNIA Z 10 FOLDOW (odchylenie standardowe) ---")
    for name in MODEL_NAMES:
        group = results.loc[results["model"] == name]
        entry = {"task_id": task_id, "dataset": task.name,
                 "model": name, "folds": len(group), "n_trees": n_trees}
        for metric in columns:
            entry[f"{metric}_mean"] = group[metric].mean()
            entry[f"{metric}_std"] = group[metric].std(ddof=1)
        summaries.append(entry)
        primary = "accuracy" if task.task_type == "classification" else "rmse"
        print(
            f"{name:<19} {primary}="
            f"{entry[f'{primary}_mean']:.8f} "
            f"+/- {entry[f'{primary}_std']:.8f} | "
            f"czas fit razem={group['fit_seconds'].sum():.1f}s"
        )

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    main_metric = "accuracy" if task.task_type == "classification" else "rmse"
    paired = results.pivot(index="fold", columns="model", values=main_metric)
    difference = (paired["HDRF full"] - paired["Homogeneous center"]).mean()
    print(f"Srednia sparowana roznica HDRF - center ({main_metric}): {difference:+.8f}")
    print(f"Zapisano: {fold_path} oraz {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=100,
                        help="Liczba drzew w kazdym modelu (domyslnie 100)")
    parser.add_argument("--out", type=Path, default=Path("results/cv_smoke"))
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees musi byc dodatnie")
    for task_id in TASK_IDS:
        evaluate_task(task_id, args.trees, args.out)


if __name__ == "__main__":
    main()
