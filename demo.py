"""Diagnostyczne demo HDRF (przed E0 i E1).

Porownujemy trzy metody na IDENTYCZNYCH podzialach i tej samej liczbie drzew:
  1. RF default: standardowy RandomForest ze scikit-learn
  2. RF center: wspolne, robocze hiperparametry z center_params()
  3. HDRF: indywidualne losowanie szesciu hiperparametrow drzew

Wyniki sa opisowe, bez doboru rozkladow pod test i bez testow istotnosci.
To nie jest pelny protokol E1 z pracy inzynierskiej.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from hdrf import HDRFClassifier, HDRFRegressor
from hdrf.sampling import center_params


B = 100                # Taka sama liczba drzew w kazdym modelu
N_SPLITS = 3          # 3-fold CV
N_REPEATS = 2         # Dwa powtorzenia podzialow -> 6 ocen kazdego modelu
SEED = 42
MODEL_NAMES = ("RF default", "RF center", "HDRF")


def show_results(results, primary_metric, higher_is_better=True):
    """Wypisz srednia, odchylenie std. i sparowane roznice wzgledem RF default."""
    print(f"{'Model':<13} {'Srednia':>11} {'Odch. std.':>13} {'Roznica vs RF':>15}")
    baseline = np.asarray(results["RF default"][primary_metric], dtype=float)
    for name in MODEL_NAMES:
        scores = np.asarray(results[name][primary_metric], dtype=float)
        mean = scores.mean()
        sd = scores.std(ddof=1)
        delta = (scores - baseline).mean()
        # Dodatni delta oznacza lepszy wynik dla accuracy i R^2;
        # ujemny delta oznacza lepszy wynik dla RMSE.
        print(f"{name:<13} {mean:>11.4f} {sd:>13.4f} {delta:>+15.4f}")
    direction = "wiecej = lepiej" if higher_is_better else "mniej = lepiej"
    print(f"Miara: {primary_metric} ({direction}); srednia z {len(baseline)} foldow")
    print("Roznica vs RF = srednia ze sparowanych roznic na tych samych foldach.")
    print("Odchylenie std. opisuje rozrzut wynikow foldow, nie niepewnosc statystyczna.")


def evaluate_classification(title, X, y):
    print(f"\n=== KLASYFIKACJA: {title} (n={len(y)}, p={X.shape[1]}) ===")
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    results = {name: {"accuracy": [], "f1_macro": []} for name in MODEL_NAMES}

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model_seed = SEED + fold_id
        center = center_params("classification", len(train_idx), X.shape[1])
        models = {
            "RF default": RandomForestClassifier(
                n_estimators=B, random_state=model_seed, n_jobs=1
            ),
            "RF center": RandomForestClassifier(
                n_estimators=B, random_state=model_seed, n_jobs=1, **center
            ),
            "HDRF": HDRFClassifier(n_estimators=B, random_state=model_seed),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            results[name]["accuracy"].append(accuracy_score(y_test, prediction))
            results[name]["f1_macro"].append(
                f1_score(y_test, prediction, average="macro", zero_division=0)
            )

    show_results(results, "accuracy")
    print("F1 macro:", " | ".join(
        f"{name}: {np.mean(results[name]['f1_macro']):.4f}" for name in MODEL_NAMES
    ))


def evaluate_regression(X, y):
    print(f"\n=== REGRESJA (n={len(y)}, p={X.shape[1]}) ===")
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    results = {name: {"RMSE": [], "R2": []} for name in MODEL_NAMES}

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model_seed = SEED + fold_id
        center = center_params("regression", len(train_idx), X.shape[1])
        models = {
            "RF default": RandomForestRegressor(
                n_estimators=B, random_state=model_seed, n_jobs=1
            ),
            "RF center": RandomForestRegressor(
                n_estimators=B, random_state=model_seed, n_jobs=1, **center
            ),
            "HDRF": HDRFRegressor(n_estimators=B, random_state=model_seed),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            results[name]["RMSE"].append(
                float(np.sqrt(mean_squared_error(y_test, prediction)))
            )
            results[name]["R2"].append(r2_score(y_test, prediction))

    show_results(results, "RMSE", higher_is_better=False)
    print("R2:", " | ".join(
        f"{name}: {np.mean(results[name]['R2']):.4f}" for name in MODEL_NAMES
    ))


if __name__ == "__main__":
    print("DEMO HDRF: powtarzane CV, wspolne podzialy, B=100, trzy modele")
    print("UWAGA: rozklady HDRF sa wstepne; nie byly jeszcze kalibrowane w E0.")

    # Zbior podobny do pierwotnego demo. Domyslnie: 2 cechy redundantne.
    X1, y1 = make_classification(
        n_samples=600, n_features=12, n_informative=6,
        n_redundant=2, random_state=42
    )
    evaluate_classification("oryginalny syntetyczny", X1, y1)

    # Drugi scenariusz: wiecej cech szumowych; nie dobrany pod wyniki HDRF.
    X2, y2 = make_classification(
        n_samples=600, n_features=30, n_informative=6,
        n_redundant=0, n_repeated=0, flip_y=0.05,
        class_sep=0.8, random_state=7
    )
    evaluate_classification("duzo cech szumowych", X2, y2)

    Xr, yr = make_regression(
        n_samples=500, n_features=8, noise=15, random_state=42
    )
    evaluate_regression(Xr, yr)

    print("\nTo demo diagnostyczne, NIE E1: w E1 zastosujemy zbiory OpenML,")
    print("ustalone foldy, skalibrowane E0 rozklady i B = 500/1000.")
