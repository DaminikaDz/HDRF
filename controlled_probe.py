"""Diagnostyczne, KONTROLOWANE porownanie stalego i losowego max_features.

Wszystkie warianty fixed/mixed poza zwyklym RF korzystaja z TEJ SAMEJ
implementacji lasu, ziaren drzew i ziaren bootstrap. Nie jest to E0/E3.
"""

from dataclasses import replace

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hdrf import HDRFClassifier, SamplingConfig


B = 100
SEED = 42

X, y = make_classification(
    n_samples=600, n_features=30, n_informative=6,
    n_redundant=0, n_repeated=0, flip_y=0.05,
    class_sep=0.8, random_state=7,
)

cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=SEED)
results = {}

for fold_no, (tr, te) in enumerate(cv.split(X, y), start=1):
    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]
    seed = SEED + fold_no - 1
    base = SamplingConfig()

    models = {
        "RF sklearn default": RandomForestClassifier(
            n_estimators=B, random_state=seed, n_jobs=1,
        ),
        "Homogeneous center": HDRFClassifier(
            n_estimators=B, randomize=(), random_state=seed,
            sampling_config=base,
        ),
        "Homogeneous 0.50": HDRFClassifier(
            n_estimators=B, randomize=(), random_state=seed,
            sampling_config=replace(base, center_classification_features=0.50),
        ),
        "Homogeneous 0.75": HDRFClassifier(
            n_estimators=B, randomize=(), random_state=seed,
            sampling_config=replace(base, center_classification_features=0.75),
        ),
        "Homogeneous 1.00": HDRFClassifier(
            n_estimators=B, randomize=(), random_state=seed,
            sampling_config=replace(base, center_classification_features=1.0),
        ),
        "HDRF only features": HDRFClassifier(
            n_estimators=B, randomize=("max_features",), random_state=seed,
            sampling_config=base,
        ),
        "HDRF full": HDRFClassifier(
            n_estimators=B, random_state=seed, sampling_config=base,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        results.setdefault(name, []).append(
            accuracy_score(y_test, model.predict(X_test))
        )

    print(f"Fold {fold_no}/6 wykonany")

print("\n=== WYNIKI: KONTROLOWANA DIAGNOSTYKA max_features ===")
base_scores = np.array(results["Homogeneous center"])
for name, scores in results.items():
    scores = np.asarray(scores)
    print(
        f"{name:<23} accuracy={scores.mean():.4f} "
        f"std={scores.std(ddof=1):.4f} "
        f"vs_center={(scores-base_scores).mean():+.4f}"
    )

print("\nUWAGA: dane syntetyczne, model niekalibrowany; to nie jest E0 ani E3.")
