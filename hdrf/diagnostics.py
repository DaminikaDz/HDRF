"""Miary jakosci lasu i diagnostyka drzew (sila, korelacja, ambiguity)."""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score


def score(model, task_type, X_test, y_test, labels=None):
    """Klasyfikacja: accuracy i log-loss; regresja: RMSE i R^2."""
    pred = model.predict(X_test)
    if task_type == "classification":
        return {"accuracy": accuracy_score(y_test, pred),
                "log_loss": log_loss(y_test, model.predict_proba(X_test), labels=labels)}
    return {"rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": r2_score(y_test, pred)}


def tree_diagnostics(model, X_test, y_test, classification):
    """Diagnostyka na zbiorze testowym (w duchu sily i korelacji Breimana).

    tree_score - srednia jakosc pojedynczego drzewa (accuracy / RMSE),
    tree_corr  - srednia korelacja bledow miedzy parami drzew
                 (regresja: reszty y - f_t; klasyfikacja: 1 - p_t(prawdziwa klasa)),
    ambiguity  - (Krogh-Vedelsby) sredni rozrzut predykcji drzew wokol lasu;
                 blad^2 lasu = sredni blad^2 drzewa - ambiguity,
    amb_share  - jaka czesc bledu drzewa usuwa usrednianie.
    """
    trees = model.estimators_
    X_test = X_test.astype(np.float32)
    errors, scores = [], []
    if classification:
        codes = np.searchsorted(model.classes_, y_test)
        for t in trees:
            proba = np.zeros((len(y_test), len(model.classes_)))
            proba[:, t.classes_.astype(int)] = t.predict_proba(X_test)
            errors.append(1.0 - proba[np.arange(len(y_test)), codes])
            scores.append(np.mean(proba.argmax(axis=1) == codes))
    else:
        for t in trees:
            resid = y_test - t.predict(X_test)
            errors.append(resid)
            scores.append(np.sqrt(np.mean(resid ** 2)))
    errors = np.vstack(errors)
    corr = np.corrcoef(errors)
    off_diag = corr[~np.eye(len(trees), dtype=bool)]
    tree_mse = float(np.mean(errors ** 2))
    ambiguity = float(np.mean(errors.var(axis=0)))
    return {"tree_score": float(np.mean(scores)), "tree_corr": float(np.nanmean(off_diag)),
            "ambiguity": ambiguity, "amb_share": ambiguity / tree_mse}
