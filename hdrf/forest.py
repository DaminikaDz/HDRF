"""HDRF v2: kontrolowane losowanie, bootstrap o dowolnym dodatnim ratio.

Bootstrap jak w scikit-learn: wagi obserwacji = liczba wylosowan (bez kopiowania
wierszy). Drzewa uczone rownolegle (joblib, watki). Bez OOB score, sample_weight
uzytkownika i macierzy sparse.
"""

from numbers import Integral

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .sampling import (
    HYPERPARAMETERS, SamplingConfig, apply_signal_rule, default_params, sample_params,
    tree_kwargs,
)


_STREAMS = {"complexity": 2, "criterion": 3, "max_features": 4, "max_samples": 5,
            "d_full": 6, "splitter": 7, "signal": 8}


def bootstrap_weights(seed, size, n_samples):
    """Wagi bootstrapu: ile razy wylosowano kazda obserwacje (jak w scikit-learn)."""
    indices = np.random.default_rng(seed).integers(0, n_samples, size=size)
    return np.bincount(indices, minlength=n_samples).astype(np.float64)


def _fit_tree(tree_cls, X, y, params, info, seed):
    weights = bootstrap_weights(info["seed"], info["size"], info["train_size"])
    tree = tree_cls(**tree_kwargs(params), random_state=seed)
    # Obserwacje z waga 0 (spoza bootstrapu) sa pomijane przez splitter CART.
    return tree.fit(X, y, sample_weight=weights)


def estimate_full_depth(tree_cls, X, y, task, n_trees, seed_seq, n_jobs=None):
    """Mediana glebokosci pelnych drzew (hiperparametry domyslne RF) na bootstrapach."""
    n_samples, n_features = X.shape
    params = default_params(task, n_samples, n_features)
    rng = np.random.default_rng(seed_seq)
    jobs = []
    for _ in range(n_trees):
        info = {"seed": int(rng.integers(0, 2**31 - 1)), "size": n_samples,
                "train_size": n_samples}
        jobs.append((info, int(rng.integers(0, 2**31 - 1))))
    trees = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_tree)(tree_cls, X, y, params, info, seed) for info, seed in jobs
    )
    return int(np.median([tree.get_depth() for tree in trees]))


def estimate_signal(tree_cls, X, y, task, n_trees, seed_seq, n_jobs=None):
    """Sila sygnalu: wynik OOB lasu z domyslnymi hiperparametrami RF.

    Klasyfikacja: OOB accuracy (y zakodowane 0..C-1), regresja: OOB R^2. Liczone na
    obserwacjach, ktore byly poza bootstrapem co najmniej jednego drzewa.
    """
    n_samples, n_features = X.shape
    params = default_params(task, n_samples, n_features)
    rng = np.random.default_rng(seed_seq)
    jobs = []
    for _ in range(n_trees):
        info = {"seed": int(rng.integers(0, 2**31 - 1)), "size": n_samples,
                "train_size": n_samples}
        jobs.append((info, int(rng.integers(0, 2**31 - 1))))
    trees = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_tree)(tree_cls, X, y, params, info, seed) for info, seed in jobs
    )
    classification = task == "classification"
    n_classes = int(y.max()) + 1 if classification else 1
    total = np.zeros((n_samples, n_classes))
    counts = np.zeros(n_samples)
    for tree, (info, _) in zip(trees, jobs):
        oob = bootstrap_weights(info["seed"], info["size"], n_samples) == 0
        if not oob.any():
            continue
        if classification:
            proba = tree.predict_proba(X[oob])
            total[np.ix_(oob, tree.classes_.astype(int))] += proba
        else:
            total[oob, 0] += tree.predict(X[oob])
        counts[oob] += 1
    seen = counts > 0
    if not seen.any():
        return float("nan")
    if classification:
        return float(np.mean(total[seen].argmax(axis=1) == y[seen]))
    pred = total[seen, 0] / counts[seen]
    resid = np.sum((y[seen] - pred) ** 2)
    spread = np.sum((y[seen] - y[seen].mean()) ** 2)
    return float(1 - resid / spread) if spread > 0 else 0.0


def _tree_predictions(estimators, X, method, n_jobs):
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(getattr(tree, method))(X) for tree in estimators
    )


class _HDRFBase(BaseEstimator):
    """Wspolne trenowanie drzew dla klasyfikacji i regresji."""

    _task = None
    _tree_cls = None

    def __init__(
        self,
        n_estimators=100,
        randomize=HYPERPARAMETERS,
        random_state=None,
        sampling_config=None,
        n_jobs=None,
        fixed_params=None,
    ):
        self.n_estimators = n_estimators
        self.randomize = randomize
        self.random_state = random_state
        self.sampling_config = sampling_config
        self.n_jobs = n_jobs
        self.fixed_params = fixed_params

    def fit(self, X, y):
        if (isinstance(self.n_estimators, bool)
                or not isinstance(self.n_estimators, Integral)
                or self.n_estimators < 1):
            raise ValueError("n_estimators musi byc dodatnia liczba calkowita")
        if isinstance(self.randomize, str):
            raise ValueError("randomize musi byc krotka/lista nazw hiperparametrow")
        try:
            enabled = tuple(self.randomize)
        except TypeError as exc:
            raise ValueError("randomize musi byc iterowalne") from exc
        unknown = set(enabled).difference(HYPERPARAMETERS)
        if unknown:
            raise ValueError(f"Nieznane hiperparametry: {sorted(unknown)}")

        config = SamplingConfig() if self.sampling_config is None else self.sampling_config
        if not isinstance(config, SamplingConfig):
            raise ValueError("sampling_config musi byc obiektem SamplingConfig")
        if (self.random_state is not None and
                (isinstance(self.random_state, bool) or
                 not isinstance(self.random_state, Integral) or
                 self.random_state < 0)):
            raise ValueError("random_state musi byc nieujemna liczba calkowita lub None")

        X, y = check_X_y(
            X, y, dtype=np.float32, ensure_all_finite="allow-nan",
            y_numeric=(self._task == "regression"),
        )
        n_samples, self.n_features_in_ = X.shape
        if self._task == "classification":
            # Kodujemy etykiety raz (0..K-1); drzewa ucza sie na kodach, co omija
            # kosztowne kodowanie etykiet tekstowych w kazdym drzewie.
            self.classes_, y = np.unique(y, return_inverse=True)
            y = y.astype(np.float64)

        # Stala kolejnosc strumieni, niezalezna od randomize / liczby drzew.
        root = np.random.SeedSequence(self.random_state)
        self.seed_entropy_ = root.entropy  # przy random_state=None pozwala odtworzyc trening
        # Indeksy strumieni sa stale: nowe grupy dopisujemy na koncu, zeby nie
        # zmieniac losowan wczesniejszych wersji (spawn(n)[i] nie zalezy od n).
        child_seeds = root.spawn(len(_STREAMS) + 2)
        bootstrap_rng = np.random.default_rng(child_seeds[0])
        tree_rng = np.random.default_rng(child_seeds[1])
        param_rngs = {
            name: np.random.default_rng(child_seeds[_STREAMS[name]])
            for name in HYPERPARAMETERS
        }

        # D_full: mediana glebokosci kilku pelnych drzew (domyslne RF) na bootstrapach.
        self.d_full_ = None
        if "complexity" in enabled:
            self.d_full_ = estimate_full_depth(
                self._tree_cls, X, y, self._task, config.n_pilot_trees,
                child_seeds[_STREAMS["d_full"]], self.n_jobs,
            )

        # Regula sygnalu (eksperymentalna): stale zlozonosci zalezne od wyniku OOB pilotazu.
        self.signal_ = None
        if "complexity" in enabled and config.signal_rule != "none":
            self.signal_ = estimate_signal(
                self._tree_cls, X, y, self._task, config.n_signal_trees,
                child_seeds[_STREAMS["signal"]], self.n_jobs,
            )
            config = apply_signal_rule(config, self.signal_)
        self.effective_config_ = config

        # Najpierw (sekwencyjnie) losujemy konfiguracje i ziarna: wynik nie zalezy od n_jobs.
        self.tree_params_ = []
        self.bootstrap_info_ = []
        self.tree_seeds_ = []
        for _ in range(int(self.n_estimators)):
            params = sample_params(
                param_rngs, self._task, n_samples, self.n_features_in_,
                enabled, config=config, d_full=self.d_full_,
                fixed=self.fixed_params,
            )
            bootstrap_seed = int(bootstrap_rng.integers(0, 2**31 - 1))
            tree_seed = int(tree_rng.integers(0, 2**31 - 1))
            # Nie ograniczamy do n: max_samples jest ratio i moze byc > 1.
            bootstrap_size = max(1, int(round(params["max_samples"] * n_samples)))
            self.tree_params_.append(params)
            self.bootstrap_info_.append({
                "seed": bootstrap_seed,
                "size": bootstrap_size,
                "train_size": n_samples,
            })
            self.tree_seeds_.append(tree_seed)

        self.estimators_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_tree)(self._tree_cls, X, y, params, info, seed)
            for params, info, seed in zip(
                self.tree_params_, self.bootstrap_info_, self.tree_seeds_
            )
        )
        return self

    def _validated_prediction_input(self, X, n_trees):
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float32, ensure_all_finite="allow-nan")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Oczekiwano {self.n_features_in_} cech, otrzymano {X.shape[1]}")
        if n_trees is None:
            count = len(self.estimators_)
        else:
            if (isinstance(n_trees, bool) or not isinstance(n_trees, Integral)
                    or not 1 <= n_trees <= len(self.estimators_)):
                raise ValueError(f"n_trees musi nalezec do [1, {len(self.estimators_)}]")
            count = int(n_trees)
        return X, count


class HDRFClassifier(ClassifierMixin, _HDRFBase):
    """HDRF dla klasyfikacji: srednia prawdopodobienstw."""

    _task = "classification"
    _tree_cls = DecisionTreeClassifier

    def predict_proba(self, X, n_trees=None):
        X, count = self._validated_prediction_input(X, n_trees)
        trees = self.estimators_[:count]
        probabilities = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for tree, part in zip(trees, _tree_predictions(trees, X, "predict_proba", self.n_jobs)):
            positions = tree.classes_.astype(np.intp)
            probabilities[:, positions] += part
        return probabilities / count

    def predict(self, X, n_trees=None):
        probabilities = self.predict_proba(X, n_trees=n_trees)
        return self.classes_[np.argmax(probabilities, axis=1)]


class HDRFRegressor(RegressorMixin, _HDRFBase):
    """HDRF dla regresji: srednia predykcji drzew."""

    _task = "regression"
    _tree_cls = DecisionTreeRegressor

    def predict(self, X, n_trees=None):
        X, count = self._validated_prediction_input(X, n_trees)
        trees = self.estimators_[:count]
        predictions = np.zeros(X.shape[0], dtype=np.float64)
        for part in _tree_predictions(trees, X, "predict", self.n_jobs):
            predictions += part
        return predictions / count
