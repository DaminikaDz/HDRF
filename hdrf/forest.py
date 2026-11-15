"""HDRF v2: kontrolowane losowanie, bootstrap o dowolnym dodatnim ratio.

Wciaz wersja sekwencyjna: bez joblib, OOB score, sample_weight, sparse.
"""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .sampling import HYPERPARAMETERS, SamplingConfig, sample_params


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
    ):
        self.n_estimators = n_estimators
        self.randomize = randomize
        self.random_state = random_state
        self.sampling_config = sampling_config

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
            X, y, dtype=np.float64, ensure_all_finite="allow-nan",
            y_numeric=(self._task == "regression"),
        )
        n_samples, self.n_features_in_ = X.shape
        if self._task == "classification":
            self.classes_ = np.unique(y)

        # Stala kolejnosc strumieni, niezalezna od randomize / liczby drzew.
        root = np.random.SeedSequence(self.random_state)
        self.seed_entropy_ = root.entropy  # przy random_state=None pozwala odtworzyc trening
        child_seeds = root.spawn(2 + len(HYPERPARAMETERS))
        bootstrap_rng = np.random.default_rng(child_seeds[0])
        tree_rng = np.random.default_rng(child_seeds[1])
        param_rngs = {
            name: np.random.default_rng(child_seeds[i + 2])
            for i, name in enumerate(HYPERPARAMETERS)
        }

        self.estimators_ = []
        self.tree_params_ = []
        self.bootstrap_info_ = []
        self.tree_seeds_ = []

        for _ in range(int(self.n_estimators)):
            params = sample_params(
                param_rngs, self._task, n_samples, self.n_features_in_,
                enabled, config=config,
            )
            bootstrap_seed = int(bootstrap_rng.integers(0, 2**31 - 1))
            tree_seed = int(tree_rng.integers(0, 2**31 - 1))

            # Nie ograniczamy do n: max_samples jest ratio i moze byc > 1.
            bootstrap_size = max(1, int(round(params["max_samples"] * n_samples)))
            indices = np.random.default_rng(bootstrap_seed).integers(
                0, n_samples, size=bootstrap_size
            )
            tree_kwargs = {key: value for key, value in params.items() if key != "max_samples"}
            tree = self._tree_cls(**tree_kwargs, random_state=tree_seed)
            tree.fit(X[indices], y[indices])

            self.estimators_.append(tree)
            self.tree_params_.append(params)
            self.bootstrap_info_.append({
                "seed": bootstrap_seed,
                "size": bootstrap_size,
                "train_size": n_samples,
            })
            self.tree_seeds_.append(tree_seed)
        return self

    def _validated_prediction_input(self, X, n_trees):
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
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
        probabilities = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for tree in self.estimators_[:count]:
            positions = np.searchsorted(self.classes_, tree.classes_)
            probabilities[:, positions] += tree.predict_proba(X)
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
        predictions = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.estimators_[:count]:
            predictions += tree.predict(X)
        return predictions / count
