"""Bootstrap przez wagi (jak w scikit-learn) i rownolegle uczenie drzew."""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier

from hdrf import HDRFClassifier, HDRFRegressor
from hdrf.forest import bootstrap_weights
from hdrf.sampling import tree_kwargs


def test_weights_sum_to_bootstrap_size():
    for size in (50, 100, 180):
        w = bootstrap_weights(seed=3, size=size, n_samples=100)
        assert w.shape == (100,)
        assert w.sum() == size
        assert np.all(w == np.floor(w))


def test_tree_matches_weights_on_full_data():
    """Drzewo HDRF = drzewo uczone na pelnym X z wagami z bootstrap_info_."""
    X, y = make_classification(n_samples=200, n_features=8, random_state=0)
    model = HDRFClassifier(n_estimators=3, randomize=(), random_state=1).fit(X, y)
    info, params = model.bootstrap_info_[0], model.tree_params_[0]
    w = bootstrap_weights(info["seed"], info["size"], info["train_size"])
    kwargs = tree_kwargs(params)
    ref = DecisionTreeClassifier(**kwargs, random_state=model.tree_seeds_[0])
    ref.fit(X.astype(np.float32), y, sample_weight=w)
    np.testing.assert_array_equal(ref.predict_proba(X), model.estimators_[0].predict_proba(X))


def test_n_jobs_does_not_change_result():
    X, y = make_regression(n_samples=300, n_features=6, noise=3, random_state=2)
    seq = HDRFRegressor(n_estimators=12, random_state=7, n_jobs=1).fit(X, y)
    par = HDRFRegressor(n_estimators=12, random_state=7, n_jobs=4).fit(X, y)
    assert seq.tree_params_ == par.tree_params_
    np.testing.assert_array_equal(seq.predict(X), par.predict(X))
    np.testing.assert_array_equal(seq.predict(X, n_trees=5), par.predict(X, n_trees=5))

    Xc, yc = make_classification(n_samples=300, n_features=6, random_state=2)
    a = HDRFClassifier(n_estimators=12, random_state=7, n_jobs=1).fit(Xc, yc)
    b = HDRFClassifier(n_estimators=12, random_state=7, n_jobs=-1).fit(Xc, yc)
    np.testing.assert_array_equal(a.predict_proba(Xc), b.predict_proba(Xc))


def test_missing_values_supported():
    X, y = make_classification(n_samples=200, n_features=5, random_state=4)
    X[::7, 1] = np.nan
    model = HDRFClassifier(n_estimators=10, random_state=0, n_jobs=2).fit(X, y)
    assert np.isfinite(model.predict_proba(X)).all()
