import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score

from hdrf import HDRFClassifier, HDRFRegressor, HYPERPARAMETERS


def test_classifier_probability_shapes_and_prefix():
    X, y = make_classification(n_samples=180, n_features=10, random_state=1)
    model = HDRFClassifier(n_estimators=20, random_state=42).fit(X, y)
    assert len(model.estimators_) == 20
    assert len(model.tree_params_) == 20
    assert model.predict_proba(X[:5]).shape == (5, 2)
    np.testing.assert_allclose(model.predict_proba(X[:5]).sum(axis=1), 1)
    np.testing.assert_allclose(model.predict_proba(X[:5], n_trees=7).sum(axis=1), 1)
    assert np.isin(model.predict(X[:5]), model.classes_).all()


def test_reproducibility():
    X, y = make_classification(n_samples=160, n_features=6, random_state=2)
    m1 = HDRFClassifier(n_estimators=15, random_state=123).fit(X, y)
    m2 = HDRFClassifier(n_estimators=15, random_state=123).fit(X, y)
    assert m1.tree_params_ == m2.tree_params_
    assert m1.bootstrap_info_ == m2.bootstrap_info_
    np.testing.assert_array_equal(m1.predict_proba(X), m2.predict_proba(X))


def test_no_randomized_params_means_identical_tree_configs():
    X, y = make_classification(n_samples=160, n_features=6, random_state=2)
    model = HDRFClassifier(n_estimators=8, randomize=(), random_state=5).fit(X, y)
    assert all(params == model.tree_params_[0] for params in model.tree_params_)


def test_one_randomized_param_only():
    X, y = make_classification(n_samples=150, n_features=6, random_state=1)
    model = HDRFClassifier(n_estimators=30, randomize=("max_samples",), random_state=8).fit(X, y)
    for parameter in HYPERPARAMETERS:
        if parameter != "max_samples":
            assert len({str(p[parameter]) for p in model.tree_params_}) == 1
    assert len({p["max_samples"] for p in model.tree_params_}) > 1


def test_missing_bootstrap_class_probabilities():
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = np.array(["common"] * 99 + ["rare"])
    model = HDRFClassifier(n_estimators=25, random_state=8).fit(X, y)
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert any(len(tree.classes_) == 1 for tree in model.estimators_)


def test_regression_and_sklearn_clone():
    X, y = make_regression(n_samples=130, n_features=7, noise=5, random_state=1)
    model = HDRFRegressor(n_estimators=15, random_state=3)
    assert clone(model).get_params() == model.get_params()
    model.fit(X, y)
    assert model.predict(X[:4]).shape == (4,)
    assert model.predict(X[:4], n_trees=5).shape == (4,)
    assert np.isfinite(model.predict(X)).all()


def test_cross_val_score():
    X, y = make_classification(n_samples=130, n_features=6, random_state=3)
    model = HDRFClassifier(n_estimators=10, random_state=42)
    scores = cross_val_score(model, X, y, cv=3)
    assert len(scores) == 3
    assert np.isfinite(scores).all()


def test_errors():
    X, y = make_classification(n_samples=100, n_features=5, random_state=2)
    with pytest.raises(ValueError):
        HDRFClassifier(n_estimators=0).fit(X, y)
    with pytest.raises(ValueError):
        HDRFClassifier(randomize=("nonexistent",)).fit(X, y)
    model = HDRFClassifier(n_estimators=5, random_state=1).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X, n_trees=6)
