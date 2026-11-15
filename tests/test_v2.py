from dataclasses import replace

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression

from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig


def _classification():
    return make_classification(n_samples=140, n_features=8, random_state=19)


def test_streams_do_not_shift_across_ablations():
    X, y = _classification()
    only_features = HDRFClassifier(n_estimators=18, randomize=("max_features",), random_state=42).fit(X, y)
    features_and_criterion = HDRFClassifier(n_estimators=18, randomize=("max_features", "criterion"), random_state=42).fit(X, y)
    assert [p["max_features"] for p in only_features.tree_params_] == [p["max_features"] for p in features_and_criterion.tree_params_]
    assert only_features.bootstrap_info_ == features_and_criterion.bootstrap_info_
    assert only_features.tree_seeds_ == features_and_criterion.tree_seeds_
    assert any(p["criterion"] != "gini" for p in features_and_criterion.tree_params_)


def test_same_bootstrap_for_fixed_and_randomized_features():
    X, y = _classification()
    config = SamplingConfig(center_classification_features=0.75)
    fixed = HDRFClassifier(12, randomize=(), random_state=55, sampling_config=config).fit(X, y)
    random = HDRFClassifier(12, randomize=("max_features",), random_state=55, sampling_config=config).fit(X, y)
    assert fixed.bootstrap_info_ == random.bootstrap_info_
    assert fixed.tree_seeds_ == random.tree_seeds_
    assert all(p["max_features"] == 0.75 for p in fixed.tree_params_)


def test_bootstrap_ratio_above_one():
    X, y = _classification()
    config = SamplingConfig(bootstrap_bounds=(1.5, 1.5), center_bootstrap=1.5)
    m = HDRFClassifier(n_estimators=10, random_state=17, sampling_config=config).fit(X, y)
    assert all(info["size"] == 210 for info in m.bootstrap_info_)
    assert all(p["max_samples"] == 1.5 for p in m.tree_params_)
    assert np.isfinite(m.predict_proba(X[:5])).all()


def test_bootstrap_ablation_keeps_seeds_but_changes_sample_sizes():
    X, y = _classification()
    config = SamplingConfig(bootstrap_bounds=(1.4, 1.8), center_bootstrap=1.0)
    fixed = HDRFClassifier(8, randomize=(), random_state=4, sampling_config=config).fit(X, y)
    mixed = HDRFClassifier(8, randomize=("max_samples",), random_state=4, sampling_config=config).fit(X, y)
    assert [d["seed"] for d in fixed.bootstrap_info_] == [d["seed"] for d in mixed.bootstrap_info_]
    assert fixed.tree_seeds_ == mixed.tree_seeds_
    assert any(a["size"] != b["size"] for a, b in zip(fixed.bootstrap_info_, mixed.bootstrap_info_))


def test_sklearn_clone_with_config():
    cfg = SamplingConfig(features_bounds=(0.7, 0.9))
    m = HDRFClassifier(n_estimators=11, sampling_config=cfg)
    cloned = clone(m)
    assert cloned.sampling_config == cfg
    assert cloned.get_params()["n_estimators"] == 11


def test_regressor_and_updated_config():
    X, y = make_regression(n_samples=140, n_features=7, random_state=23)
    cfg = replace(SamplingConfig(), bootstrap_bounds=(1.0, 1.8))
    model = HDRFRegressor(n_estimators=10, random_state=42, sampling_config=cfg).fit(X, y)
    assert model.predict(X[:5]).shape == (5,)
    assert any(p["max_samples"] > 1.0 for p in model.tree_params_)


def test_input_validation():
    X, y = _classification()
    with pytest.raises(ValueError):
        SamplingConfig(bootstrap_bounds=(0, 2))
    with pytest.raises(ValueError):
        SamplingConfig(features_bounds=(0.3, 2))
    with pytest.raises(ValueError):
        HDRFClassifier(random_state=-1).fit(X, y)
    with pytest.raises(ValueError):
        HDRFClassifier(sampling_config={}).fit(X, y)
