"""Rozklady v3: tryby zlozonosci, F4+F1 dla max_features, BR, srodek empiryczny."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression

from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig, empirical_center
from hdrf.sampling import depth_bounds, log_uniform_int, sample_params


def _classification():
    return make_classification(n_samples=300, n_features=16, random_state=19)


def test_streams_do_not_shift_across_ablations():
    X, y = _classification()
    a = HDRFClassifier(18, randomize=("max_features",), random_state=42).fit(X, y)
    b = HDRFClassifier(18, randomize=("max_features", "criterion"), random_state=42).fit(X, y)
    assert [p["max_features"] for p in a.tree_params_] == [p["max_features"] for p in b.tree_params_]
    assert a.bootstrap_info_ == b.bootstrap_info_
    assert a.tree_seeds_ == b.tree_seeds_


def test_log_uniform_int_endpoints_not_halved():
    rng = np.random.default_rng(0)
    counts = Counter(log_uniform_int(rng, 1, 10) for _ in range(200000))
    for k in range(1, 11):
        expected = np.log((k + 1) / k) / np.log(11)
        assert abs(counts[k] / 200000 - expected) < 0.005


def test_complexity_modes_never_limit_twice():
    rng = np.random.default_rng(1)
    cfg = SamplingConfig()
    modes = Counter()
    for _ in range(5000):
        p = sample_params(rng, "classification", 10000, 20, ("complexity",), cfg, d_full=30)
        modes[p["mode"]] += 1
        assert p["min_samples_split"] == 2
        if p["mode"] == "depth":
            assert p["min_samples_leaf"] == 1
            assert p["max_depth"] is None or 14 <= p["max_depth"] < 30
        else:
            assert p["max_depth"] is None and 1 <= p["min_samples_leaf"] <= cfg.leaf_max
    assert abs(modes["depth"] / 5000 - cfg.depth_mode_prob) < 0.03
    assert depth_bounds(10000, 30, cfg) == (14, 30)
    assert depth_bounds(10000, 10, cfg) == (10, 10)


def test_max_features_median_is_k0_when_symmetric():
    rng = np.random.default_rng(2)
    cfg = SamplingConfig(features_down=2.0, features_up=2.0)
    for p in (10, 33, 100):
        ks = [sample_params(rng, "classification", 1000, p, ("max_features",), cfg)["max_features"]
              for _ in range(4000)]
        assert int(np.median(ks)) == int(np.rint(np.sqrt(p)))
        assert min(ks) >= max(1, int(np.rint(np.sqrt(p) / 2))) and max(ks) <= p


def test_regression_k0_variants():
    rng = np.random.default_rng(3)
    for k0, expected in (("p", 33), ("p/3", 11)):
        cfg = SamplingConfig(regression_k0=k0, features_default_prob=1.0)
        assert sample_params(rng, "regression", 500, 33, ("max_features",), cfg)["max_features"] == expected


def test_fixed_params_and_defaults():
    X, y = _classification()
    m = HDRFClassifier(5, randomize=(), random_state=0).fit(X, y)
    assert all(p["max_features"] == 4 and p["max_depth"] is None for p in m.tree_params_)
    m = HDRFClassifier(5, randomize=(), random_state=0,
                       fixed_params={"min_samples_leaf": 3, "max_samples": 0.8}).fit(X, y)
    assert all(p["min_samples_leaf"] == 3 for p in m.tree_params_)
    assert all(info["size"] == 240 for info in m.bootstrap_info_)
    with pytest.raises(ValueError):
        HDRFClassifier(5, fixed_params={"bogus": 1}).fit(X, y)


def test_d_full_estimated_only_for_complexity():
    X, y = _classification()
    assert HDRFClassifier(5, randomize=("criterion",), random_state=0).fit(X, y).d_full_ is None
    m = HDRFClassifier(5, random_state=0).fit(X, y)
    assert m.d_full_ >= 1


def test_bootstrap_ratio_above_one():
    X, y = _classification()
    cfg = SamplingConfig(bootstrap_bounds=(1.5, 1.5))
    m = HDRFClassifier(n_estimators=10, random_state=17, sampling_config=cfg).fit(X, y)
    assert all(info["size"] == 450 for info in m.bootstrap_info_)


def test_empirical_center():
    X, y = _classification()
    ref = HDRFClassifier(40, random_state=0, n_jobs=2).fit(X, y)
    for limit in ("leaf", "depth"):
        center, info = empirical_center(X, y, "classification", reference=ref, limit=limit)
        assert center["criterion"] == "gini"
        assert center["max_features"] == 4
        assert 0.7 <= center["max_samples"] <= 1.0
        assert info["target_leaves"] > 1
        model = HDRFClassifier(5, randomize=(), fixed_params=center, random_state=0).fit(X, y)
        assert np.isfinite(model.predict_proba(X)).all()


def test_sklearn_clone_with_config():
    cfg = SamplingConfig(features_down=3.0)
    m = HDRFRegressor(n_estimators=11, sampling_config=cfg, fixed_params={"min_samples_leaf": 2})
    assert clone(m).get_params() == m.get_params()


def test_input_validation():
    X, y = _classification()
    with pytest.raises(ValueError):
        SamplingConfig(bootstrap_bounds=(0, 2))
    with pytest.raises(ValueError):
        SamplingConfig(features_down=0.5)
    with pytest.raises(ValueError):
        SamplingConfig(regression_k0="p/2")
    with pytest.raises(ValueError):
        HDRFClassifier(random_state=-1).fit(X, y)
    with pytest.raises(ValueError):
        HDRFClassifier(sampling_config={}).fit(X, y)
    Xr, yr = make_regression(n_samples=120, n_features=5, random_state=1)
    HDRFRegressor(5, random_state=0, sampling_config=replace(SamplingConfig(), regression_k0="p/3")).fit(Xr, yr)
