
import numpy as np
from sklearn.datasets import make_classification

from hdrf import HDRFClassifier, SamplingConfig


X, y = make_classification(
    n_samples=200,
    n_features=10,
    random_state=42
)

config = SamplingConfig(
    center_classification_features=0.75
)

fixed = HDRFClassifier(
    n_estimators=5,
    randomize=(),
    sampling_config=config,
    random_state=42
)

mixed = HDRFClassifier(
    n_estimators=5,
    randomize=("max_features",),
    sampling_config=config,
    random_state=42
)

fixed.fit(X, y)
mixed.fit(X, y)

print("Czy bootstrap_info_ jest identyczne?")
print(fixed.bootstrap_info_ == mixed.bootstrap_info_)

print("\nCzy ziarna CART sa identyczne?")
print(fixed.tree_seeds_ == mixed.tree_seeds_)

print("\nPorownanie drzew:")

for i in range(5):
    print(f"\nDrzewo {i + 1}")

    print(
        "Fixed max_features:",
        fixed.tree_params_[i]["max_features"]
    )

    print(
        "Mixed max_features:",
        mixed.tree_params_[i]["max_features"]
    )

    print(
        "Bootstrap seed:",
        fixed.bootstrap_info_[i]["seed"]
    )

# Odtwarzamy indeksy bootstrap dla pierwszego drzewa
info_fixed = fixed.bootstrap_info_[0]
info_mixed = mixed.bootstrap_info_[0]

idx_fixed = np.random.default_rng(
    info_fixed["seed"]
).integers(
    0,
    info_fixed["train_size"],
    size=info_fixed["size"]
)

idx_mixed = np.random.default_rng(
    info_mixed["seed"]
).integers(
    0,
    info_mixed["train_size"],
    size=info_mixed["size"]
)

print("\nCzy indeksy bootstrap sa identyczne?")
print(np.array_equal(idx_fixed, idx_mixed))
