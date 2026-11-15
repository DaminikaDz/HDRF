
from hdrf import HDRFClassifier, SamplingConfig
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=200,
    n_features=10,
    random_state=42
)

for mode in ("default", "center"):

    model = HDRFClassifier(
        n_estimators=5,
        randomize=("max_features",),
        sampling_config=SamplingConfig(
            fixed_params_mode=mode
        ),
        random_state=42
    )

    model.fit(X, y)

    print(f"\n=== TRYB {mode.upper()} ===")

    print("Pierwsze drzewo:")
    print(model.tree_params_[0])

    print(
        "Bootstrap size:",
        model.bootstrap_info_[0]["size"]
    )
