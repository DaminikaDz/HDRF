"""Read downloaded OpenML E0 tasks and their original cross-validation folds.

Preprocessing (as in the M2 data report):
- J1: drop columns with > 50% missing values, constant columns and ID-like
  columns listed in J1_ID_COLUMNS (from the M2 quality report).
- Categorical (nominal) features: ordinal encoding in the order of levels
  declared in the ARFF header; missing values stay NaN (no imputation).
- J7: n > 50,000 -> uniform subsample of 50,000 rows (fixed seed); every kept
  row keeps its original OpenML fold, so test folds still partition the data.
Train/test splits are always the original OpenML folds.

Usage from the project root::

    python data_loader.py

Or import::

    from data_loader import load_e0_task
    task = load_e0_task(361055)
    X_train, X_test, y_train, y_test = task.get_fold(0)
"""

from dataclasses import dataclass
from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd
from scipy.io import arff


DEFAULT_ROOT = Path("data/e0/raw")
J7_MAX_ROWS = 50_000
J7_SEED = 0
# ID-like columns found in the M2 quality report (kol_typu_id).
J1_ID_COLUMNS = {361070: ("lineNo",)}


@dataclass(frozen=True)
class OpenMLFold:
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class E0Task:
    task_id: int
    name: str
    task_type: str
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    folds: tuple[OpenMLFold, ...]

    def get_fold(self, fold: int = 0, repeat: int = 0):
        """Return X_train, X_test, y_train, y_test for one original OpenML fold."""
        matches = [f for f in self.folds if f.fold == fold and f.repeat == repeat]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous repeat={repeat}, fold={fold}")
        split = matches[0]
        return (
            self.X[split.train_idx],
            self.X[split.test_idx],
            self.y[split.train_idx],
            self.y[split.test_idx],
        )


def _read_arff(folder: Path, stem: str, with_meta: bool = False):
    """Open a single .arff or .arff.gz file without changing column ordering."""
    candidates = [folder / f"{stem}.arff", folder / f"{stem}.arff.gz"]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {stem}.arff[.gz] in {folder}; found {present}"
        )
    path = present[0]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        records, meta = arff.loadarff(stream)
    frame = pd.DataFrame.from_records(records)

    # scipy.io.arff returns nominal values as bytes.
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(
            lambda value: value.decode("utf-8", errors="replace")
            if isinstance(value, bytes) else value
        )
    return (frame, meta) if with_meta else frame


def _load_splits(folder: Path, n_rows: int, keep: np.ndarray | None = None) -> tuple[OpenMLFold, ...]:
    splits = _read_arff(folder, "splits")
    required = {"repeat", "fold", "rowid", "type"}
    if not required.issubset(splits.columns):
        raise ValueError(f"Missing OpenML split columns: {required - set(splits.columns)}")

    for key in ("repeat", "fold", "rowid"):
        values = pd.to_numeric(splits[key], errors="raise").to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Invalid non-integer split column: {key}")
        splits[key] = values.astype(np.int64)
    splits["type"] = splits["type"].astype(str).str.upper().str.strip()
    if not set(splits["type"]).issubset({"TRAIN", "TEST"}):
        raise ValueError(f"Invalid split types: {splits['type'].unique()}")

    folds = []
    for (repeat, fold), part in splits.groupby(["repeat", "fold"], sort=True):
        train = np.sort(part.loc[part["type"] == "TRAIN", "rowid"].to_numpy(dtype=np.int64))
        test = np.sort(part.loc[part["type"] == "TEST", "rowid"].to_numpy(dtype=np.int64))

        # Every fold must partition exactly the original dataset, with no overlap.
        combined = np.concatenate((train, test))
        if (
            not len(train) or not len(test)
            or combined.size != n_rows
            or not np.array_equal(np.sort(combined), np.arange(n_rows))
        ):
            raise ValueError(f"Invalid OpenML partition repeat={repeat}, fold={fold}")
        folds.append(OpenMLFold(int(repeat), int(fold), train, test))

    if len(folds) != 10 or {f.repeat for f in folds} != {0} or {f.fold for f in folds} != set(range(10)):
        raise ValueError("E0 expects exactly 10 folds: repeat=0, fold=0..9")

    if keep is not None:
        # J7: restrict every fold to the kept rows and renumber them 0..len(keep)-1.
        new_index = np.full(n_rows, -1, dtype=np.int64)
        new_index[keep] = np.arange(len(keep))
        folds = [
            OpenMLFold(f.repeat, f.fold,
                       new_index[f.train_idx[new_index[f.train_idx] >= 0]],
                       new_index[f.test_idx[new_index[f.test_idx] >= 0]])
            for f in folds
        ]
        n_rows = len(keep)
        if any(not len(f.train_idx) or not len(f.test_idx) for f in folds):
            raise ValueError("Empty fold after J7 subsampling")

    # In 10-fold CV, every observation is test data exactly once.
    test_frequency = np.bincount(np.concatenate([f.test_idx for f in folds]), minlength=n_rows)
    if not np.all(test_frequency == 1):
        raise ValueError("Test folds do not cover each observation exactly once")
    return tuple(folds)


def load_e0_task(task_id: int, root: Path | str = DEFAULT_ROOT) -> E0Task:
    """Load one E0 task: J1, ordinal encoding, J7 and the original OpenML folds."""
    root = Path(root)
    matches = sorted(root.glob(f"*/{int(task_id)}_*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one folder for task {task_id} under {root}, found {matches}. "
            "Download first using python fetch_e0.py --task TASK_ID"
        )
    folder = matches[0]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task_id") != int(task_id) or manifest.get("role") != "E0 development":
        raise ValueError(f"Unexpected task ID or role in {folder / 'manifest.json'}")
    task_type = folder.parent.name
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unknown task type: {task_type}")

    data, meta = _read_arff(folder, "data", with_meta=True)
    target = manifest["target_from_task"]
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} not found in {folder}")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate feature names in ARFF")
    if len(data) != manifest.get("expected_rows_from_M2"):
        raise ValueError("Row count differs from M2; verify the downloaded dataset")

    features = data.drop(columns=[target])
    # Ordinal encoding of nominal features (ARFF level order); '?' / unknown -> NaN.
    for column in features.columns:
        kind, levels = meta[column]
        if kind == "nominal":
            codes = {level: float(i) for i, level in enumerate(levels)}
            features[column] = features[column].map(codes).astype(np.float64)
    non_numeric = list(features.select_dtypes(exclude="number").columns)
    if non_numeric:
        raise NotImplementedError(f"Unsupported non-numeric columns: {non_numeric[:8]}")

    # J1: > 50% missing, constant, ID-like.
    drop = [c for c in features.columns
            if features[c].isna().mean() > 0.5 or features[c].nunique(dropna=True) <= 1]
    drop += [c for c in J1_ID_COLUMNS.get(int(task_id), ()) if c in features.columns]
    features = features.drop(columns=sorted(set(drop)))
    if features.shape[1] != manifest.get("expected_features_after_J1_from_M2"):
        raise ValueError(
            f"Feature count after J1 ({features.shape[1]}) differs from M2 "
            f"({manifest.get('expected_features_after_J1_from_M2')})"
        )
    X = features.to_numpy(dtype=np.float64)
    if np.isinf(X).any():
        raise ValueError("Features contain +/- infinity")

    target_values = data[target]
    if target_values.isna().any() or target_values.astype(str).eq("?").any():
        raise ValueError("Target contains missing values")
    if task_type == "regression":
        y = pd.to_numeric(target_values, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("Regression target must contain finite numbers")
    else:
        y = target_values.to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("Classification requires at least two classes")

    keep = None
    if len(data) > J7_MAX_ROWS:
        keep = np.sort(np.random.RandomState(J7_SEED).choice(len(data), J7_MAX_ROWS, replace=False))
        X, y = X[keep], y[keep]
    folds = _load_splits(folder, n_rows=len(data), keep=keep)
    return E0Task(
        task_id=int(task_id), name=manifest["expected_name_from_M2"],
        task_type=task_type, feature_names=tuple(features.columns),
        X=X, y=y, folds=folds,
    )


def main() -> None:
    for task_id in (361055, 361077):
        task = load_e0_task(task_id)
        X_train, X_test, y_train, y_test = task.get_fold(fold=0)
        print(f"\n{task.name} | task={task.task_id} | {task.task_type}")
        print(f"X.shape={task.X.shape}, y.shape={task.y.shape}, foldow={len(task.folds)}")
        print(f"Fold 0: train={X_train.shape}, test={X_test.shape}")
        print(f"Target: train={y_train.shape}, test={y_test.shape}")
        print(f"Brakujace wartosci X: {np.isnan(task.X).sum()}")


if __name__ == "__main__":
    main()
