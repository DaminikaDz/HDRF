"""Inspect raw OpenML E0 tasks and their original train/test folds.

Read-only: does not modify ARFF files or fit models.
"""
from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path("data/e0/raw")
TASK_FOLDERS = [
    ROOT / "classification" / "361055_credit",
    ROOT / "regression" / "361077_Ailerons",
]


def read_arff(folder: Path, prefix: str) -> pd.DataFrame:
    files = sorted(p for p in folder.glob(prefix + ".arff*") if p.suffix in {".arff", ".gz"})
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one {prefix}.arff[.gz] in {folder}, found: {files}")
    opener = gzip.open if files[0].suffix == ".gz" else open
    with opener(files[0], "rt", encoding="utf-8", errors="replace") as stream:
        records, _ = arff.loadarff(stream)
    df = pd.DataFrame.from_records(records)
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].map(lambda value: value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value)
    return df


def inspect(folder: Path) -> None:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    target = manifest["target_from_task"]
    data = read_arff(folder, "data")
    splits = read_arff(folder, "splits")

    print(f"\n{'=' * 60}")
    print(f"TASK {manifest['task_id']}  |  {manifest['expected_name_from_M2']}")
    print(f"Wymiary surowych danych: {data.shape}")
    print(f"Target z OpenML: {target!r}")
    if target not in data.columns:
        raise KeyError(f"Target {target!r} not found. Available columns: {list(data.columns)}")

    features = data.drop(columns=[target])
    print(f"Liczba surowych cech: {features.shape[1]}")
    print(f"Liczba cech po J1 wg M2: {manifest['expected_features_after_J1_from_M2']}")
    print(f"Kolumny numeryczne: {features.select_dtypes(include='number').shape[1]}")
    print(f"Kolumny nienumeryczne: {features.select_dtypes(exclude='number').shape[1]}")
    missing = features.isna().sum().sum() + features.eq('?').sum().sum()
    print(f"Liczba brakow (komorek cech): {int(missing)}")
    print("Pierwsze cechy:", list(features.columns[:8]))
    if manifest["role"] == "E0 development" and folder.parent.name == "classification":
        print("Klasy i liczebnosci:", data[target].value_counts(dropna=False).to_dict())
    else:
        goal = pd.to_numeric(data[target], errors="coerce")
        print(f"Target regresji: min={goal.min():.5g}, max={goal.max():.5g}")

    required = {"repeat", "fold", "rowid", "type"}
    if not required.issubset(splits.columns):
        raise ValueError(f"Unexpected split columns: {list(splits.columns)}")
    for column in ("repeat", "fold", "rowid"):
        splits[column] = splits[column].astype("int64")
    splits["type"] = splits["type"].astype(str).str.upper()
    if not set(splits["type"]).issubset({"TRAIN", "TEST"}):
        raise ValueError(f"Unexpected split types: {splits['type'].unique()}")

    print(f"Plik podzialow: {len(splits):,} wierszy")
    print(f"Liczba powtorzen: {splits['repeat'].nunique()}")
    print(f"Liczba foldow: {splits[['repeat', 'fold']].drop_duplicates().shape[0]}")

    valid = True
    for (repeat, fold), rows in splits.groupby(["repeat", "fold"]):
        train = rows.loc[rows["type"] == "TRAIN", "rowid"].to_numpy()
        test = rows.loc[rows["type"] == "TEST", "rowid"].to_numpy()
        partition_ok = (
            len(np.intersect1d(train, test)) == 0
            and len(np.unique(train)) == len(train)
            and len(np.unique(test)) == len(test)
            and len(train) + len(test) == len(data)
            and np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(data)))
        )
        valid &= partition_ok
        if repeat == splits["repeat"].min() and fold == splits["fold"].min():
            print(f"Pierwszy fold (repeat={repeat}, fold={fold}): train={len(train)}, test={len(test)}")
        if not partition_ok:
            print(f"  UWAGA: niepoprawny podzial repeat={repeat}, fold={fold}")
    print(f"Podzialy rozlaczne i kompletne: {valid}")
    if not valid:
        raise ValueError("Split validation failed")


if __name__ == "__main__":
    for folder in TASK_FOLDERS:
        inspect(folder)
    print("\nGotowe: kontrola metadanych i podzialow zakonczona.")
