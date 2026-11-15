import json
import numpy as np
import pytest

from data_loader import load_e0_task


def create_example(root, task_id=361055, task_type="classification", extra_feature=False, bad_split=False, big=False):
    folder = root / task_type / f"{task_id}_example"
    folder.mkdir(parents=True)
    col = "@ATTRIBUTE cat {a,b}\n" if extra_feature else ""
    values = []
    for i in range(20):
        target = str(i % 2) if task_type == "classification" else str(float(i) / 10.0)
        row = f"{i},{i/2.0}"
        if extra_feature:
            row += ",a" if i % 2 else ",b"
        values.append(f"{row},{target}")
    target_type = "{0,1}" if task_type == "classification" else "NUMERIC"
    content = ("@RELATION example\n@ATTRIBUTE a NUMERIC\n@ATTRIBUTE b NUMERIC\n" + col
               + f"@ATTRIBUTE goal {target_type}\n@DATA\n" + "\n".join(values) + "\n")
    (folder / "data.arff").write_text(content)
    split_records = []
    for fold in range(10):
        test_rows = {2*fold, 2*fold+1}
        for rowid in range(20):
            category = "TEST" if rowid in test_rows else "TRAIN"
            if bad_split and fold == 0 and rowid == 0:
                category = "TRAIN"
            split_records.append(f"0,{fold},{rowid},{category}")
    splits = ("@RELATION splits\n@ATTRIBUTE repeat NUMERIC\n@ATTRIBUTE fold NUMERIC\n"
              "@ATTRIBUTE rowid NUMERIC\n@ATTRIBUTE type {TRAIN,TEST}\n@DATA\n" + "\n".join(split_records) + "\n")
    (folder / "splits.arff").write_text(splits)
    manifest = {"role":"E0 development", "task_id":task_id, "target_from_task":"goal",
                "expected_name_from_M2":"example", "expected_rows_from_M2": 50_001 if big else 20,
                "expected_features_after_J1_from_M2":2 if not extra_feature else 3}
    (folder / "manifest.json").write_text(json.dumps(manifest))


def test_classification(tmp_path):
    create_example(tmp_path)
    task = load_e0_task(361055, root=tmp_path)
    assert task.X.shape == (20, 2)
    assert task.y.shape == (20,)
    assert len(task.folds) == 10
    assert set(task.y.tolist()) == {"0", "1"}
    Xtr, Xte, ytr, yte = task.get_fold(0)
    assert Xtr.shape == (18, 2) and Xte.shape == (2, 2)
    assert set(Xte[:, 0].tolist()) == {0.0, 1.0}
    assert set(yte.tolist()) == {"0", "1"}


def test_regression(tmp_path):
    create_example(tmp_path, 361077, "regression")
    task = load_e0_task(361077, root=tmp_path)
    assert task.y.dtype == np.float64
    assert task.task_type == "regression"
    assert np.isclose(task.y[1], 0.1)


def test_bad_split_rejected(tmp_path):
    create_example(tmp_path, bad_split=True)
    with pytest.raises(ValueError, match="Invalid OpenML partition|Test folds do not cover"):
        load_e0_task(361055, root=tmp_path)


def test_categorical_not_silently_encoded(tmp_path):
    create_example(tmp_path, extra_feature=True)
    with pytest.raises(NotImplementedError, match="Ordinal encoding"):
        load_e0_task(361055, root=tmp_path)


def test_j7_not_silently_ignored(tmp_path):
    create_example(tmp_path, big=True)
    with pytest.raises(NotImplementedError, match="J7"):
        load_e0_task(361055, root=tmp_path)


def test_missing_task(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_e0.py"):
        load_e0_task(361055, root=tmp_path)
