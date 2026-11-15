"""Download raw E0 development tasks from OpenML (REST API v1).

No third-party dependencies. Downloads the original dataset, task metadata,
associated dataset metadata and the original OpenML split file.

This script does NOT preprocess data or train models.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

API = "https://www.openml.org/api/v1/json"

# Exact task IDs from M2, section 2.4 (development data, not test benchmarks).
TASKS = {
    361282: ("classification", "albert", 58252, 31),
    361055: ("classification", "credit", 16714, 10),
    361275: ("classification", "default-of-credit-card-clients", 13272, 20),
    361070: ("classification", "eye_movements", 7608, 19),
    361278: ("classification", "heloc", 10000, 22),
    361274: ("classification", "jannis", 57580, 54),
    361065: ("classification", "MagicTelescope", 13376, 10),
    361068: ("classification", "MiniBooNE", 72998, 50),
    361077: ("regression", "Ailerons", 13750, 33),
    361292: ("regression", "Allstate_Claims_Severity", 188318, 124),
    361082: ("regression", "Bike_Sharing_Demand", 17379, 6),
    361074: ("regression", "elevators", 16599, 16),
    361079: ("regression", "house_16H", 22784, 16),
    361097: ("regression", "Mercedes_Benz_Greener_Manufacturing", 4209, 359),
    361085: ("regression", "sulfur", 10081, 6),
    361279: ("regression", "yprop_4_1", 8885, 42),
}


def request(url):
    return Request(url, headers={"User-Agent": "HDRF-E0-research/0.1"})


def fetch_json(url):
    with urlopen(request(url), timeout=90) as response:
        return json.load(response)


def download(url, destination):
    """Stream download to a temp file and rename only on success."""
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"  cached: {destination.name}")
        return
    temp = destination.with_name(destination.name + ".part")
    try:
        with urlopen(request(url), timeout=120) as response, temp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        temp.replace(destination)
        print(f"  saved: {destination.name} ({destination.stat().st_size:,} B)")
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def task_input(task, name):
    inputs = task.get("input", [])
    if isinstance(inputs, dict):
        inputs = [inputs]
    matching = [item for item in inputs if item.get("name") == name]
    if len(matching) != 1:
        raise ValueError(f"Expected input {name!r}, found {len(matching)}")
    return matching[0]


def extension_from_url(url):
    path = urlsplit(url).path.lower()
    if path.endswith(".arff.gz"):
        return ".arff.gz"
    if path.endswith(".gz"):
        return ".gz"
    return ".arff"


def download_task(task_id, base, metadata_only=False):
    group, expected_name, expected_n, expected_p = TASKS[task_id]
    print(f"\n[{group}] task {task_id} | {expected_name}")
    task_payload = fetch_json(f"{API}/task/{task_id}")
    task = task_payload["task"]
    source = task_input(task, "source_data")
    dataset_id = int(source["data_set"]["data_set_id"])
    # In the OpenML task JSON, target_feature lives inside source_data.data_set.
    target_name = source["data_set"].get("target_feature")
    procedure = task_input(task, "estimation_procedure")["estimation_procedure"]
    split_url = procedure.get("data_splits_url")
    if not split_url:
        raise ValueError(f"Task {task_id} has no train/test split URL")

    dataset_payload = fetch_json(f"{API}/data/{dataset_id}")
    dataset = dataset_payload["data_set_description"]
    if not target_name:
        target_name = dataset.get("default_target_attribute")
    data_url = dataset.get("url")
    if not data_url:
        raise ValueError(f"Dataset {dataset_id} has no data URL")

    folder = base / group / f"{task_id}_{expected_name}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, obj in (("task.json", task_payload), ("dataset.json", dataset_payload)):
        (folder / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "role": "E0 development", "task_id": task_id,
        "dataset_id": dataset_id, "expected_name_from_M2": expected_name,
        "dataset_name_openml": dataset.get("name"),
        "expected_rows_from_M2": expected_n, "expected_features_after_J1_from_M2": expected_p,
        "target_from_task": target_name, "split_procedure": procedure.get("type"),
        "raw_data_url": data_url, "original_split_url": split_url,
        "preprocessing_applied": False,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  dataset ID: {dataset_id} | target: {target_name}")
    print(f"  split procedure: {procedure.get('type')}")
    if metadata_only:
        print("  metadata only; no ARFF files downloaded")
        return

    download(data_url, folder / ("data" + extension_from_url(data_url)))
    download(split_url, folder / ("splits" + extension_from_url(split_url)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="download all 16 E0 tasks")
    group.add_argument("--task", type=int, choices=sorted(TASKS), help="download one M2 task")
    parser.add_argument("--metadata-only", action="store_true", help="only metadata (no large files)")
    parser.add_argument("--output", type=Path, default=Path("data/e0/raw"), help="output directory")
    args = parser.parse_args()
    ids = list(TASKS) if args.all else [args.task] if args.task else [361055, 361077]
    errors = []
    for task_id in ids:
        try:
            download_task(task_id, args.output, args.metadata_only)
        except Exception as exc:
            errors.append(task_id)
            print(f"  ERROR for task {task_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"\nFinished: {len(ids)-len(errors)}/{len(ids)} tasks downloaded successfully")
    if errors:
        print("Failed task IDs:", ", ".join(map(str, errors)), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
