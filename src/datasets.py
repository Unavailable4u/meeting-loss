"""Datasets and their accuracy targets (setting study, D-031). No torch import: the sweep and analysis scripts use it.

A run's dataset is the config key `dataset` (default "mnist": every earlier config and log is MNIST and unchanged). Each dataset has a
fixed train/validation split (src/data.py, decision D-008) and two accuracy targets. MNIST keeps 0.97 and 0.975 (D-003). Any other
dataset gets its targets from a PROBE (scripts/probe_dataset.py), fixed before any other arm is run on it:

    plateau = the median over the probe seeds of the mean validation accuracy of the last 5 evaluations of arm A (the MNIST-tuned
    configuration, unchanged) at 3,000 steps;  high = floor((plateau - 0.005) / 0.005) * 0.005;  low = high - 0.005.

The rule reproduces the MNIST targets from arm A's MNIST plateau (0.9804 -> 0.975 and 0.97). The targets are written to
configs/datasets/<dataset>_targets.json, committed, and never edited.
"""
import json
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
DATASETS = {                    # name -> torchvision class, normalisation (mean, std of the training pixels)
    "mnist": dict(tv="MNIST", mean=0.1307, std=0.3081),
    "fmnist": dict(tv="FashionMNIST", mean=0.2860, std=0.3530),
}
MNIST_TARGETS = (0.97, 0.975)
PLATEAU_MARGIN, TARGET_STEP = 0.005, 0.005


def check_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return name


def targets_from_plateau(plateau):
    """(low, high) from arm A's plateau accuracy (module docstring)."""
    high = math.floor((plateau - PLATEAU_MARGIN) / TARGET_STEP + 1e-9) * TARGET_STEP
    return (round(high - TARGET_STEP, 4), round(high, 4))


def targets_path(name, repo=REPO):
    return pathlib.Path(repo) / "configs" / "datasets" / f"{name}_targets.json"


def dataset_targets(name, repo=REPO):
    """(low, high) accuracy targets of a dataset; for a dataset other than MNIST they come from its probe."""
    check_dataset(name)
    if name == "mnist":
        return MNIST_TARGETS
    p = targets_path(name, repo)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run `python scripts/probe_dataset.py --dataset {name}` first and commit the result")
    t = json.loads(p.read_text())["targets"]
    return (float(t[0]), float(t[1]))
