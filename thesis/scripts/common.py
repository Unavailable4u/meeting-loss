"""Shared loaders for the thesis figures and tables.

Everything is computed from the raw run logs of the repository (results/raw/**), using the repository's own
scoring code (src/sweep.py, src/crossing.py), so that no number in the thesis is typed by hand.
"""
import glob
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(os.environ.get("MEETING_LOSS_REPO", "/home/claude/meeting-loss"))
sys.path.insert(0, str(REPO))
from src.sweep import load_run, score_run, paired_pct_change  # noqa: E402

RAW = REPO / "results" / "raw"
TARGETS = (0.97, 0.975)

# group name -> (directory under results/raw, file pattern)
STAGE3 = {
    "A":      (".", "20261023_final-t1-base-two_base_s*.jsonl"),
    "A2":     (".", "20261023_final-t2-base-two_base_s*.jsonl"),
    "B":      (".", "20261023_final-t1-deepsup-two_deepsup_s*.jsonl"),
    "D":      (".", "20261023_final-t1-meet-two_meet_s*.jsonl"),
    "C":      (".", "20261023_final-t1-ae-two_ae_s*.jsonl"),
    "E":      (".", "20261023_final-t1-base_lowlr-two_base_s*.jsonl"),
    "B-code": (".", "20261023_final-t1-deepsup_code-two_deepsup_s*.jsonl"),
    "F":      (".", "20261023_final-t1-meet_grad-two_meet_grad_s*.jsonl"),
    "V1":     (".", "20261023_final-t1-meet_noise-two_meet_s*.jsonl"),
    "V2":     (".", "20261023_final-t1-meet_ao-two_meet_ao_s*.jsonl"),
    "V3":     (".", "20261023_final-t1-meet_frozen-two_meet_frozen_s*.jsonl"),
}
PREREG2 = {
    "A1":  "20261027_final-t1-base-two_base_s*.jsonl",
    "A2":  "20261027_final-t2-base-two_base_s*.jsonl",
    "A5":  "20261027_final-t5-base-two_base_s*.jsonl",
    "A6":  "20261027_final-t6-base-two_base_s*.jsonl",
    "A7":  "20261027_final-t7-base-two_base_s*.jsonl",
    "A8":  "20261027_final-t8-base-two_base_s*.jsonl",
    "B48": "20261027_final-t9-base_b48-two_base_s*.jsonl",
    "S30": "20261027_prereg2-S30_selective_s*.jsonl",
    "S20": "20261027_prereg2-S20_selective_s*.jsonl",
    "S30T": "20261027_final-t10-selective30-two_selective_s*.jsonl",
    "R30": "20261027_prereg2-R30_selective_s*.jsonl",
}
PREREG3 = {
    "A1":  "20261030_final-t1-base-two_base_s*.jsonl",
    "A2":  "20261030_final-t2-base-two_base_s*.jsonl",
    "A5":  "20261030_final-t5-base-two_base_s*.jsonl",
    "A6":  "20261030_final-t6-base-two_base_s*.jsonl",
    "A7":  "20261030_final-t7-base-two_base_s*.jsonl",
    "A8":  "20261030_final-t8-base-two_base_s*.jsonl",
    "S20w0": "20261030_prereg3-S20w0_selective_s*.jsonl",
    "S30": "20261030_prereg3-S30_selective_s*.jsonl",
    "SBT": "20261030_final-t11-sb-two_sb_s*.jsonl",
    "S20": "20261030_prereg3-S20_selective_s*.jsonl",
    "S15": "20261030_prereg3-S15_selective_s*.jsonl",
    "S20w10": "20261030_prereg3-S20w10_selective_s*.jsonl",
    "R20w0": "20261030_prereg3-R20w0_selective_s*.jsonl",
}


def _load_glob(directory, pattern):
    runs = {}
    for p in sorted(glob.glob(str(RAW / directory / pattern))):
        hdr, rows = load_run(p)
        runs[hdr["config"]["seed"]] = dict(rows=rows, header=hdr)
    return runs


def load_stage3():
    return {g: _load_glob(d, pat) for g, (d, pat) in STAGE3.items()}


def load_prereg2():
    return {g: _load_glob("prereg_v2", pat) for g, pat in PREREG2.items()}


def load_prereg3():
    return {g: _load_glob("prereg_v3", pat) for g, pat in PREREG3.items()}


def scores(group_runs, target, key="score"):
    """dict seed -> FLOPs (or steps) to the target under the D-018 rule; never-reached runs get full-run FLOPs."""
    out = {}
    for s, r in group_runs.items():
        d = score_run(r["rows"], target)
        if key == "steps":
            out[s] = d["steps"] if d["reached"] else np.nan
        else:
            out[s] = d[key]
    return out


def arr(group_runs, target, seeds=None, key="score"):
    sc = scores(group_runs, target, key)
    seeds = sorted(sc) if seeds is None else seeds
    return np.array([sc[s] for s in seeds], dtype=float)


def n_never(group_runs, target):
    return sum(not score_run(r["rows"], target)["reached"] for r in group_runs.values())


def pct(ref, grp, n_boot=10000, alpha=0.05, seed=0):
    return paired_pct_change(ref, grp, n_boot=n_boot, alpha=alpha, seed=seed)


def curve(group_runs, grid=None, n=500, xkey="flops"):
    """Validation accuracy of every seed on a common x grid (FLOPs by default), linear interpolation per seed."""
    seeds = sorted(group_runs)
    xs = [np.array([r[xkey] for r in group_runs[s]["rows"]], float) for s in seeds]
    ys = [np.array([r["val_acc"] for r in group_runs[s]["rows"]], float) for s in seeds]
    if grid is None:
        grid = np.linspace(0, min(x[-1] for x in xs), n)
    Y = np.array([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    return grid, Y


def curve_steps(group_runs):
    seeds = sorted(group_runs)
    steps = np.array([r["step"] for r in group_runs[seeds[0]]["rows"]], float)
    Y = np.array([[r["val_acc"] for r in group_runs[s]["rows"]] for s in seeds], float)
    return steps, Y
