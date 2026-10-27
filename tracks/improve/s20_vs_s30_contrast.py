"""Improve track, look 1 (docs/notes/improve-s20-vs-s30-contrast.md): paired S20 against S30 on the prereg-v2 logs.

Exploratory, reads existing logs only. Usage: python tracks/improve/s20_vs_s30_contrast.py
"""
import csv
import glob
import json
import os
import re

import numpy as np

RAW = "results/raw/prereg_v2"
OUT = "results/processed/improve_s20_vs_s30.csv"
SEEDS = list(range(1300, 1320))
TARGETS = [0.90, 0.95, 0.96, 0.97, 0.975]
N_BOOT, BOOT_SEED, LO, HI = 10000, 0, 1.25, 98.75  # 97.5% two-sided, as in the protocol


def load(group):
    runs = {}
    for f in glob.glob(f"{RAW}/**/*{group}_selective_s*.jsonl", recursive=True):
        lines = [json.loads(x) for x in open(f)]
        seed = lines[0]["config"]["seed"]
        runs[seed] = [r for r in lines[1:] if "val_acc" in r]
    assert sorted(runs) == SEEDS, (group, sorted(runs))
    return runs


def cross(rows, t):
    """First of two consecutive evaluations at or above t (D-018); (flops, step) or (nan, nan) if never."""
    for i in range(len(rows) - 1):
        if rows[i]["val_acc"] >= t and rows[i + 1]["val_acc"] >= t:
            return rows[i]["flops"], rows[i]["step"]
    return np.nan, np.nan


def main():
    g = {"S30": load("prereg2-S30"), "S20": load("prereg2-S20")}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(SEEDS), (N_BOOT, len(SEEDS)))
    out = []
    print(f"{'target':>7} {'metric':>6} | S30 mean | S20 mean | paired change [97.5% interval] | seeds S20<S30 | never S30/S20")
    for t in TARGETS:
        c = {k: np.array([cross(v[s], t) for s in SEEDS]) for k, v in g.items()}  # shape (20, 2)
        for j, metric in enumerate(["FLOPs", "steps"]):
            a, b = c["S20"][:, j], c["S30"][:, j]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < len(SEEDS):
                print(f"{t:7.3f} {metric:>6} | skipped: a run never reached the target (n_never S30={int(np.isnan(b).sum())}, S20={int(np.isnan(a).sum())})")
                out.append([t, metric, "", "", "", "", "", int(np.isnan(b).sum()), int(np.isnan(a).sum())])
                continue
            pt = (a.mean() / b.mean() - 1) * 100
            bs = (a[idx].mean(1) / b[idx].mean(1) - 1) * 100
            lo, hi = np.percentile(bs, LO), np.percentile(bs, HI)
            wins = int((a < b).sum())
            scale = 1e9 if metric == "FLOPs" else 1
            print(f"{t:7.3f} {metric:>6} | {b.mean()/scale:8.1f} | {a.mean()/scale:8.1f} | {pt:+6.1f}% [{lo:+6.1f}, {hi:+6.1f}] | {wins:2d}/20 | 0/0")
            out.append([t, metric, round(b.mean() / scale, 2), round(a.mean() / scale, 2), round(pt, 2), round(lo, 2), round(hi, 2), 0, 0])
    # per-step cost after warm-up and warm-up share
    for k, v in g.items():
        rows = v[SEEDS[0]]
        steps = [r["step"] for r in rows]
        f = [r["flops"] for r in rows]
        i100 = steps.index(100)
        per_step = (f[-1] - f[i100]) / (steps[-1] - steps[i100])
        print(f"{k}: FLOPs per step after warm-up {per_step/1e6:.2f} M, warm-up (100 full steps) {f[i100]/1e9:.2f} G")
    s30_cost = np.mean([cross(g["S30"][s], 0.975)[0] for s in SEEDS])
    s20_cost = np.mean([cross(g["S20"][s], 0.975)[0] for s in SEEDS])
    w = np.mean([next(r["flops"] for r in g["S30"][s] if r["step"] == 100) for s in SEEDS])
    print(f"warm-up share of the cost to 0.975: S30 {w/s30_cost*100:.1f}%, S20 {w/s20_cost*100:.1f}%")
    # validation accuracy at equal steps
    print("\nmean validation accuracy at equal steps (S20 - S30, percentage points; sd of paired difference in brackets)")
    for st in (100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000):
        d = np.array([next(r["val_acc"] for r in g["S20"][s] if r["step"] == st) - next(r["val_acc"] for r in g["S30"][s] if r["step"] == st) for s in SEEDS]) * 100
        print(f"  step {st:>4}: {d.mean():+.3f} [{d.std(ddof=1):.3f}]  S20 ahead in {int((d > 0).sum())}/20 seeds")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["target", "metric", "S30_mean", "S20_mean", "paired_change_pct", "lo", "hi", "n_never_S30", "n_never_S20"])
        w_.writerows(out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
