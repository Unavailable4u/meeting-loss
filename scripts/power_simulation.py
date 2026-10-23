"""Seed noise and power of the pre-registered H1 / H6 rule (decision D-019; supports the choice of seeds).

1) Measures the run-to-run noise of steps-to-target in the tuning sweeps: for every trial whose two selection
   seeds both reached the target (adopted crossing rule), the SD of log(steps) per run is estimated from the
   difference between the two seeds; trials are pooled per arm and overall. (Trials that never reach the
   target are excluded, so slow, noisy configurations are under-represented.)
2) Simulates the three-way outcome of D-019 (margin 15%, two-sided 97.5% paired-bootstrap interval) for a single
   target: noise is log-normal, arms are independent (a conservative choice: pairing by seed can only help),
   and the true change of mean FLOPs-to-target is varied. The real rule needs two targets to agree, so its power
   is somewhat lower than shown.

Usage: python scripts/power_simulation.py [--sims 1000] [--boots 1000]
Writes results/processed/power_stage3.csv. Exploratory planning aid, not evidence about the methods.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src.sweep import collect_sweep


def measure_noise(root="results/raw/sweeps"):
    arms = ["base", "deepsup", "meet", "meet_grad", "deepsup_code", "meet_frozen"]
    sets = {a: collect_sweep("t1", [a], root)[a] for a in arms}
    sets["base (t2)"] = collect_sweep("t2", ["base"], root)["base"]
    rows, pooled = [], []
    for name, d in sets.items():
        diffs = []
        for t in d["trials"]:
            ps = list(t["per_seed"].values())
            if all(p["reached"] for p in ps):
                diffs.append(np.log(ps[0]["steps"]) - np.log(ps[1]["steps"]))
        diffs = np.array(diffs)
        rows.append((name, len(diffs), float(np.sqrt(np.mean(diffs ** 2) / 2))))
        pooled.append((len(diffs), float(np.mean(diffs ** 2) / 2)))
    n = np.array([p[0] for p in pooled]); v = np.array([p[1] for p in pooled])
    return rows, float(np.sqrt(np.sum(n * v) / np.sum(n)))


def outcomes(rng, n, sigma, delta, sims, boots, alpha=0.025, margin=-15.0):
    out = dict(supported=0, refuted=0, inconclusive=0)
    for _ in range(sims):
        a = np.exp(rng.normal(0, sigma, n))
        d = (1 + delta / 100) * np.exp(rng.normal(0, sigma, n))
        idx = rng.integers(0, n, (boots, n))
        stat = 100 * (d[idx].mean(1) / a[idx].mean(1) - 1)
        point = 100 * (d.mean() / a.mean() - 1)
        lo, hi = np.quantile(stat, [alpha, 1 - alpha])
        if point <= margin and hi < 0:
            out["supported"] += 1
        elif lo > margin:
            out["refuted"] += 1
        else:
            out["inconclusive"] += 1
    return {k: v / sims for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--root", default="results/raw/sweeps")
    ap.add_argument("--out", default="results/processed/power_stage3.csv")
    args = ap.parse_args()

    rows, pooled = measure_noise(args.root)
    print("per-run SD of log steps-to-target (trials with both selection seeds reaching 0.975):")
    for name, n, sd in rows:
        print(f"  {name:>13}: {sd:.3f}  ({n} trials)")
    print(f"  pooled: {pooled:.3f}")

    rng = np.random.default_rng(1)
    table = []
    print("\nthree-way outcome probabilities (single target, 97.5% interval, margin 15%)")
    for sigma in sorted({0.12, round(pooled, 3), 0.16}):
        for delta in (0, -10, -15, -20):
            res = {n: outcomes(rng, n, sigma, delta, args.sims, args.boots) for n in (10, 20)}
            print(f"  sigma {sigma:.3f} true change {delta:+4d}%:  " + " | ".join(
                f"n={n}: supported {r['supported']:.2f} refuted {r['refuted']:.2f} inconclusive {r['inconclusive']:.2f}"
                for n, r in res.items()))
            for n, r in res.items():
                table.append(dict(sigma=sigma, true_change_pct=delta, n_seeds=n, **r))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
