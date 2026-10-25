"""Pilot: does a selective backward pass keep the speed of training while saving FLOPs?

Runs the tuned baseline (arm A) and the selective-backward variants with the SAME tuned hyperparameters
(learning rate, weight decay, activation, schedule) on 3 new seeds and compares FLOPs-to-target under the
shared crossing rule (src/crossing.py, D-018). Exploratory: it only decides whether a pre-registered test is
worth building. Plan and decision rule (written before the run): docs/notes/selective-backward-pilot.md.

Usage:
  python scripts/pilot_selective.py --tag pilot1                 # real run: 4 variants x 3 seeds x 3,000 steps
  python scripts/pilot_selective.py --tag sf1 --sel-fracs 0.1 0.2 0.3 0.5 --seeds 1210 1211 1212
                                                                 # Look 3 (docs/notes/selective-backward-prereg-prep.md): the sel_frac curve and its knee
  python scripts/pilot_selective.py --synthetic --seeds 0 --max-steps 200 --variants S30 --tag smoke \
         --out-dir /tmp/pilot --csv /tmp/pilot.csv                # plumbing test, no MNIST
"""
import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from exploratory.looks import frac_name, knee  # noqa: E402
from src.crossing import crossing_index  # noqa: E402
from src.sweep import _git_info, config_hash  # noqa: E402
from src.train import run  # noqa: E402

TARGETS = (0.97, 0.975)
VARIANTS = {   # overrides on top of the tuned baseline config
    "S30": dict(arm="selective", sel_frac=0.3, sel_warmup=100, sel_mode="top"),
    "S50": dict(arm="selective", sel_frac=0.5, sel_warmup=100, sel_mode="top"),
    "R30": dict(arm="selective", sel_frac=0.3, sel_warmup=100, sel_mode="random"),   # control: random rows
}
MAIN = ("S30", "S50")        # R30 is a control and never decides the verdict
# ---- pre-registered pilot rule (docs/notes/selective-backward-pilot.md) ----
GO_RATIO = 0.85              # FLOPs-to-target, ratio of means vs A, at BOTH targets, with no missed seed
STOP_RATIO = 1.00            # at the 0.975 target; also STOP if 2 or more seeds miss it


def synthetic_loaders(batch_size, seed, n=6000):
    g = torch.Generator().manual_seed(0)
    means = torch.randn(10, 784, generator=g) * 0.6

    def make(n_, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.randint(0, 10, (n_,), generator=gg)
        return means[y] + torch.randn(n_, 784, generator=gg) * 3.0, y

    x, y = make(n, 1)
    xv, yv = make(1000, 2)
    return (DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(seed)),
            DataLoader(TensorDataset(xv, yv), batch_size=500))


def crossing_cost(rows, target):
    """(flops, steps, reached) at the crossing; a run that never reaches the target costs its full budget."""
    i = crossing_index([r["val_acc"] for r in rows], target)
    if i is None:
        return rows[-1]["flops"], None, False
    return rows[i]["flops"], rows[i]["step"], True


def summarize(runs):
    """runs: {variant: {seed: rows}} (must contain "A"). Returns {variant: {target: stats}}."""
    base = runs["A"]
    out = {}
    for name, by_seed in runs.items():
        out[name] = {}
        for t in TARGETS:
            cost = {s: crossing_cost(rows, t) for s, rows in by_seed.items()}
            ref = {s: crossing_cost(rows, t) for s, rows in base.items()}
            seeds = sorted(set(cost) & set(ref))
            f = np.array([cost[s][0] for s in seeds], float)
            fa = np.array([ref[s][0] for s in seeds], float)
            st = [cost[s][1] for s in seeds if cost[s][2]]
            sta = [ref[s][1] for s in seeds if ref[s][2]]
            out[name][t] = dict(
                flops_mean=float(f.mean()), ratio_of_means=float(f.mean() / fa.mean()),
                paired_ratio=float(np.mean(f / fa)), misses=sum(1 for s in seeds if not cost[s][2]),
                n=len(seeds), steps_mean=float(np.mean(st)) if st else float("nan"),
                steps_ratio=float(np.mean(st) / np.mean(sta)) if st and sta else float("nan"),
                final_val=float(np.mean([by_seed[s][-1]["val_acc"] for s in seeds])))
    return out


def pilot_verdict(summary):
    """GO / STOP / TUNE from the pre-registered rule. Only S30 and S50 decide; R30 is reported as a control.
    GO    a main variant has ratio_of_means <= 0.85 and no missed seed at BOTH targets
    STOP  every main variant has ratio_of_means >= 1.0 at 0.975, or 2 or more missed seeds there
    TUNE  anything else"""
    present = [v for v in MAIN if v in summary]
    if not present:
        return "TUNE"
    if any(all(summary[v][t]["ratio_of_means"] <= GO_RATIO and summary[v][t]["misses"] == 0 for t in TARGETS)
           for v in present):
        return "GO"
    if all(summary[v][0.975]["ratio_of_means"] >= STOP_RATIO or summary[v][0.975]["misses"] >= 2 for v in present):
        return "STOP"
    return "TUNE"


def load_all(out_dir):
    runs = {}
    for f in sorted(pathlib.Path(out_dir).glob("*_s*.jsonl")):
        name, seed = f.stem.rsplit("_s", 1)
        lines = [json.loads(l) for l in open(f) if l.strip()]
        runs.setdefault(name, {})[int(seed)] = lines[1:]
    return runs


def write_run(path, cfg, log):
    commit, dirty = _git_info()
    header = {"_header": True, "run_id": path.stem, "config": cfg, "config_hash": config_hash(cfg),
              "git_commit": commit, "git_dirty": dirty, "pilot": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in log:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def print_summary(summary):
    print("\n=== FLOPs-to-target vs the tuned baseline A (ratio < 1 means cheaper); crossing rule: two consecutive evaluations ===")
    print("variant  target | GFLOPs  ratio(means) paired ratio | steps ratio | missed seeds | final val")
    for name in summary:
        for t in TARGETS:
            s = summary[name][t]
            print(f"{name:7s}  {t:.3f}  | {s['flops_mean'] / 1e9:6.1f}   {s['ratio_of_means']:6.3f}      {s['paired_ratio']:6.3f}     |   {s['steps_ratio']:6.3f}    |   {s['misses']}/{s['n']}        | {s['final_val']:.4f}")
    if "R30" in summary and "S30" in summary:
        d = summary["R30"][0.975]["ratio_of_means"] / summary["S30"][0.975]["ratio_of_means"] - 1
        print(f"\nControl: random 30% costs {100 * d:+.1f}% more FLOPs-to-0.975 than the top-loss 30% "
              f"({'selection by loss matters' if d > 0.05 else 'NOT clearly better than random rows: selection by loss may not matter'}).")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuned/t1_base_two.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[600, 601, 602])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--sel-fracs", type=float, nargs="+", default=None,
                    help="Look 3: run selection fractions (top-loss rows, sel_warmup 100) instead of --variants; reports the knee")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tag", default="pilot1")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--synthetic", action="store_true", help="plumbing test on synthetic data (no MNIST)")
    args = ap.parse_args(argv)

    base = yaml.safe_load(open(REPO / args.config if not os.path.isabs(args.config) and not os.path.exists(args.config) else args.config))
    if args.max_steps:
        base["max_steps"] = args.max_steps
    device = "cpu" if base.get("device", "auto") == "auto" else base["device"]
    out_dir = pathlib.Path(args.out_dir or REPO / "results" / "raw" / "pilot" / args.tag)
    variants, fracs = list(args.variants), None
    if args.sel_fracs:
        fracs = {frac_name(f): f for f in args.sel_fracs}
        for n, f in fracs.items():
            VARIANTS.setdefault(n, dict(arm="selective", sel_frac=f, sel_warmup=100, sel_mode="top"))
        variants = list(fracs)

    for seed in args.seeds:
        for name in ["A"] + variants:
            path = out_dir / f"{name}_s{seed}.jsonl"
            if path.exists():
                print(f"skip {path.name} (exists)")
                continue
            cfg = {**base, **(VARIANTS[name] if name != "A" else {}), "seed": seed, "stage": f"pilot-{args.tag}"}
            data = synthetic_loaders(cfg["batch_size"], seed) if args.synthetic else None
            print(f"run {name} seed {seed} ({cfg['arm']}, {cfg['max_steps']} steps)", flush=True)
            write_run(path, cfg, run(cfg, device, data=data))

    runs = {k: v for k, v in load_all(out_dir).items() if k == "A" or k in variants}
    summary = summarize(runs)
    print_summary(summary)
    if fracs:
        verdict = knee(summary, fracs)
        print("\nLook 3 (descriptive; the pre-registered pilot rule is not applied here). Per-step cost is fixed by sel_frac: "
              "the forward pass of the whole batch is 44.4% of arm A's step, so no selection can save more than about 56% per step.")
        print(f"Knee = smallest sel_frac with ratio <= 0.85 at both targets and no missed seed: "
              f"{verdict if verdict is not None else 'none among ' + str(sorted(fracs.values()))}\n")
    else:
        verdict = pilot_verdict(summary)
        print(f"\nPre-registered pilot rule: GO if S30 or S50 has FLOPs-to-target <= {GO_RATIO} x A at BOTH targets with no missed seed; "
              f"STOP if both have ratio >= {STOP_RATIO} at 0.975 or 2+ missed seeds; otherwise TUNE.\nVERDICT: {verdict}\n")
    csv_path = pathlib.Path(args.csv or REPO / "results" / "processed" / f"pilot_selective_{args.tag}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["variant", "target", "flops_mean", "ratio_of_means", "paired_ratio", "steps_ratio", "misses", "n", "final_val"])
        for name, per in summary.items():
            for t, s in per.items():
                w.writerow([name, t, int(s["flops_mean"]), round(s["ratio_of_means"], 4), round(s["paired_ratio"], 4),
                            round(s["steps_ratio"], 4), s["misses"], s["n"], round(s["final_val"], 4)])
    print(f"wrote {csv_path}")
    return verdict


if __name__ == "__main__":
    main()
