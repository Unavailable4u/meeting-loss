"""Exploratory checks A (direction geometry) and L (linking measurement) of the meeting signal.

Trains the tuned arm D (meeting loss) and the tuned arm A (plain baseline) on the same new seeds and, at
checkpoints, measures (one training run serves both checks):
  check A  parallel/orthogonal split of the meeting gradient, one-step probe, and the cosines between the
           directions backprop / meeting loss / perfect local inverse (= Gauss-Newton) / damped logit-space inverse
  check L  how much of the task gradient and of the meeting gradient lives in the top-30% rows by loss
Decision rules were written before any run: docs/notes/mechanism-a-direction-geometry.md and
docs/notes/linking-measurement-concentration.md. Exploratory (D-015).

Threads: the Stage 3 runs used one torch thread (E-011); --threads defaults to 1.

Usage:
  python scripts/mechanism_check.py --tag ma1          # real run (MNIST, 3 seeds); time the smoke run first
  python scripts/mechanism_check.py --synthetic --max-steps 40 --checkpoints 10 20 40 --seeds 0 --batches 2 --probe-batches 1 --tag smoke
"""
import argparse
import csv
import itertools
import pathlib
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere
from exploratory.mechanism import (PAIRS, RULE_STEPS, decide_link, decide_q1, decide_q2, decide_q3,  # noqa: E402
                                   run_measured)

COLUMNS = ["seed", "arm", "step", "quantity", "value"]


def synthetic_loaders(batch_size, seed, n=6000):
    """Plumbing test only: 10 Gaussian clusters in 784 dimensions (same as scripts/gradient_concentration.py)."""
    g = torch.Generator().manual_seed(0)
    means = torch.randn(10, 784, generator=g) * 0.6

    def make(n_, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.randint(0, 10, (n_,), generator=gg)
        return means[y] + torch.randn(n_, 784, generator=gg) * 3.0, y

    x, y = make(n, 1)
    xv, yv = make(1000, 2)
    train = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed), drop_last=True)
    return train, DataLoader(TensorDataset(xv, yv), batch_size=500)


def mean_by_step(rows, arm, quantity):
    """{step: mean over seeds} of one quantity for one arm."""
    out = {}
    for _, a, step, q, v in rows:
        if a == arm and q == quantity:
            out.setdefault(step, []).append(v)
    return {s: float(np.mean(v)) for s, v in out.items()}


def summarize(rows, checkpoints):
    m = lambda arm, q: mean_by_step(rows, arm, q)  # noqa: E731
    f = lambda d, s: f"{d.get(s, float('nan')):8.3f}"  # noqa: E731

    print("\n=== Check A, parameter space (arm D, mean over seeds) ===")
    print("step   val_acc | cos(task,meet) par_share perp/task meet/task |  dL(task)   dL(perp)  |dL perp|/|dL task|")
    cos, par, perp, mr = m("meet", "cos_task_meet"), m("meet", "par_share"), m("meet", "perp_ratio"), m("meet", "meet_ratio")
    dlt, dlp, acc = m("meet", "dl_task"), m("meet", "dl_perp"), m("meet", "val_acc")
    for s in checkpoints:
        ratio = abs(dlp.get(s, float("nan"))) / (abs(dlt.get(s, float("nan"))) + 1e-30)
        print(f"{s:5d} {f(acc, s)} | {f(cos, s)} {f(par, s)} {f(perp, s)} {f(mr, s)} | {dlt.get(s, float('nan')):10.2e} "
              f"{dlp.get(s, float('nan')):10.2e}  {ratio:8.3f}")

    print("\n=== Check A, h-space cosines (arm D; weighted by the backprop norm) ===")
    names = [f"cos_{a}_{b}_w" for a, b in PAIRS]
    print("step  " + " ".join(f"{n[4:-2]:>11s}" for n in names) + "  norm_ratio skipped")
    series = {n: m("meet", n) for n in names}
    nr, sk = m("meet", "norm_ratio"), m("meet", "skipped_share")
    for s in checkpoints:
        print(f"{s:5d} " + " ".join(f"{f(series[n], s):>11s}" for n in names) + f" {f(nr, s)} {f(sk, s)}")

    print("\n=== Check L, concentration in the top-30% rows by loss ===")
    print("step | task share A  task share D | meet share  meet cos  overlap(unsel) | corr(loss,|u_meet|) corr(loss,|g_h|) | confident rows (D)")
    tsa, tsd = m("base", "task_norm_share"), m("meet", "task_norm_share")
    ms, mc, ov = m("meet", "meet_norm_share"), m("meet", "meet_cos"), m("meet", "overlap_cos_unselected")
    cu, cg, cf = m("meet", "corr_loss_umeet"), m("meet", "corr_loss_gbp"), m("meet", "confident_share")
    for s in checkpoints:
        print(f"{s:5d} | {f(tsa, s)}  {f(tsd, s)} | {f(ms, s)} {f(mc, s)} {f(ov, s)} | {f(cu, s)}  {f(cg, s)} | {f(cf, s)}")

    v = {
        "Q1": decide_q1(cos, dlp, dlt),
        "Q2": decide_q2(m("meet", "cos_meet_pinv_w"), m("meet", "cos_meet_bp_w")),
        "Q2_lsq": decide_q2(m("meet", "cos_meet_lsq_w"), m("meet", "cos_meet_bp_w")),
        "Q3": decide_q3(m("meet", "cos_bp_pinv_w")),
        "L": decide_link(ms, cu),
    }
    print(f"\nPre-registered rules (steps {RULE_STEPS}; Q2 uses steps 500 and 1000):")
    print(f"  Q1 redundancy of the signal in parameter space : {v['Q1']}")
    print(f"  Q2 does G approximate the curvature direction? : {v['Q2']}   (descriptive, with the lsq direction: {v['Q2_lsq']})")
    print(f"  Q3 does curvature matter for this network?     : {v['Q3']}")
    print(f"  L  is the meeting gradient already selective?  : {v['L']}\n")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-d", default="configs/tuned/t1_meet_two.yaml")
    ap.add_argument("--config-a", default="configs/tuned/t1_base_two.yaml")
    ap.add_argument("--arms", nargs="+", default=["meet", "base"], choices=["meet", "base"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[600, 601, 602])
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[100, 200, 500, 1000, 1500, 2000, 3000])
    ap.add_argument("--batches", type=int, default=16, help="measurement batches per checkpoint")
    ap.add_argument("--probe-batches", type=int, default=8, help="batches (disjoint from the pool) for the one-step probe")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1, help="torch threads; 1 matches the Stage 3 runs (E-011)")
    ap.add_argument("--tag", default="ma1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true", help="plumbing test on synthetic data (no MNIST)")
    ap.add_argument("--with-linking", action="store_true", help="accepted for the command in the plan note; check L is always measured")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    print(f"torch threads = {torch.get_num_threads()}", flush=True)
    cfgs = {"meet": yaml.safe_load(open(args.config_d)), "base": yaml.safe_load(open(args.config_a))}
    out = pathlib.Path(args.out or f"results/processed/mechanism_a_{args.tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.synthetic:
        from src.data import get_data
    loader_fn = synthetic_loaders if args.synthetic else get_data

    rows = []
    for seed in args.seeds:
        for arm in args.arms:
            cfg = {**cfgs[arm], "seed": seed, "device": "cpu"}
            if args.max_steps:
                cfg["max_steps"] = args.max_steps
            bs = cfg["batch_size"]
            # a separate shuffling stream for the measurement pool and the probe; one epoch is a permutation,
            # so the pool batches and the probe batches are disjoint
            batches = list(itertools.islice(iter(loader_fn(bs, 10_000 + seed)[0]), args.batches + args.probe_batches))
            pool = [(x.clone(), y.clone()) for x, y in batches[:args.batches]]
            probe = (torch.cat([x for x, _ in batches[args.batches:]]), torch.cat([y for _, y in batches[args.batches:]]))
            train_loader, val_loader = loader_fn(bs, seed)
            print(f"seed {seed} arm {arm}: {cfg['max_steps']} steps, measuring at {args.checkpoints} on {len(pool)} fixed "
                  f"batches of {bs} (probe {probe[0].shape[0]} examples)", flush=True)
            r, _ = run_measured(cfg, train_loader, val_loader, pool, probe, args.checkpoints, "cpu")
            rows.extend(r)

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"wrote {out}")
    return summarize(rows, [c for c in args.checkpoints if c <= (args.max_steps or max(c for c in args.checkpoints))])


if __name__ == "__main__":
    main()
