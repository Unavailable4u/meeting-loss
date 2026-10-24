"""Exploratory check: how much of the gradient comes from the highest-loss examples?

Trains the tuned baseline (arm A) for 3,000 steps on several new seeds and, at checkpoints, measures
what share of the batch gradient the top-30% highest-loss examples carry (and whether they point the
same way as the full gradient). The decision rule was written before the run:
docs/notes/gradient-concentration-check.md.

Usage:
  python scripts/gradient_concentration.py --tag gc1                    # real run (MNIST, 3 seeds, about 15 minutes)
  python scripts/gradient_concentration.py --synthetic --max-steps 60 --checkpoints 20 40 60 --seeds 0 --tag smoke
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
from exploratory.gradconc import PRIMARY_FRACTION, RULE_STEPS, decide, run_with_checkpoints  # noqa: E402

COLUMNS = ["seed", "step", "mode", "fraction", "norm_share", "proj_share", "cos", "omitted", "loss_share",
           "val_acc", "pool_loss", "n_batches"]


def synthetic_loaders(batch_size, seed, n=6000):
    """Plumbing test only: 10 Gaussian clusters in 784 dimensions."""
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


def summarize(rows, checkpoints, fractions):
    def mean_over_seeds(mode, f, step, key):
        v = [r[key] for r in rows if r["mode"] == mode and r["fraction"] == f and r["step"] == step]
        return (float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)) if v else (float("nan"),) * 3 + (0,)

    f0 = PRIMARY_FRACTION
    print(f"\n=== Top {int(f0 * 100)}% highest-loss examples (mean over seeds; [min, max]) vs a random {int(f0 * 100)}% ===")
    print("step   val_acc | norm_share      | cos(top, full)  | proj_share | omitted | random: norm_share")
    ns, cs = {}, {}
    for s in checkpoints:
        n = mean_over_seeds("top", f0, s, "norm_share")
        c = mean_over_seeds("top", f0, s, "cos")
        p = mean_over_seeds("top", f0, s, "proj_share")
        o = mean_over_seeds("top", f0, s, "omitted")
        a = mean_over_seeds("top", f0, s, "val_acc")
        r = mean_over_seeds("random", f0, s, "norm_share")
        ns[s], cs[s] = n[0], c[0]
        print(f"{s:5d}  {a[0]:.4f} | {n[0]:.3f} [{n[1]:.2f},{n[2]:.2f}] | {c[0]:.3f} [{c[1]:.2f},{c[2]:.2f}] |   {p[0]:.3f}    |  {o[0]:.3f}  | {r[0]:.3f}")
    print("\n=== Share of gradient norm vs fraction of examples kept (top by loss), per checkpoint ===")
    print("step  " + "  ".join(f"top{int(f * 100):>3d}%" for f in fractions))
    for s in checkpoints:
        print(f"{s:5d} " + "  ".join(f"{mean_over_seeds('top', f, s, 'norm_share')[0]:7.3f}" for f in fractions))
    verdict = decide(ns, cs)
    print(f"\nPre-registered rule (steps {RULE_STEPS}): PROMISING if norm_share >= 0.80 and cos >= 0.90 at all three; "
          f"DROP if norm_share < 0.60 at step 1000.\nVERDICT: {verdict}\n")
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuned/t1_base_two.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[500, 501, 502])
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[100, 200, 500, 1000, 1500, 2000, 3000])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--batches", type=int, default=16, help="measurement batches per checkpoint")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tag", default="gc1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true", help="plumbing test on synthetic data (no MNIST)")
    args = ap.parse_args(argv)

    cfg0 = yaml.safe_load(open(args.config))
    if args.max_steps:
        cfg0["max_steps"] = args.max_steps
    cfg0["device"] = "cpu" if cfg0.get("device", "auto") == "auto" else cfg0["device"]
    out = pathlib.Path(args.out or f"results/processed/gradient_concentration_{args.tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.synthetic:
        from src.data import get_data

    rows = []
    for seed in args.seeds:
        cfg = {**cfg0, "seed": seed}
        bs = cfg["batch_size"]
        loader_fn = synthetic_loaders if args.synthetic else get_data
        pool_loader, _ = loader_fn(bs, 10_000 + seed)                  # separate shuffling stream for the measurement pool
        pool = [(x.clone(), y.clone()) for x, y in itertools.islice(iter(pool_loader), args.batches)]
        train_loader, val_loader = loader_fn(bs, seed)
        print(f"seed {seed}: training {cfg['max_steps']} steps, measuring at {args.checkpoints} "
              f"on {len(pool)} fixed batches of {bs} training examples", flush=True)
        r, _ = run_with_checkpoints(cfg, train_loader, val_loader, pool, args.checkpoints, args.fractions, cfg["device"])
        rows.extend(r)

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})
    print(f"wrote {out}")
    return summarize(rows, [c for c in args.checkpoints if c <= cfg0["max_steps"]], args.fractions)


if __name__ == "__main__":
    main()
