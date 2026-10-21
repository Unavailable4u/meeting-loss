"""Per-step FLOP overhead of every arm relative to arm A (deterministic count, no data needed).

Usage: python scripts/flops_overhead.py [--batch-size 128] [--out results/processed/flops_overhead_v1.1.csv]
Counts one full training step (forward + backward, including G, heads, and extra forwards) with
torch.utils.flop_counter. Optimizer FLOPs are ignored. Settings that change the count:
sigma_inv > 0 adds an extra forward through the upper half and through G.
"""
import argparse
import csv
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src.flops import flops_per_step
from src.model import MeetNet

CFG = dict(lam_inv=1.0, lam_meet=0.1, warmup_steps=1, lam_aux=0.3, eta_hat=0.1,
           max_steps=100, lr=1e-3, wd=1e-4)
ROWS = [("base", {}), ("deepsup", {}), ("ae", {}), ("meet", {}), ("meet", {"sigma_inv": 0.1}),
        ("meet_ao", {}), ("meet_frozen", {}), ("meet_grad", {})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default="results/processed/flops_overhead_v1.1.csv")
    args = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    batch = (torch.randn(args.batch_size, 784, generator=g), torch.randint(0, 10, (args.batch_size,), generator=g))
    out, base = [], None
    for arm, extra in ROWS:
        cfg = {**CFG, **extra}
        model = MeetNet(arm=arm)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        f = flops_per_step(model, opt, batch, cfg)
        base = f if base is None else base
        name = arm + ("+sigma_inv" if extra else "")
        out.append(dict(arm=name, flops_per_step=f, overhead_vs_A_pct=round(100 * (f / base - 1), 2)))
        print(f"{name:18s} {f:>14,d}  {100 * (f / base - 1):+7.2f}%")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "flops_per_step", "overhead_vs_A_pct"], lineterminator="\n")
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
