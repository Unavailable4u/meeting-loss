"""Look 1 of track S: does the FLOPs saving of the selective backward pass translate into wall-clock time?

Times the training step of arm A (plain) and of S30 (selective backward on the highest-loss 30% of each batch, with
arm A's tuned hyperparameters) on the same machine and device: the full step (zero_grad, forward, backward, optimizer,
scheduler), including the top-k selection and the row gather of S30, on batches prefetched to the device (data loading is
not timed). Per variant: a warm-up of 200 steps (which also covers S30's 100 full-backward steps), then 1,000 timed
steps, 3 repeats, the variants alternating in order, median seconds per step. Descriptive (D-015); the rule (the ratio S30 / A
at most 0.85: translates; below 1.00: partial; otherwise: does not translate) is in docs/notes/selective-backward-prereg-prep.md.
The verdict is the one-thread reading (the project convention, E-011); torch's default thread count is reported as a
second, descriptive reading.

  python scripts/time_selective.py --tag sw1                        # real run (MNIST), a few minutes
  python scripts/time_selective.py --tag smoke --synthetic --warmup 20 --timed 50 --repeats 1 --csv /tmp/t.csv
"""
import argparse
import csv
import itertools
import pathlib
import platform
import sys
import time

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from exploratory.looks import s30_config, summarize_timing  # noqa: E402

N_BATCHES = 100          # prefetched batches, cycled through


def synthetic_loaders(batch_size, seed, n=6000):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    means = torch.randn(10, 784, generator=g) * 0.6
    gg = torch.Generator().manual_seed(1)
    y = torch.randint(0, 10, (n,), generator=gg)
    x = means[y] + torch.randn(n, 784, generator=gg) * 3.0
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True,
                      generator=torch.Generator().manual_seed(seed)), None


def make_step(cfg, device):
    """A function step(x, y) doing one full training step exactly as src.train.run does."""
    from src.model import MeetNet, compute_losses
    from src.selective import selective_step
    from src.train import build_optimizer
    from src.utils import set_seed
    set_seed(cfg["seed"])
    model = MeetNet(arm=cfg["arm"], act=cfg.get("act", "relu")).to(device)
    opt, sched = build_optimizer(model, cfg)
    selective = cfg["arm"] == "selective"
    state = {"step": 0}

    def step(x, y):
        opt.zero_grad(set_to_none=True)
        if selective:
            selective_step(model, x, y, cfg, state["step"])
        else:
            loss, _, _, _ = compute_losses(model, x, y, cfg, state["step"])
            loss.backward()
        opt.step()
        sched.step()
        state["step"] += 1

    return step, model, opt


def sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def seconds_per_step(cfg, device, batches, warmup, timed):
    step, _, _ = make_step(cfg, device)
    n = len(batches)
    for i in range(warmup):
        step(*batches[i % n])
    sync(device)
    t0 = time.perf_counter()
    for i in range(timed):
        step(*batches[(warmup + i) % n])
    sync(device)
    return (time.perf_counter() - t0) / timed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/tuned/t1_base_two.yaml")
    ap.add_argument("--seed", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--timed", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 0],
                    help="torch thread counts to time; 0 = torch's default. The first one gives the verdict (default 1).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="sw1")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--synthetic", action="store_true", help="plumbing test on synthetic data (no MNIST)")
    args = ap.parse_args(argv)

    import torch
    default_threads = torch.get_num_threads()
    cfg_path = pathlib.Path(args.config)
    base = yaml.safe_load(open(cfg_path if cfg_path.exists() else REPO / args.config))
    base["seed"] = args.seed
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfgs = {"A": base, "S30": s30_config(base, base["stage"])}
    if args.synthetic:
        loader = synthetic_loaders(base["batch_size"], args.seed)[0]
    else:
        from src.data import get_data
        loader = get_data(base["batch_size"], args.seed)[0]
    batches = [(x.to(device), y.to(device)) for x, y in itertools.islice(iter(loader), N_BATCHES)]

    print(f"torch {torch.__version__}, device {device}, python {platform.python_version()}, {platform.platform()}, "
          f"torch default threads {default_threads}\nbatches of {base['batch_size']}, prefetched {len(batches)}; "
          f"{args.warmup} warm-up + {args.timed} timed steps, {args.repeats} repeats", flush=True)
    rows, results = [], {}
    for t in args.threads:
        threads = default_threads if t == 0 else t
        torch.set_num_threads(threads)
        times = {"A": [], "S30": []}
        for r in range(args.repeats):
            order = ["A", "S30"] if r % 2 == 0 else ["S30", "A"]        # alternate to cancel drift
            for name in order:
                s = seconds_per_step(cfgs[name], device, batches, args.warmup, args.timed)
                times[name].append(s)
                rows.append([device, threads, r, name, 1000.0 * s])
                print(f"threads {threads:2d} repeat {r} {name:3s}: {1000.0 * s:7.3f} ms/step", flush=True)
        results[t] = summarize_timing(times)
        res = results[t]
        print(f"threads {threads}: median A {res['median_ms']['A']:.3f} ms, S30 {res['median_ms']['S30']:.3f} ms, "
              f"ratio {res['ratio']:.3f} -> {res['verdict']}", flush=True)

    first = args.threads[0]
    print(f"\nDevice: {device}. A result on this device does not transfer to another one.")
    print("Pre-registered rule: ratio S30 / A at most 0.85: FLOPS SAVING TRANSLATES; below 1.00: PARTIAL; otherwise: DOES NOT TRANSLATE.")
    print(f"VERDICT (threads = {default_threads if first == 0 else first}): {results[first]['verdict']}\n")
    out = pathlib.Path(args.csv or REPO / "results" / "processed" / f"timing_{args.tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["device", "threads", "repeat", "variant", "ms_per_step"])
        w.writerows(rows)
    print(f"wrote {out}")
    return results[first]["verdict"]


if __name__ == "__main__":
    main()
