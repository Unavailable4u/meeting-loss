"""Training loop and CLI. UNTESTED starter code (docs/RISKS.md R-10).

Usage:  python -m src.train --config configs/stage1_baseline.yaml [--seed 1]
Writes an append-only run log to results/raw/<run_id>.jsonl (first line = header with the
config, config hash and git commit). Never edit raw results by hand.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import time

import torch
import yaml

from src.data import get_data
from src.diagnostics import grad_alignment, mid_state_variance
from src.flops import flops_per_step
from src.model import MeetNet, compute_losses
from src.utils import set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(-1).cpu()
        correct += (pred == y).sum().item()
        n += y.numel()
    model.train()
    return correct / n


def run(cfg, device="cpu"):
    set_seed(cfg["seed"])
    train_loader, val_loader = get_data(cfg["batch_size"], cfg["seed"])
    model = MeetNet(arm=cfg["arm"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    bx, by = next(iter(train_loader))
    fps = flops_per_step(model, opt, (bx.to(device), by.to(device)), cfg)

    log, step, t0 = [], 0, time.time()
    while step < cfg["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss, parts, _, _ = compute_losses(model, x, y, cfg, step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % cfg["eval_every"] == 0:
                row = dict(step=step, flops=step * fps, time=time.time() - t0,
                           val_acc=evaluate(model, val_loader, device),
                           **{k: v.item() for k, v in parts.items()})
                if cfg["arm"] == "meet":
                    row["align"] = grad_alignment(model, x, y)
                    row["h_var"] = mid_state_variance(model, x)
                log.append(row)
            if step >= cfg["max_steps"]:
                break
    return log


def steps_to_target(log, target):
    """First (step, flops) at which validation accuracy reaches target; None if never.
    A run that never reaches the target must be recorded, never dropped."""
    for row in log:
        if row["val_acc"] >= target:
            return row["step"], row["flops"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    run_id = f"{datetime.date.today():%Y%m%d}_{cfg['stage']}_{cfg['arm']}_s{cfg['seed']}"

    log = run(cfg, device)

    out = pathlib.Path("results/raw") / f"{run_id}.jsonl"
    with open(out, "x") as f:                         # "x": never overwrite a raw result
        f.write(json.dumps({"_header": True, "run_id": run_id, "config": cfg,
                            "config_hash": cfg_hash, "git_commit": commit}) + "\n")
        for row in log:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
