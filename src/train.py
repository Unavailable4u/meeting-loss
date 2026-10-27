"""Training loop and CLI (method v1.1).

Usage:  python -m src.train --config configs/stage1_baseline.yaml [--seed 1]
Writes an append-only run log to results/raw/<run_id>.jsonl (first line = header with the
config, config hash and git commit). Never edit raw results by hand.
See configs/template_all_options.yaml for every config key.
"""
import argparse
import datetime
import hashlib
import json
import math
import pathlib
import subprocess
import time

import torch
import yaml

from src.crossing import DEFAULT_CROSSING, crossing_index
from src.diagnostics import grad_alignment, mid_state_variance
from src.flops import flops_per_step
from src.model import MEET_ARMS, MeetNet, compute_losses
from src.selective import OsgdSchedule, selective_step
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


def lr_factors(cfg):
    """Learning-rate factors as functions of the step, for (low, rest, aux) parameter groups.

    lr_schedule:  "constant" (default) or "cosine" (decays to zero over max_steps).
    aux_lr_decay: "none" (default) or "power_inv" (DSN reference code: (1 + 1e-4*step)^-0.75 on
                  the companion head, on top of the base schedule).
    """
    T = max(1, cfg.get("max_steps", 1))
    if cfg.get("lr_schedule", "constant") == "cosine":
        base = lambda s: 0.5 * (1.0 + math.cos(math.pi * min(1.0, s / T)))
    else:
        base = lambda s: 1.0
    if cfg.get("aux_lr_decay", "none") == "power_inv":
        aux = lambda s: base(s) * (1.0 + 1e-4 * s) ** -0.75
    else:
        aux = base
    return base, base, aux


def build_optimizer(model, cfg):
    """AdamW with parameter groups: lower layers (lr * low_lr_mult, control E), all other
    trainable parameters, and the companion head of arm B in its own group. With default
    settings this is mathematically identical to one AdamW over model.parameters()."""
    lr = cfg["lr"]
    low = list(model.low.parameters())
    aux = list(model.aux.parameters()) if hasattr(model, "aux") else []
    skip = {id(p) for p in low + aux}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in skip]
    f_low, f_rest, f_aux = lr_factors(cfg)
    entries = [(low, lr * cfg.get("low_lr_mult", 1.0), f_low), (rest, lr, f_rest)]
    if aux:
        entries.append((aux, lr, f_aux))
    entries = [e for e in entries if e[0]]             # drop empty groups
    opt = torch.optim.AdamW([dict(params=ps, lr=l) for ps, l, _ in entries],
                            lr=lr, weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[f for _, _, f in entries])
    return opt, sched


def run(cfg, device="cpu", data=None):
    """Train one run. `data` may be a (train_loader, val_loader) pair (used by tests);
    by default the fixed MNIST split from src/data.py (decision D-008) is loaded."""
    if cfg["arm"] == "sb":                          # Selective-Backprop has its own loop (src/sb.py, D-025)
        from src.sb import run_sb
        return run_sb(cfg, device, data)
    set_seed(cfg["seed"])
    if data is None:
        from src.data import get_data               # imported lazily: needs torchvision
        data = get_data(cfg["batch_size"], cfg["seed"])
    train_loader, val_loader = data
    model = MeetNet(arm=cfg["arm"], act=cfg.get("act", "relu")).to(device)
    opt, sched = build_optimizer(model, cfg)
    bx, by = next(iter(train_loader))
    fps = flops_per_step(model, opt, (bx.to(device), by.to(device)), cfg)
    selective = cfg["arm"] == "selective"
    warm = cfg.get("sel_warmup", 0) if selective else 0
    fps_warm = flops_per_step(model, opt, (bx.to(device), by.to(device)), cfg, step=0) if warm else fps
    dyn = selective and cfg.get("sel_schedule", "none") != "none"      # a schedule changes k, hence the cost per step
    osgd = OsgdSchedule() if dyn and cfg["sel_schedule"] == "osgd" else None
    cost_by_k, steps_per_epoch = {}, len(train_loader)

    log, step, t0, cum_flops = [], 0, time.time(), 0
    while step < cfg["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if selective:
                if dyn:
                    stats = {}
                    loss_t, k, _ = selective_step(model, x, y, cfg, step, frac=osgd.frac if osgd else None, stats=stats)
                    if osgd:
                        osgd.observe(stats)
                else:
                    loss_t, k, _ = selective_step(model, x, y, cfg, step)
                parts = {"task": loss_t, "k": torch.tensor(float(k))}
            else:
                loss, parts, _, _ = compute_losses(model, x, y, cfg, step)
                loss.backward()
            opt.step()
            sched.step()
            if dyn:
                if k not in cost_by_k:
                    cost_by_k[k] = flops_per_step(model, opt, (bx.to(device), by.to(device)), cfg, step=step, k=k)
                cum_flops += cost_by_k[k]
            else:
                cum_flops += fps_warm if step < warm else fps
            step += 1
            if osgd and step % steps_per_epoch == 0:
                osgd.end_epoch()
            if step % cfg["eval_every"] == 0:
                row = dict(step=step, flops=cum_flops, time=time.time() - t0,
                           val_acc=evaluate(model, val_loader, device),
                           **{k: v.item() for k, v in parts.items()})
                if cfg["arm"] in MEET_ARMS or cfg["arm"] in ("meet_grad", "meet_rho"):
                    row["align"] = grad_alignment(model, x, y, cfg)
                    row["h_var"] = mid_state_variance(model, x)
                log.append(row)
            if step >= cfg["max_steps"]:
                break
    return log


def steps_to_target(log, target, rule=DEFAULT_CROSSING):
    """(step, flops) at which validation accuracy reaches target under `rule` (src/crossing.py;
    default D-018: two consecutive evaluations at or above the target, reported at the first of them);
    None if never. A run that never reaches the target must be recorded, never dropped."""
    i = crossing_index([row["val_acc"] for row in log], target, rule)
    return None if i is None else (log[i]["step"], log[i]["flops"])


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
