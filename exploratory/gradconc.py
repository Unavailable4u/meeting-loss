"""Gradient concentration: how much of the gradient comes from the highest-loss examples?

Exploratory check described in docs/notes/gradient-concentration-check.md (decision rule written
BEFORE the run). Per-example gradients are computed with torch.func; nothing here touches the
global random state, so measuring does not change the training trajectory.
"""
import math

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from src.model import MeetNet, compute_losses
from src.train import build_optimizer, evaluate
from src.utils import set_seed

# ---- pre-registered decision rule (docs/notes/gradient-concentration-check.md) ----
PRIMARY_FRACTION = 0.3
RULE_STEPS = (500, 1000, 2000)
PROMISING_NORM_SHARE = 0.80
PROMISING_COS = 0.90
DROP_NORM_SHARE = 0.60          # at step 1000


def per_example_grads(model, x, y):
    """Flattened per-example gradients of the cross-entropy loss, shape [B, P], and per-example losses [B]."""
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_fn(p, xi, yi):
        logits = functional_call(model, (p, buffers), (xi.unsqueeze(0),))
        return F.cross_entropy(logits, yi.unsqueeze(0))

    g = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, x, y)
    flat = torch.cat([g[k].reshape(x.shape[0], -1) for k in params], dim=1)
    with torch.no_grad():
        losses = F.cross_entropy(model(x), y, reduction="none")
    return flat, losses


def subset_metrics(flat, losses, idx):
    """How well does the subset `idx` represent the full batch gradient (sum over examples)?

    norm_share  sum of per-example gradient norms in the subset / sum over the batch
    proj_share  <subset gradient, full gradient> / ||full gradient||^2  (share of the update kept)
    cos         cosine between the subset gradient and the full gradient (direction fidelity)
    omitted     ||gradient of the examples left out|| / ||full gradient||
    loss_share  share of the total loss held by the subset
    """
    norms = flat.norm(dim=1)
    G = flat.sum(0)
    gS = flat[idx].sum(0)
    eps = 1e-30
    return dict(
        norm_share=(norms[idx].sum() / (norms.sum() + eps)).item(),
        proj_share=((gS @ G) / (G @ G + eps)).item(),
        cos=F.cosine_similarity(gS, G, dim=0, eps=1e-12).item(),
        omitted=((G - gS).norm() / (G.norm() + eps)).item(),
        loss_share=(losses[idx].sum() / (losses.sum() + eps)).item(),
    )


def concentration(flat, losses, fractions, gen):
    """Metrics for the top-loss subset and for a random subset of the same size, per fraction."""
    B = flat.shape[0]
    order = torch.argsort(losses, descending=True)
    rand = torch.randperm(B, generator=gen)
    out = {}
    for f in fractions:
        k = min(B, max(1, math.ceil(f * B)))
        out[("top", f)] = subset_metrics(flat, losses, order[:k])
        out[("random", f)] = subset_metrics(flat, losses, rand[:k])
    return out


def measure_pool(model, pool, fractions, gen, device="cpu"):
    """Average the metrics over a fixed pool of training batches. Returns {(mode, fraction): {metric: mean}} and mean loss."""
    acc, mean_loss = {}, 0.0
    for x, y in pool:
        flat, losses = per_example_grads(model, x.to(device), y.to(device))
        mean_loss += losses.mean().item() / len(pool)
        for key, m in concentration(flat, losses, fractions, gen).items():
            d = acc.setdefault(key, {k: 0.0 for k in m})
            for k, v in m.items():
                d[k] += v / len(pool)
    return acc, mean_loss


def run_with_checkpoints(cfg, train_loader, val_loader, pool, checkpoints, fractions, device="cpu", measure=True):
    """Train arm A (plain network) and measure gradient concentration at the checkpoints.
    Returns (rows, model). Measuring uses its own generator and never changes the training."""
    if cfg["arm"] != "base":
        raise ValueError("gradient concentration is measured on the plain baseline (arm base)")
    set_seed(cfg["seed"])
    model = MeetNet(arm=cfg["arm"], act=cfg.get("act", "relu")).to(device)
    opt, sched = build_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(12345 + cfg["seed"])
    cps = set(checkpoints)
    rows, step = [], 0
    while step < cfg["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss, _, _, _ = compute_losses(model, x, y, cfg, step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if measure and step in cps:
                res, mean_loss = measure_pool(model, pool, fractions, gen, device)
                acc = evaluate(model, val_loader, device)
                for (mode, f), m in res.items():
                    rows.append(dict(seed=cfg["seed"], step=step, mode=mode, fraction=f, val_acc=acc,
                                     pool_loss=mean_loss, n_batches=len(pool), **m))
            if step >= cfg["max_steps"]:
                break
    return rows, model


def decide(norm_share_by_step, cos_by_step):
    """Apply the pre-registered rule to {step: mean metric at the primary fraction}.
    PROMISING  norm_share >= 0.80 and cos >= 0.90 at steps 500, 1000 and 2000
    DROP       norm_share < 0.60 at step 1000
    otherwise  INCONCLUSIVE"""
    if all(norm_share_by_step.get(s, 0) >= PROMISING_NORM_SHARE and cos_by_step.get(s, 0) >= PROMISING_COS
           for s in RULE_STEPS):
        return "PROMISING"
    if norm_share_by_step.get(1000, 1.0) < DROP_NORM_SHARE:
        return "DROP"
    return "INCONCLUSIVE"
