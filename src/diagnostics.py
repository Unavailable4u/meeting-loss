"""Diagnostics for H3 (gradient alignment) and for collapse. Uses the same code path as training."""
import torch
import torch.nn.functional as F

from src.model import compute_losses

_DEFAULT_CFG = dict(lam_inv=1.0, lam_meet=0.1, warmup_steps=1, lam_aux=0.0)


def grad_alignment(model, x, y, cfg=None):
    """Cosine similarity between the task gradient and the meeting-loss gradient on the lower
    layers. Returns NaN for arms without a meeting loss."""
    if not hasattr(model, "G") and model.arm != "meet_grad":
        return float("nan")
    cfg = {**_DEFAULT_CFG, **(cfg or {})}
    params = list(model.low.parameters())
    _, parts, _, _ = compute_losses(model, x, y, cfg, step=10**9)
    g_task = torch.autograd.grad(parts["task"], params, retain_graph=True)
    g_meet = torch.autograd.grad(parts["meet"], params, retain_graph=True)
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
    return F.cosine_similarity(flat(g_task), flat(g_meet), dim=0).item()


@torch.no_grad()
def mid_state_variance(model, x):
    """Mean per-feature variance of h across the batch (collapse indicator)."""
    return model.low(x).var(dim=0).mean().item()
