"""Selective backward pass (arm "selective").

Forward the WHOLE batch (this also gives every example's loss), then run the backward pass only for the
highest-loss fraction of the batch. E-009 found that once training is under way a small share of the
examples carries nearly all of the gradient, so most of the backward work can be skipped.

The backward pass is written by hand for the plain MLP of src/model.py and reuses the stored forward
activations, so the saved FLOPs are real and visible to torch.utils.flop_counter. tests/test_selective.py
checks that it matches autograd exactly. The update is the full-batch gradient minus the omitted part:
the selected rows' gradient is divided by the FULL batch size B, not by the number selected.

Config keys: sel_frac (default 0.3), sel_warmup (steps with the full backward pass, default 0),
sel_mode ("top" = highest loss, or "random" = a random subset of the same size: the control).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _layers(model):
    """The four Linear layers and the activation kind of the plain MLP (low: L-act-L, up: act-L-act-L)."""
    lin = [model.low[0], model.low[2], model.up[1], model.up[3]]
    if not all(isinstance(m, nn.Linear) for m in lin):
        raise ValueError("selective backward is written for the plain MeetNet layout")
    act = model.low[1]
    if isinstance(act, nn.ReLU):
        return lin, "relu"
    if isinstance(act, nn.Tanh):
        return lin, "tanh"
    raise ValueError(f"unsupported activation {type(act).__name__}")


def _act(z, kind):
    return torch.relu(z) if kind == "relu" else torch.tanh(z)


def _dact(z, a, kind):
    return (z > 0).to(z.dtype) if kind == "relu" else 1.0 - a * a


def n_selected(batch_size, step, cfg):
    """Number of rows that go through the backward pass at this step."""
    frac = cfg.get("sel_frac", 0.3)
    if step < cfg.get("sel_warmup", 0) or frac >= 1.0:
        return batch_size
    return min(batch_size, max(1, math.ceil(frac * batch_size)))


def select_indices(losses, k, mode="top", generator=None):
    """Rows to back-propagate: the k highest-loss rows (top) or k random rows (random control)."""
    if mode == "top":
        return torch.topk(losses, k).indices
    if mode == "random":
        return torch.randperm(losses.shape[0], generator=generator)[:k]
    raise ValueError(f"unknown sel_mode {mode!r}")


def selective_step(model, x, y, cfg, step):
    """Forward the full batch, back-propagate only the selected rows, and write .grad on every parameter.
    Returns (mean loss over the full batch, number of rows back-propagated, selected indices or None)."""
    (l1, l2, l3, l4), kind = _layers(model)
    B = x.shape[0]
    with torch.no_grad():
        a0 = x
        z1 = F.linear(a0, l1.weight, l1.bias); a1 = _act(z1, kind)
        z2 = F.linear(a1, l2.weight, l2.bias); a2 = _act(z2, kind)
        z3 = F.linear(a2, l3.weight, l3.bias); a3 = _act(z3, kind)
        z4 = F.linear(a3, l4.weight, l4.bias)
        losses = F.cross_entropy(z4, y, reduction="none")

        k = n_selected(B, step, cfg)
        if k == B:
            idx = None
            take = lambda t: t
        else:
            idx = select_indices(losses, k, cfg.get("sel_mode", "top"))
            take = lambda t: t[idx]

        # backward, only for the selected rows; gradients are divided by the FULL batch size B
        p = torch.softmax(take(z4), dim=1)
        d4 = (p - F.one_hot(take(y), p.shape[1]).to(p.dtype)) / B
        l4.weight.grad = d4.t() @ take(a3);                 l4.bias.grad = d4.sum(0)
        d3 = (d4 @ l4.weight) * _dact(take(z3), take(a3), kind)
        l3.weight.grad = d3.t() @ take(a2);                 l3.bias.grad = d3.sum(0)
        d2 = (d3 @ l3.weight) * _dact(take(z2), take(a2), kind)
        l2.weight.grad = d2.t() @ take(a1);                 l2.bias.grad = d2.sum(0)
        d1 = (d2 @ l2.weight) * _dact(take(z1), take(a1), kind)
        l1.weight.grad = d1.t() @ take(a0);                 l1.bias.grad = d1.sum(0)
    return losses.mean(), k, idx
