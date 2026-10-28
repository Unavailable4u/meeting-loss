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

Optional keys added for the Improve track (D-024, D-025; every default leaves the behaviour above unchanged):
sel_score ("loss" = rank rows by their loss, default; "gradnorm" = rank by the norm of the last-layer gradient,
norm(softmax - one_hot), the bound of Katharopoulos and Fleuret 2018, free in the same forward pass);
sel_schedule ("none" default; "osgd" = the kept fraction follows the default rule of Ordered SGD, Kawaguchi and Lu 2020:
1, 1/2, 1/4, 1/8, 1/16 of the batch once the running training accuracy of the last epoch passes 80, 90, 95, 99.5 percent;
"linear" = from sel_frac_start to sel_frac_end over the steps after the warm-up);
sel_scale ("B" = divide the summed gradient of the kept rows by the batch size, i.e. the full-batch gradient minus the part
of the omitted rows, default; "k" = divide by the number of kept rows, the mean over the kept rows as in Ordered SGD; the two
are equal whenever all rows are kept; added for look 2 of the Improve track, DR-015).
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


def linear_frac(step, cfg):
    """Kept fraction of the linear schedule: sel_frac_start at the end of the warm-up, sel_frac_end at max_steps."""
    warm = cfg.get("sel_warmup", 0)
    span = max(1, cfg["max_steps"] - warm)
    t = min(1.0, max(0.0, (step - warm) / span))
    return cfg["sel_frac_start"] + (cfg["sel_frac_end"] - cfg["sel_frac_start"]) * t


def n_selected(batch_size, step, cfg, frac=None):
    """Number of rows that go through the backward pass at this step. `frac` (a schedule's current value) overrides
    sel_frac; with sel_schedule "linear" the fraction is computed from the step."""
    if frac is None:
        if cfg.get("sel_schedule", "none") == "linear":
            frac = linear_frac(step, cfg)
        else:
            frac = cfg.get("sel_frac", 0.3)
    if step < cfg.get("sel_warmup", 0) or frac >= 1.0:
        return batch_size
    return min(batch_size, max(1, math.ceil(frac * batch_size)))


OSGD_RULE = ((0.995, 16), (0.95, 8), (0.90, 4), (0.80, 2))     # (training accuracy at least, divisor of the batch size)


def osgd_divisor(train_acc):
    """Ordered SGD's default rule for q (Kawaguchi and Lu 2020): q = s, s/2, s/4, s/8, s/16."""
    for thr, div in OSGD_RULE:
        if train_acc >= thr:
            return div
    return 1


class OsgdSchedule:
    """Kept fraction 1/divisor, updated at the end of every epoch from the running training accuracy of that epoch
    (the accuracy of the forward predictions on the batches seen, an approximation of the paper's training accuracy)."""

    def __init__(self):
        self.div, self.correct, self.n = 1, 0, 0

    @property
    def frac(self):
        return 1.0 / self.div

    def observe(self, stats):
        self.correct += stats["correct"]
        self.n += stats["n"]

    def end_epoch(self):
        if self.n:
            self.div = osgd_divisor(self.correct / self.n)
        self.correct = self.n = 0


def select_indices(losses, k, mode="top", generator=None):
    """Rows to back-propagate: the k highest-loss rows (top) or k random rows (random control)."""
    if mode == "top":
        return torch.topk(losses, k).indices
    if mode == "random":
        return torch.randperm(losses.shape[0], generator=generator)[:k]
    raise ValueError(f"unknown sel_mode {mode!r}")


def selective_step(model, x, y, cfg, step, k=None, frac=None, stats=None):
    """Forward the full batch, back-propagate only the selected rows, and write .grad on every parameter.
    Returns (mean loss over the full batch, number of rows back-propagated, selected indices or None).
    Optional: k (number of rows, overrides everything), frac (current kept fraction of a schedule), stats (a dict that
    receives the number of correct predictions and rows of the full batch)."""
    (l1, l2, l3, l4), kind = _layers(model)
    B = x.shape[0]
    with torch.no_grad():
        a0 = x
        z1 = F.linear(a0, l1.weight, l1.bias); a1 = _act(z1, kind)
        z2 = F.linear(a1, l2.weight, l2.bias); a2 = _act(z2, kind)
        z3 = F.linear(a2, l3.weight, l3.bias); a3 = _act(z3, kind)
        z4 = F.linear(a3, l4.weight, l4.bias)
        losses = F.cross_entropy(z4, y, reduction="none")

        if stats is not None:
            stats["correct"] = int((z4.argmax(1) == y).sum())
            stats["n"] = B
        if k is None:
            k = n_selected(B, step, cfg, frac)
        k = min(k, B)
        if k == B:
            idx = None
            take = lambda t: t
        else:
            score = losses
            if cfg.get("sel_score", "loss") == "gradnorm":
                score = (torch.softmax(z4, dim=1) - F.one_hot(y, z4.shape[1]).to(z4.dtype)).norm(dim=1)
            idx = select_indices(score, k, cfg.get("sel_mode", "top"))
            take = lambda t: t[idx]

        # backward, only for the selected rows; gradients are divided by the FULL batch size B (sel_scale "B", the
        # default) or by the number of kept rows k (sel_scale "k")
        p = torch.softmax(take(z4), dim=1)
        d4 = (p - F.one_hot(take(y), p.shape[1]).to(p.dtype)) / (k if cfg.get("sel_scale", "B") == "k" else B)
        l4.weight.grad = d4.t() @ take(a3);                 l4.bias.grad = d4.sum(0)
        d3 = (d4 @ l4.weight) * _dact(take(z3), take(a3), kind)
        l3.weight.grad = d3.t() @ take(a2);                 l3.bias.grad = d3.sum(0)
        d2 = (d3 @ l3.weight) * _dact(take(z2), take(a2), kind)
        l2.weight.grad = d2.t() @ take(a1);                 l2.bias.grad = d2.sum(0)
        d1 = (d2 @ l2.weight) * _dact(take(z1), take(a1), kind)
        l1.weight.grad = d1.t() @ take(a0);                 l1.bias.grad = d1.sum(0)
    return losses.mean(), k, idx
