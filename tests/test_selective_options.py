"""Tests for the options added to the selective backward pass for the Improve track (D-024, D-025):
sel_score "gradnorm", the "osgd" and "linear" kept-fraction schedules, and the optional k, frac and stats arguments."""
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.model import MeetNet
from src.selective import (OsgdSchedule, linear_frac, n_selected, osgd_divisor, selective_step)
from src.train import run

B = 32
FWD_PER_ROW, BWD_PER_ROW = 537_600, 673_792
CFG = dict(stage="t", arm="selective", seed=0, act="relu", lr=1e-3, wd=1e-4, batch_size=B, max_steps=40, eval_every=10,
           lr_schedule="cosine", sel_frac=0.3, sel_warmup=0, sel_mode="top")


def _batch(n=B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 784, generator=g), torch.randint(0, 10, (n,), generator=g)


def _loaders(sep=0.6, noise=3.0, seed=0):
    g = torch.Generator().manual_seed(7)
    means = torch.randn(10, 784, generator=g) * sep
    def make(n, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.randint(0, 10, (n,), generator=gg)
        return means[y] + torch.randn(n, 784, generator=gg) * noise, y
    x, y = make(1024, 1); xv, yv = make(256, 2)
    return (DataLoader(TensorDataset(x, y), batch_size=B, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(xv, yv), batch_size=128))


def test_gradnorm_score_selects_the_largest_last_layer_gradient_and_matches_autograd():
    torch.manual_seed(0)
    m = MeetNet(arm="selective")
    x, y = _batch()
    with torch.no_grad():
        z = m(x)
        norm = (torch.softmax(z, 1) - F.one_hot(y, 10).float()).norm(dim=1)
    cfg = {**CFG, "sel_score": "gradnorm"}
    m.zero_grad(set_to_none=True)
    _, k, idx = selective_step(m, x, y, cfg, step=10**9)
    assert k == math.ceil(0.3 * B) and set(idx.tolist()) == set(torch.topk(norm, k).indices.tolist())
    got = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(set_to_none=True)
    (F.cross_entropy(m(x), y, reduction="none")[idx].sum() / B).backward()
    for a, b in zip(got, [p.grad for p in m.parameters()]):
        assert torch.allclose(a, b, atol=1e-6)


def test_default_score_is_the_loss():
    torch.manual_seed(0)
    m = MeetNet(arm="selective")
    x, y = _batch()
    _, k, idx_default = selective_step(m, x, y, CFG, step=10**9)
    _, _, idx_loss = selective_step(m, x, y, {**CFG, "sel_score": "loss"}, step=10**9)
    assert torch.equal(idx_default, idx_loss)


def test_k_override_frac_override_and_stats():
    torch.manual_seed(0)
    m = MeetNet(arm="selective")
    x, y = _batch()
    stats = {}
    _, k, idx = selective_step(m, x, y, CFG, step=10**9, k=5, stats=stats)
    assert k == 5 and len(idx) == 5
    assert stats["n"] == B and stats["correct"] == int((m(x).argmax(1) == y).sum())
    assert selective_step(m, x, y, CFG, step=10**9, frac=0.5)[1] == B // 2
    assert selective_step(m, x, y, CFG, step=10**9, frac=1.0)[2] is None            # full backward, no selection
    assert selective_step(m, x, y, {**CFG, "sel_warmup": 100}, step=3, k=5)[1] == 5   # k wins over the warm-up


def test_osgd_divisor_follows_the_published_rule():
    cases = [(0.0, 1), (0.79, 1), (0.80, 2), (0.89, 2), (0.90, 4), (0.949, 4), (0.95, 8), (0.994, 8), (0.995, 16), (1.0, 16)]
    for acc, div in cases:
        assert osgd_divisor(acc) == div, acc


def test_osgd_schedule_updates_at_the_end_of_the_epoch_only():
    s = OsgdSchedule()
    assert s.frac == 1.0
    s.observe(dict(correct=92, n=100))
    assert s.frac == 1.0                                    # not yet
    s.end_epoch()
    assert s.frac == 1 / 4 and s.n == 0
    s.observe(dict(correct=50, n=100)); s.end_epoch()
    assert s.frac == 1.0                                    # a pure function of the last epoch's accuracy
    s.end_epoch()                                           # an empty epoch changes nothing
    assert s.frac == 1.0


def test_linear_schedule_runs_from_start_to_end_after_the_warmup():
    cfg = {**CFG, "sel_schedule": "linear", "sel_frac_start": 0.1, "sel_frac_end": 0.5, "sel_warmup": 10, "max_steps": 110}
    assert n_selected(128, 5, cfg) == 128                    # warm-up: full backward
    assert linear_frac(10, cfg) == 0.1 and linear_frac(110, cfg) == 0.5 and abs(linear_frac(60, cfg) - 0.3) < 1e-12
    assert n_selected(128, 10, cfg) == math.ceil(0.1 * 128) and n_selected(128, 110, cfg) == 64


def test_linear_schedule_flops_are_the_exact_sum_over_the_steps():
    cfg = {**CFG, "sel_schedule": "linear", "sel_frac_start": 0.5, "sel_frac_end": 0.1, "sel_warmup": 4, "max_steps": 40, "eval_every": 10}
    log = run(cfg, device="cpu", data=_loaders())
    want = 0
    for step in range(40):
        k = n_selected(B, step, cfg)
        want += B * FWD_PER_ROW + k * BWD_PER_ROW
        if (step + 1) % 10 == 0:
            assert log[(step + 1) // 10 - 1]["flops"] == want
    assert log[-1]["k"] == n_selected(B, 39, cfg)


def test_osgd_run_shrinks_k_once_training_accuracy_is_high_and_saves_flops():
    cfg = {**CFG, "sel_schedule": "osgd", "sel_frac": 1.0, "max_steps": 160, "eval_every": 16, "lr": 3e-3}
    easy = lambda: _loaders(sep=3.0, noise=0.5)
    log = run(cfg, device="cpu", data=easy())
    base = run({**cfg, "arm": "base"}, device="cpu", data=easy())
    ks = [r["k"] for r in log]
    assert ks[0] == B                                        # starts with the whole batch (no warm-up needed)
    assert ks[-1] <= B // 8 and len(set(ks)) >= 2
    assert log[-1]["flops"] < base[-1]["flops"]


def test_default_schedule_option_leaves_the_selective_run_unchanged():
    cfg = {**CFG, "max_steps": 30, "eval_every": 10, "sel_warmup": 10}
    a = run(cfg, device="cpu", data=_loaders())
    b = run({**cfg, "sel_schedule": "none", "sel_score": "loss"}, device="cpu", data=_loaders())
    strip = lambda log: [{k: v for k, v in r.items() if k != "time"} for r in log]
    assert strip(a) == strip(b)


def _autograd_grads(m, x, y, idx, denom):
    m.zero_grad(set_to_none=True)
    (F.cross_entropy(m(x), y, reduction="none")[idx].sum() / denom).backward()
    return [p.grad.clone() for p in m.parameters()]


def test_sel_scale_k_is_the_mean_over_the_kept_rows_and_B_the_default():
    torch.manual_seed(0)
    m = MeetNet(arm="selective")
    x, y = _batch()
    m.zero_grad(set_to_none=True)
    _, k, idx = selective_step(m, x, y, {**CFG, "sel_scale": "k"}, step=10**9)
    got_k = [p.grad.clone() for p in m.parameters()]
    for a, b in zip(got_k, _autograd_grads(m, x, y, idx, denom=k)):
        assert torch.allclose(a, b, atol=1e-6)
    m.zero_grad(set_to_none=True)
    selective_step(m, x, y, CFG, step=10**9)                                  # default: divided by the batch size
    got_B = [p.grad.clone() for p in m.parameters()]
    for a, b in zip(got_B, _autograd_grads(m, x, y, idx, denom=B)):
        assert torch.allclose(a, b, atol=1e-6)
    for a, b in zip(got_k, got_B):                                            # the two differ exactly by the factor B / k
        assert torch.allclose(a, b * (B / k), atol=1e-6)


def test_sel_scale_has_no_effect_when_all_rows_are_kept_and_explicit_B_equals_the_default():
    torch.manual_seed(0)
    m = MeetNet(arm="selective")
    x, y = _batch()
    grads = {}
    for name, cfg in [("default", CFG), ("B", {**CFG, "sel_scale": "B"}), ("k_full", {**CFG, "sel_scale": "k", "sel_frac": 1.0}),
                      ("B_full", {**CFG, "sel_frac": 1.0})]:
        m.zero_grad(set_to_none=True)
        selective_step(m, x, y, cfg, step=10**9)
        grads[name] = [p.grad.clone() for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(grads["default"], grads["B"]))
    assert all(torch.equal(a, b) for a, b in zip(grads["k_full"], grads["B_full"]))


def test_sel_scale_k_changes_the_run_but_not_its_flops():
    cfg = {**CFG, "max_steps": 30, "eval_every": 10, "sel_warmup": 10, "sel_frac": 0.25}
    a = run(cfg, device="cpu", data=_loaders())
    b = run({**cfg, "sel_scale": "k"}, device="cpu", data=_loaders())
    assert [r["flops"] for r in a] == [r["flops"] for r in b] and [r["k"] for r in a] == [r["k"] for r in b]
    assert [r["task"] for r in a] != [r["task"] for r in b]                    # the training trajectory is different
