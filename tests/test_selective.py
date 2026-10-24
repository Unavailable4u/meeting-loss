"""Tests for the selective backward pass (src/selective.py, arm "selective")."""
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.flops import flops_per_step
from src.model import MeetNet
from src.selective import n_selected, select_indices, selective_step
from src.train import build_optimizer, run

B = 128
CFG = dict(lr=1e-3, wd=1e-4, sel_frac=0.3, sel_warmup=0, sel_mode="top", max_steps=40, eval_every=10, batch_size=32)


def batch(n=B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 784, generator=g), torch.randint(0, 10, (n,), generator=g)


def autograd_grads(model, x, y, idx=None):
    """Gradient of (1/B) * sum over the selected rows of the cross-entropy, by autograd."""
    model.zero_grad(set_to_none=True)
    losses = F.cross_entropy(model(x), y, reduction="none")
    sel = losses if idx is None else losses[idx]
    (sel.sum() / x.shape[0]).backward()
    return [p.grad.clone() for p in model.parameters()]


def manual_grads(model, x, y, cfg, step=10**9):
    model.zero_grad(set_to_none=True)
    loss, k, idx = selective_step(model, x, y, cfg, step)
    return [p.grad.clone() for p in model.parameters()], loss, k, idx


def test_full_backward_matches_autograd_exactly():
    for act in ("relu", "tanh"):
        torch.manual_seed(0)
        m = MeetNet(arm="selective", act=act)
        x, y = batch()
        ref = autograd_grads(m, x, y)
        got, loss, k, idx = manual_grads(m, x, y, {**CFG, "sel_frac": 1.0})
        assert k == B and idx is None
        assert math.isclose(loss.item(), F.cross_entropy(m(x), y).item(), rel_tol=1e-6)
        for a, b in zip(ref, got):
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-4), act


def test_subset_backward_matches_autograd_on_the_same_rows():
    for act in ("relu", "tanh"):
        torch.manual_seed(1)
        m = MeetNet(arm="selective", act=act)
        x, y = batch(seed=2)
        got, _, k, idx = manual_grads(m, x, y, CFG)
        assert k == math.ceil(0.3 * B) and idx.shape[0] == k
        ref = autograd_grads(m, x, y, idx)
        for a, b in zip(ref, got):
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-4), act


def test_top_mode_selects_the_highest_loss_rows():
    torch.manual_seed(3)
    m = MeetNet(arm="selective")
    x, y = batch(seed=4)
    with torch.no_grad():
        losses = F.cross_entropy(m(x), y, reduction="none")
    _, _, k, idx = manual_grads(m, x, y, CFG)
    assert set(idx.tolist()) == set(torch.topk(losses, k).indices.tolist())
    assert losses[idx].min() >= torch.sort(losses, descending=True).values[k - 1] - 1e-9


def test_random_mode_has_the_same_size_but_not_the_same_rows():
    torch.manual_seed(5)
    m = MeetNet(arm="selective")
    x, y = batch(seed=6)
    _, _, k, top = manual_grads(m, x, y, CFG)
    _, _, k2, rnd = manual_grads(m, x, y, {**CFG, "sel_mode": "random"})
    assert k == k2 and set(top.tolist()) != set(rnd.tolist())
    assert torch.unique(rnd).numel() == k


def test_warmup_uses_the_full_backward_pass():
    assert n_selected(B, 0, {**CFG, "sel_warmup": 100}) == B
    assert n_selected(B, 99, {**CFG, "sel_warmup": 100}) == B
    assert n_selected(B, 100, {**CFG, "sel_warmup": 100}) == math.ceil(0.3 * B)
    assert n_selected(B, 5, {**CFG, "sel_frac": 1.0}) == B


def test_flop_count_is_exact():
    """forward 2*B*sum(in*out); weight gradients per selected row; input gradients for layers 2 to 4."""
    macs = 784 * 256 + 256 * 128 + 128 * 256 + 256 * 10           # 268,800
    macs_input = 256 * 128 + 128 * 256 + 256 * 10                   # layers 2-4: 68,096
    x, y = batch()
    for frac in (1.0, 0.5, 0.3):
        m = MeetNet(arm="selective")
        opt, _ = build_optimizer(m, CFG)
        k = n_selected(B, 10**9, {**CFG, "sel_frac": frac})
        expected = 2 * B * macs + 2 * k * macs + 2 * k * macs_input
        assert flops_per_step(m, opt, (x, y), {**CFG, "sel_frac": frac}) == expected
    # the full-backward case equals the plain baseline's count (155,058,176 at B = 128)
    mb = MeetNet(arm="base")
    optb, _ = build_optimizer(mb, CFG)
    base = flops_per_step(mb, optb, (x, y), {**CFG, "lam_aux": 0.0})
    assert base == 155_058_176 == 2 * B * macs + 2 * B * macs + 2 * B * macs_input


def test_selective_with_all_rows_trains_like_the_baseline():
    def final(arm):
        torch.manual_seed(7)
        m = MeetNet(arm=arm)
        opt, _ = build_optimizer(m, CFG)
        x, y = batch(64, seed=8)
        for s in range(6):
            opt.zero_grad(set_to_none=True)
            if arm == "selective":
                selective_step(m, x, y, {**CFG, "sel_frac": 1.0}, s)
            else:
                F.cross_entropy(m(x), y).backward()
            opt.step()
        return [p.detach() for p in m.parameters()]
    for a, b in zip(final("base"), final("selective")):
        assert torch.allclose(a, b, atol=1e-4)


def _loaders(seed=0):
    g = torch.Generator().manual_seed(7)
    means = torch.randn(10, 784, generator=g) * 0.6
    def make(n, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.randint(0, 10, (n,), generator=gg)
        return means[y] + torch.randn(n, 784, generator=gg) * 3.0, y
    x, y = make(1024, 1); xv, yv = make(256, 2)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(xv, yv), batch_size=128))


def test_run_logs_cumulative_flops_with_warmup_and_learns():
    cfg = {**CFG, "arm": "selective", "seed": 0, "max_steps": 60, "eval_every": 20, "sel_warmup": 20, "sel_frac": 0.5, "batch_size": 32}
    log = run(cfg, device="cpu", data=_loaders())
    base = run({**cfg, "arm": "base"}, device="cpu", data=_loaders())
    assert [r["step"] for r in log] == [20, 40, 60]
    full, sel = base[0]["flops"] // 20, None
    # first 20 steps cost a full backward pass (same as the baseline), later steps cost less
    assert log[0]["flops"] == base[0]["flops"]
    assert log[1]["flops"] - log[0]["flops"] < base[1]["flops"] - base[0]["flops"]
    assert log[-1]["flops"] < base[-1]["flops"]
    assert log[-1]["val_acc"] > 0.5 and all("k" in r for r in log)
