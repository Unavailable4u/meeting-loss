"""Tests for method v1.1: arms, variants, controls, optimizer groups, schedules, smoke runs.
Run with: pytest"""
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.diagnostics import grad_alignment
from src.model import ARMS, MEET_ARMS, MeetNet, compute_losses
from src.train import build_optimizer, lr_factors, run, steps_to_target

CFG = dict(lam_inv=1.0, rho=1.0, rho_warmup_steps=1, lam_meet=0.1, warmup_steps=1, lam_aux=0.3, eta_hat=0.1,
           max_steps=100, lr=1e-3, wd=1e-4)


def batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 784, generator=g), torch.randint(0, 10, (n,), generator=g)


def test_all_arms_run_and_backward():
    x, y = batch()
    for arm in ARMS:
        torch.manual_seed(0)
        m = MeetNet(arm=arm)
        total, parts, h, z = compute_losses(m, x, y, CFG, step=10)
        assert torch.isfinite(total), arm
        total.backward()
        assert all(p.grad is not None for p in m.low.parameters()), arm


def test_tanh_option_has_no_relu():
    m = MeetNet(arm="meet", act="tanh")
    mods = list(m.modules())
    assert any(isinstance(k, torch.nn.Tanh) for k in mods)
    assert not any(isinstance(k, torch.nn.ReLU) for k in mods)


def test_inverse_loss_gradient_reaches_only_G():
    x, y = batch()
    for cfg in (CFG, {**CFG, "sigma_inv": 0.1}):
        m = MeetNet(arm="meet")
        _, parts, _, _ = compute_losses(m, x, y, cfg, step=10)
        parts["inv"].backward()
        assert all(p.grad is not None for p in m.G.parameters())
        assert all(p.grad is None for p in m.low.parameters())
        assert all(p.grad is None for p in m.up.parameters())


def test_meet_gradient_reaches_only_low_for_variants():
    x, y = batch()
    for arm, extra in (("meet", {"sigma_inv": 0.1}), ("meet_ao", {}), ("meet_frozen", {})):
        m = MeetNet(arm=arm)
        _, parts, _, _ = compute_losses(m, x, y, {**CFG, **extra}, step=10)
        parts["meet"].backward()
        assert all(p.grad is not None for p in m.low.parameters()), arm
        assert all(p.grad is None for p in m.up.parameters()), arm
        assert all(p.grad is None for p in m.G.parameters()), arm


def test_ao_variant_input_and_fixed_features():
    m = MeetNet(arm="meet_ao", d_aux=64)
    assert m.G[0].in_features == m.n_cls + 64
    assert "R" not in dict(m.named_parameters())          # a buffer, never trained
    R0 = m.R.clone()
    x, y = batch()
    opt, _ = build_optimizer(m, {**CFG, "lr": 1e-2})
    total, *_ = compute_losses(m, x, y, CFG, step=10)
    total.backward()
    opt.step()
    assert torch.equal(m.R, R0)


def test_frozen_G_is_not_updated_and_has_no_inverse_loss():
    m = MeetNet(arm="meet_frozen")
    assert all(not p.requires_grad for p in m.G.parameters())
    G0 = [p.detach().clone() for p in m.G.parameters()]
    x, y = batch()
    opt, _ = build_optimizer(m, {**CFG, "lr": 1e-2})
    total, parts, _, _ = compute_losses(m, x, y, CFG, step=10)
    assert parts["inv"].item() == 0.0
    total.backward()
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(G0, m.G.parameters()))


def test_exact_gradient_target_is_parallel_to_task_gradient():
    x, y = batch()
    m = MeetNet(arm="meet_grad")
    _, parts, h, _ = compute_losses(m, x, y, CFG, step=10)
    g_task = torch.autograd.grad(parts["task"], h, retain_graph=True)[0]
    g_meet = torch.autograd.grad(parts["meet"], h, retain_graph=True)[0]
    cos = F.cosine_similarity(g_task.flatten(), g_meet.flatten(), dim=0)
    assert cos.item() > 0.999


def test_deep_supervision_schedule_and_threshold():
    x, y = batch()
    m = MeetNet(arm="deepsup")
    cfg = {**CFG, "aux_sched": "linear", "max_steps": 100}
    t0, p0, _, _ = compute_losses(m, x, y, cfg, step=0)
    t1, p1, _, _ = compute_losses(m, x, y, cfg, step=100)
    assert t0.item() > p0["task"].item() + 1e-6            # companion loss active at step 0
    assert math.isclose(t1.item(), p1["task"].item(), rel_tol=1e-6)   # decayed to zero
    t2, p2, _, _ = compute_losses(m, x, y, {**CFG, "aux_gamma": 1e6}, step=0)
    assert math.isclose(t2.item(), p2["task"].item(), rel_tol=1e-6)   # below threshold: vanishes


def test_default_optimizer_matches_plain_adamw():
    x, y = batch()
    def train(grouped):
        torch.manual_seed(3)
        m = MeetNet(arm="base")
        if grouped:
            opt, sched = build_optimizer(m, CFG)
        else:
            opt, sched = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"]), None
        for s in range(5):
            total, *_ = compute_losses(m, x, y, CFG, step=s)
            opt.zero_grad()
            total.backward()
            opt.step()
            if sched:
                sched.step()
        return [p.detach().clone() for p in m.parameters()]
    a, b = train(True), train(False)
    assert all(torch.allclose(p, q, atol=1e-7) for p, q in zip(a, b))


def test_low_lr_multiplier_and_schedules():
    m = MeetNet(arm="base")
    opt, _ = build_optimizer(m, {**CFG, "low_lr_mult": 4.0})
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert math.isclose(lrs[0], CFG["lr"]) and math.isclose(lrs[-1], 4 * CFG["lr"])
    base, _, aux = lr_factors({**CFG, "lr_schedule": "cosine", "aux_lr_decay": "power_inv", "max_steps": 100})
    assert math.isclose(base(0), 1.0) and base(100) < 1e-9 and base(30) > base(60)
    assert aux(50) < base(50)


def _loaders(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 784, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True,
                       generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(x[:64], y[:64]), batch_size=64))


def test_smoke_run_every_arm():
    for arm in ARMS:
        cfg = {**CFG, "arm": arm, "seed": 0, "batch_size": 32, "max_steps": 30, "eval_every": 10,
               "lr_schedule": "cosine", "act": "tanh"}
        log = run(cfg, device="cpu", data=_loaders())
        assert len(log) == 3, arm
        assert all(math.isfinite(r["val_acc"]) and r["flops"] > 0 for r in log), arm
        assert log[-1]["flops"] > log[0]["flops"], arm
        if arm in MEET_ARMS or arm == "meet_grad":
            assert all(math.isfinite(r["align"]) for r in log), arm
        assert steps_to_target(log, 2.0) is None


def test_grad_alignment_is_nan_without_meeting_loss():
    x, y = batch()
    assert math.isnan(grad_alignment(MeetNet(arm="base"), x, y, CFG))
