"""Stage 2 tests (docs/GUIDE.md, Section 6). Run with: pytest

Coverage of the guide's required tests:
1. zero lambdas match baseline ........ test_zero_lambda_matches_baseline
2. meet gradient reaches only low ..... test_meet_gradient_reaches_only_low
3. L_inv decreases ..................... test_inverse_loss_decreases_when_G_trained_alone
4. target carries no gradient .......... covered by test 2 (G and up get no gradient from L_meet)
5. FLOP counter on a known layer ....... test_flop_counter_linear_layer
6. reproducibility (CPU) ............... test_reproducible_given_seed
"""
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.model import MeetNet, compute_losses

CFG0 = dict(lam_inv=0.0, lam_meet=0.0, warmup_steps=1, lam_aux=0.0)
CFG = dict(lam_inv=1.0, lam_meet=0.1, warmup_steps=1, lam_aux=0.0)


def batch(n=32):
    torch.manual_seed(0)
    return torch.randn(n, 784), torch.randint(0, 10, (n,))


def test_zero_lambda_matches_baseline():
    m = MeetNet(arm="meet")
    base = MeetNet(arm="base")
    base.low.load_state_dict(m.low.state_dict())
    base.up.load_state_dict(m.up.state_dict())
    x, y = batch()
    for net in (m, base):
        loss, *_ = compute_losses(net, x, y, CFG0, step=100)
        loss.backward()
    for a, b in zip(list(m.low.parameters()) + list(m.up.parameters()),
                    list(base.low.parameters()) + list(base.up.parameters())):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_meet_gradient_reaches_only_low():
    m = MeetNet(arm="meet")
    x, y = batch()
    _, parts, _, _ = compute_losses(m, x, y, CFG, step=100)
    parts["meet"].backward()
    assert all(p.grad is not None for p in m.low.parameters())
    assert all(p.grad is None for p in m.up.parameters())
    assert all(p.grad is None for p in m.G.parameters())


def test_inverse_loss_decreases_when_G_trained_alone():
    m = MeetNet(arm="meet")
    x, y = batch(256)
    opt = torch.optim.Adam(m.G.parameters(), lr=1e-2)
    first = last = None
    for _ in range(200):
        _, parts, _, _ = compute_losses(m, x, y, CFG, step=100)
        opt.zero_grad()
        parts["inv"].backward()
        opt.step()
        first = parts["inv"].item() if first is None else first
        last = parts["inv"].item()
    assert last < first


def test_flop_counter_linear_layer():
    lin = torch.nn.Linear(64, 32, bias=False)
    x = torch.randn(8, 64)
    with FlopCounterMode(display=False) as fc:
        lin(x)
    assert fc.get_total_flops() == 2 * 8 * 64 * 32


def _train_steps(seed, n=5):
    torch.manual_seed(seed)
    m = MeetNet(arm="meet")
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x, y = batch()
    for s in range(n):
        loss, *_ = compute_losses(m, x, y, CFG, step=s)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in m.parameters()]


def test_reproducible_given_seed():
    a, b = _train_steps(1), _train_steps(1)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
