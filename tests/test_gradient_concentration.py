"""Tests for the exploratory gradient-concentration check (exploratory/gradconc.py)."""
import itertools
import math

import torch
import torch.nn.functional as F

from exploratory.gradconc import (concentration, decide, measure_pool, per_example_grads,
                                  run_with_checkpoints, subset_metrics)
from scripts_loader import load_script
from src.model import MeetNet

CFG = dict(arm="base", seed=0, lr=1e-3, wd=1e-4, batch_size=32, max_steps=40, eval_every=10, act="relu",
           lr_schedule="constant", warmup_steps=1, lam_aux=0.0, lam_inv=0.0, lam_meet=0.0)


def clusters(n=1024, seed=0):
    g = torch.Generator().manual_seed(7)
    means = torch.randn(10, 784, generator=g) * 0.6
    gg = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=gg)
    return means[y] + torch.randn(n, 784, generator=gg) * 3.0, y


def loaders(seed=0):
    from torch.utils.data import DataLoader, TensorDataset
    x, y = clusters(1024, 1)
    xv, yv = clusters(256, 2)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True,
                       generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(xv, yv), batch_size=128))


def test_per_example_grads_match_autograd():
    torch.manual_seed(0)
    m = MeetNet(arm="base")
    x, y = clusters(8, 3)
    flat, losses = per_example_grads(m, x, y)
    assert flat.shape[0] == 8 and losses.shape == (8,)
    for i in (0, 3, 7):
        m.zero_grad()
        F.cross_entropy(m(x[i:i + 1]), y[i:i + 1]).backward()
        ref = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
        assert torch.allclose(flat[i], ref, atol=1e-6)


def test_mean_of_per_example_grads_is_the_batch_gradient():
    torch.manual_seed(0)
    m = MeetNet(arm="base")
    x, y = clusters(16, 4)
    flat, _ = per_example_grads(m, x, y)
    m.zero_grad()
    F.cross_entropy(m(x), y).backward()
    batch = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
    assert torch.allclose(flat.mean(0), batch, atol=1e-6)


def test_subset_metrics_edge_cases():
    g = torch.randn(20, 50)
    losses = torch.rand(20)
    full = subset_metrics(g, losses, torch.arange(20))
    assert math.isclose(full["norm_share"], 1.0, rel_tol=1e-6) and math.isclose(full["cos"], 1.0, rel_tol=1e-5)
    assert math.isclose(full["proj_share"], 1.0, rel_tol=1e-5) and full["omitted"] < 1e-5
    # one dominant example: the subset containing it holds most of the norm and points the same way
    g = torch.randn(20, 50) * 0.01
    g[3] = 10 * torch.ones(50)
    s = subset_metrics(g, torch.rand(20), torch.tensor([3]))
    assert s["norm_share"] > 0.9 and s["cos"] > 0.95


def test_random_subset_share_is_close_to_its_fraction():
    torch.manual_seed(1)
    g = torch.randn(4000, 20)
    losses = torch.rand(4000)
    gen = torch.Generator().manual_seed(0)
    r = concentration(g, losses, [0.3], gen)[("random", 0.3)]
    assert abs(r["norm_share"] - 0.3) < 0.03


def test_high_loss_examples_carry_more_gradient_than_random_ones():
    x, y = clusters(32, 5)
    (train, val) = loaders(0)
    pool = [(a, b) for a, b in itertools.islice(iter(train), 4)]
    rows, model = run_with_checkpoints({**CFG, "max_steps": 60}, train, val, pool, [60], [0.3])
    top = next(r for r in rows if r["mode"] == "top")
    rnd = next(r for r in rows if r["mode"] == "random")
    assert top["norm_share"] > rnd["norm_share"] + 0.05
    assert 0.0 <= top["norm_share"] <= 1.0


def test_measuring_does_not_change_training():
    def final_params(measure):
        train, val = loaders(0)
        pool = [(a, b) for a, b in itertools.islice(iter(loaders(9)[0]), 3)]
        _, model = run_with_checkpoints(CFG, train, val, pool, [10, 20, 30], [0.3, 0.5], measure=measure)
        return [p.detach().clone() for p in model.parameters()]
    a, b = final_params(True), final_params(False)
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_decision_rule():
    steps = (500, 1000, 2000)
    assert decide({s: 0.85 for s in steps}, {s: 0.95 for s in steps}) == "PROMISING"
    assert decide({s: 0.85 for s in steps}, {s: 0.80 for s in steps}) == "INCONCLUSIVE"   # concentrated but not aligned
    assert decide({500: 0.7, 1000: 0.55, 2000: 0.6}, {s: 0.95 for s in steps}) == "DROP"
    assert decide({500: 0.9, 1000: 0.7, 2000: 0.9}, {s: 0.95 for s in steps}) == "INCONCLUSIVE"


def test_script_smoke_run(tmp_path):
    mod = load_script("gradient_concentration")
    out = tmp_path / "gc.csv"
    verdict = mod.main(["--synthetic", "--seeds", "0", "--max-steps", "30", "--checkpoints", "10", "20", "30",
                        "--batches", "2", "--tag", "smoke", "--out", str(out)])
    assert verdict in ("PROMISING", "DROP", "INCONCLUSIVE")
    lines = out.read_text().splitlines()
    assert lines[0].startswith("seed,step,mode,fraction,norm_share")
    assert len(lines) == 1 + 3 * 4 * 2          # 3 checkpoints x 4 fractions x (top, random)
