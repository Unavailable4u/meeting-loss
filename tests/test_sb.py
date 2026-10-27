"""Tests for the Selective-Backprop reference arm (src/sb.py, arm "sb"; D-025)."""
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.flops import flops_per_step
from src.model import MeetNet
from src.sb import LossHistory, forward_flops, keep_probability, run_sb
from src.train import build_optimizer, run

BATCH = 32
FWD_PER_ROW, BWD_PER_ROW = 537_600, 673_792            # the counter's values for the 784-256-128-256-10 MLP (see test_selective)
CFG = dict(stage="t", arm="sb", seed=0, act="relu", lr=1e-3, wd=1e-4, batch_size=BATCH, max_steps=40, eval_every=10,
           lr_schedule="cosine", sb_beta=2.0, sb_history=1024, sb_prime_steps=10)


def _loaders(seed=0):
    g = torch.Generator().manual_seed(7)
    means = torch.randn(10, 784, generator=g) * 0.6
    def make(n, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.randint(0, 10, (n,), generator=gg)
        return means[y] + torch.randn(n, 784, generator=gg) * 3.0, y
    x, y = make(1024, 1); xv, yv = make(256, 2)
    return (DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(xv, yv), batch_size=128))


def _strip(log):
    return [{k: v for k, v in r.items() if k != "time"} for r in log]


def test_loss_history_is_a_ring_buffer_of_the_last_values():
    h = LossHistory(4)
    h.extend(torch.tensor([1.0, 2.0, 3.0]))
    assert h.n == 3
    h.extend(torch.tensor([4.0, 5.0, 6.0]))                # 1 and 2 drop out: history = {3, 4, 5, 6}
    assert h.n == 4 and sorted(h.buf.tolist()) == [3.0, 4.0, 5.0, 6.0]
    h.extend(torch.arange(10.0, 20.0))                      # longer than the buffer: the last 4 remain
    assert sorted(h.buf.tolist()) == [16.0, 17.0, 18.0, 19.0]


def test_percentile_is_the_share_strictly_below_and_probability_its_power():
    h = LossHistory(8)
    h.extend(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    pct = h.percentile(torch.tensor([0.5, 2.0, 5.0]))
    assert pct.tolist() == [0.0, 0.25, 1.0]
    p = keep_probability(pct, beta=2.0)
    assert torch.allclose(p, torch.tensor([0.0, 0.0625, 1.0]))
    assert keep_probability(torch.tensor([0.0]), 2.0, sampling_min=0.1).item() == pytest.approx(0.1)


def test_expected_keep_rate_is_one_over_beta_plus_one():
    g = torch.Generator().manual_seed(0)
    h = LossHistory(1024)
    h.extend(torch.rand(1024, generator=g))
    losses = torch.rand(200000, generator=g)
    for beta in (1.0, 2.0, 3.0):
        rate = keep_probability(h.percentile(losses), beta).mean().item()
        assert rate == pytest.approx(1 / (1 + beta), abs=0.01)


def test_sb_network_starts_like_arm_a():
    torch.manual_seed(5); a = MeetNet(arm="base")
    torch.manual_seed(5); b = MeetNet(arm="sb")
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_priming_is_bit_identical_to_arm_a():
    cfg = {**CFG, "sb_prime_steps": 40}
    sb = run(cfg, device="cpu", data=_loaders())
    base = run({**cfg, "arm": "base"}, device="cpu", data=_loaders())
    assert [(r["step"], r["flops"], r["val_acc"]) for r in sb] == [(r["step"], r["flops"], r["val_acc"]) for r in base]
    assert [r["flops_cached"] for r in sb] == [r["flops"] for r in sb]            # no selection phase yet


def test_per_step_costs_are_the_counters_values():
    m = MeetNet(arm="sb")
    opt, _ = build_optimizer(m, CFG)
    x, y = next(iter(_loaders()[0]))
    assert forward_flops(m, x) == BATCH * FWD_PER_ROW
    assert flops_per_step(m, opt, (x, y), CFG) == BATCH * (FWD_PER_ROW + BWD_PER_ROW)


def test_flops_are_the_exact_sum_of_selection_forwards_and_updates():
    log = run(CFG, device="cpu", data=_loaders())
    fwd, upd = BATCH * FWD_PER_ROW, BATCH * (FWD_PER_ROW + BWD_PER_ROW)
    prime = CFG["sb_prime_steps"]
    assert [r["step"] for r in log] == [10, 20, 30, 40]
    for r in log:
        sel_forwards = r["n_fwd"] - prime                                           # forward batches after priming
        assert r["flops"] == upd * r["step"] + fwd * sel_forwards
        assert r["flops_cached"] == upd * prime + (upd - fwd) * (r["step"] - prime) + fwd * sel_forwards
    assert log[-1]["flops_cached"] < log[-1]["flops"]


def test_each_update_uses_exactly_one_batch_of_kept_rows():
    log = run(CFG, device="cpu", data=_loaders())
    prime = CFG["sb_prime_steps"]
    for r in log:
        seen_after_priming = r["rows_seen"] - prime * BATCH
        kept = round(r["sel_rate"] * seen_after_priming)
        assert kept - BATCH * (r["step"] - prime) == r["queue"] >= 0


def test_beta_zero_keeps_every_row_so_one_update_per_forward_batch():
    log = run({**CFG, "sb_beta": 0.0}, device="cpu", data=_loaders())
    assert all(r["n_fwd"] == r["step"] and r["queue"] == 0 and r["sel_rate"] == 1.0 for r in log)


def test_higher_beta_needs_more_forward_batches_per_update():
    n = {b: run({**CFG, "sb_beta": b}, device="cpu", data=_loaders())[-1]["n_fwd"] for b in (0.0, 1.0, 3.0)}
    assert n[0.0] < n[1.0] < n[3.0]


def test_run_is_deterministic_and_learns():
    cfg = {**CFG, "max_steps": 120, "eval_every": 40}
    a = run(cfg, device="cpu", data=_loaders())
    b = run(cfg, device="cpu", data=_loaders())
    assert _strip(a) == _strip(b)
    assert a[-1]["val_acc"] > 0.4


def test_too_strict_selection_stops_instead_of_looping():
    with pytest.raises(RuntimeError, match="selection is too strict"):
        run_sb({**CFG, "sb_beta": 1000.0, "sb_prime_steps": 0, "sb_max_forward_batches": 50}, device="cpu", data=_loaders())
