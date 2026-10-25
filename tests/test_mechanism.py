"""Tests for checks A and L (exploratory/mechanism.py, scripts/mechanism_check.py)."""
import csv
import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from exploratory.gradconc import per_example_grads
from exploratory.mechanism import (decide_link, decide_q1, decide_q2, decide_q3, directions, geometry_param_space,
                                   link_metrics, measure_checkpoint, per_example_low_meet_grads, run_measured,
                                   softmax_hessian, spearman, split_parallel, up_jacobian, up_jacobian_autograd)
from scripts_loader import load_script
from src.model import MeetNet, compute_losses
from src.train import run

CFG = dict(arm="meet", seed=0, lr=1e-3, wd=1e-4, batch_size=32, max_steps=40, eval_every=10, act="tanh",
           lr_schedule="constant", warmup_steps=5, lam_aux=0.0, lam_inv=0.1, lam_meet=0.05, sigma_inv=0.0)


def clusters(n=1024, seed=0):
    g = torch.Generator().manual_seed(7)
    means = torch.randn(10, 784, generator=g) * 0.6
    gg = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=gg)
    return means[y] + torch.randn(n, 784, generator=gg) * 3.0, y


def loaders(seed=0):
    x, y = clusters(1024, 1)
    xv, yv = clusters(256, 2)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True,
                       generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(xv, yv), batch_size=128))


# ---------------------------------------------------------------- geometry in h-space
@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_closed_form_jacobian_matches_autograd(act):
    torch.manual_seed(0)
    m = MeetNet(arm="base", act=act).double()
    h = torch.randn(6, 128, dtype=torch.double)
    assert torch.allclose(up_jacobian(m, h), up_jacobian_autograd(m, h), atol=1e-10)


def test_bp_direction_is_minus_the_gradient_of_the_cross_entropy_at_h():
    torch.manual_seed(1)
    m = MeetNet(arm="meet", act="tanh").double()
    x, y = clusters(12, 3)
    x = x.double()
    d, n_kept, _ = directions(m, x, y, True)
    assert n_kept == 12
    h = m.low(x).detach().requires_grad_()
    g = torch.autograd.grad(F.cross_entropy(m.up(h), y, reduction="sum"), h)[0]
    assert torch.allclose(d["bp"], -g, atol=1e-10)


def test_pinv_is_the_inverse_the_gauss_newton_step_and_lsq_is_the_damped_inverse():
    torch.manual_seed(2)
    m = MeetNet(arm="meet", act="relu").double()
    m.up = torch.nn.Sequential(torch.nn.Identity(), m.up[1], torch.nn.Identity(), m.up[3])   # linear top half
    x, y = clusters(10, 5)
    x = x.double()
    d, n_kept, _ = directions(m, x, y, False)
    assert n_kept == 10
    with torch.no_grad():
        h = m.low(x)
        p = m.up(h).softmax(-1)
        delta = F.one_hot(y, 10).double() - p
        J, S = up_jacobian(m, h), softmax_hessian(p)
        # (1) the perfect local inverse moves p to e_y to first order: S J u = e_y - p
        assert torch.allclose(torch.einsum("bkd,bd->bk", S @ J, d["pinv"]), delta, atol=1e-8)
        # (2) the Gauss-Newton step (J^T S J)^+ J^T (e_y - p) is the same vector (J has full row rank)
        A = J.transpose(1, 2) @ S @ J
        rhs = torch.einsum("bkd,bk->bd", J, delta)
        gn = torch.einsum("bde,be->bd", torch.linalg.pinv(A, hermitian=True), rhs)
        assert torch.allclose(gn, d["pinv"], atol=1e-6)
        # (3) lsq equals the damped least-squares form (J^T J + alpha I)^-1 J^T (e_y - p)
        JJt = J @ J.transpose(1, 2)
        alpha = 0.01 * JJt.diagonal(dim1=1, dim2=2).mean(-1)
        B = J.transpose(1, 2) @ J + alpha[:, None, None] * torch.eye(128, dtype=torch.double)
        assert torch.allclose(torch.linalg.solve(B, rhs.unsqueeze(-1)).squeeze(-1), d["lsq"], atol=1e-8)


def test_confident_rows_are_skipped():
    torch.manual_seed(3)
    m = MeetNet(arm="meet").double()
    with torch.no_grad():
        m.up[3].bias.copy_(torch.tensor([30.0] + [0.0] * 9))      # class 0 is certain for every row
    x, y = clusters(20, 6)
    y[:10] = 0
    y[10:] = (y[10:] % 9) + 1                                    # the other rows are never class 0
    d, n_kept, n_total = directions(m, x.double(), y, False)
    assert n_total == 20 and n_kept == 10 and d["bp"].shape[0] == 10


def test_split_parallel_is_orthogonal_and_adds_up():
    torch.manual_seed(4)
    gt, gm = torch.randn(500), torch.randn(500)
    par, perp = split_parallel(gt, gm)
    assert abs(float(perp @ gt)) < 1e-3 and torch.allclose(par + perp, gm, atol=1e-5)
    assert torch.allclose(par, (gm @ gt) / (gt @ gt) * gt, atol=1e-5)


# ---------------------------------------------------------------- parameter space and check L
def test_geometry_param_space_restores_the_parameters_exactly():
    torch.manual_seed(5)
    m = MeetNet(arm="meet", act="tanh")
    x, y = clusters(32, 6)
    before = [p.detach().clone() for p in m.parameters()]
    out = geometry_param_space(m, CFG, x, y, clusters(64, 7))
    assert all(torch.equal(a, b.detach()) for a, b in zip(before, m.parameters()))
    assert abs(out["par_share"] - out["cos_task_meet"] ** 2) < 1e-12
    assert all(math.isfinite(v) for v in out.values())


def test_per_example_meeting_gradients_average_to_the_batch_gradient():
    torch.manual_seed(6)
    m = MeetNet(arm="meet", act="tanh")
    x, y = clusters(16, 4)
    flat = per_example_low_meet_grads(m, x, y)
    _, parts, _, _ = compute_losses(m, x, y, CFG, step=10 ** 9)
    ref = torch.cat([g.reshape(-1) for g in torch.autograd.grad(parts["meet"], list(m.low.parameters()))])
    assert torch.allclose(flat.mean(0), ref, atol=1e-6)


def test_lower_layer_columns_come_first_in_the_per_example_task_gradients():
    torch.manual_seed(7)
    m = MeetNet(arm="meet")
    x, y = clusters(6, 5)
    flat, _ = per_example_grads(m, x, y)
    n_low = sum(p.numel() for p in m.low.parameters())
    for i in (0, 4):
        m.zero_grad()
        F.cross_entropy(m(x[i:i + 1]), y[i:i + 1]).backward()
        ref = torch.cat([p.grad.reshape(-1) for p in m.low.parameters()])
        assert torch.allclose(flat[i, :n_low], ref, atol=1e-6)


def test_keeping_every_row_gives_share_and_cosine_one():
    torch.manual_seed(8)
    m = MeetNet(arm="meet", act="tanh")
    x, y = clusters(24, 8)
    out = link_metrics(m, x, y, True, fraction=1.0)
    for k in ("task_norm_share", "task_cos", "meet_norm_share", "meet_cos"):
        assert math.isclose(out[k], 1.0, rel_tol=1e-5), k
    assert "overlap_cos_unselected" not in out
    part = link_metrics(m, x, y, True, fraction=0.25)
    assert 0 < part["meet_norm_share"] < 1 and -1 <= part["overlap_cos_unselected"] <= 1


def test_spearman():
    a = torch.randn(50)
    assert math.isclose(spearman(a, a ** 3), 1.0, rel_tol=1e-9)
    assert math.isclose(spearman(a, -a), -1.0, rel_tol=1e-9)


def test_measure_checkpoint_keys_per_arm():
    torch.manual_seed(9)
    pool = [clusters(16, s) for s in (10, 11)]
    probe = clusters(32, 12)
    qd = measure_checkpoint(MeetNet(arm="meet", act="tanh"), CFG, pool, probe)
    for k in ("cos_meet_bp_w", "cos_meet_pinv_w", "cos_bp_lsq_u", "norm_ratio", "cos_task_meet", "dl_perp",
              "task_norm_share", "meet_norm_share", "corr_loss_umeet", "overlap_cos_unselected"):
        assert k in qd, k
    qa = measure_checkpoint(MeetNet(arm="base", act="relu"), {**CFG, "arm": "base"}, pool, probe)
    assert "task_norm_share" in qa and not any(k.startswith(("meet_", "cos_")) for k in qa)


# ---------------------------------------------------------------- training equivalence
def _acc(rows):
    return {r[2]: r[4] for r in rows if r[3] == "val_acc"}


def test_measuring_does_not_change_training():
    pool = [clusters(16, s) for s in (10, 11)]
    probe = clusters(32, 12)
    cps = [10, 20, 30, 40]
    a, _ = run_measured(CFG, *loaders(0), pool, probe, cps, measure=False)
    b, _ = run_measured(CFG, *loaders(0), pool, probe, cps, measure=True)
    assert _acc(a) == _acc(b) and len(_acc(a)) == 4


def test_run_measured_follows_src_train_run():
    pool = [clusters(16, 10)]
    ref = {r["step"]: r["val_acc"] for r in run(CFG, "cpu", data=loaders(0))}
    rows, _ = run_measured(CFG, *loaders(0), pool, clusters(32, 12), [10, 20, 30, 40], measure=False)
    assert _acc(rows) == ref


# ---------------------------------------------------------------- decision rules
def test_decide_q1():
    ok = {500: 0.7, 1000: 0.7, 2000: 0.7}
    tiny, big = {500: 0.01, 1000: 0.01, 2000: 0.01}, {500: -0.5, 1000: -0.5, 2000: -0.5}
    assert decide_q1(ok, tiny, big) == "REDUNDANT"
    assert decide_q1(ok, big, big) == "NOT REDUNDANT"
    assert decide_q1({500: 0.5, 1000: 0.7, 2000: 0.7}, tiny, big) == "INCONCLUSIVE"
    assert decide_q1({500: 0.7}, tiny, big).startswith("N/A")


def test_decide_q2_q3_and_link():
    assert decide_q2({500: 0.8, 1000: 0.8}, {500: 0.5, 1000: 0.6}) == "YES"
    assert decide_q2({500: 0.4, 1000: 0.5}, {500: 0.5, 1000: 0.6}) == "NO"
    assert decide_q2({500: 0.8, 1000: 0.55}, {500: 0.5, 1000: 0.6}) == "UNCLEAR"
    assert decide_q3({500: 0.95, 1000: 0.93, 2000: 0.92}) == "BACKPROP ~ GN"
    assert decide_q3({500: 0.7, 1000: 0.6, 2000: 0.75}) == "GN DIFFERS"
    assert decide_q3({500: 0.95, 1000: 0.85, 2000: 0.92}) == "GREY"
    hi, lo = {500: 0.95, 1000: 0.93, 2000: 0.96}, {500: 0.5, 1000: 0.4, 2000: 0.55}
    corr = {500: 0.6, 1000: 0.7, 2000: 0.6}
    assert decide_link(hi, corr) == "ALREADY SELECTIVE"
    assert decide_link(lo, corr) == "NOT SELECTIVE"
    assert decide_link({500: 0.95, 1000: 0.7, 2000: 0.95}, corr) == "PARTIAL"
    assert decide_link(hi, {500: 0.6, 1000: 0.2, 2000: 0.6}) == "PARTIAL"


# ---------------------------------------------------------------- the script, end to end on synthetic data
def test_script_runs_on_synthetic_data(tmp_path):
    mc = load_script("mechanism_check")
    out = tmp_path / "m.csv"
    verdicts = mc.main(["--synthetic", "--max-steps", "40", "--checkpoints", "10", "20", "40", "--seeds", "0",
                        "--batches", "2", "--probe-batches", "1", "--tag", "t", "--out", str(out)])
    rows = list(csv.DictReader(open(out)))
    assert rows and set(rows[0]) == {"seed", "arm", "step", "quantity", "value"}
    assert {r["arm"] for r in rows} == {"meet", "base"}
    assert any(r["quantity"] == "cos_meet_pinv_w" for r in rows)
    assert set(verdicts) == {"Q1", "Q2", "Q2_lsq", "Q3", "L"} and all(v.startswith("N/A") for v in verdicts.values())
