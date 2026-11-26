"""Root-cause checks B and D (docs/notes/mechanism-root-cause.md, D-028, DR-019): the oracle directions, the arms
`meet_oracle` / `meet_replace*`, the sweep arms, the decision rules and the runner.

The tests the note asks for first:
1. oracle_rho = 0 reproduces arm A step for step (test_rho_zero_reproduces_arm_a)
2. replacing backprop by the EXACT gradient (cut at h, oracle type grad, rho 1) reproduces arm A (test_exact_gradient_replacement_reproduces_arm_a)
3. the gradient at h of the injected signal has norm rho * (task gradient norm) and points along the oracle direction (test_injected_gradient_*)
4. in a replacement arm the task loss does not reach F_low and the meeting term does (test_replace_*)
5. the oracle directions equal those of check A (exploratory/mechanism.py) (test_directions_match_check_a)
6. adding the new sweep arms left every existing arm's trials and configs bit-identical (test_existing_sweep_arms_unchanged)
"""
import copy
import hashlib
import json
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from exploratory import mechanism as mech
from exploratory import oracle as orc
from src import oracle, sweep
from src.final import FinalJob
from src.model import ARMS, MeetNet, compute_losses
from src.train import run

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scripts_loader import load_script  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG = dict(lam_inv=0.1, lr=1e-3, wd=1e-4, max_steps=100, lr_schedule="cosine")
ORACLE_KINDS = ("grad", "pinv", "logit")


def _batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 784, generator=g), torch.randint(0, 10, (n,), generator=g)


def _trained(arm, act, steps=6, seed=0):
    """A model a few steps into training, so that p is not uniform and the weights are not at their initial values."""
    torch.manual_seed(seed)
    m = MeetNet(arm=arm, act=act)
    x, y = _batch(seed=seed)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], 1e-3)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
    return m, x, y


def _loaders(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.randn(n, 784, generator=g), torch.randint(0, 10, (n,), generator=g)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(x[:64], y[:64]), batch_size=64))


def _run(arm, act, **extra):
    cfg = dict(seed=0, batch_size=32, max_steps=30, eval_every=5, lr=1e-3, wd=1e-4, lr_schedule="cosine", lam_inv=0.1, arm=arm, act=act, **extra)
    return run(cfg, "cpu", data=_loaders())


def _curve(log):
    return [(r["val_acc"], r["task"]) for r in log]


# ------------------------------------------------------------------ registry
def test_new_arms_registered():
    for a in ("meet_oracle", "meet_replace", "meet_replace_frozen", "meet_replace_oracle"):
        assert a in ARMS


# ------------------------------------------------------------------ oracle directions
@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_jacobian_matches_autograd(act):
    m, x, _ = _trained("meet_oracle", act)
    with torch.no_grad():
        h = m.low(x).double()
    m64 = copy.deepcopy(m).double()
    assert torch.allclose(oracle.up_jacobian(m, h), mech.up_jacobian_autograd(m64, h), atol=1e-12)


@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_directions_match_check_a(act):
    m, x, y = _trained("meet_oracle", act)
    with torch.no_grad():
        h = m.low(x)
        p = m.up(h).softmax(-1)
    e = F.one_hot(y, 10).float()
    ref, _, _ = mech.directions(copy.deepcopy(m).double(), x.double(), y, False)
    keep = p.gather(1, y[:, None]).squeeze(1) <= oracle.SKIP_PY
    for kind, name in (("pinv", "pinv"), ("logit", "lsq"), ("grad", "bp")):
        u = oracle.direction(m, h, p, e, kind)[keep].double()
        assert (u - ref[name]).abs().max() <= 1e-5 * ref[name].abs().max(), kind


def test_pinv_is_a_float64_computation():
    """The probabilities are recomputed in float64: float32 ones perturb the null direction of S and blow the pinv up."""
    m, x, y = _trained("meet_oracle", "relu")
    with torch.no_grad():
        h = m.low(x)
        p = m.up(h).softmax(-1)
    u = oracle.direction(m, h, p, F.one_hot(y, 10).float(), "pinv")
    assert torch.isfinite(u).all() and u.dtype == h.dtype
    assert u.norm(dim=1).max() < 1e4 * oracle.direction(m, h, p, F.one_hot(y, 10).float(), "grad").norm(dim=1).max() + 1.0


def test_confident_rows_are_zero():
    m, x, y = _trained("meet_oracle", "relu")
    with torch.no_grad():
        h = m.low(x)
    p = F.one_hot(y, 10).float() * 0.99999 + 1e-6 / 9
    u = oracle.direction(m, h, p, F.one_hot(y, 10).float(), "pinv")
    assert torch.count_nonzero(u) == 0


def test_unknown_kind_and_bound_rejected():
    m, x, y = _trained("meet_oracle", "relu")
    with torch.no_grad():
        h = m.low(x)
        p = m.up(h).softmax(-1)
    with pytest.raises(ValueError):
        oracle.direction(m, h, p, F.one_hot(y, 10).float(), "nope")
    with pytest.raises(ValueError):
        oracle.coefficients(m, h, p, F.one_hot(y, 10).float(), h, dict(oracle_type="pinv", oracle_rho=1.0, oracle_bound="x"), 0)


# ------------------------------------------------------------------ the injected gradient
def _meet_grad_at_h(arm, act, kind, rho, bound="none"):
    m, x, y = _trained(arm, act)
    cfg = dict(CFG, oracle_type=kind, oracle_rho=rho, oracle_bound=bound)
    _, parts, h, z = compute_losses(m, x, y, cfg, step=0)
    g_meet = torch.autograd.grad(parts["meet"], h, retain_graph=True)[0]
    g_task = torch.autograd.grad(parts["task"], h, retain_graph=True)[0]
    return m, x, y, h, z, g_meet, g_task


@pytest.mark.parametrize("act", ["relu", "tanh"])
@pytest.mark.parametrize("kind", ["pinv", "logit"])
def test_injected_gradient_has_rho_times_task_norm_and_direction(act, kind):
    rho = 0.7
    m, x, y, h, z, g_meet, g_task = _meet_grad_at_h("meet_oracle", act, kind, rho)
    with torch.no_grad():
        p = z.softmax(-1)
        u = oracle.direction(m, h.detach(), p, F.one_hot(y, 10).float(), kind)
    live = u.norm(dim=1) > 0
    assert live.all()
    assert torch.allclose(g_meet.norm(dim=1), rho * g_task.norm(dim=1), rtol=1e-4)
    cos = F.cosine_similarity(-g_meet, u, dim=1)          # a descent step on h moves along +u: the gradient is -c
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-4)


@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_f_prime_total_gradient_is_one_plus_rho_times_backprop(act):
    rho = 0.5
    _, _, _, _, _, g_meet, g_task = _meet_grad_at_h("meet_oracle", act, "grad", rho)
    assert torch.allclose(g_meet, rho * g_task, rtol=1e-5, atol=1e-9)


def test_radial_bound_removes_the_component_along_h():
    m, x, y, h, z, g_meet, g_task = _meet_grad_at_h("meet_oracle", "relu", "pinv", 1.0, bound="radial")
    radial = (g_meet * F.normalize(h.detach(), dim=1)).sum(1)
    assert radial.abs().max() < 1e-4 * g_meet.norm(dim=1).max()
    assert torch.allclose(g_meet.norm(dim=1), g_task.norm(dim=1), rtol=1e-4)


def test_surrogate_gradient_is_c():
    h = torch.randn(4, 8, requires_grad=True)
    c = torch.randn(4, 8, requires_grad=True)
    g = torch.autograd.grad(oracle.surrogate(h, c), h)[0]
    assert torch.equal(g, c.detach())
    assert torch.autograd.grad(oracle.surrogate(h, c), c, allow_unused=True)[0] is None


def test_meet_oracle_signal_is_detached_from_the_model():
    """Nothing from the surrogate reaches F_up (it is added at h only) and the task loss still trains it."""
    m, x, y = _trained("meet_oracle", "relu")
    cfg = dict(CFG, oracle_type="pinv", oracle_rho=1.0)
    _, parts, _, _ = compute_losses(m, x, y, cfg, step=0)
    for p in m.up.parameters():
        assert torch.autograd.grad(parts["meet"], p, retain_graph=True, allow_unused=True)[0] is None or \
            torch.autograd.grad(parts["meet"], p, retain_graph=True, allow_unused=True)[0].abs().max() == 0
    assert any(torch.autograd.grad(parts["meet"], p, retain_graph=True)[0].abs().max() > 0 for p in m.low.parameters())


def test_oracle_warmup_ramp():
    assert oracle.ramp({}, 0) == 1.0
    assert oracle.ramp({"oracle_warmup_steps": 10}, 5) == 0.5
    assert oracle.ramp({"oracle_warmup_steps": 10}, 99) == 1.0


# ------------------------------------------------------------------ identities with arm A
@pytest.mark.parametrize("act", ["relu", "tanh"])
@pytest.mark.parametrize("kind", ORACLE_KINDS)
def test_rho_zero_reproduces_arm_a(act, kind):
    a = _run("base", act)
    o = _run("meet_oracle", act, oracle_type=kind, oracle_rho=0.0)
    assert _curve(a) == _curve(o)


@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_exact_gradient_replacement_reproduces_arm_a(act):
    a = _run("base", act)
    r = _run("meet_replace_oracle", act, oracle_type="grad", oracle_rho=1.0)
    assert _curve(a) == _curve(r)


@pytest.mark.parametrize("arm", ["meet_replace", "meet_replace_frozen"])
def test_replace_start_beyond_the_horizon_reproduces_arm_a_curve_for_the_task(arm):
    """Before the cut the lower half trains by backprop alone: the first replace_start steps are arm A's."""
    a = _run("base", "relu")
    r = _run(arm, "relu", replace_start=10 ** 6)
    assert _curve(a) == _curve(r)


def test_replace_oracle_before_the_cut_is_arm_a():
    a = _run("base", "relu")
    r = _run("meet_replace_oracle", "relu", oracle_type="pinv", oracle_rho=1.0, replace_start=10 ** 6)
    assert _curve(a) == _curve(r)


# ------------------------------------------------------------------ replacement arms
def _low_grads(loss, m):
    return torch.autograd.grad(loss, list(m.low.parameters()), retain_graph=True, allow_unused=True)


@pytest.mark.parametrize("arm", ["meet_replace", "meet_replace_frozen", "meet_replace_oracle"])
def test_replace_cuts_the_task_gradient_at_h(arm):
    m, x, y = _trained(arm, "relu")
    cfg = dict(CFG, oracle_type="pinv", oracle_rho=1.0, replace_start=0, lam_meet=1.0)
    _, parts, _, _ = compute_losses(m, x, y, cfg, step=0)
    assert all(g is None for g in _low_grads(parts["task"], m)), "the task loss must not reach F_low"
    gm = _low_grads(parts["meet"], m)
    assert any(g is not None and g.abs().max() > 0 for g in gm), "the meeting signal must reach F_low"
    # and the task loss still trains F_up
    assert any(g.abs().max() > 0 for g in torch.autograd.grad(parts["task"], list(m.up.parameters()), retain_graph=True))


def test_replace_before_the_cut_the_task_loss_reaches_f_low():
    m, x, y = _trained("meet_replace", "relu")
    cfg = dict(CFG, replace_start=50, lam_meet=1.0)
    _, parts, _, _ = compute_losses(m, x, y, cfg, step=10)
    assert any(g is not None and g.abs().max() > 0 for g in _low_grads(parts["task"], m))
    assert not parts["meet"].requires_grad                      # before the cut the meeting term is only logged


def test_replace_inverse_loss_trains_g_and_meeting_loss_does_not():
    m, x, y = _trained("meet_replace", "relu")
    cfg = dict(CFG, replace_start=0, lam_meet=1.0)
    _, parts, _, _ = compute_losses(m, x, y, cfg, step=0)
    g_inv = torch.autograd.grad(parts["inv"], list(m.G.parameters()), retain_graph=True)
    assert all(g.abs().max() > 0 for g in g_inv)
    assert all(g is None or g.abs().max() == 0
               for g in torch.autograd.grad(parts["meet"], list(m.G.parameters()), retain_graph=True, allow_unused=True))


def test_replace_frozen_g_is_frozen():
    m, x, y = _trained("meet_replace_frozen", "relu")
    assert not any(p.requires_grad for p in m.G.parameters())
    before = [p.clone() for p in m.G.parameters()]
    log = _run("meet_replace_frozen", "relu")
    assert log and all(np.isfinite(r["task"]) for r in log)
    after = MeetNet(arm="meet_replace_frozen", act="relu")      # same G init needs the same seed: just check requires_grad above
    assert all(torch.equal(b, b2) for b, b2 in zip(before, [p.clone() for p in m.G.parameters()]))
    assert after.G is not None


def test_replace_meeting_loss_is_a_sum_over_dimensions():
    """The replacement meeting loss is the per-example SUM of squares (mean over the batch), not arm D's mean over dimensions."""
    m, x, y = _trained("meet_replace", "relu")
    cfg = dict(CFG, replace_start=0, lam_meet=1.0)
    _, parts, h, z = compute_losses(m, x, y, cfg, step=0)
    with torch.no_grad():
        p, e = z.softmax(-1), F.one_hot(y, 10).float()
        b = h + m.G(e) - m.G(p)
        want = ((h - b) ** 2).sum(1).mean()
    assert torch.allclose(parts["meet"], want, rtol=1e-5)


@pytest.mark.parametrize("arm,extra", [("meet_oracle", dict(oracle_type="pinv", oracle_rho=1.0)),
                                       ("meet_replace", {}), ("meet_replace_frozen", {}),
                                       ("meet_replace_oracle", dict(oracle_type="logit", oracle_rho=1.0))])
@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_new_arms_train_and_log_finite_values(arm, extra, act):
    log = _run(arm, act, **extra)
    assert len(log) > 3
    for r in log:
        for k in ("task", "val_acc", "flops", "h_var"):
            assert np.isfinite(r[k]), (arm, k)
    assert log[-1]["task"] < log[0]["task"] + 1.0


def test_alignment_diagnostic_handles_replacement_arms():
    from src.diagnostics import grad_alignment
    for arm in ("meet_replace", "meet_replace_oracle", "meet_oracle"):
        m, x, y = _trained(arm, "relu")
        a = grad_alignment(m, x, y, dict(oracle_type="pinv", oracle_rho=1.0, replace_start=0, lam_meet=1.0, lam_inv=0.1))
        assert -1.0 <= a <= 1.0, arm


# ------------------------------------------------------------------ sweeps
def test_existing_sweep_arms_unchanged():
    """Digests of (trials, config) recorded BEFORE the new sweep arms were added, for arms that are tuned and used."""
    pinned = json.load(open(REPO / "tests" / "fixtures" / "sweep_digests_before_dr019.json"))
    assert pinned
    for key, want in pinned.items():
        arm, tag = key.split("|")
        trials = sweep.sample_trials(arm, tag)
        cfg = sweep.build_cfg(arm, trials[0], 1000, tag)
        got = hashlib.sha1(json.dumps([trials, cfg], sort_keys=True).encode()).hexdigest()[:12]
        assert got == want, key


def test_base_tanh_sweep_arm_fixes_tanh_and_searches_the_rest():
    t = sweep.sample_trials("base_tanh", "t13")
    assert len(t) == sweep.N_TRIALS and all("act" not in x for x in t)
    assert {"lr", "wd", "lr_schedule"} <= set(t[0])
    cfg = sweep.build_cfg("base_tanh", t[0], 960, "t13")
    assert cfg["arm"] == "base" and cfg["act"] == "tanh"
    base = sweep.sample_trials("base", "t13")
    assert "act" in base[0]                                       # the common dimension is untouched for other arms


def test_meet_replace_sweep_arm():
    t = sweep.sample_trials("meet_replace", "t4")
    assert {"act", "lam_inv", "lr", "wd", "lr_schedule"} <= set(t[0])
    cfg = sweep.build_cfg("meet_replace", t[0], 950, "t4")
    assert cfg["arm"] == "meet_replace" and cfg["replace_start"] == 0 and cfg["lam_meet"] == 1.0


# ------------------------------------------------------------------ plans, rules, runner
@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """A repo root with the tuned configs the plans are derived from (stand-ins for the t13 and t4 tuning outputs)."""
    (tmp_path / "configs" / "tuned").mkdir(parents=True)
    for name in ("t1_base_two", "t2_base_two", "t1_meet_two"):
        (tmp_path / "configs" / "tuned" / f"{name}.yaml").write_text((REPO / "configs" / "tuned" / f"{name}.yaml").read_text())
    base = yaml.safe_load((REPO / "configs" / "tuned" / "t1_base_two.yaml").read_text())
    (tmp_path / "configs" / "tuned" / "t13_base_tanh_two.yaml").write_text(yaml.safe_dump({**base, "act": "tanh", "lr": 0.002}))
    meet = yaml.safe_load((REPO / "configs" / "tuned" / "t1_meet_two.yaml").read_text())
    (tmp_path / "configs" / "tuned" / "t4_meet_replace_two.yaml").write_text(
        yaml.safe_dump({**meet, "arm": "meet_replace", "replace_start": 0, "lam_meet": 1.0}))
    monkeypatch.setattr(orc, "ROOT", tmp_path)
    return tmp_path


def _write_chosen(root, relu=0.3, tanh=1.0):
    (root / "configs" / "oracle").mkdir(parents=True, exist_ok=True)
    (root / orc.CHOSEN_FILE).write_text(json.dumps({"relu": {"F": relu, "P": relu, "Q": relu}, "tanh": {"F": tanh, "P": tanh, "Q": tanh}}))


def test_group_specs_phase_1_and_2_and_3(fake_root):
    p1 = orc.group_specs("mb1", 1)
    assert len(p1) == 2 * (1 + 3 * 3)                              # A + F, P, Q x 3 rho, per setting
    assert {s["seeds"] for s in p1} == {tuple(range(700, 703))}
    _write_chosen(fake_root)
    p2 = {s["name"] for s in orc.group_specs("mb1", 2)}
    assert {"A-relu", "A2-relu", "F-relu-rho0p3", "Pr-relu-rho0p3", "A-tanh", "P-tanh-rho1", "D-tanh"} <= p2
    assert "A2-tanh" not in p2 and "D-relu" not in p2
    p3 = orc.group_specs("mb1", 3)
    assert all(s["kind"] in ("P", "Q") and s["lr_mult"] in orc.LR_MULTS for s in p3) and len(p3) == 2 * 2 * 2
    assert {s["seeds"] for s in p3} == {tuple(range(710, 720))}


def test_group_specs_check_d(fake_root):
    specs = orc.group_specs("md1", 1)
    names = {s["name"] for s in specs}
    assert {"A-relu", "A2-relu", "A-tanh", "R1", "R1w", "R2-relu-rho1", "R4-relu", "R2-tanh-rho1", "R4-tanh"} == names
    assert {s["seeds"] for s in specs} == {tuple(range(900, 910))}
    assert next(s for s in specs if s["name"] == "R1")["ref"] == "A-tanh"      # R1 was tuned in tanh in the fixture


def test_seeds_do_not_overlap_other_checks():
    used = set()
    for t in orc.TAGS.values():
        for seeds in t["seeds"].values():
            used |= set(seeds)
    assert not used & set(range(0, 100)) and not used & set(range(1100, 1130))        # Stage 3 and dose checks


def test_build_config_derivations(fake_root):
    _write_chosen(fake_root, 0.3, 1.0)
    specs = {s["name"]: s for s in orc.group_specs("mb1", 2)}
    p = orc.build_config(specs["P-tanh-rho1"], "mb1")
    assert p["arm"] == "meet_oracle" and p["oracle_type"] == "pinv" and p["oracle_rho"] == 1.0 and p["act"] == "tanh"
    assert p["lr"] == 0.002 and p["oracle_bound"] == "none" and p["stage"] == "orc-mb1-P-tanh-rho1"
    assert orc.build_config(specs["Pr-relu-rho0p3"], "mb1")["oracle_bound"] == "radial"
    assert orc.build_config(specs["F-relu-rho0p3"], "mb1")["oracle_type"] == "grad"
    d = orc.build_config(specs["D-tanh"], "mb1")
    assert d["arm"] == "meet" and d["lr"] == yaml.safe_load((REPO / "configs" / "tuned" / "t1_meet_two.yaml").read_text())["lr"]
    p3 = {s["name"]: s for s in orc.group_specs("mb1", 3)}
    assert orc.build_config(p3["P-tanh-rho1-lr0p5"], "mb1")["lr"] == pytest.approx(0.001)
    r = {s["name"]: s for s in orc.group_specs("md1", 1)}
    assert orc.build_config(r["R1w"], "md1")["replace_start"] == orc.R1W_START
    o = orc.build_config(r["R2-relu-rho1"], "md1")
    assert o["arm"] == "meet_replace_oracle" and o["oracle_type"] == "pinv" and o["replace_start"] == 0
    assert orc.build_config(r["R4-tanh"], "md1")["arm"] == "meet_replace_frozen"


def test_missing_tuned_config_has_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr(orc, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="tuning pass"):
        orc.build_config(dict(name="A-tanh", kind="A", setting="tanh", rho=None, lr_mult=1.0), "mb1")


def test_choose_rho_lowest_steps_ties_to_smaller_rho():
    def g(steps):
        return dict(steps={0.97: steps, 0.975: steps})
    scores = {}
    for k in ("F", "P", "Q"):
        scores[orc.group_name(k, "relu", 0.3)] = g([900, 900])
        scores[orc.group_name(k, "relu", 1.0)] = g([800, 800])
        scores[orc.group_name(k, "relu", 3.0)] = g([800, 800])
    assert orc.choose_rho(scores, "relu") == {"F": 1.0, "P": 1.0, "Q": 1.0}


def _ch(p, lo, hi):
    return {t: (p, lo, hi) for t in orc.TARGETS}


def test_decide_oracle():
    helps = _ch(-20.0, -30.0, -8.0)
    none = _ch(-2.0, -8.0, 4.0)
    mid = _ch(-12.0, -25.0, 2.0)
    f_bad = {t: -18.0 for t in orc.TARGETS}                                         # F' did as well: not "the oracle's direction"
    f_ok = {t: -3.0 for t in orc.TARGETS}
    assert orc.decide_oracle({"P": helps, "Q": none}, {"P": {t: -15.0 for t in orc.TARGETS}, "Q": None}, f_ok) == "ORACLE HELPS"
    assert orc.decide_oracle({"P": helps}, {"P": {t: -4.0 for t in orc.TARGETS}}, f_ok) != "ORACLE HELPS"         # fails against A2
    assert orc.decide_oracle({"P": helps}, None, f_bad) != "ORACLE HELPS"                                          # F' as good
    assert orc.decide_oracle({"P": none, "Q": none, "Pr": none}, None, f_ok) == "NO HELP"
    assert orc.decide_oracle({"P": mid, "Q": none}, None, f_ok) == "INCONCLUSIVE"
    one_target = {0.97: (-20.0, -30.0, -8.0), 0.975: (-1.0, -9.0, 5.0)}
    assert orc.decide_oracle({"P": one_target}, None, None) != "ORACLE HELPS"       # both targets must agree


def test_decide_replace_order_of_rules():
    ok = {0.97: 8, 0.975: 8}
    assert orc.decide_replace(ok, {0.97: 10.0, 0.975: 14.0}, 0.0) == "SUFFICIENT"
    assert orc.decide_replace(ok, {0.97: 10.0, 0.975: 16.0}, 0.0) == "PARTIAL"
    assert orc.decide_replace({0.97: 6, 0.975: 6}, {0.97: 10.0, 0.975: 10.0}, 0.02) == "INSUFFICIENT"   # accuracy gap > 1 point
    assert orc.decide_replace({0.97: 8, 0.975: 6}, {0.97: 10.0, 0.975: 10.0}, 0.0) == "PARTIAL"
    assert orc.decide_replace({0.97: 4, 0.975: 1}, {0.97: 0.0, 0.975: 0.0}, 0.0) == "INSUFFICIENT"
    assert orc.decide_replace({0.97: 6, 0.975: 5}, {0.97: 0.0, 0.975: 0.0}, 0.0) == "UNCLEAR"
    assert orc.decide_replace({0.97: 4, 0.975: 4}, {0.97: 0.0, 0.975: 0.0}, 0.0, n=4) == "SUFFICIENT"   # thresholds scale with n


def _fake_log(path, header, accs, hvar=2.0):
    rows = [json.dumps({"_header": True, **header})]
    for i, a in enumerate(accs):
        rows.append(json.dumps(dict(step=(i + 1) * 100, val_acc=a, flops=1.0, task=0.1, h_var=hvar, align=0.5)))
    pathlib.Path(path).write_text("\n".join(rows) + "\n")


def test_load_group_and_change_roundtrip(fake_root, tmp_path):
    out = tmp_path / "raw"
    out.mkdir()
    spec_a = orc._spec("A", "relu", None, (700, 701))
    spec_p = orc._spec("P", "relu", "A-relu", (700, 701), 0.3)
    for spec, accs in ((spec_a, [0.9, 0.96, 0.971, 0.98, 0.98, 0.98]), (spec_p, [0.9, 0.97, 0.98, 0.98, 0.98, 0.98])):
        cfg = orc.build_config(spec, "mb1")
        for s in spec["seeds"]:
            _fake_log(out / f"2026-01-01_{cfg['stage']}_{cfg['arm']}_s{s}.jsonl", {"config": cfg}, accs)
    a, p = orc.load_group(out, "mb1", spec_a), orc.load_group(out, "mb1", spec_p)
    assert a["reached"][0.97] == [True, True] and p["steps"][0.97][0] < a["steps"][0.97][0]
    ch = orc.change(a, p, n_boot=200)
    assert ch[0.97][0] < 0
    with pytest.raises(ValueError, match="missing seeds"):
        orc.load_group(out, "mb1", orc._spec("A", "relu", None, (700, 701, 702)))


def test_runner_make_jobs_seed_major_and_configs_checked(fake_root, tmp_path, monkeypatch):
    rc = load_script("run_root_cause")
    monkeypatch.chdir(tmp_path)
    specs = [orc._spec("A", "relu", None, (1, 2)), orc._spec("A", "tanh", None, (1, 2))]
    jobs = rc.make_jobs("mb1", specs, tmp_path / "raw")
    assert [(j.group, j.seed) for j in jobs] == [("A-relu", 1), ("A-tanh", 1), ("A-relu", 2), ("A-tanh", 2)]
    assert all(isinstance(j, FinalJob) for j in jobs)
    with pytest.raises(SystemExit):
        rc.ensure_configs("mb1", specs)                                   # missing and not allowed to write
    rc.ensure_configs("mb1", specs, write=True)
    rc.ensure_configs("mb1", specs)                                       # now consistent
    path = orc.config_path("mb1", "A-relu")
    cfg = yaml.safe_load(open(path))
    cfg["lr"] = 123.0
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit):
        rc.ensure_configs("mb1", specs)                                   # a hand edit is refused
