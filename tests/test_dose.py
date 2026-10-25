"""Check G0 (docs/notes/mechanism-dose-response.md): the arm `meet_rho`, the dose-response runner and its decision rule.

The four tests the note asks for first:
1. rho = 0 reproduces arm A's validation curve step for step (test_rho_zero_reproduces_arm_a)
2. the gradient at h of L_meet_rho has per-example norm rho * (task gradient norm) (test_gradient_at_h_has_rho_times_task_norm)
3. the meeting direction is detached: nothing from L_meet_rho reaches G or the upper layers (test_meet_gradient_reaches_only_low)
4. the FLOP counter's overhead equals the analytic count (test_flop_overhead_matches_analytic_count)
"""
import json
import math
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from exploratory import dose
from src.final import FinalJob, run_final_job
from src.flops import flops_per_step
from src.model import ARMS, MeetNet, compute_losses
from src.train import build_optimizer, run

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from scripts_loader import load_script  # noqa: E402

CFG = dict(lam_inv=0.1, rho=1.0, rho_warmup_steps=1, lr=1e-3, wd=1e-4, max_steps=100)


def batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 784, generator=g), torch.randint(0, 10, (n,), generator=g)


def _loaders(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 784, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True,
                       generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(x[:64], y[:64]), batch_size=64))


def test_arm_registered_and_requires_rho():
    assert "meet_rho" in ARMS
    m = MeetNet(arm="meet_rho")
    x, y = batch()
    with pytest.raises(KeyError):                       # no silent default dose
        compute_losses(m, x, y, {k: v for k, v in CFG.items() if k != "rho"}, step=10)


@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_rho_zero_reproduces_arm_a(act):
    """Test 1: same seed, same data order, G trained but never reaching F_low -> identical curves."""
    base = dict(CFG, seed=0, batch_size=32, max_steps=40, eval_every=5, lr_schedule="cosine", act=act)
    a = run(dict(base, arm="base"), device="cpu", data=_loaders())
    r = run(dict(base, arm="meet_rho", rho=0.0), device="cpu", data=_loaders())
    assert len(a) == len(r) == 8
    assert [x["val_acc"] for x in a] == [x["val_acc"] for x in r]
    assert [x["task"] for x in a] == [x["task"] for x in r]          # exact, not approximate
    assert all(math.isfinite(x["inv"]) for x in r)                    # G was trained meanwhile


def test_rho_positive_changes_the_run():
    base = dict(CFG, seed=0, batch_size=32, max_steps=40, eval_every=5, act="tanh")
    a = run(dict(base, arm="base"), device="cpu", data=_loaders())
    r = run(dict(base, arm="meet_rho", rho=3.0), device="cpu", data=_loaders())
    assert [x["task"] for x in a] != [x["task"] for x in r]


@pytest.mark.parametrize("rho", [0.3, 1.0, 3.0])
def test_gradient_at_h_has_rho_times_task_norm(rho):
    """Test 2: per example, ||d L_meet / d h_i|| = rho * ||d L_task / d h_i|| (mean-loss units), along the learned direction."""
    torch.manual_seed(1)
    m = MeetNet(arm="meet_rho", act="tanh")
    x, y = batch(16)
    cfg = dict(CFG, rho=rho, rho_warmup_steps=1)
    _, parts, h, _ = compute_losses(m, x, y, cfg, step=10)
    g_meet = torch.autograd.grad(parts["meet"], h, retain_graph=True)[0]
    g_task = torch.autograd.grad(parts["task"], h, retain_graph=True)[0]
    assert torch.allclose(g_meet.norm(dim=1), rho * g_task.norm(dim=1), rtol=1e-4, atol=1e-9)
    # direction: descent on h moves along u = G(e_y) - G(p), so the gradient is -rho * n * u / ||u||
    with torch.no_grad():
        p = m(x).softmax(-1)
        u = m.G(F.one_hot(y, 10).float()) - m.G(p)
        cos = F.cosine_similarity(-g_meet, u, dim=1)
    assert torch.all(cos > 0.9999)


def test_confident_rows_get_no_meeting_gradient():
    torch.manual_seed(1)
    m = MeetNet(arm="meet_rho")
    x, y = batch(16)
    y = torch.full_like(y, 7)                            # one class, so a constant logit vector is right for every row
    with torch.no_grad():
        m.up[3].bias.zero_()
        m.up[3].weight.mul_(1e-3)                        # small but not zero, so the task gradient at h is not exactly 0
        m.up[3].bias[7] = 30.0                           # every row is (numerically) certain and correct
    _, parts, h, z = compute_losses(m, x, y, CFG, step=10)
    assert torch.all(z.softmax(-1).gather(1, y[:, None]) > 0.9999)
    g_meet = torch.autograd.grad(parts["meet"], h, retain_graph=True)[0]
    assert torch.all(g_meet == 0)
    # a mixed batch: only the rows whose label is the certain class are masked
    y_mixed = y.clone()
    y_mixed[8:] = 3
    _, parts, h, z = compute_losses(m, x, y_mixed, CFG, step=10)
    g_meet = torch.autograd.grad(parts["meet"], h, retain_graph=True)[0]
    assert torch.all(g_meet[:8] == 0)
    assert torch.all(g_meet[8:].norm(dim=1) > 0)


def test_ramp_is_linear_and_zero_at_step_zero():
    torch.manual_seed(1)
    m = MeetNet(arm="meet_rho", act="tanh")
    x, y = batch(16)
    cfg = dict(CFG, rho=2.0, rho_warmup_steps=200)
    norms = []
    for step in (0, 50, 100, 200, 400):
        _, parts, h, _ = compute_losses(m, x, y, cfg, step=step)
        g_meet = torch.autograd.grad(parts["meet"], h, retain_graph=True)[0]
        g_task = torch.autograd.grad(parts["task"], h, retain_graph=True)[0]
        norms.append((g_meet.norm(dim=1) / g_task.norm(dim=1)).mean().item())
    assert norms[0] == 0.0
    assert norms[1] == pytest.approx(0.5, rel=1e-3) and norms[2] == pytest.approx(1.0, rel=1e-3)
    assert norms[3] == pytest.approx(2.0, rel=1e-3) and norms[4] == pytest.approx(2.0, rel=1e-3)


def test_meet_gradient_reaches_only_low():
    """Test 3: the direction and the per-example scale are detached; G's only loss is L_inv."""
    m = MeetNet(arm="meet_rho")
    x, y = batch()
    _, parts, _, _ = compute_losses(m, x, y, CFG, step=10)
    parts["meet"].backward()
    assert all(p.grad is not None for p in m.low.parameters())
    assert all(p.grad is None for p in m.up.parameters())
    assert all(p.grad is None for p in m.G.parameters())
    m2 = MeetNet(arm="meet_rho")
    _, parts2, _, _ = compute_losses(m2, x, y, CFG, step=10)
    parts2["inv"].backward()                              # L_inv trains G only
    assert all(p.grad is not None for p in m2.G.parameters())
    assert all(p.grad is None for p in m2.low.parameters()) and all(p.grad is None for p in m2.up.parameters())


def test_flop_overhead_matches_analytic_count():
    """Test 4. Batch 128. Arm A: 155,058,176 FLOPs per step (F-001). Overhead of meet_rho, matmuls only:
    G forward on p and on e_y: 2 * 2*B*(10*256 + 256*128) = 18,087,936;
    G backward for L_inv (weight grads of both layers, input grad of the second): 2*B*(256*128 + 10*256) + 2*B*256*128 = 17,432,576;
    one extra backward of L_task to h (input grads of the two upper layers that carry h): 2*B*(256*10 + 128*256) = 9,043,968.
    Arm D has the first two (+22.9%, F-003); meet_rho adds the third (+28.7% in total)."""
    B = 128
    x, y = batch(B)
    def fps(arm, **extra):
        m = MeetNet(arm=arm)
        opt, _ = build_optimizer(m, CFG)
        return flops_per_step(m, opt, (x, y), dict(CFG, **extra))
    a = fps("base")
    assert a == 155_058_176
    g_fwd = 2 * 2 * B * (10 * 256 + 256 * 128)
    g_bwd = 2 * B * (256 * 128 + 10 * 256) + 2 * B * 256 * 128
    extra_grad = 2 * B * (256 * 10 + 128 * 256)
    assert (g_fwd, g_bwd, extra_grad) == (18_087_936, 17_432_576, 9_043_968)
    d = fps("meet", lam_meet=0.1, warmup_steps=1)
    assert d - a == g_fwd + g_bwd                         # arm D, as in F-003
    r = fps("meet_rho")
    assert r - a == g_fwd + g_bwd + extra_grad
    assert fps("meet_rho", rho=0.0) == r                  # the dose does not change the count


# ----------------------------------------------------------------------------- analysis and decision rule
def _p(point, lo=None, hi=None):
    lo = point - 3 if lo is None else lo
    hi = point + 3 if hi is None else hi
    return (point, lo, hi)


def _table(**by_rho):
    """by_rho: rho -> (point at 0.97, point at 0.975); intervals +-3 around the point."""
    return {float(k.replace("r", "").replace("_", ".")): {0.97: _p(v[0]), 0.975: _p(v[1])} for k, v in by_rho.items()}


def test_decide_helps_harms_flat_mixed():
    helps = _table(r0_1=(0, 0), r0_3=(-12, -11), r1=(-15, -14), r3=(-5, -4))
    helps[1.0][0.97] = (-15, -19, -8)
    assert dose.decide(helps) == "DOSE HELPS"
    # a point estimate beyond -10% but an interval that includes 0 is not enough
    weak = _table(r0_1=(0, 0), r0_3=(-12, -12), r1=(-5, -4), r3=(-5, -4))
    weak[0.3][0.97] = (-12, -25, 1)
    weak[0.3][0.975] = (-12, -25, 1)
    assert dose.decide(weak) == "MIXED"
    harms = _table(r0_1=(1, 1), r0_3=(8, 9), r1=(12, 14), r3=(25, 30))
    assert dose.decide(harms) == "DOSE HARMS"
    not_monotone = _table(r0_1=(1, 1), r0_3=(30, 30), r1=(12, 14), r3=(25, 30))     # decreases from 0.3 to 1
    assert dose.decide(not_monotone) == "MIXED"
    only_one_target = _table(r0_1=(1, 1), r0_3=(8, 9), r1=(12, 4), r3=(25, 30))
    assert dose.decide(only_one_target) == "MIXED"
    flat = _table(r0_1=(1, -2), r0_3=(2, 3), r1=(-3, 2), r3=(3, -1))
    assert dose.decide(flat) == "FLAT"
    not_flat = _table(r0_1=(1, -2), r0_3=(2, 3), r1=(-6, 2), r3=(3, -1))
    assert dose.decide(not_flat) == "MIXED"
    flat_but_excludes_zero = _table(r0_1=(1, 1), r0_3=(2, 3), r1=(3, 4), r3=(4, 3))
    flat_but_excludes_zero[3.0][0.97] = (4, 1, 7)         # within +-5 but the interval excludes 0
    assert dose.decide(flat_but_excludes_zero) == "MIXED"


def test_decide_overlap_of_helps_and_harms_is_mixed():
    both = _table(r0_1=(-12, -12), r0_3=(8, 9), r1=(12, 14), r3=(25, 30))
    both[0.1][0.97] = (-12, -20, -5)
    both[0.1][0.975] = (-12, -20, -5)
    assert dose.decide(both) == "MIXED"                    # pre-run clarification: report the curve


def _fake_log(seed, reach_step, align=0.7):
    rows = []
    for i, step in enumerate(range(50, 3001, 50)):
        acc = 0.99 if step >= reach_step else 0.90
        rows.append(dict(step=step, flops=step * 200_000, val_acc=acc, task=0.1, inv=0.2, meet=-0.01,
                         align=align, h_var=1.0, time=0.0))
    return rows


def _write_run(path, stage, arm, seed, rows, rho):
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(stage=stage, arm=arm, seed=seed, rho=rho, max_steps=3000)
    with open(path, "w") as f:
        f.write(json.dumps({"_header": True, "run_id": path.stem, "config": cfg, "config_hash": "x"}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_analyze_end_to_end_on_synthetic_logs(tmp_path):
    seeds = list(range(1100, 1110))
    steps = {0.0: 1000, 0.1: 1000, 0.3: 1000, 1.0: 1500, 3.0: 2000}      # harm grows with the dose
    for rho, reach in steps.items():
        for i, s in enumerate(seeds):
            stage = f"dose-mr1-rho{dose.rho_tag(rho)}"
            _write_run(tmp_path / f"20261025_{stage}_meet_rho_s{s}.jsonl", stage, "meet_rho", s,
                       _fake_log(s, reach + 50 * (i % 3)), rho)
    res = dose.analyze(tmp_path, "mr1", seeds, rhos=tuple(steps), n_boot=500)
    assert res["verdict"] == "DOSE HARMS"
    assert res["table"][3.0][0.97][0] > res["table"][1.0][0.97][0] > 0
    assert res["n_never"][3.0][0.975] == 0
    # a run that never reaches the target is charged the full run, not dropped
    stage = f"dose-mr1-rho{dose.rho_tag(3.0)}"
    (tmp_path / f"20261025_{stage}_meet_rho_s1100.jsonl").unlink()
    _write_run(tmp_path / f"20261025_{stage}_meet_rho_s1100.jsonl", stage, "meet_rho", 1100,
               _fake_log(1100, 10**6), 3.0)
    res2 = dose.analyze(tmp_path, "mr1", seeds, rhos=tuple(steps), n_boot=500)
    assert res2["n_never"][3.0][0.97] == 1


def test_analyze_refuses_incomplete_groups(tmp_path):
    seeds = [1100, 1101]
    stage = "dose-mr1-rho0"
    _write_run(tmp_path / f"20261025_{stage}_meet_rho_s1100.jsonl", stage, "meet_rho", 1100, _fake_log(1100, 1000), 0.0)
    with pytest.raises(ValueError, match="missing"):
        dose.analyze(tmp_path, "mr1", seeds, rhos=(0.0,), n_boot=50)


# ----------------------------------------------------------------------------- configs and runner
def test_configs_are_derived_from_arm_a_and_arm_d(tmp_path):
    cfg = dose.build_config(0.3, "mr1")
    a = yaml.safe_load(open(dose.ROOT / "configs" / "tuned" / "t1_base_two.yaml"))
    d = yaml.safe_load(open(dose.ROOT / "configs" / "tuned" / "t1_meet_two.yaml"))
    assert cfg["arm"] == "meet_rho" and cfg["rho"] == 0.3 and cfg["rho_warmup_steps"] == 200
    assert cfg["stage"] == "dose-mr1-rho0p3"
    for k in ("act", "lr", "wd", "batch_size", "max_steps", "eval_every", "lr_schedule", "low_lr_mult"):
        assert cfg[k] == a[k], k
    assert cfg["lam_inv"] == d["lam_inv"] and cfg["sigma_inv"] == 0.0
    assert cfg["max_steps"] == 3000 and cfg["act"] == "relu" and cfg["lr_schedule"] == "cosine"
    assert [dose.rho_tag(r) for r in dose.RHOS] == ["0", "0p1", "0p3", "1", "3"]


def test_committed_configs_match_the_derivation():
    for rho in dose.RHOS:
        path = dose.config_path("mr1", rho)
        assert path.exists(), path
        assert yaml.safe_load(open(path)) == {k: v for k, v in dose.build_config(rho, "mr1").items()}


def test_runner_writes_logs_and_skips_finished_runs(tmp_path):
    mod = load_script("run_dose_response")
    jobs = mod.make_jobs("mr1", [1100, 1101], dose.RHOS, tmp_path)
    assert len(jobs) == 10 and jobs[0].seed == 1100 and jobs[-1].seed == 1101       # seed-major
    calls = []
    def fake_runner(cfg, device):
        calls.append((cfg["stage"], cfg["seed"]))
        return _fake_log(cfg["seed"], 1000)
    assert [run_final_job(j, "cpu", fake_runner) for j in jobs[:3]] == ["done"] * 3
    assert run_final_job(jobs[0], "cpu", fake_runner) == "skipped" and len(calls) == 3
    first = next(tmp_path.glob("*rho0_meet_rho_s1100.jsonl"))
    header = json.loads(first.read_text().splitlines()[0])
    assert header["config"]["arm"] == "meet_rho" and "git_commit" in header and header["plan_group"] == "rho0"


# ----------------------------------------------------------------------------- checks G1 and G2 (docs/notes/mechanism-dose-follow-up.md)
def test_tag_table_for_g0_g1_g2():
    t1, t2, t3 = dose.tag_spec("mr1"), dose.tag_spec("mr2"), dose.tag_spec("mr3")
    assert (t1["act"], t1["rule"], tuple(t1["rhos"]), tuple(t1["seeds"])) == ("relu", "g0", dose.RHOS, tuple(range(1100, 1110)))
    assert (t2["act"], t2["rule"], tuple(t2["rhos"]), tuple(t2["seeds"])) == \
        ("relu", "low", (0.0, 0.001, 0.003, 0.01, 0.03), tuple(range(1110, 1120)))
    assert (t3["act"], t3["rule"], tuple(t3["rhos"]), tuple(t3["seeds"])) == ("tanh", "g0", dose.RHOS, tuple(range(1120, 1130)))
    blocks = [set(t["seeds"]) for t in (t1, t2, t3)]
    assert not (blocks[0] & blocks[1]) and not (blocks[0] & blocks[2]) and not (blocks[1] & blocks[2])
    assert all(min(b) >= 1100 and max(b) < 1200 for b in blocks)                  # final seeds 0-99 and track S (1200+) untouched
    assert dose.rho_tag(0.001) == "0p001" and dose.rho_tag(0.03) == "0p03"
    assert dose.tag_spec("smoke-x")["rule"] == "g0"                                # unknown tags behave like G0 (plumbing only)


def test_g2_config_changes_only_the_activation_and_the_stage():
    c1, c3 = dose.build_config(0.3, "mr1"), dose.build_config(0.3, "mr3")
    assert c3["act"] == "tanh" and c1["act"] == "relu"
    diff = {k for k in set(c1) | set(c3) if c1.get(k) != c3.get(k)}
    assert diff == {"act", "stage"}
    assert c3["stage"] == "dose-mr3-rho0p3"
    c2 = dose.build_config(0.001, "mr2")
    assert c2["act"] == "relu" and c2["rho"] == 0.001 and c2["stage"] == "dose-mr2-rho0p001"


@pytest.mark.parametrize("tag", ["mr1", "mr2", "mr3"])
def test_committed_configs_match_the_derivation_for_every_tag(tag):
    for rho in dose.tag_spec(tag)["rhos"]:
        path = dose.config_path(tag, rho)
        assert path.exists(), path
        assert yaml.safe_load(open(path)) == dose.build_config(rho, tag)


def _lt(**by_rho):
    return {float(k.replace("r", "").replace("_", ".")): {0.97: _p(v[0]), 0.975: _p(v[1])} for k, v in by_rho.items()}


def test_decide_low_dose_rule():
    helps = _lt(r0_001=(-12, -11), r0_003=(-3, -2), r0_01=(1, 2), r0_03=(2, 3))
    helps[0.001][0.97] = (-12, -18, -6)
    helps[0.001][0.975] = (-11, -17, -5)
    assert dose.decide_low(helps) == "LOW DOSE HELPS"
    weak = _lt(r0_001=(-12, -11), r0_003=(-3, -2), r0_01=(1, 2), r0_03=(2, 3))     # intervals +-3 around -12 are below 0 ...
    weak[0.001][0.97] = (-12, -25, 2)                                                   # ... unless made to include 0
    assert dose.decide_low(weak) == "MIXED"
    harms = _lt(r0_001=(1, 2), r0_003=(4, 5), r0_01=(12, 14), r0_03=(25, 30))
    assert dose.decide_low(harms) == "LOW DOSE HARMS"
    not_monotone = _lt(r0_001=(1, 2), r0_003=(20, 21), r0_01=(12, 14), r0_03=(25, 30))
    assert dose.decide_low(not_monotone) == "MIXED"
    one_target = _lt(r0_001=(1, 2), r0_003=(4, 5), r0_01=(12, 14), r0_03=(25, 4))
    assert dose.decide_low(one_target) == "MIXED"
    flat = _lt(r0_001=(1, -2), r0_003=(2, 3), r0_01=(-3, 2), r0_03=(3, -1))
    assert dose.decide_low(flat) == "FLAT"
    both = _lt(r0_001=(-12, -11), r0_003=(4, 5), r0_01=(12, 14), r0_03=(25, 30))
    both[0.001][0.97] = (-12, -18, -6)
    both[0.001][0.975] = (-11, -17, -5)
    assert dose.decide_low(both) == "MIXED"


def test_analyze_uses_the_rule_of_the_tag(tmp_path):
    seeds = list(range(1110, 1120))
    spec = dose.tag_spec("mr2")
    reach = {0.0: 1000, 0.001: 1000, 0.003: 1000, 0.01: 1500, 0.03: 2200}              # harm grows with the dose
    for rho, r in reach.items():
        stage = f"dose-mr2-rho{dose.rho_tag(rho)}"
        for i, s in enumerate(seeds):
            _write_run(tmp_path / f"20261026_{stage}_meet_rho_s{s}.jsonl", stage, "meet_rho", s,
                       _fake_log(s, r + 50 * (i % 3)), rho)
    res = dose.analyze(tmp_path, "mr2", n_boot=500)                                     # seeds and doses from the tag table
    assert res["seeds"] == list(spec["seeds"]) and res["rhos"] == list(spec["rhos"])
    assert res["verdict"] == "LOW DOSE HARMS"


def test_runner_refuses_unknown_tags_for_real_runs(monkeypatch, tmp_path):
    mod = load_script("run_dose_response")
    monkeypatch.setattr(sys, "argv", ["run_dose_response.py", "--tag", "zz9", "--dry-run"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "tag" in str(e.value)
