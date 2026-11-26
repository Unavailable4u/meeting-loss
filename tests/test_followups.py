"""Follow-ups of the root-cause checks (D-029, docs/notes/mechanism-root-cause-2.md): the radially bounded replacement, the sweep arm
`meet_replace_rb`, the plans of mb2 (check B in tanh) and md2 (bounded replacement in relu), and their reading rules.

The tests the note asks for first:
1. `replace_bound` unset or "none" changes nothing (test_unbounded_replacement_is_unchanged)
2. with `replace_bound: radial` the meeting gradient at h has no component along h and equals the unbounded one minus it (test_radial_*)
3. the plans, derived configs and seeds of mb2 and md2 (test_plan_*, test_build_config_*)
4. the reading rules (test_diagnose_bounded, test_analyze_b2)
"""
import json
import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_oracle import CFG, _curve, _run, _trained, fake_root  # noqa: E402,F401  (fixtures are imported by name)

from exploratory import oracle as orc  # noqa: E402
from src import oracle, sweep  # noqa: E402
from src.model import compute_losses  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ the bound in the replacement arms
@pytest.mark.parametrize("arm", ["meet_replace", "meet_replace_frozen"])
def test_unbounded_replacement_is_unchanged(arm):
    a = _run(arm, "relu")
    b = _run(arm, "relu", replace_bound="none")
    assert _curve(a) == _curve(b)


@pytest.mark.parametrize("arm", ["meet_replace", "meet_replace_frozen"])
@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_radial_replacement_gradient_is_the_unbounded_one_without_its_radial_part(arm, act):
    m, x, y = _trained(arm, act)
    cfg = dict(CFG, replace_start=0, lam_meet=1.0)
    grads = {}
    for bound in ("none", "radial"):
        total, parts, h, _ = compute_losses(m, x, y, dict(cfg, replace_bound=bound), step=0)
        grads[bound] = torch.autograd.grad(total, h, retain_graph=True)[0]          # the task loss does not reach h: meeting term only
        grads[bound + "_h"] = h.detach()
    assert grads["none_h"].equal(grads["radial_h"])
    h = grads["none_h"]
    h_hat = h / h.norm(dim=1, keepdim=True)
    g_none, g_rad = grads["none"], grads["radial"]
    along = (g_rad * h_hat).sum(1)
    assert along.abs().max() < 1e-5 * g_none.norm(dim=1).max()
    expect = g_none - (g_none * h_hat).sum(1, keepdim=True) * h_hat
    assert torch.allclose(g_rad, expect, atol=1e-6, rtol=1e-4)
    assert (g_none * h_hat).sum(1).abs().max() > 1e-6, "the test needs a signal that has a radial part to remove"


def test_radial_replacement_meeting_part_carries_the_training_gradient():
    """parts['meet'] is what the alignment diagnostic differentiates: its gradient on F_low must be the one the total sends there."""
    m, x, y = _trained("meet_replace", "relu")
    cfg = dict(CFG, replace_start=0, lam_meet=0.5, replace_bound="radial")
    total, parts, _, _ = compute_losses(m, x, y, cfg, step=0)
    low = list(m.low.parameters())
    g_total = torch.autograd.grad(total, low, retain_graph=True)
    g_meet = torch.autograd.grad(parts["meet"], low, retain_graph=True)
    for a, b in zip(g_total, g_meet):
        assert torch.allclose(a, 0.5 * b, atol=1e-6, rtol=1e-4)


def test_radial_replacement_trains_and_stays_finite():
    log = _run("meet_replace", "relu", replace_bound="radial", replace_start=0, lam_meet=1.0)
    assert all(np.isfinite(r["task"]) and np.isfinite(r["val_acc"]) for r in log)


def test_unknown_replace_bound_is_refused():
    m, x, y = _trained("meet_replace", "relu")
    with pytest.raises(ValueError, match="replace_bound"):
        compute_losses(m, x, y, dict(CFG, replace_start=0, replace_bound="sphere"), step=0)


def test_remove_radial_is_orthogonal_and_keeps_zero_rows_zero():
    v, h = torch.randn(5, 8), torch.randn(5, 8)
    v[2] = 0
    out = oracle.remove_radial(v, h)
    assert (out * h).sum(1).abs().max() < 1e-5 and out[2].abs().max() == 0


# ------------------------------------------------------------------ sweep arm
def test_meet_replace_rb_sweep_arm():
    t = sweep.sample_trials("meet_replace_rb", "t15")
    assert len(t) == sweep.N_TRIALS and all("act" not in x for x in t) and all("lam_inv" in x for x in t)
    cfg = sweep.build_cfg("meet_replace_rb", t[0], 970, "t15")
    assert cfg["arm"] == "meet_replace" and cfg["act"] == "relu" and cfg["replace_bound"] == "radial"
    assert cfg["replace_start"] == 0 and cfg["lam_meet"] == 1.0


def test_t4_sweep_arm_is_unchanged_by_the_follow_ups():
    cfg = sweep.build_cfg("meet_replace", sweep.sample_trials("meet_replace", "t4")[0], 4000, "t4")
    assert "replace_bound" not in cfg


# ------------------------------------------------------------------ plans
@pytest.fixture
def fake_root2(fake_root):
    """fake_root plus stand-ins for the t14 and t15 tuning outputs and the chosen rho of mb1."""
    tuned = fake_root / "configs" / "tuned"
    base = yaml.safe_load((tuned / "t13_base_tanh_two.yaml").read_text())
    (tuned / "t14_base_tanh_two.yaml").write_text(yaml.safe_dump({**base, "lr": 0.003, "wd": 1e-5}))
    meet = yaml.safe_load((tuned / "t4_meet_replace_two.yaml").read_text())
    (tuned / "t15_meet_replace_rb_two.yaml").write_text(yaml.safe_dump({**meet, "act": "relu", "replace_bound": "radial", "lr": 0.004}))
    (fake_root / "configs" / "oracle").mkdir(parents=True, exist_ok=True)
    chosen = {"relu": {"F": 0.3, "P": 0.3, "Q": 0.3}, "tanh": {"F": 1.0, "P": 0.3, "Q": 0.3}}     # the values of the real mb1 phase 1
    (fake_root / orc.CHOSEN_FILE).write_text(json.dumps(chosen))
    return fake_root


def test_tags_registered_with_fresh_seeds():
    assert orc.TAGS["mb2"]["seeds"][1] == tuple(range(720, 730)) and orc.TAGS["md2"]["seeds"][1] == tuple(range(930, 940))
    seen = {}
    for tag, t in orc.TAGS.items():
        for seeds in t["seeds"].values():
            for s in seeds:
                assert seen.setdefault(s, tag) == tag or tag == "mb1" and seen[s] == "mb1", f"seed {s} used by {seen[s]} and {tag}"
    assert orc.single_phase("mb2") and orc.single_phase("md2") and orc.single_phase("md1") and not orc.single_phase("mb1")


def test_plan_mb2(fake_root2):
    specs = {s["name"]: s for s in orc.group_specs("mb2", 1)}
    assert set(specs) == {"A-tanh", "A-tanh-lr0p5", "A2-tanh", "F-tanh-rho1", "Q-tanh-rho0p3", "Q-tanh-rho0p3-lr0p5"}
    assert specs["Q-tanh-rho0p3-lr0p5"]["ref"] == "A-tanh-lr0p5" and specs["Q-tanh-rho0p3"]["ref"] == "A-tanh"
    assert specs["A-tanh-lr0p5"]["ref"] == "A-tanh" and specs["A2-tanh"]["ref"] == "A-tanh" and specs["A-tanh"]["ref"] is None
    assert {s["seeds"] for s in specs.values()} == {tuple(range(720, 730))}
    assert orc.group_specs("mb2", 1, ("relu",)) == orc.group_specs("mb2", 1)         # tanh only, whatever --settings says


def test_plan_md2(fake_root2):
    specs = {s["name"]: s for s in orc.group_specs("md2", 1)}
    assert set(specs) == {"A-relu", "R1b-relu", "R4b-relu", "R2q-relu-rho1", "R2qb-relu-rho1", "R2pb-relu-rho1"}
    assert all(s["ref"] == "A-relu" for n, s in specs.items() if n != "A-relu")
    assert {s["seeds"] for s in specs.values()} == {tuple(range(930, 940))}


def test_build_config_mb2(fake_root2):
    specs = {s["name"]: s for s in orc.group_specs("mb2", 1)}
    a = orc.build_config(specs["A-tanh"], "mb2")
    half = orc.build_config(specs["A-tanh-lr0p5"], "mb2")
    assert half["lr"] == pytest.approx(a["lr"] * 0.5) and {k: v for k, v in half.items() if k not in ("lr", "stage")} == \
        {k: v for k, v in a.items() if k not in ("lr", "stage")}
    a2 = orc.build_config(specs["A2-tanh"], "mb2")
    assert a2["lr"] == 0.003 and a2["wd"] == 1e-5 and a2["act"] == "tanh"
    q, qh = orc.build_config(specs["Q-tanh-rho0p3"], "mb2"), orc.build_config(specs["Q-tanh-rho0p3-lr0p5"], "mb2")
    assert q["arm"] == "meet_oracle" and q["oracle_type"] == "logit" and q["oracle_rho"] == 0.3 and q["lr"] == a["lr"]
    assert qh["lr"] == pytest.approx(a["lr"] * 0.5)
    f = orc.build_config(specs["F-tanh-rho1"], "mb2")
    assert f["oracle_type"] == "grad" and f["oracle_rho"] == 1.0
    assert q["stage"] == "orc-mb2-Q-tanh-rho0p3"


def test_build_config_md2(fake_root2):
    specs = {s["name"]: s for s in orc.group_specs("md2", 1)}
    c = {n: orc.build_config(s, "md2") for n, s in specs.items()}
    assert c["A-relu"]["arm"] == "base" and c["A-relu"]["act"] == "relu"
    assert c["R1b-relu"]["arm"] == "meet_replace" and c["R1b-relu"]["replace_bound"] == "radial" and c["R1b-relu"]["lr"] == 0.004
    assert c["R4b-relu"]["arm"] == "meet_replace_frozen" and c["R4b-relu"]["replace_bound"] == "radial"
    assert c["R4b-relu"]["lr"] == c["A-relu"]["lr"] and c["R4b-relu"]["act"] == "relu"
    q, qb, pb = c["R2q-relu-rho1"], c["R2qb-relu-rho1"], c["R2pb-relu-rho1"]
    assert all(x["arm"] == "meet_replace_oracle" and x["oracle_rho"] == 1.0 and x["replace_start"] == 0 for x in (q, qb, pb))
    assert (q["oracle_type"], q["oracle_bound"]) == ("logit", "none") and (qb["oracle_type"], qb["oracle_bound"]) == ("logit", "radial")
    assert (pb["oracle_type"], pb["oracle_bound"]) == ("pinv", "radial")


def test_r1b_refuses_a_config_without_the_bound(fake_root2):
    path = fake_root2 / "configs" / "tuned" / "t15_meet_replace_rb_two.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg.pop("replace_bound")
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="replace_bound radial"):
        orc.build_config(orc._spec("R1b", "relu", "A-relu", (930,)), "md2")


def test_existing_plans_are_unchanged_by_the_follow_ups(fake_root2):
    """mb1 and md1 derive the same groups and configs as before the follow-ups (their config files are committed and checked)."""
    assert len(orc.group_specs("mb1", 1)) == 2 * (1 + 3 * 3)
    a = orc.build_config(orc._spec("A", "tanh", None, (700,)), "mb1")
    assert a["lr"] == 0.002 and "replace_bound" not in a
    assert orc.build_config(orc._spec("A2", "relu", None, (710,)), "mb1")["lr"] == \
        yaml.safe_load((REPO / "configs" / "tuned" / "t2_base_two.yaml").read_text())["lr"]


# ------------------------------------------------------------------ reading rules
def test_diagnose_bounded_readings():
    v = dict(R1b="INSUFFICIENT", R4b="INSUFFICIENT", R2q="INSUFFICIENT", R2qb="INSUFFICIENT", R2pb="INSUFFICIENT")
    assert "does not rescue" in orc.diagnose_bounded(v)[0] and "Neither" in orc.diagnose_bounded(v)[1]
    assert "same verdict" in orc.diagnose_bounded(v)[2]
    v.update(R2qb="PARTIAL", R4b="PARTIAL")
    d = orc.diagnose_bounded(v)
    assert d[0].startswith("The bound rescues") and "at least 0.97" in d[1] and "differs" in d[2]
    v.update(R2q="SUFFICIENT")
    assert "already trains" in orc.diagnose_bounded(v)[0]
    v.update(R2q="UNCLEAR", R2qb="UNCLEAR")
    assert "unclear" in orc.diagnose_bounded(v)[0]


def _scores(steps97, steps975=None, reached=True, final=0.98):
    n = len(steps97)
    steps975 = steps975 or steps97
    return dict(seeds=list(range(n)), steps={0.97: list(steps97), 0.975: list(steps975)},
                reached={0.97: [reached] * n, 0.975: [reached] * n}, final=[final] * n, h_var=3.0, align=0.5)


def test_analyze_b2_verdicts(fake_root2):
    specs = orc.group_specs("mb2", 1)
    base = [1000, 1100, 1050, 1000, 1200, 1100, 1000, 1050, 1150, 1100]
    scores = {"A-tanh": _scores(base), "A-tanh-lr0p5": _scores([int(b * 1.1) for b in base]), "A2-tanh": _scores([int(b * 1.2) for b in base]),
              "F-tanh-rho1": _scores(base), "Q-tanh-rho0p3": _scores(base), "Q-tanh-rho0p3-lr0p5": _scores([int(b * 1.1) for b in base])}
    r = orc.analyze_b2(specs, scores, n_boot=500)
    assert r["verdict"] == "NO HELP" and r["q_matched"][0.97][0] == pytest.approx(0.0, abs=0.5)
    assert r["lr_effect"][0.97][0] == pytest.approx(10.0, abs=0.5) and r["a2"][0.97][0] == pytest.approx(20.0, abs=0.5)
    # Q at the halved lr is 15% faster than A at the halved lr (matched) at both targets: the oracle helps
    scores["Q-tanh-rho0p3-lr0p5"] = _scores([int(b * 1.1 * 0.8) for b in base])
    r = orc.analyze_b2(specs, scores, n_boot=500)
    assert r["verdict"] == "ORACLE HELPS" and r["q_matched"][0.97][0] == pytest.approx(-20.0, abs=0.5)
    assert "TANH ORACLE HELPS" in orc.report_b2(r)
    # a gain that is only the lr effect is not a help: Q at the halved lr is 10% faster than A, A at the halved lr is 10% faster too
    scores["A-tanh-lr0p5"] = _scores([int(b * 0.9) for b in base])
    scores["Q-tanh-rho0p3-lr0p5"] = _scores([int(b * 0.9) for b in base])
    r = orc.analyze_b2(specs, scores, n_boot=500)
    assert r["q_matched"][0.97][0] == pytest.approx(0.0, abs=0.5) and r["q_half_vs_a"][0.97][0] == pytest.approx(-10.0, abs=0.5)
    assert r["verdict"] == "NO HELP"                  # the unmatched comparison (-10%) is reported but does not decide: it is the lr effect
