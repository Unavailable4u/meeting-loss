"""Study 1b, the meeting network (docs/notes/meeting-network.md, D-030, DR-023): src/meetnet.py, the sweep arms, the plan and the
decision rules of exploratory/meetnet.py, and the runner.

The tests the note asks for first:
1. the equal-cost rule gives k = 1 for the 4-layer MNIST MLP and the mirrored backward chain ends at the meeting node (test_equal_cost_*, test_chain_shapes)
2. both chains receive the meeting gradient; the forward-continues head does NOT reach the forward chain (test_gradients_*)
3. noise code: zero at test time, random in training; multi-hot code: fixed and distinct (test_*_code)
4. every readout is logged and val_acc is the primary one (test_train_logs_readouts)
5. the group verdicts and readings follow the pre-written rules (test_decide_group, test_diagnose_study)
6. the plan, the derived configs and the runner (test_group_specs, test_build_config, test_runner_*)
"""
import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from exploratory import meetnet as mn
from src import meetnet, sweep
from src.flops import flops_per_step
from src.model import compute_losses
from src.train import build_optimizer, run

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scripts_loader import load_script  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def _xy(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 784, generator=g), torch.randint(0, 10, (n,), generator=g)


def _loaders(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(10, 784, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    x = proto[y] + 2.0 * torch.randn(n, 784, generator=g)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(x[:128], y[:128]), batch_size=128))


def _cfg(arm, **extra):
    return dict(seed=0, batch_size=32, max_steps=60, eval_every=10, lr=2e-3, wd=1e-4, arm=arm, act="relu", mn_k=1, **extra)


# ------------------------------------------------------------------ the equal-cost rule and the chains
def test_equal_cost_rule_for_the_mnist_mlp():
    assert meetnet.layer_macs() == [200704, 32768, 32768, 2560]
    assert meetnet.equal_cost_k() == 1
    assert meetnet.MeetingNet().k == 1                           # default k comes from the rule
    assert meetnet.equal_cost_k((10, 10, 10, 10, 10)) == 2       # equal layers: the balanced split, ties to the smaller k
    assert meetnet.equal_cost_k((8, 8, 8)) == 1


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("code,d_code", [("onehot", 10), ("noise", 18), ("multihot", 16)])
def test_chain_shapes(k, code, d_code):
    m = meetnet.MeetingNet("mn_match", k=k, code=code)
    d_k = meetnet.DIMS[k]
    x, _ = _xy()
    assert m.state(x).shape == (32, d_k)
    assert m.targets().shape == (10, d_k)                        # the backward chain ends at the same node
    assert len(m.fwd) == k and len(m.back) == 4 - k
    assert m.back[0].in_features == d_code
    assert [l.out_features for l in m.back] == list(meetnet.DIMS[::-1][1:5 - k])


def test_post_activation_state_is_nonnegative_in_relu_and_pre_activation_can_be_negative():
    x, _ = _xy()
    assert (meetnet.MeetingNet("mn_match", k=1).state(x) >= 0).all()
    assert (meetnet.MeetingNet("mn_match", k=1, post_act=False).state(x) < 0).any()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        meetnet.MeetingNet("nope")
    with pytest.raises(ValueError):
        meetnet.MeetingNet("mn_match", k=4)
    with pytest.raises(ValueError):
        meetnet.MeetingNet("mn_match", code="dense")


def test_cosine_logits_are_bounded_by_the_scale():
    m = meetnet.MeetingNet("mn_match")
    x, y = _xy()
    _, parts, s, z = compute_losses(m, x, y, {"mn_scale": 7.0}, 0)
    assert z.abs().max() <= 7.0 + 1e-4 and parts["meet"].item() > 0


# ------------------------------------------------------------------ gradients
def _grads(arm, **kw):
    torch.manual_seed(0)
    m = meetnet.MeetingNet(arm, **kw)
    x, y = _xy()
    loss, *_ = compute_losses(m, x, y, {}, 0)
    loss.backward()
    return m, {n: p.grad for n, p in m.named_parameters()}


def test_gradients_reach_both_chains():
    m, g = _grads("mn_match")
    assert all(g[n] is not None and g[n].abs().sum() > 0 for n in g if n.startswith(("fwd", "back")))


def test_gradients_forward_head_does_not_reach_the_meeting_state():
    m, g = _grads("mn_fwd")
    _, g_match = _grads("mn_match")
    assert all(g[n] is not None and g[n].abs().sum() > 0 for n in g if n.startswith("head"))
    for n in g_match:                                            # same seed: the chains see exactly the matching-only gradient
        assert torch.allclose(g[n], g_match[n], atol=1e-6), n


def test_table_ablation_has_no_backward_chain():
    m, g = _grads("mn_table")
    assert not hasattr(m, "back") and g["table"].abs().sum() > 0
    assert m.targets().shape == (10, 256)


def test_shallow_is_ordinary_cross_entropy():
    m, g = _grads("shallow")
    assert not hasattr(m, "back") and isinstance(m.head, torch.nn.Linear) and m.head.in_features == 256
    x, y = _xy()
    loss, parts, _, z = compute_losses(m, x, y, {}, 0)
    assert set(parts) == {"task"} and torch.allclose(loss, torch.nn.functional.cross_entropy(m(x), y))


# ------------------------------------------------------------------ codes
def test_noise_code():
    m = meetnet.MeetingNet("mn_match", code="noise", sigma=0.5)
    a, b = m.targets(train=True), m.targets(train=True)
    assert not torch.allclose(a, b)                              # random in training
    assert torch.allclose(m.targets(), m.targets())             # zero noise at test time
    assert torch.allclose(m._codes(False)[:, 10:], torch.zeros(10, 8))


def test_multihot_code_is_fixed_and_distinct():
    a, b = meetnet.multihot_codes(), meetnet.multihot_codes()
    assert torch.equal(a, b) and a.shape == (10, 16) and set(a.unique().tolist()) == {-1.0, 1.0}
    assert len({tuple(r.tolist()) for r in a}) == 10
    m1, m2 = meetnet.MeetingNet("mn_match", code="multihot"), meetnet.MeetingNet("mn_match", code="multihot")
    assert torch.equal(m1.codes, m2.codes)


# ------------------------------------------------------------------ training, logging and cost
@pytest.mark.parametrize("arm,primary,logged", [("mn_match", "match", {"match"}), ("mn_fwd", "fwd", {"fwd", "match"}),
                                                  ("mn_table", "match", {"match"}), ("shallow", "fwd", {"fwd"})])
def test_train_logs_readouts(arm, primary, logged):
    log = run(_cfg(arm), "cpu", data=_loaders())
    assert {k[len("val_acc_"):] for k in log[0] if k.startswith("val_acc_")} == logged
    assert all(r["val_acc"] == r[f"val_acc_{primary}"] for r in log)
    assert log[-1]["flops"] > 0 and log[-1]["val_acc"] > 0.3     # learns the easy synthetic task


@pytest.mark.parametrize("arm,kw", [("mn_match", {}), ("mn_fwd", {}), ("mn_match", dict(mn_code="noise")), ("mn_match", dict(mn_k=2)),
                                     ("mn_fwd", dict(mn_k=3))])
def test_arms_learn(arm, kw):
    cfg = {**_cfg(arm), **kw, "max_steps": 120}
    log = run(cfg, "cpu", data=_loaders())
    assert log[-1]["val_acc"] > 0.8


def test_run_is_deterministic():
    a, b = run(_cfg("mn_fwd"), "cpu", data=_loaders()), run(_cfg("mn_fwd"), "cpu", data=_loaders())
    assert [r["val_acc"] for r in a] == [r["val_acc"] for r in b] and [r["flops"] for r in a] == [r["flops"] for r in b]


def test_flops_count_the_backward_chain_and_the_head():
    batch = _xy(128)
    cost = {}
    for arm in ("mn_match", "mn_fwd", "shallow"):
        m = meetnet.MeetingNet(arm)
        opt = build_optimizer(m, dict(lr=1e-3, wd=1e-4))[0]
        cost[arm] = flops_per_step(m, opt, batch, {})
    assert cost["mn_fwd"] > cost["mn_match"] > 0
    assert cost["mn_match"] > cost["shallow"] * 0.95            # the backward chain costs something, the 256x10 head is cheaper than the chain
    assert cost["mn_fwd"] > cost["shallow"]


def test_optimizer_has_no_empty_group_for_the_meeting_network():
    opt, sched = build_optimizer(meetnet.MeetingNet("mn_fwd"), dict(lr=1e-3, wd=1e-4))
    assert len(opt.param_groups) == 1 and len(sched.lr_lambdas) == 1


def test_existing_arms_unchanged_by_the_new_dispatch():
    from src.model import MeetNet
    m = MeetNet(arm="base")
    assert not hasattr(m, "mn_losses")
    opt, _ = build_optimizer(m, dict(lr=1e-3, wd=1e-4, low_lr_mult=0.5))
    assert len(opt.param_groups) == 2 and opt.param_groups[0]["lr"] == pytest.approx(5e-4)


# ------------------------------------------------------------------ sweep arms
def test_sweep_arms():
    for arm in sweep.MN_SWEEP_ARMS:
        cfg_arm, fixed, extra = sweep.SWEEP_ARMS[arm]
        assert cfg_arm in meetnet.MN_ARMS and fixed["mn_k"] in (1, 2, 3)
        cfg = sweep.build_cfg(arm, sweep.sample_trials(arm, "t16", 1)[0], 1000, "t16")
        assert cfg["arm"] == cfg_arm and cfg["mn_k"] == fixed["mn_k"] and cfg["max_steps"] == sweep.SWEEP_STEPS
        assert ("mn_scale" in cfg) == (arm != "shallow")
        meetnet.build_model(cfg)                                 # constructible
    assert sweep.SWEEP_ARMS["mn_match"][0] == "mn_match" and sweep.SWEEP_ARMS["mn_noise"][1]["mn_code"] == "noise"
    assert sweep.SWEEP_ARMS["mn_mh"][1]["mn_code"] == "multihot"
    assert [sweep.SWEEP_ARMS[f"mn_match_k{k}"][1]["mn_k"] for k in (2, 3)] == [2, 3]
    assert sweep.expected_steps(sweep.MN_SWEEP_ARMS) == 3000
    assert meetnet.equal_cost_k() == sweep.SWEEP_ARMS["mn_match"][1]["mn_k"]


# ------------------------------------------------------------------ plan and configs
@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    (tmp_path / "configs" / "tuned").mkdir(parents=True)
    for name in ("t1_base_two", "t1_meet_two"):
        (tmp_path / "configs" / "tuned" / f"{name}.yaml").write_text((REPO / "configs" / "tuned" / f"{name}.yaml").read_text())
    for g, arm in mn.GROUP_ARM.items():
        hp = dict(lr=1e-3, wd=1e-4, act="relu", lr_schedule="cosine", **({} if arm == "shallow" else {"mn_scale": 12.0}))
        cfg = sweep.build_cfg(arm, hp, 1000, "t16")
        cfg.update(seed=0, stage=f"final-t16-{arm}-two")
        (tmp_path / "configs" / "tuned" / f"t16_{arm}_two.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(mn, "ROOT", tmp_path)
    return tmp_path


def test_group_specs(fake_root):
    specs = mn.group_specs("mn1")
    assert [s["name"] for s in specs][:3] == ["A", "S", "D"] and len(specs) == 12
    assert {s["seeds"] for s in specs} == {tuple(range(1500, 1510))}
    assert {s["role"] for s in specs if s["name"] in ("A", "S")} == {"reference"}


def test_seeds_are_disjoint_from_the_other_studies():
    from exploratory import oracle as orc
    used = set()
    for t in orc.TAGS.values():
        for seeds in t["seeds"].values():
            used |= set(seeds)
    mine = set(mn.TAGS["mn1"]["seeds"])
    assert not used & mine and not mine & set(range(0, 100)) and not mine & set(range(1100, 1130)) and not mine & set(range(1300, 1320))
    assert min(mine) >= 100


def test_build_config(fake_root):
    specs = {s["name"]: s for s in mn.group_specs("mn1")}
    c = mn.build_config(specs["M2f"], "mn1")
    assert c["arm"] == "mn_fwd" and c["mn_k"] == 2 and c["stage"] == "mn-mn1-M2f" and c["mn_scale"] == 12.0
    assert mn.build_config(specs["M1n"], "mn1")["mn_code"] == "noise" and mn.build_config(specs["M1h"], "mn1")["mn_code"] == "multihot"
    assert mn.build_config(specs["A"], "mn1")["arm"] == "base" and mn.build_config(specs["D"], "mn1")["arm"] == "meet"
    stages = [mn.build_config(s, "mn1")["stage"] for s in specs.values()]
    assert len(set(stages)) == len(stages)
    p = fake_root / "configs" / "tuned" / "t16_mn_match_k2_two.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["mn_k"] = 1
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="mn_k"):
        mn.build_config(specs["M2m"], "mn1")


def test_missing_tuned_config_has_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(mn, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="t16"):
        mn.build_config({"name": "M1m", "seeds": (1,)}, "mn1")


# ------------------------------------------------------------------ decision rules
def _ch(point, lo, hi):
    return {t: (point, lo, hi) for t in mn.TARGETS}


R10 = {t: 10 for t in mn.TARGETS}


def test_decide_group():
    win = _ch(-20, -30, -10)
    assert mn.decide_group(R10, win, win, 0.001) == "EXCELS"
    assert mn.decide_group(R10, win, win, 0.003) == "FASTER, LESS ACCURATE"
    assert mn.decide_group(R10, win, _ch(-5, -9, 0), 0.0) != "EXCELS"                # against S too
    assert mn.decide_group(R10, _ch(-20, -30, 2), win, 0.0) != "EXCELS"              # interval includes 0
    assert mn.decide_group({0.97: 4, 0.975: 2}, win, win, 0.0) == "FAILS"
    assert mn.decide_group(R10, win, win, 0.02) == "FASTER, LESS ACCURATE"           # fast and inaccurate stays what it is
    assert mn.decide_group(R10, _ch(5, -3, 15), _ch(10, 0, 20), 0.0) == "COMPETITIVE"
    assert mn.decide_group(R10, _ch(5, -3, 15), _ch(40, 30, 50), 0.0) == "SLOWER"
    assert mn.decide_group(R10, _ch(5, -3, 15), _ch(5, -3, 15), 0.02) == "FAILS"
    assert mn.decide_group({0.97: 7, 0.975: 6}, _ch(5, -3, 15), _ch(5, -3, 15), 0.0) == "UNCLEAR"


def _score(rng, mean_flops, acc=0.98, n=10, steps=1000, reached=True, readouts=None):
    f = rng.normal(mean_flops, mean_flops * 0.01, n)
    return dict(seeds=list(range(n)), flops={t: list(f) for t in mn.TARGETS}, steps={t: [steps] * n for t in mn.TARGETS},
                reached={t: [reached] * n for t in mn.TARGETS}, final=[acc] * n, readouts=readouts or {})


def test_diagnose_study_readings():
    rng = np.random.default_rng(0)
    sc = {"A": _score(rng, 100), "S": _score(rng, 90), "M1m": _score(rng, 90),
          "M1t": _score(rng, 90), "M1n": _score(rng, 90), "M1h": _score(rng, 90), "M2m": _score(rng, 120), "M3m": _score(rng, 150),
          "M1f": _score(rng, 95, readouts={"fwd": [0.985] * 10, "match": [0.98] * 10}), "M2f": _score(rng, 80), "M3f": _score(rng, 120)}
    r = mn.diagnose_study(sc, n_boot=2000)
    assert r["chain_vs_table"] == "CHAIN NOT NEEDED" and r["label_M1n"] == "NO EFFECT" and r["shallow"] == "SAME AS SHALLOW NET"
    assert r["meeting_point_match"]["verdict"] == "EQUAL-COST RULE CONFIRMED" and r["meeting_point_match"]["best_k"] == 1
    assert r["meeting_point_fwd"]["verdict"] == "EQUAL-COST RULE NOT CONFIRMED" and r["meeting_point_fwd"]["best_k"] == 2
    assert r["readout"]["verdict"] == "FORWARD BETTER"
    sc["M1t"] = _score(rng, 120)                                 # a table that needs 33% more FLOPs: the chain helps
    sc["M1n"] = _score(rng, 70)                                  # a richer label that saves 22%
    r = mn.diagnose_study(sc, n_boot=2000)
    assert r["chain_vs_table"] == "CHAIN HELPS" and r["label_M1n"] == "RICHER LABEL HELPS"
    sc["M1m"] = _score(rng, 60)                                  # faster than S by a third
    assert mn.diagnose_study(sc, n_boot=2000)["shallow"] == "BETTER"


def test_analyze_and_report_run_on_a_full_set():
    rng = np.random.default_rng(1)
    sc = {n: _score(rng, 100 + 7 * i) for i, n in enumerate(["A", "S", "D", *[g for g in mn.GROUP_ARM if g != "S"]])}
    res = mn.analyze(sc, n_boot=500)
    assert set(res["groups"]) == {g for g in mn.GROUP_ARM if g != "S"}
    assert "verdict" in mn.report(sc, res) or "SLOWER" in mn.report(sc, res)


# ------------------------------------------------------------------ the runner
def test_runner_writes_configs_and_refuses_edits(fake_root, monkeypatch, capsys):
    rm = load_script("run_meeting_net")
    monkeypatch.setattr(rm.mn, "ROOT", fake_root)
    specs = mn.group_specs("mn1")
    with pytest.raises(SystemExit):
        rm.ensure_configs("mn1", specs)                          # missing, not asked to write
    rm.ensure_configs("mn1", specs, write=True)
    rm.ensure_configs("mn1", specs)                              # now identical to the derivation
    p = mn.config_path("mn1", "M1m")
    p.write_text(p.read_text().replace("lr:", "lr: 9\n#", 1))
    with pytest.raises(SystemExit):
        rm.ensure_configs("mn1", specs)


def test_runner_job_order_is_seed_major(fake_root):
    rm = load_script("run_meeting_net")
    jobs = rm.make_jobs("mn1", mn.group_specs("mn1"), "out")
    assert len(jobs) == 12 * 10
    assert [j.seed for j in jobs[:12]] == [1500] * 12 and jobs[0].group == "A"


def test_runner_end_to_end_on_a_tiny_run(fake_root, tmp_path, monkeypatch):
    """Two groups, two seeds, a few steps on synthetic loaders: the run files exist, load_group reads them, the analysis runs."""
    rm = load_script("run_meeting_net")
    monkeypatch.setattr(rm.mn, "ROOT", fake_root)
    specs = [s for s in mn.group_specs("mn1", seeds=(1, 2)) if s["name"] in ("A", "S", "M1m", "M1f")]
    rm.ensure_configs("mn1", specs, write=True)
    import src.train as train
    from src.final import run_final_job
    out = tmp_path / "out"
    for job in rm.make_jobs("mn1", specs, out):
        assert run_final_job(job, "cpu", runner=lambda cfg, dev: run({**cfg, "max_steps": 60, "eval_every": 10, "batch_size": 32}, dev, data=_loaders())) == "done"
    g = mn.load_group(out, "mn1", specs[3])
    assert g["name"] == "M1f" and set(g["readouts"]) == {"fwd", "match"} and len(g["final"]) == 2
    sc = {s["name"]: mn.load_group(out, "mn1", s) for s in specs}
    res = mn.analyze(sc, n_boot=200)
    assert set(res["groups"]) == {"M1m", "M1f"} and "readout" in res["readings"]
