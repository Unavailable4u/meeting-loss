"""Tests for the sweep and analysis helpers (src/sweep.py; decisions D-014 and D-017).

Selection-rule tests use hand-made logs, so their expected values are computed by hand.
The end-to-end tests train for a few steps on synthetic data (MNIST is not needed).
"""
import json
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import sweep
from src.model import ARMS
from src.train import run


def _synthetic():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(256, 784, generator=g), torch.randint(0, 10, (256,), generator=g)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True,
                       generator=torch.Generator().manual_seed(0), drop_last=True),
            DataLoader(TensorDataset(x[:64], y[:64]), batch_size=64))


def _fake_log(accs, flops_per_eval=100.0, eval_every=50):
    return [dict(step=(i + 1) * eval_every, flops=(i + 1) * flops_per_eval, val_acc=a)
            for i, a in enumerate(accs)]


# --- search spaces and sampler -----------------------------------------------------------------
def test_sampler_is_deterministic_and_depends_on_tag_and_arm():
    a = sweep.sample_trials("meet", "t1")
    assert a == sweep.sample_trials("meet", "t1")
    assert a != sweep.sample_trials("meet", "t2")
    assert a != sweep.sample_trials("meet_ao", "t1")
    assert len(a) == sweep.N_TRIALS


@pytest.mark.parametrize("arm", sorted(sweep.SWEEP_ARMS))
def test_sampled_values_are_inside_the_d014_ranges(arm):
    extra = sweep.SWEEP_ARMS[arm][2]
    dims = {**sweep.COMMON, **extra}
    for hp in sweep.sample_trials(arm, "t1", n=60):
        assert set(hp) == set(dims)
        for k, spec in dims.items():
            if spec[0] == "choice":
                assert hp[k] in spec[1], (arm, k)
            else:
                assert spec[1] * 0.999 <= hp[k] <= spec[2] * 1.001, (arm, k, hp[k])


def test_search_space_matches_d014():
    c = sweep.COMMON
    assert c["lr"] == ("log", 1e-4, 1e-2) and c["wd"] == ("log", 1e-6, 1e-2)
    assert c["act"][1] == ["relu", "tanh"] and c["lr_schedule"][1] == ["constant", "cosine"]
    meet = sweep.SWEEP_ARMS["meet"][2]
    assert meet["lam_meet"] == ("log", 1e-3, 10.0) and meet["lam_inv"] == ("log", 0.1, 10.0)
    assert meet["warmup_steps"][1] == [200, 500, 1000]
    assert sweep.SWEEP_ARMS["base_lowlr"][2]["low_lr_mult"] == ("log", 0.5, 8.0)
    assert sweep.SWEEP_ARMS["deepsup"][2]["lam_aux"] == ("log", 1e-2, 1.0)
    assert "lam_inv" not in sweep.SWEEP_ARMS["meet_frozen"][2]          # no inverse loss there


def test_every_sweep_arm_maps_to_a_real_arm_and_config_is_complete():
    for name, (arm, fixed, _) in sweep.SWEEP_ARMS.items():
        assert arm in ARMS, name
        cfg = sweep.build_cfg(name, sweep.sample_trials(name, "t1")[0], seed=1000, tag="t1")
        assert cfg["arm"] == arm and cfg["seed"] == 1000 and cfg["max_steps"] == sweep.SWEEP_STEPS
        assert cfg["stage"] == "sweept1"
        for k, v in fixed.items():
            assert cfg[k] == v
    # control E is arm A with a lower-layer learning-rate multiplier, not a new arm
    e = sweep.build_cfg("base_lowlr", sweep.sample_trials("base_lowlr", "t1")[0], 1000, "t1")
    assert e["arm"] == "base" and "low_lr_mult" in e


# --- protocol guards ---------------------------------------------------------------------------
def test_selection_seeds_must_not_overlap_final_seeds():
    with pytest.raises(ValueError):
        sweep.check_protocol("t1", 25, 3000, [5, 6])
    assert sweep.check_protocol("t1", 25, 3000, [1000, 1001]) is False


def test_non_protocol_sweeps_need_a_smoke_tag():
    with pytest.raises(ValueError):
        sweep.check_protocol("t1", 5, 3000, [1000, 1001])
    with pytest.raises(ValueError):
        sweep.check_protocol("t1", 25, 6000, [1000, 1001])
    assert sweep.check_protocol("smoke1", 2, 100, [100, 101]) is True


def test_job_order_is_trial_major_so_a_stopped_sweep_stays_balanced():
    jobs = sweep.plan_jobs("t1", ["base", "meet"], [1000, 1001], n_trials=3)
    assert len(jobs) == 3 * 2 * 2
    assert [(j.trial, j.arm) for j in jobs[:4]] == [(0, "base"), (0, "base"), (0, "meet"), (0, "meet")]


# --- selection rule (D-017) and crossing definition (D-018) ---------------------------------------
def test_default_score_needs_two_consecutive_evaluations():
    # one lucky evaluation at 0.976 (3rd) is followed by a drop; the real crossing is at the 5th
    rows = _fake_log([0.90, 0.96, 0.976, 0.97, 0.976, 0.977, 0.98])
    s = sweep.score_run(rows)
    assert s["reached"] and s["steps"] == 250 and s["flops"] == 500.0 and s["score"] == 500.0
    f = sweep.score_run(rows, crossing="first")                  # the pre-D-018 definition
    assert f["steps"] == 150 and f["score"] == 300.0


def test_a_lone_final_evaluation_at_target_does_not_count():
    s = sweep.score_run(_fake_log([0.90, 0.96, 0.976]))
    assert not s["reached"] and s["score"] == 300.0              # scored with the full budget


def test_run_that_never_reaches_target_is_scored_with_full_budget():
    rows = _fake_log([0.90, 0.95, 0.97, 0.974])
    s = sweep.score_run(rows)
    assert not s["reached"] and s["steps"] is None and s["score"] == 400.0 == s["budget"]


def test_trial_score_is_the_mean_over_seeds_with_misses_at_full_budget():
    hit = sweep.score_run(_fake_log([0.9, 0.98, 0.98, 0.98]))       # reaches at FLOPs 200
    miss = sweep.score_run(_fake_log([0.9, 0.9, 0.9, 0.9]))         # full budget 400
    t = sweep.summarize_trial({1000: hit, 1001: miss})
    assert t["score"] == 300.0 and t["n_reached"] == 1 and t["n_seeds"] == 2


def test_extra_compute_is_counted_by_the_selection_score():
    cheap = sweep.score_run(_fake_log([0.9, 0.976, 0.976], flops_per_eval=100))
    costly = sweep.score_run(_fake_log([0.9, 0.976, 0.976], flops_per_eval=123))   # same steps, +23% FLOPs
    assert costly["steps"] == cheap["steps"] and costly["score"] > cheap["score"]


def test_ties_go_to_higher_late_accuracy_then_lower_trial_index():
    trials = [dict(trial=0, score=400.0, late_val=0.90),
              dict(trial=1, score=400.0, late_val=0.95),
              dict(trial=2, score=400.0, late_val=0.95),
              dict(trial=3, score=500.0, late_val=0.99)]
    assert sweep.select_best(trials)["trial"] == 1
    assert sweep.select_best([dict(trial=4, score=300.0, late_val=0.1)] + trials)["trial"] == 4


# --- end to end: run, resume, collect, select ----------------------------------------------------
def _tiny_sweep(tmp_path, arms=("base", "meet", "meet_ao"), n_trials=2, seeds=(100, 101), steps=20):
    root = tmp_path / "sweeps"
    jobs = sweep.plan_jobs("smoke1", list(arms), list(seeds), n_trials, steps, root)
    for a in arms:
        sweep.write_manifest("smoke1", a, list(seeds), n_trials, steps, root)
    return root, jobs


def test_run_job_writes_a_complete_log_and_never_overwrites(tmp_path):
    root, jobs = _tiny_sweep(tmp_path, arms=("meet",))
    job = jobs[0]
    runner = lambda cfg, device: run({**cfg, "eval_every": 5}, device="cpu", data=_synthetic())
    assert sweep.run_job(job, "cpu", runner) == "done"
    hdr, rows = sweep.load_run(job.path)
    assert hdr["_header"] and hdr["sweep"]["arm"] == "meet" and hdr["config"]["seed"] == job.seed
    assert hdr["config_hash"] == sweep.config_hash(hdr["config"]) and "git_commit" in hdr
    assert [r["step"] for r in rows] == [5, 10, 15, 20]
    before = job.path.read_text()
    assert sweep.run_job(job, "cpu", lambda c, d: pytest.fail("must not rerun")) == "skipped"
    assert job.path.read_text() == before


def test_manifest_refuses_a_changed_search_space(tmp_path):
    root = tmp_path / "sweeps"
    path = sweep.write_manifest("smoke1", "meet", [100, 101], 2, 20, root)
    doc = json.loads(path.read_text())
    doc["trials"][0]["lr"] = 0.5
    path.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError):
        sweep.write_manifest("smoke1", "meet", [100, 101], 2, 20, root)


def test_collect_and_select_on_a_tiny_synthetic_sweep(tmp_path):
    root, jobs = _tiny_sweep(tmp_path)
    runner = lambda cfg, device: run({**cfg, "eval_every": 5}, device="cpu", data=_synthetic())
    for j in jobs:
        assert sweep.run_job(j, "cpu", runner) == "done"
    data = sweep.collect_sweep("smoke1", root=root, target=0.0)       # target 0: every run "reaches" at once
    assert set(data) == {"base", "meet", "meet_ao"}                   # "meet" must not swallow "meet_ao"
    for arm, d in data.items():
        assert len(d["trials"]) == 2 and d["incomplete"] == []
        for t in d["trials"]:
            assert sorted(t["per_seed"]) == [100, 101] and t["n_reached"] == 2
            assert math.isfinite(t["score"]) and t["score"] > 0
        best = sweep.select_best(d["trials"])
        assert best["score"] == min(t["score"] for t in d["trials"])
    # costlier arms pay for their extra compute in the score
    base_first = data["base"]["trials"][0]["score"]
    assert data["meet_ao"]["trials"][0]["score"] > base_first


def test_incomplete_trials_are_reported_not_scored(tmp_path):
    root, jobs = _tiny_sweep(tmp_path, arms=("base",))
    runner = lambda cfg, device: run({**cfg, "eval_every": 5}, device="cpu", data=_synthetic())
    for j in jobs:
        if not (j.trial == 1 and j.seed == 101):                       # leave one run missing
            sweep.run_job(j, "cpu", runner)
    d = sweep.collect_sweep("smoke1", ["base"], root, target=0.0)["base"]
    assert [t["trial"] for t in d["trials"]] == [0] and d["incomplete"] == [1]


def test_tuned_config_keeps_hyperparameters_and_resets_seed():
    cfg = sweep.build_cfg("meet", sweep.sample_trials("meet", "t1")[3], seed=1001, tag="t1")
    text = sweep.tuned_config(cfg, "t1", "meet", "note")
    import yaml
    out = yaml.safe_load(text)
    assert out["seed"] == 0 and out["max_steps"] == cfg["max_steps"]
    assert out["stage"] == "final-t1-meet-two"                    # the crossing rule is part of the stage name
    assert sweep.tuned_config(cfg, "t1", "meet", "note", crossing="first").count("final-t1-meet-first") == 1
    assert {k: v for k, v in out.items() if k not in ("seed", "stage")} == \
           {k: v for k, v in cfg.items() if k not in ("seed", "stage")}


# --- statistics --------------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_less_spread():
    rng = np.random.default_rng(0)
    wide, narrow = rng.normal(100, 10, 10), rng.normal(100, 1, 10)
    lo, hi = sweep.bootstrap_ci(wide, n_boot=4000)
    assert lo < wide.mean() < hi
    lo2, hi2 = sweep.bootstrap_ci(narrow, n_boot=4000)
    assert (hi2 - lo2) < (hi - lo)


def test_paired_pct_change_recovers_a_known_effect_and_uses_pairing():
    rng = np.random.default_rng(1)
    ref = rng.normal(100.0, 20.0, 10)                  # large seed-to-seed spread...
    grp = ref * 0.8 + rng.normal(0, 0.5, 10)           # ...but a consistent 20% saving per seed
    p, lo, hi = sweep.paired_pct_change(ref, grp, n_boot=4000)
    assert -21 < p < -19 and hi < 0 and (hi - lo) < 4  # pairing keeps the CI tight and excludes zero
    p0, lo0, hi0 = sweep.paired_pct_change(ref, ref.copy(), n_boot=4000)
    assert p0 == 0 and lo0 == hi0 == 0
    with pytest.raises(ValueError):
        sweep.paired_pct_change(ref, grp[:5])


def test_group_scores_rejects_two_runs_with_the_same_seed(tmp_path):
    def write(name, seed):
        p = tmp_path / name
        p.write_text(json.dumps({"_header": True, "config": {"seed": seed}}) + "\n"
                     + "\n".join(json.dumps(r) for r in _fake_log([0.9, 0.98])) + "\n")
        return str(p)
    ok = sweep.group_scores([write("a.jsonl", 0), write("b.jsonl", 1)], [0.97], crossing="first")
    assert sorted(ok) == [0, 1] and ok[0][0.97]["reached"]
    with pytest.raises(ValueError):
        sweep.group_scores([write("c.jsonl", 3), write("d.jsonl", 3)], [0.97])


# --- selection-bias check ------------------------------------------------------------------------
def test_cross_seed_check_selects_on_one_seed_and_scores_on_the_other():
    def t(i, s_a, s_b):
        return dict(trial=i, per_seed={100: dict(score=s_a), 101: dict(score=s_b)})
    trials = [t(0, 100.0, 300.0), t(1, 200.0, 150.0), t(2, 400.0, 100.0)]
    mean, picks = sweep.cross_seed_check(trials, [100, 101])
    # picked on seed 100 -> trial 0, held-out score 300; picked on seed 101 -> trial 2, held-out score 400
    assert picks == [(0, 300.0), (2, 400.0)] and mean == 350.0
    with pytest.raises(ValueError):
        sweep.cross_seed_check(trials, [100])
