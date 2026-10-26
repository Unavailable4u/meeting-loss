"""Machinery for prereg-v2 (decision D-023, DR-011): the sweep arms of t9 and t10, the plan, the lock tag read from the
plan, the shift rules, the H7a / H7b analysis and the noise-floor outcome. Written before the code.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import yaml

from src import final, prereg2, sweep
from src.sweep import margin_verdict, overall_verdict

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scripts_loader import load_script  # noqa: E402


# ----------------------------------------------------------------------------- sweep arms t9 (batch 48) and t10 (tuned S)
def test_new_sweep_arms_and_their_fixed_settings():
    assert {"base_b48", "selective30"} <= set(sweep.SWEEP_ARMS)
    cfg = sweep.build_cfg("base_b48", sweep.sample_trials("base_b48", "t9")[0], 9000, "t9", max_steps=8000)
    assert cfg["arm"] == "base" and cfg["batch_size"] == 48 and cfg["eval_every"] == 125 and cfg["max_steps"] == 8000
    cfg = sweep.build_cfg("selective30", sweep.sample_trials("selective30", "t10")[0], 9100, "t10")
    assert cfg["arm"] == "selective" and cfg["sel_frac"] == 0.3 and cfg["sel_warmup"] == 100 and cfg["sel_mode"] == "top"
    assert cfg["max_steps"] == 3000 and cfg["batch_size"] == 128
    # same search space as arm A (lr, wd, act, lr_schedule): nothing else is tuned
    for arm in ("base_b48", "selective30"):
        assert set(sweep.sample_trials(arm, "t9")[0]) == set(sweep.COMMON)
        assert len(sweep.sample_trials(arm, "t9")) == sweep.N_TRIALS == 25


def test_trials_of_the_new_arms_are_deterministic_and_distinct_from_arm_a():
    a = sweep.sample_trials("base", "t9")
    b = sweep.sample_trials("base_b48", "t9")
    assert b == sweep.sample_trials("base_b48", "t9") and a != b                 # the draw depends on (tag, arm)


def test_the_evaluation_spacing_divides_the_horizon():
    for arm, steps in sweep.ARM_STEPS.items():
        ev = sweep.SWEEP_ARMS[arm][1]["eval_every"]
        assert steps % ev == 0, "the last evaluation must fall on the last step (collect_sweep checks completeness that way)"


def test_horizon_per_arm_and_protocol_check():
    assert sweep.expected_steps(["base"]) == 3000 and sweep.expected_steps(["base_b48"]) == 8000
    assert sweep.expected_steps(["selective30"]) == 3000
    with pytest.raises(ValueError, match="different horizons"):
        sweep.expected_steps(["base", "base_b48"])
    # the protocol (25 trials, 2 seeds) at the arm's own horizon is accepted without a smoke tag
    assert sweep.check_protocol("t9", 25, 8000, (9000, 9001), arms=["base_b48"]) is False
    assert sweep.check_protocol("t10", 25, 3000, (9100, 9101), arms=["selective30"]) is False
    with pytest.raises(ValueError, match="non-protocol"):
        sweep.check_protocol("t9", 25, 3000, (9000, 9001), arms=["base_b48"])      # the wrong horizon for this arm
    with pytest.raises(ValueError, match="non-protocol"):
        sweep.check_protocol("t5", 25, 8000, (5000, 5001), arms=["base"])
    assert sweep.check_protocol("smoke1", 2, 100, (100, 101), arms=["base_b48"]) is True
    assert sweep.check_protocol("t5", 25, 3000, (5000, 5001)) is False             # old call without arms still works


def test_selection_seeds_of_t9_and_t10_are_disjoint_from_everything_reserved():
    t9, t10 = (9000, 9001), (9100, 9101)
    assert min(t9 + t10) >= sweep.MIN_SELECTION_SEED
    used = set(range(0, 100)) | set(range(600, 603)) | set(range(700, 720)) | set(range(800, 830)) | set(range(900, 910)) \
        | set(range(950, 952)) | set(range(1000, 1002)) | set(range(1100, 1130)) | set(range(1200, 1213)) \
        | set(range(1300, 1320)) | {2000, 2001, 5000, 5001, 6000, 6001, 7000, 7001, 8000, 8001}
    assert not (set(t9 + t10) & used)


# ----------------------------------------------------------------------------- shift rules and the margin of 0
def test_shift_value_rules():
    changes = [+34.0, -8.0, +5.0, -12.0, +20.0]          # A2, t5, t6, t7, t8 against A1
    assert sweep.shift_value(changes, "max_abs") == 34.0
    assert sweep.shift_value(changes, "max_gain") == 12.0                                  # the best further tuning is 12% better
    assert sweep.shift_value(changes, "median_abs") == 12.0
    assert sweep.shift_value([+10.0, +20.0], "max_gain") == 0.0                          # no further tuning beats A1
    with pytest.raises(ValueError):
        sweep.shift_value(changes, "mean")


def test_margin_verdict_with_a_zero_margin():
    assert margin_verdict(-12.0, -20.0, -4.0, margin=0.0) == "supported"
    assert margin_verdict(-12.0, -20.0, 3.0, margin=0.0) == "inconclusive"                # interval includes 0
    assert margin_verdict(5.0, 1.0, 9.0, margin=0.0) == "refuted"
    assert margin_verdict(2.0, -3.0, 7.0, margin=0.0) == "inconclusive"


# ----------------------------------------------------------------------------- plan: seed block, horizons, out dir, lock tag
def _write_cfg(repo, name, stage, arm, steps):
    p = repo / "configs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(dict(stage=stage, arm=arm, max_steps=steps, seed=0)))
    return f"configs/{name}"


def _plan(tmp_path, groups, **top):
    doc = dict(horizon_steps=3000, seed_block="1300-1303", out_dir="results/raw/prereg_v2", lock_tag="prereg-v2",
               groups=groups, **top)
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_plan_with_a_seed_block_a_group_horizon_and_metadata(tmp_path):
    a = _write_cfg(tmp_path, "a.yaml", "final-t1-base-two", "base", 3000)
    b = _write_cfg(tmp_path, "b.yaml", "final-t9-base_b48-two", "base", 8000)
    p = _plan(tmp_path, [dict(name="A1", config=a, seeds="1300-1303", role="reference"),
                         dict(name="B48", config=b, seeds="1300-1303", role="control", horizon_steps=8000)])
    groups = final.load_plan(p, repo=tmp_path)
    assert [g["name"] for g in groups] == ["A1", "B48"] and groups[1]["horizon"] == 8000 and groups[0]["horizon"] == 3000
    meta = final.plan_meta(p)
    assert meta["lock_tag"] == "prereg-v2" and meta["out_dir"] == "results/raw/prereg_v2" and meta["seed_block"] == [1300, 1303]
    # Stage 3 plan keeps its defaults
    m3 = final.plan_meta(REPO / "configs" / "stage3_plan.yaml")
    assert m3["lock_tag"] == "prereg-v1" and m3["out_dir"] == "results/raw" and m3["seed_block"] is None


def test_plan_rejects_seeds_outside_the_block_and_a_wrong_group_horizon(tmp_path):
    a = _write_cfg(tmp_path, "a.yaml", "final-t1-base-two", "base", 3000)
    p = _plan(tmp_path, [dict(name="A1", config=a, seeds="1300-1304", role="reference")])
    with pytest.raises(ValueError, match="seeds"):
        final.load_plan(p, repo=tmp_path)
    p = _plan(tmp_path, [dict(name="A1", config=a, seeds="1300-1303", role="reference", horizon_steps=8000)])
    with pytest.raises(ValueError, match="max_steps"):
        final.load_plan(p, repo=tmp_path)
    p = _plan(tmp_path, [dict(name="A1", config=a, seeds="1300-1303", role="boss")])
    with pytest.raises(ValueError, match="role"):
        final.load_plan(p, repo=tmp_path)


def test_plan_can_report_missing_configs_when_allowed(tmp_path):
    a = _write_cfg(tmp_path, "a.yaml", "final-t1-base-two", "base", 3000)
    p = _plan(tmp_path, [dict(name="A1", config=a, seeds="1300-1303", role="reference"),
                         dict(name="B48", config="configs/missing.yaml", seeds="1300-1303", role="control")])
    with pytest.raises(FileNotFoundError):
        final.load_plan(p, repo=tmp_path)
    groups = final.load_plan(p, repo=tmp_path, allow_missing=True)
    assert groups[1]["cfg"] is None and groups[0]["cfg"]["arm"] == "base"


def test_lock_status_accepts_a_tag_and_paths_of_its_own(tmp_path):
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "src" / "a.py").write_text("v1\n")
    (repo / "configs" / "plan.yaml").write_text("v1\n")
    run = lambda *a: subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=repo, capture_output=True, text=True, check=True)
    run("init", "-q"); run("add", "-A"); run("commit", "-qm", "lock"); run("tag", "prereg-v2")
    assert final.lock_status(repo, tag="prereg-v2", paths=["src", "configs/plan.yaml"]) == (True, "ok")
    (repo / "configs" / "plan.yaml").write_text("v2\n")
    ok, msg = final.lock_status(repo, tag="prereg-v2", paths=["src", "configs/plan.yaml"])
    assert not ok and "plan.yaml" in msg
    ok, msg = final.lock_status(repo, tag="prereg-v1", paths=["src"])
    assert not ok and "does not exist" in msg


# ----------------------------------------------------------------------------- the committed prereg-v2 plan and derived configs
def test_derived_configs_are_arm_a1_with_only_the_selective_keys_changed():
    a1 = yaml.safe_load(open(REPO / "configs" / "tuned" / "t1_base_two.yaml"))
    for name, frac, mode in (("S30", 0.3, "top"), ("S20", 0.2, "top"), ("R30", 0.3, "random")):
        want = prereg2.derive_config(name)
        got = yaml.safe_load(open(REPO / "configs" / "prereg_v2" / f"{name}.yaml"))
        assert got == want
        diff = {k for k in set(a1) | set(want) if a1.get(k) != want.get(k)}
        assert diff == {"stage", "arm", "sel_frac", "sel_warmup", "sel_mode"}, name
        assert want["arm"] == "selective" and want["sel_frac"] == frac and want["sel_mode"] == mode and want["sel_warmup"] == 100
        assert want["stage"] == f"prereg2-{name}"
    with pytest.raises(ValueError):
        prereg2.derive_config("S99")


def test_committed_plan_matches_d023():
    path = REPO / "configs" / "prereg_v2_plan.yaml"
    groups = final.load_plan(path, allow_missing=True)
    meta = final.plan_meta(path)
    assert meta["lock_tag"] == "prereg-v2" and meta["out_dir"] == "results/raw/prereg_v2" and meta["seed_block"] == [1300, 1319]
    assert meta["shift_rule"] == "max_gain"                                        # decided 2026-10-27 (D-023, plan note)
    by = {g["name"]: g for g in groups}
    assert set(by) == {"A1", "S30", "B48", "A2", "A5", "A6", "A7", "A8", "S20", "S30T", "R30"}
    assert all(g["seeds"] == list(range(1300, 1320)) for g in groups)
    roles = {n: g["role"] for n, g in by.items()}
    assert roles["A1"] == "reference" and roles["S30"] == "primary" and roles["B48"] == "control"
    assert {n for n, r in roles.items() if r == "shift"} == {"A2", "A5", "A6", "A7", "A8"}
    assert by["B48"]["horizon"] == 8000 and all(g["horizon"] == 3000 for n, g in by.items() if n != "B48")
    # only the two tunings that do not exist yet may be missing
    missing = {n for n, g in by.items() if g["cfg"] is None}
    assert missing <= {"B48", "S30T"}
    assert by["A1"]["config"] == "configs/tuned/t1_base_two.yaml" and by["A2"]["config"] == "configs/tuned/t2_base_two.yaml"
    for n, t in (("A5", "t5"), ("A6", "t6"), ("A7", "t7"), ("A8", "t8")):
        assert by[n]["config"] == f"configs/tuned/{t}_base_two.yaml"
    assert by["S30"]["config"] == "configs/prereg_v2/S30.yaml" and by["R30"]["config"] == "configs/prereg_v2/R30.yaml"
    assert sum(len(g["seeds"]) for g in groups) == 220


# ----------------------------------------------------------------------------- H7a, H7b and the noise floor
def _scores(flops_by_seed_target, steps=None):
    """{seed: {target: score dict}} with a given FLOPs-to-target per (seed, target)."""
    return {s: {t: dict(reached=True, steps=int(f // 1000), flops=f, score=f, budget=f, final_val=0.98, late_val=0.98)
                for t, f in d.items()} for s, d in flops_by_seed_target.items()}


def _group(rng, base, rel, seeds=range(1300, 1320), sd=0.04):
    out = {}
    for s in seeds:
        n = float(np.exp(rng.normal(0, sd)))
        out[s] = {0.97: base * rel[0] * n, 0.975: base * rel[1] * n}
    return _scores(out)


def test_h7_verdicts_supported_and_the_shift_rule():
    rng = np.random.default_rng(0)
    g = {"A1": _group(rng, 100e9, (1.0, 1.0)), "S30": _group(rng, 100e9, (0.60, 0.62)), "B48": _group(rng, 100e9, (0.95, 0.97)),
         "A2": _group(rng, 100e9, (1.20, 1.15)), "A5": _group(rng, 100e9, (0.92, 0.95)),
         "A6": _group(rng, 100e9, (1.05, 1.03)), "A7": _group(rng, 100e9, (0.97, 0.99)), "A8": _group(rng, 100e9, (1.02, 1.04))}
    roles = {"A1": "reference", "S30": "primary", "B48": "control", **{k: "shift" for k in ("A2", "A5", "A6", "A7", "A8")}}
    res = prereg2.h7(g, roles, shift_rule="max_abs", n_boot=2000)
    assert res["H7a"]["overall"] == "supported"                          # about -40% against A1, beyond the largest shift (+20%)
    assert res["H7b"]["overall"] == "supported"                          # about -37% against the small batch, margin 0
    assert res["shift"][0.97] == pytest.approx(20.0, abs=3.0)
    # a further tuning of A that is as good as S makes the verdict inconclusive under max_abs (and max_gain)
    g2 = dict(g, A5=_group(rng, 100e9, (0.58, 0.60)))
    assert prereg2.h7(g2, roles, shift_rule="max_abs", n_boot=2000)["H7a"]["overall"] == "inconclusive"
    assert prereg2.h7(g2, roles, shift_rule="max_gain", n_boot=2000)["H7a"]["overall"] == "inconclusive"
    # a much WORSE further tuning matters under max_abs only
    g3 = dict(g, A2=_group(rng, 100e9, (1.80, 1.70)))
    assert prereg2.h7(g3, roles, shift_rule="max_abs", n_boot=2000)["H7a"]["overall"] == "inconclusive"
    assert prereg2.h7(g3, roles, shift_rule="max_gain", n_boot=2000)["H7a"]["overall"] == "supported"


def test_h7_refuted_and_unpaired_inputs_are_rejected():
    rng = np.random.default_rng(1)
    g = {"A1": _group(rng, 100e9, (1.0, 1.0)), "S30": _group(rng, 100e9, (0.99, 1.0)), "B48": _group(rng, 100e9, (0.88, 0.90)),
         "A2": _group(rng, 100e9, (1.1, 1.1))}
    roles = {"A1": "reference", "S30": "primary", "B48": "control", "A2": "shift"}
    res = prereg2.h7(g, roles, shift_rule="max_abs", n_boot=2000)
    assert res["H7a"]["overall"] == "refuted" and res["H7b"]["overall"] == "refuted"
    g["S30"] = {s: v for s, v in g["S30"].items() if s != 1305}
    with pytest.raises(ValueError, match="same seeds"):
        prereg2.h7(g, roles, shift_rule="max_abs", n_boot=100)
    with pytest.raises(ValueError, match="exactly one"):
        prereg2.h7({"A1": g["A1"]}, {"A1": "reference"}, shift_rule="max_abs", n_boot=100)


def test_noise_floor_outcome():
    large = {"a": {0.97: 100.0, 0.975: 100.0}, "b": {0.97: 120.0, 0.975: 118.0}, "c": {0.97: 105.0, 0.975: 104.0}}
    assert prereg2.noise_floor(large)["outcome"] == "LARGE"
    small = {"a": {0.97: 100.0, 0.975: 100.0}, "b": {0.97: 108.0, 0.975: 105.0}}
    assert prereg2.noise_floor(small)["outcome"] == "SMALL"
    mixed = {"a": {0.97: 100.0, 0.975: 100.0}, "b": {0.97: 120.0, 0.975: 105.0}}
    r = prereg2.noise_floor(mixed)
    assert r["outcome"] == "MIXED" and r["ratio"][0.97] == pytest.approx(1.2) and r["ratio"][0.975] == pytest.approx(1.05)


def _write_run(out_dir, stage, arm, seed, group, steps_total, reach, per_step=1000, eval_every=50, horizon=None):
    rows = []
    for step in range(eval_every, steps_total + 1, eval_every):
        acc = 0.985 if step >= reach else 0.90
        rows.append(dict(step=step, flops=step * per_step, val_acc=acc))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"20261101_{stage}_{arm}_s{seed}.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"_header": True, "config": dict(stage=stage, arm=arm, seed=seed, max_steps=horizon or steps_total),
                            "plan_group": group}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_cli_end_to_end_and_refuses_incomplete_groups(tmp_path):
    names = {"A1": "reference", "S30": "primary", "B48": "control", "A2": "shift", "R30": "descriptive"}
    reach = {"A1": 1000, "S30": 700, "B48": 950, "A2": 1100, "R30": 1200}
    groups = []
    for n, role in names.items():
        stage, arm, steps = f"st-{n}", ("selective" if n in ("S30", "R30") else "base"), (8000 if n == "B48" else 3000)
        cfg = _write_cfg(tmp_path, f"{n}.yaml", stage, arm, steps)
        g = dict(name=n, config=cfg, seeds="1300-1305", role=role)
        if n == "B48":
            g["horizon_steps"] = 8000
        groups.append(g)
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump(dict(horizon_steps=3000, seed_block="1300-1305", out_dir=str(tmp_path / "raw"),
                                        lock_tag="prereg-v2", shift_rule="max_abs", groups=groups)))
    mod = load_script("analyze_prereg2")
    out_dir = tmp_path / "raw"
    for n in names:
        stage, arm = f"st-{n}", ("selective" if n in ("S30", "R30") else "base")
        for s in range(1300, 1306):
            _write_run(out_dir, stage, arm, s, n, 8000 if n == "B48" else 3000, reach[n] + 50 * (s % 3))
    res = mod.analyze(plan, repo=tmp_path, n_boot=500)
    assert res["H7a"]["overall"] == "supported" and res["H7b"]["overall"] == "supported"
    assert set(res["noise_floor"]["ratio"]) == {0.97, 0.975}
    (next(out_dir.glob("*st-S30_selective_s1303.jsonl"))).unlink()
    with pytest.raises(ValueError, match="missing"):
        mod.analyze(plan, repo=tmp_path, n_boot=100)


def test_finished_horizon_is_required(tmp_path):
    names = {"A1": "reference", "S30": "primary"}
    groups = []
    for n, role in names.items():
        cfg = _write_cfg(tmp_path, f"{n}.yaml", f"st-{n}", "base" if n == "A1" else "selective", 3000)
        groups.append(dict(name=n, config=cfg, seeds="1300-1301", role=role))
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump(dict(horizon_steps=3000, seed_block="1300-1301", out_dir=str(tmp_path / "raw"),
                                        lock_tag="prereg-v2", shift_rule="max_abs", groups=groups)))
    out_dir = tmp_path / "raw"
    for n in names:
        for s in (1300, 1301):
            _write_run(out_dir, f"st-{n}", "base" if n == "A1" else "selective", s, n, 3000 if (n, s) != ("S30", 1301) else 1500, 700, horizon=3000)
    mod = load_script("analyze_prereg2")
    with pytest.raises(ValueError, match="unfinished"):
        mod.analyze(plan, repo=tmp_path, n_boot=100)


# ----------------------------------------------------------------------------- runner: lock tag and output directory come from the plan
def test_run_final_reads_the_lock_tag_and_out_dir_from_the_plan(tmp_path):
    a = str(tmp_path / _write_cfg(tmp_path, "a.yaml", "final-t1-base-two", "base", 3000))
    plan = _plan(tmp_path, [dict(name="A1", config=a, seeds="1300-1303", role="reference")])
    r = subprocess.run([sys.executable, "scripts/run_final.py", "--plan", str(plan), "--dry-run"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "prereg-v2" in r.stdout and "results/raw/prereg_v2" in r.stdout


def _fake_sweep_run(root, tag, arm, trial, seed, hp, steps, reach, eval_every):
    cfg = sweep.build_cfg(arm, hp, seed, tag, max_steps=steps)
    rows = [dict(step=st, flops=st * 1000, val_acc=0.985 if st >= reach else 0.9) for st in range(eval_every, steps + 1, eval_every)]
    job = sweep.Job(tag, arm, trial, seed, hp, steps, str(root))
    job.path.parent.mkdir(parents=True, exist_ok=True)
    with open(job.path, "w") as f:
        f.write(json.dumps({"_header": True, "config": cfg, "sweep": dict(tag=tag, arm=arm, trial=trial, hp=hp)}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_selection_and_final_config_of_the_batch_48_sweep(tmp_path):
    arm, tag, steps = "base_b48", "t9", 8000
    trials = sweep.sample_trials(arm, tag, 3)
    sweep.write_manifest(tag, arm, (9000, 9001), 3, steps, root=tmp_path)
    for i, hp in enumerate(trials):
        for s in (9000, 9001):
            _fake_sweep_run(tmp_path, tag, arm, i, s, hp, steps, 3000 + 500 * i, 125)
    data = sweep.collect_sweep(tag, [arm], root=tmp_path)
    best = sweep.select_best(data[arm]["trials"])
    assert best["trial"] == 0
    text = sweep.tuned_config(best["cfg"], tag, arm, "test", "two")
    cfg = yaml.safe_load(text)
    assert cfg["stage"] == "final-t9-base_b48-two" and cfg["arm"] == "base" and cfg["batch_size"] == 48
    assert cfg["max_steps"] == 8000 and cfg["eval_every"] == 125 and cfg["seed"] == 0


def test_dry_runs_of_the_new_sweeps_and_the_plan():
    r = subprocess.run([sys.executable, "scripts/run_sweep.py", "--tag", "t9", "--arms", "base_b48", "--seeds", "9000", "9001", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "50 runs of 8000 steps" in r.stdout
    r = subprocess.run([sys.executable, "scripts/run_sweep.py", "--tag", "t10", "--arms", "selective30", "--seeds", "9100", "9101", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "50 runs of 3000 steps" in r.stdout
    r = subprocess.run([sys.executable, "scripts/run_sweep.py", "--tag", "t9", "--arms", "base_b48", "--seeds", "9000", "9001",
                        "--max-steps", "3000", "--dry-run"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "non-protocol" in r.stderr
    r = subprocess.run([sys.executable, "scripts/run_final.py", "--plan", "configs/prereg_v2_plan.yaml", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "lock tag prereg-v2" in r.stdout and "results/raw/prereg_v2" in r.stdout
