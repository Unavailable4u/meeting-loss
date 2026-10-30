"""Machinery for prereg-v3 (decision D-026): derived configs, the plan, the analysis of H8, H9 and H10, the secondary readings
(cached count, warm-up credit), the provenance report, and the runner's dry run. Written before the lock; no torch needed."""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import yaml

from src import final, prereg3

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scripts_loader import load_script  # noqa: E402

PLAN = REPO / "configs" / "prereg_v3_plan.yaml"
SEEDS = range(2300, 2320)


# ----------------------------------------------------------------------------- constants, derived configs, the committed plan
def test_error_budget_is_shared_by_the_three_hypotheses_and_margins_are_as_decided():
    assert prereg3.ALPHA == pytest.approx(0.05 / 3) and prereg3.N_HYPOTHESES == 3
    assert prereg3.MARGINS == {"H8": 15.0, "H9": 10.0, "H10": 15.0}
    assert prereg3.REFERENCES == {"H8": "A1", "H9": "S30", "H10": "SBT"} and prereg3.PRIMARY == "S20w0"
    assert prereg3.TARGETS == (0.97, 0.975)


def test_derived_configs_are_arm_a1_with_only_the_selective_keys_changed_and_match_the_look_2_configs():
    a1 = yaml.safe_load(open(REPO / "configs" / "tuned" / "t1_base_two.yaml"))
    expect = {"S20w0": (0.2, 0, "top"), "S30": (0.3, 100, "top"), "S20": (0.2, 100, "top"), "S15": (0.15, 100, "top"),
              "S20w10": (0.2, 10, "top"), "R20w0": (0.2, 0, "random")}
    assert set(prereg3.DERIVED) == set(expect)
    for name, (frac, warm, mode) in expect.items():
        want = prereg3.derive_config(name)
        got = yaml.safe_load(open(REPO / "configs" / "prereg_v3" / f"{name}.yaml"))
        assert got == want
        assert {k for k in set(a1) | set(want) if a1.get(k) != want.get(k)} == {"stage", "arm", "sel_frac", "sel_warmup", "sel_mode"}, name
        assert want["arm"] == "selective" and (want["sel_frac"], want["sel_warmup"], want["sel_mode"]) == (frac, warm, mode)
        assert want["stage"] == f"prereg3-{name}" and want["max_steps"] == 3000
        look2 = REPO / "tracks" / "improve" / "configs_look2" / f"l2-{name}.yaml"
        if look2.exists():                                   # the confirmatory arm is the exploratory arm of look 2, same keys and values
            old = yaml.safe_load(open(look2))
            assert {k: v for k, v in old.items() if k != "stage"} == {k: v for k, v in want.items() if k != "stage"}, name
    with pytest.raises(ValueError):
        prereg3.derive_config("S99")


def test_committed_plan_matches_d026():
    groups = final.load_plan(PLAN)                           # every config file exists
    meta = final.plan_meta(PLAN)
    assert meta["lock_tag"] == "prereg-v3" and meta["out_dir"] == "results/raw/prereg_v3" and meta["seed_block"] == [2300, 2319]
    assert meta["shift_rule"] == "max_gain" and meta["horizon_steps"] == 3000
    assert "configs/prereg_v3" in meta["locked_paths"] and "src" in meta["locked_paths"] and "configs/tuned" in meta["locked_paths"]
    by = {g["name"]: g for g in groups}
    assert set(by) == {"A1", "S20w0", "S30", "SBT", "A2", "A5", "A6", "A7", "A8", "S20", "S15", "S20w10", "R20w0"}
    assert all(g["seeds"] == list(SEEDS) and g["horizon"] == 3000 for g in groups)
    assert sum(len(g["seeds"]) for g in groups) == 260
    roles = {n: g["role"] for n, g in by.items()}
    assert roles["A1"] == "reference" and roles["S20w0"] == "primary" and roles["S30"] == "control" and roles["SBT"] == "control"
    assert {n for n, r in roles.items() if r == "shift"} == {"A2", "A5", "A6", "A7", "A8"}
    assert {n for n, r in roles.items() if r == "descriptive"} == {"S20", "S15", "S20w10", "R20w0"}
    assert by["SBT"]["config"] == "configs/tuned/t11_sb_two.yaml" and by["SBT"]["cfg"]["arm"] == "sb"
    assert by["A1"]["config"] == "configs/tuned/t1_base_two.yaml"
    for n, t in (("A2", "t2"), ("A5", "t5"), ("A6", "t6"), ("A7", "t7"), ("A8", "t8")):
        assert by[n]["config"] == f"configs/tuned/{t}_base_two.yaml"
    for n in prereg3.DERIVED:
        assert by[n]["config"] == f"configs/prereg_v3/{n}.yaml"
    assert not {g["cfg"]["stage"] for g in groups} & {"prereg2-S30", "l2-S20w0"}            # no log of an earlier study can be picked up
    assert all(2300 <= s <= 2319 for g in groups for s in g["seeds"])


def test_dry_run_of_the_plan_lists_13_groups_and_260_runs():
    r = subprocess.run([sys.executable, "scripts/run_final.py", "--plan", "configs/prereg_v3_plan.yaml", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "lock tag prereg-v3" in r.stdout and "results/raw/prereg_v3" in r.stdout and "13 groups, 260 runs" in r.stdout


# ----------------------------------------------------------------------------- H8, H9, H10 on synthetic scores
def _scores(by_seed):
    return {s: {t: dict(reached=True, steps=int(f // 1000), flops=f, score=f, budget=f, final_val=0.98, late_val=0.98)
                for t, f in d.items()} for s, d in by_seed.items()}


def _group(rng, base, rel, sd=0.04):
    out = {}
    for s in SEEDS:
        n = float(np.exp(rng.normal(0, sd)))
        out[s] = {0.97: base * rel[0] * n, 0.975: base * rel[1] * n}
    return _scores(out)


ROLES = {"A1": "reference", "S20w0": "primary", "S30": "control", "SBT": "control",
         **{k: "shift" for k in ("A2", "A5", "A6", "A7", "A8")}, "S20": "descriptive"}


def _synthetic(rng, s20w0=(0.50, 0.49), s30=(0.65, 0.62), sbt=(1.30, 1.20), shifts=None):
    shifts = shifts or {"A2": (1.2, 1.15), "A5": (0.95, 0.97), "A6": (1.05, 1.03), "A7": (0.98, 0.99), "A8": (1.02, 1.04)}
    g = {"A1": _group(rng, 100e9, (1.0, 1.0)), "S20w0": _group(rng, 100e9, s20w0), "S30": _group(rng, 100e9, s30),
         "SBT": _group(rng, 100e9, sbt), "S20": _group(rng, 100e9, (0.58, 0.56))}
    g.update({k: _group(rng, 100e9, v) for k, v in shifts.items()})
    return g


def test_all_three_hypotheses_supported_and_the_shift_applies_to_h8_only():
    res = prereg3.h8_h10(_synthetic(np.random.default_rng(0)), ROLES, "max_abs", n_boot=2000)
    assert [res[h]["overall"] for h in ("H8", "H9", "H10")] == ["supported"] * 3
    assert res["H8"]["per_target"][0.97]["shift"] == pytest.approx(20.0, abs=3.0)
    assert res["H9"]["per_target"][0.97]["shift"] is None and res["H10"]["per_target"][0.97]["shift"] is None
    assert res["H8"]["margin"] == 15.0 and res["H9"]["margin"] == 10.0 and res["H10"]["margin"] == 15.0
    assert all(r["n_better"] == 20 and r["n_seeds"] == 20 for r in res["H9"]["per_target"].values())
    assert res["alpha"] == pytest.approx(0.05 / 3)


def test_a_tuning_of_a_as_good_as_s20w0_makes_h8_inconclusive_but_not_h9_or_h10():
    rng = np.random.default_rng(1)
    g = _synthetic(rng, shifts={"A2": (1.1, 1.1), "A5": (0.48, 0.5), "A6": (1.0, 1.0), "A7": (1.0, 1.0), "A8": (1.0, 1.0)})
    res = prereg3.h8_h10(g, ROLES, "max_gain", n_boot=2000)
    assert res["H8"]["overall"] == "inconclusive" and res["H9"]["overall"] == "supported" and res["H10"]["overall"] == "supported"


def test_each_hypothesis_has_its_own_margin():
    rng = np.random.default_rng(2)
    # 8% below S30 clears no 10% margin (inconclusive or refuted), 8% below nothing else is needed for H8 and H10
    res = prereg3.h8_h10(_synthetic(rng, s20w0=(0.60, 0.58), s30=(0.65, 0.63)), ROLES, "max_gain", n_boot=2000)
    assert res["H9"]["overall"] in ("inconclusive", "refuted") and res["H9"]["overall"] != "supported"
    assert res["H8"]["overall"] == "supported" and res["H10"]["overall"] == "supported"
    # S20w0 no better than S30: refuted for H9; a Selective-Backprop as cheap as S20w0 is refuted for H10
    res = prereg3.h8_h10(_synthetic(np.random.default_rng(3), s20w0=(0.65, 0.62), s30=(0.65, 0.62), sbt=(0.66, 0.63)), ROLES, "max_gain", n_boot=2000)
    assert res["H9"]["overall"] == "refuted" and res["H10"]["overall"] == "refuted"


def test_the_two_targets_must_agree():
    rng = np.random.default_rng(4)
    res = prereg3.h8_h10(_synthetic(rng, sbt=(1.30, 0.50)), ROLES, "max_gain", n_boot=2000)   # SBT as cheap as S20w0 at 0.975 only
    assert res["H10"]["per_target"][0.97]["verdict"] == "supported" and res["H10"]["per_target"][0.975]["verdict"] == "refuted"
    assert res["H10"]["overall"] == "inconclusive"


def test_wrong_roles_and_unpaired_inputs_are_rejected():
    rng = np.random.default_rng(5)
    g = _synthetic(rng)
    with pytest.raises(ValueError, match="exactly one"):
        prereg3.h8_h10({"A1": g["A1"]}, {"A1": "reference"}, "max_gain", n_boot=100)
    with pytest.raises(ValueError, match="SBT"):
        prereg3.h8_h10({k: v for k, v in g.items() if k != "SBT"}, {k: v for k, v in ROLES.items() if k != "SBT"}, "max_gain", n_boot=100)
    with pytest.raises(ValueError, match="role 'control'"):
        prereg3.h8_h10(g, dict(ROLES, SBT="descriptive"), "max_gain", n_boot=100)
    with pytest.raises(ValueError, match="primary group"):
        prereg3.h8_h10(g, dict(ROLES, S20w0="descriptive", S20="primary"), "max_gain", n_boot=100)
    bad = dict(g, S30={s: v for s, v in g["S30"].items() if s != 2305})
    with pytest.raises(ValueError, match="same seeds"):
        prereg3.h8_h10(bad, ROLES, "max_gain", n_boot=100)


def test_noise_floor_is_the_rule_of_prereg_v2():
    large = {"a": {0.97: 100.0, 0.975: 100.0}, "b": {0.97: 120.0, 0.975: 118.0}}
    assert prereg3.noise_floor(large)["outcome"] == "LARGE"


# ----------------------------------------------------------------------------- secondary quantities from a log
def _rows(total=3000, every=50, warm=100, full=1000.0, sel=400.0, reach=700, cached_share=None):
    rows = []
    for step in range(every, total + 1, every):
        flops = min(step, warm) * full + max(step - warm, 0) * sel
        r = dict(step=step, flops=flops, val_acc=0.985 if step >= reach else 0.90)
        if cached_share is not None:
            r["flops_cached"] = flops * cached_share
        rows.append(r)
    return rows


def test_step_costs_credit_and_cached_score_from_a_log():
    rows = _rows(warm=100)
    assert prereg3.selective_step_flops(rows, 100) == pytest.approx(400.0)
    assert prereg3.full_step_flops(_rows(warm=10 ** 9)) == pytest.approx(1000.0)
    # crossing at step 700 (two consecutive evaluations; reported at the first): all 100 warm-up steps were taken
    assert prereg3.warmup_credit(rows, 100, 0.97, 1000.0) == pytest.approx(100 * 600.0)
    # crossing before the warm-up has ended: only the steps taken count
    early = _rows(warm=100, reach=50)
    assert prereg3.warmup_credit(early, 100, 0.97, 1000.0) == pytest.approx(50 * 600.0)
    # a short warm-up that ends between two evaluations: the cost per selective step is still read correctly
    assert prereg3.selective_step_flops(_rows(warm=10), 10) == pytest.approx(400.0)
    assert prereg3.warmup_credit(_rows(warm=10), 10, 0.97, 1000.0) == pytest.approx(10 * 600.0)
    assert prereg3.warmup_credit(_rows(warm=0), 0, 0.97, 1000.0) == 0.0
    # cached count: taken at the same crossing, None when the log has none, whole run when the target is never reached
    sb = _rows(warm=0, cached_share=0.8)
    assert prereg3.score_run_cached(sb, 0.97) == pytest.approx(0.8 * 700 * 400.0)
    assert prereg3.score_run_cached(rows, 0.97) is None
    assert prereg3.score_run_cached(_rows(warm=0, reach=10 ** 6, cached_share=0.8), 0.97) == pytest.approx(0.8 * 3000 * 400.0)


# ----------------------------------------------------------------------------- the script end to end
def _write_cfg(repo, name, stage, arm, warm=None):
    cfg = dict(stage=stage, arm=arm, max_steps=3000, seed=0)
    if warm is not None:
        cfg["sel_warmup"] = warm
    p = repo / "configs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return f"configs/{name}"


def _fake_study(tmp_path, commit="abc123", dirty=False, short=None):
    """Groups of the plan with fake logs: full step 1000, selective step 400 (200 for S20w0's rows), SBT 2.3x per update with a cached count."""
    spec = {  # name: (role, arm, warm, reach step, per-selective-step cost)
        "A1": ("reference", "base", None, 1400, None), "S20w0": ("primary", "selective", 0, 800, 400.0), "S30": ("control", "selective", 100, 1000, 500.0),
        "SBT": ("control", "sb", None, 800, 2300.0), "A2": ("shift", "base", None, 1500, None), "S20": ("descriptive", "selective", 100, 900, 400.0),
        "S20w10": ("descriptive", "selective", 10, 820, 400.0), "R20w0": ("descriptive", "selective", 0, 1300, 400.0)}
    out_dir = tmp_path / "raw"
    groups = []
    for n, (role, arm, warm, reach, sel) in spec.items():
        cfg = _write_cfg(tmp_path, f"{n}.yaml", f"st-{n}", arm, warm)
        groups.append(dict(name=n, config=cfg, seeds="2300-2305", role=role))
        for s in range(2300, 2306):
            r = reach + 50 * (s % 3)
            if arm == "base":
                rows = _rows(warm=10 ** 9, full=1000.0, reach=r)
            elif arm == "sb":
                rows = _rows(warm=0, sel=sel, reach=r, cached_share=0.8)
            else:
                rows = _rows(warm=warm, full=1000.0, sel=sel, reach=r)
            if short == (n, s):
                rows = rows[:30]
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"20261101_st-{n}_{arm}_s{s}.jsonl", "w") as f:
                f.write(json.dumps({"_header": True, "config": dict(stage=f"st-{n}", arm=arm, seed=s, max_steps=3000), "plan_group": n,
                                    "git_commit": commit, "git_dirty": dirty}) + "\n")
                for row in rows:
                    f.write(json.dumps(row) + "\n")
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump(dict(horizon_steps=3000, seed_block="2300-2305", out_dir=str(out_dir), lock_tag="prereg-v3",
                                        shift_rule="max_gain", groups=groups)))
    return plan, out_dir


def test_cli_end_to_end_secondary_readings_and_provenance(tmp_path):
    plan, out_dir = _fake_study(tmp_path)
    mod = load_script("analyze_prereg3")
    res = mod.analyze(plan, repo=tmp_path, n_boot=500)
    assert res["H8"]["overall"] == "supported" and res["H9"]["overall"] == "supported" and res["H10"]["overall"] == "supported"
    assert res["provenance_ok"] and list(res["provenance"].values()) == [48]
    sec = res["secondary"]
    assert {"H10_cached", "S20w0_vs_S20", "S20w0_vs_S20_credited", "S20w0_vs_S20w10", "control_R20w0", "S20w0_vs_S20_steps"} <= set(sec)
    # S20 (warm-up 100, same selective cost) against S20w0: credited FLOPs differ from the logged ones by 100 x (1000 - 400) = 60,000 FLOPs
    logged = sec["S20w0_vs_S20"]["change"][0.97][0]
    credited = sec["S20w0_vs_S20_credited"]["change"][0.97][0]
    assert credited > logged                                  # charging the warm-up like selective steps removes part of S20w0's advantage
    # the cached count of SBT is 0.8 of its logged count, so S20w0 looks worse against it than against the uncached count
    assert sec["H10_cached"]["change"][0.97][0] > res["H10"]["per_target"][0.97]["point"]
    text = mod.report(res)
    for needle in ("98.33%", "VERDICT H8: supported", "VERDICT H9", "VERDICT H10", "run headers: abc123|dirty=False x48", "noise floor", "cached count"):
        assert needle in text, needle
    mod.write_csv(res, tmp_path / "out.csv")
    header = (tmp_path / "out.csv").read_text().splitlines()[0]
    assert "verdict" in header and "vs_S30_pct" in header


def test_a_dirty_tree_or_a_second_commit_is_reported(tmp_path):
    plan, out_dir = _fake_study(tmp_path, dirty=True)
    mod = load_script("analyze_prereg3")
    res = mod.analyze(plan, repo=tmp_path, n_boot=100)
    assert not res["provenance_ok"] and "WARNING" in mod.report(res)
    # rewrite the header of one run with another commit
    f = next(out_dir.glob("*st-S30_selective_s2301.jsonl"))
    lines = f.read_text().splitlines()
    hdr = json.loads(lines[0]); hdr["git_commit"], hdr["git_dirty"] = "zzz999", False
    f.write_text("\n".join([json.dumps(hdr)] + lines[1:]) + "\n")
    assert not mod.analyze(plan, repo=tmp_path, n_boot=100)["provenance_ok"]


def test_incomplete_or_unfinished_groups_are_refused(tmp_path):
    plan, out_dir = _fake_study(tmp_path, short=("S30", 2303))
    mod = load_script("analyze_prereg3")
    with pytest.raises(ValueError, match="unfinished"):
        mod.analyze(plan, repo=tmp_path, n_boot=100)
    plan2, out_dir2 = _fake_study(tmp_path / "second")
    next(out_dir2.glob("*st-SBT_sb_s2304.jsonl")).unlink()
    with pytest.raises(ValueError, match="missing"):
        mod.analyze(plan2, repo=tmp_path / "second", n_boot=100)
