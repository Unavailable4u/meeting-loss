"""Tests for the Stage 3 plan, the lock check and the three-way outcome (decision D-019)."""
import json
import pathlib
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import final
from src.sweep import margin_verdict, overall_verdict
from src.train import run

REPO = pathlib.Path(__file__).resolve().parents[1]


# --- the plan ------------------------------------------------------------------------------------
def test_parse_seeds():
    assert final.parse_seeds("0-19") == list(range(20))
    assert final.parse_seeds("3") == [3] and final.parse_seeds("0-2,7") == [0, 1, 2, 7]
    with pytest.raises(ValueError):
        final.parse_seeds("0-3,2")


def test_stage3_plan_matches_d019():
    groups = final.load_plan()
    by = {g["name"]: g for g in groups}
    assert [g["name"] for g in groups][:4] == ["A", "D", "B", "A2"]
    assert sum(len(g["seeds"]) for g in groups) == 150
    for name in ("A", "D", "B", "A2"):
        assert by[name]["seeds"] == list(range(20)), name
    for name in set(by) - {"A", "D", "B", "A2"}:
        assert by[name]["seeds"] == list(range(10)), name
    assert {n for n, g in by.items() if g["role"] == "primary"} == {"A", "D"}
    assert {n for n, g in by.items() if g["role"] == "secondary"} == {"B"}
    assert {n for n, g in by.items() if g["role"] == "gated"} == {"C", "E"}
    assert all(g["cfg"]["max_steps"] == 3000 and g["cfg"]["stage"].endswith("-two") for g in groups)
    assert by["A"]["cfg"]["stage"] == "final-t1-base-two" and by["A2"]["cfg"]["stage"] == "final-t2-base-two"
    assert by["D"]["cfg"]["arm"] == "meet" and by["B"]["cfg"]["arm"] == "deepsup"
    assert all(max(g["seeds"]) < final.RESERVED_SEED_LIMIT for g in groups)


def test_jobs_are_seed_major_so_a_stopped_run_stays_balanced():
    jobs = final.plan_jobs(final.load_plan())
    assert len(jobs) == 150
    first = [(j.group, j.seed) for j in jobs[:11]]
    assert {s for _, s in first} == {0} and len({g for g, _ in first}) == 11
    assert [j.seed for j in jobs[-4:]] == [19] * 4


def test_plan_rejects_colliding_groups_and_bad_roles(tmp_path):
    cfg = REPO / "configs" / "tuned" / "t1_base_two.yaml"
    bad = tmp_path / "plan.yaml"
    bad.write_text(f"horizon_steps: 3000\ngroups:\n  - {{name: X, config: configs/tuned/t1_base_two.yaml, seeds: '0-1', role: primary}}\n"
                   f"  - {{name: Y, config: configs/tuned/t1_base_two.yaml, seeds: '0-1', role: primary}}\n")
    with pytest.raises(ValueError, match="collide"):
        final.load_plan(bad)
    bad.write_text("horizon_steps: 3000\ngroups:\n  - {name: X, config: configs/tuned/t1_base_two.yaml, seeds: '0-1', role: boss}\n")
    with pytest.raises(ValueError, match="role"):
        final.load_plan(bad)
    bad.write_text("horizon_steps: 3000\ngroups:\n  - {name: X, config: configs/tuned/t1_base_two.yaml, seeds: '1000', role: primary}\n")
    with pytest.raises(ValueError, match="seeds"):
        final.load_plan(bad)


# --- runner --------------------------------------------------------------------------------------
def _synthetic():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(256, 784, generator=g), torch.randint(0, 10, (256,), generator=g)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True,
                       generator=torch.Generator().manual_seed(0), drop_last=True),
            DataLoader(TensorDataset(x[:64], y[:64]), batch_size=64))


def test_run_final_job_writes_a_log_and_skips_finished_runs_even_with_another_date(tmp_path):
    groups = final.load_plan()
    job = next(j for j in final.plan_jobs(groups, tmp_path) if j.group == "D" and j.seed == 3)
    runner = lambda cfg, device: run({**cfg, "eval_every": 5, "max_steps": 20}, device="cpu", data=_synthetic())
    assert final.run_final_job(job, "cpu", runner) == "done"
    (path,) = list(tmp_path.glob("*_final-t1-meet-two_meet_s3.jsonl"))
    hdr = json.loads(path.read_text().splitlines()[0])
    assert hdr["config"]["seed"] == 3 and hdr["plan_group"] == "D" and "git_commit" in hdr
    assert len(path.read_text().splitlines()) == 1 + 4
    path.rename(tmp_path / "20200101_final-t1-meet-two_meet_s3.jsonl")      # same run, other date
    assert final.run_final_job(job, "cpu", lambda c, d: pytest.fail("must not rerun")) == "skipped"
    assert not list(tmp_path.glob("2026*"))


# --- lock check ----------------------------------------------------------------------------------
def _git(repo, *a):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=repo,
                          capture_output=True, text=True, check=True)


def test_lock_status_requires_the_tag_and_unchanged_locked_files(tmp_path):
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    (repo / "configs" / "tuned").mkdir(parents=True)
    (repo / "docs").mkdir()
    for f in ["src/a.py", "configs/stage3_plan.yaml", "configs/tuned/x.yaml", "docs/HYPOTHESES.md", "docs/METHOD.md", "notes.txt"]:
        (repo / f).write_text("v1\n")
    _git(repo, "init", "-q"); _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "lock")
    ok, msg = final.lock_status(repo)
    assert not ok and "does not exist" in msg
    _git(repo, "tag", final.LOCK_TAG)
    assert final.lock_status(repo) == (True, "ok")
    (repo / "notes.txt").write_text("v2\n")                                    # an unlocked file may change
    assert final.lock_status(repo)[0]
    (repo / "src" / "a.py").write_text("v2\n")                                 # a locked file may not
    ok, msg = final.lock_status(repo)
    assert not ok and "src/a.py" in msg
    _git(repo, "checkout", "src/a.py")
    (repo / "configs" / "tuned" / "new.yaml").write_text("x\n")                # nor may a new tuned config appear
    ok, msg = final.lock_status(repo)
    assert not ok and "new.yaml" in msg


def test_runner_refuses_without_the_tag():
    r = subprocess.run([sys.executable, "scripts/run_final.py", "--groups", "A", "--limit", "1"],
                       cwd=REPO, capture_output=True, text=True)
    ok, _ = final.lock_status(REPO)
    if not ok:
        assert r.returncode != 0 and "LOCK CHECK FAILED" in (r.stdout + r.stderr)


# --- three-way outcome ---------------------------------------------------------------------------
def test_margin_verdict_cases():
    assert margin_verdict(-30.0, -35.0, -25.0) == "supported"
    assert margin_verdict(-15.0, -20.0, -1.0) == "supported"                  # point exactly at the margin
    assert margin_verdict(-16.0, -25.0, 2.0) == "inconclusive"                # interval includes zero
    assert margin_verdict(-10.0, -20.0, -1.0) == "inconclusive"               # gain too small, interval not clear of 15%
    assert margin_verdict(-5.0, -12.0, 3.0) == "refuted"                      # interval excludes a 15% gain
    assert margin_verdict(8.0, 2.0, 14.0) == "refuted"                        # worse than the reference
    assert margin_verdict(-14.0, -14.5, -13.5) == "refuted"                   # tight interval above -15


def test_a_supported_gain_must_exceed_the_retuning_shift():
    assert margin_verdict(-30.0, -35.0, -25.0, shift=-10.0) == "supported"
    assert margin_verdict(-30.0, -35.0, -25.0, shift=12.0) == "supported"     # shift is compared in size
    assert margin_verdict(-30.0, -35.0, -25.0, shift=-30.0) == "inconclusive"
    assert margin_verdict(-30.0, -35.0, -25.0, shift=-41.0) == "inconclusive"
    assert margin_verdict(-5.0, -12.0, 3.0, shift=-40.0) == "refuted"          # the shift never turns refuted into supported


def test_overall_verdict_needs_every_primary_target_to_agree():
    assert overall_verdict(["supported", "supported"]) == "supported"
    assert overall_verdict(["refuted", "refuted"]) == "refuted"
    assert overall_verdict(["supported", "refuted"]) == "inconclusive"
    assert overall_verdict(["supported", "inconclusive"]) == "inconclusive"
    assert overall_verdict([]) == "inconclusive"


def _write_group(tmp_path, name, seeds, flops_per_eval, cross_at=5, n_evals=12):
    for s in seeds:
        accs = [0.90 + 0.005 * i for i in range(cross_at)] + [0.98] * (n_evals - cross_at)
        rows = [dict(step=50 * (i + 1), flops=flops_per_eval * (i + 1) * (1 + 0.0 * s), val_acc=a) for i, a in enumerate(accs)]
        (tmp_path / f"{name}_s{s}.jsonl").write_text(
            json.dumps({"_header": True, "config": {"seed": s}}) + "\n" + "\n".join(json.dumps(r) for r in rows) + "\n")


def _compare(tmp_path, *extra):
    cmd = [sys.executable, "scripts/analyze.py", "compare", "--ref", "A", "--targets", "0.97", "0.975",
           "--n-boot", "500", "--alpha", "0.025", "--margin", "15", "--primary-targets", "0.97", "0.975",
           "--group", f"A={tmp_path}/A_s*.jsonl", "--group", f"D={tmp_path}/D_s*.jsonl", *extra]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def test_compare_cli_reports_supported_refuted_and_the_shift_rule(tmp_path):
    _write_group(tmp_path, "A", range(6), 100.0)
    _write_group(tmp_path, "D", range(6), 70.0)                                # 30% fewer FLOPs per run
    r = _compare(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "D: supported" in r.stdout and "97.5% CI" in r.stdout
    _write_group(tmp_path, "A2", range(6), 70.0)                               # a second tuning of A that is as good as D
    r = _compare(tmp_path, "--group", f"A2={tmp_path}/A2_s*.jsonl", "--shift-group", "A2")
    assert "D: inconclusive" in r.stdout and "A2:" not in r.stdout.split("three-way outcome")[1]
    for p in tmp_path.glob("D_s*.jsonl"):
        p.unlink()
    _write_group(tmp_path, "D", range(6), 100.0)                               # no change at all
    r = _compare(tmp_path)
    assert "D: refuted" in r.stdout
