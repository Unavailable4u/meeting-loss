"""Tests that need torch: the timing script (Look 1) and the sel_frac option of the pilot script (Look 3), on synthetic data."""
import csv

import torch

from scripts_loader import load_script


def test_each_arm_changes_the_parameters_in_one_step():
    mod = load_script("time_selective")
    base = dict(stage="x", arm="base", seed=0, act="relu", lr=1e-3, wd=1e-4, batch_size=64, max_steps=100, eval_every=10,
                lam_aux=0.0, lr_schedule="constant")
    x, y = next(iter(mod.synthetic_loaders(64, 0)[0]))
    for cfg in (base, {**base, "arm": "selective", "sel_frac": 0.3, "sel_warmup": 0, "sel_mode": "top"}):
        step, model, _ = mod.make_step(cfg, "cpu")
        before = [p.detach().clone() for p in model.parameters()]
        step(x, y)
        assert any(not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters())), cfg["arm"]


def test_timing_script_runs_on_synthetic_data(tmp_path):
    mod = load_script("time_selective")
    out = tmp_path / "t.csv"
    verdict = mod.main(["--synthetic", "--warmup", "5", "--timed", "20", "--repeats", "2", "--threads", "1",
                        "--device", "cpu", "--csv", str(out)])
    assert verdict in ("FLOPS SAVING TRANSLATES", "PARTIAL", "DOES NOT TRANSLATE")
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4 and {r["variant"] for r in rows} == {"A", "S30"} and all(float(r["ms_per_step"]) > 0 for r in rows)


def test_pilot_script_sel_fracs_mode_reports_a_knee_or_none(tmp_path):
    pilot = load_script("pilot_selective")
    knee = pilot.main(["--synthetic", "--seeds", "0", "--max-steps", "120", "--sel-fracs", "0.3", "0.5", "--tag", "t",
                       "--out-dir", str(tmp_path / "raw"), "--csv", str(tmp_path / "p.csv")])
    assert knee in (None, 0.3, 0.5)
    assert {p.name for p in (tmp_path / "raw").glob("*.jsonl")} == {"A_s0.jsonl", "S30_s0.jsonl", "S50_s0.jsonl"}
    assert len(list(csv.reader(open(tmp_path / "p.csv")))) == 1 + 3 * 2          # header + (A, S30, S50) x two targets
