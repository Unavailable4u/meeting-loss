"""Tests for the pure logic of the track-S looks (exploratory/looks.py) and the small-batch control pipeline
(scripts/run_batch_control.py, run here with a fake runner, so no torch and no MNIST are needed)."""
import json
import math

import pytest

from exploratory import looks as L
from scripts_loader import load_script


def rows(accs, flops_per_eval=100.0, every=50):
    return [dict(step=every * (i + 1), flops=flops_per_eval * (i + 1), val_acc=a) for i, a in enumerate(accs)]


FAST = [0.90, 0.972, 0.973, 0.978, 0.979, 0.980]      # reaches 0.97 at index 1 (two consecutive), 0.975 at index 3
SLOW = [0.90, 0.95, 0.972, 0.973, 0.976, 0.977]       # reaches 0.97 at index 2, 0.975 at index 4


# ---------------------------------------------------------------- design
def test_horizon_and_evaluation_spacing_keep_the_samples_constant():
    assert [L.horizon_steps(b) for b in (32, 48, 64, 128)] == [12000, 8000, 6000, 3000]
    assert [L.eval_every(b) for b in (32, 48, 64, 128)] == [200, 133, 100, 50]
    for b in (32, 64, 128):                                   # exact for the sizes that divide 6,400
        assert L.eval_every(b) * b == 6400


def test_sampled_learning_rates_are_deterministic_in_range_and_depend_on_tag_and_batch():
    a = L.sample_lrs("sb1", 32)
    assert a == L.sample_lrs("sb1", 32) and len(a) == L.N_TRIALS
    assert all(L.LR_RANGE[0] <= lr <= L.LR_RANGE[1] for lr in a)
    assert a != L.sample_lrs("sb1", 48) and a != L.sample_lrs("sb2", 32)
    assert len(L.sample_lrs("sb1", 64, 3)) == 3


def test_trial_and_variant_configs_change_only_what_the_design_says():
    base = dict(stage="x", arm="base", lr=0.0013, wd=1e-4, batch_size=128, max_steps=3000, eval_every=50, act="relu")
    t = L.trial_config(base, 48, 0.002, "s")
    assert t == {**base, "stage": "s", "batch_size": 48, "lr": 0.002, "max_steps": 8000, "eval_every": 133}
    s = L.s30_config(base, "s")
    assert s == {**base, "stage": "s", "arm": "selective", "sel_frac": 0.3, "sel_warmup": 100, "sel_mode": "top"}
    assert L.eval_config(base, "A", "e") == {**base, "stage": "e"}
    best = {"b32": {**base, "batch_size": 32, "lr": 0.003}}
    assert L.eval_config(base, "b32", "e32", best) == {**base, "batch_size": 32, "lr": 0.003, "stage": "e32"}
    assert L.eval_stage("sb1", "A") == "sb1-eval-A" and L.trial_stage("sb1", 32, 3) == "sb1-tune-b32-t03"


# ---------------------------------------------------------------- scoring and ratios
def test_crossing_cost_uses_two_consecutive_evaluations_and_charges_misses_the_full_budget():
    assert L.crossing_cost(rows(FAST), 0.97) == (200.0, 100, True)
    assert L.crossing_cost(rows(FAST), 0.975) == (400.0, 200, True)
    assert L.crossing_cost(rows([0.9, 0.99, 0.9, 0.9]), 0.97) == (400.0, None, False)    # a single spike is not a crossing


def test_ratios_are_ratios_of_mean_flops_to_target_on_the_common_seeds():
    runs = {"A": {0: rows(FAST), 1: rows(FAST)},
            "X": {0: rows(FAST, flops_per_eval=50.0), 1: rows(SLOW, flops_per_eval=50.0), 2: rows(FAST)}}
    r = L.ratios(runs, "A")
    assert r["A"][0.97]["ratio_of_means"] == 1.0
    assert r["X"][0.97]["n"] == 2                                       # seed 2 is not in the reference
    assert math.isclose(r["X"][0.97]["ratio_of_means"], (100 + 150) / (200 + 200))   # X: 2*50 and 3*50 ; A: 2*100 twice
    assert r["X"][0.97]["misses"] == 0


def test_tuning_picks_the_cheapest_trial_and_a_miss_costs_the_full_budget():
    trial_runs = {0: {1200: rows(SLOW), 1201: rows(SLOW)},
                  1: {1200: rows(FAST), 1201: rows(FAST)},
                  2: {1200: rows([0.9] * 6), 1201: rows(FAST)}}
    table, best = L.tune_table(trial_runs)
    assert best["trial"] == 1 and best["n_reached"] == 2
    assert table[2]["n_reached"] == 1 and table[2]["score"] == (600.0 + 200.0) / 2


# ---------------------------------------------------------------- verdicts
def _summary(r32, r48, r64, s30):
    def one(x):
        return {t: dict(flops_mean=x * 1e9, ratio_of_means=x, misses=0, n=3, steps_ratio=1.0) for t in L.TARGETS}
    return {"A": one(1.0), "b32": one(r32), "b48": one(r48), "b64": one(r64), "S30": one(s30)}


def test_batch_control_verdicts():
    small = ["b32", "b48", "b64"]
    problem = L.batch_control_verdict(_summary(0.80, 1.1, 1.2, 0.61), small)
    assert problem["verdict"] == "BASELINE PROBLEM" and problem["best"] == "b32" and problem["s_beats_best"]
    assert L.batch_control_verdict(_summary(1.05, 1.1, 1.3, 0.61), small)["verdict"] == "A WAS FINE"
    assert L.batch_control_verdict(_summary(0.95, 1.1, 1.2, 0.61), small)["verdict"] == "PARTIAL"
    close = L.batch_control_verdict(_summary(0.70, 1.1, 1.2, 0.65), small)       # S30 / best = 0.93 > 0.85
    assert close["verdict"] == "BASELINE PROBLEM" and not close["s_beats_best"]


def test_best_small_batch_uses_the_worst_case_over_the_targets():
    s = _summary(0.9, 0.9, 0.9, 0.6)
    s["b48"][0.975]["ratio_of_means"] = 1.2
    s["b64"][0.97]["ratio_of_means"] = 0.8
    assert L.best_small_batch(s, ["b32", "b48", "b64"]) == "b32"      # max ratios: 0.9, 1.2, 0.9 -> tie goes to the first


def test_timing_verdict_and_summary():
    assert L.timing_verdict(0.80) == "FLOPS SAVING TRANSLATES" and L.timing_verdict(0.85) == "FLOPS SAVING TRANSLATES"
    assert L.timing_verdict(0.95) == "PARTIAL" and L.timing_verdict(1.0) == "DOES NOT TRANSLATE"
    res = L.summarize_timing({"A": [0.010, 0.011, 0.012], "S30": [0.008, 0.007, 0.009]})
    assert math.isclose(res["ratio"], 0.008 / 0.011) and res["verdict"] == "FLOPS SAVING TRANSLATES"
    assert math.isclose(res["median_ms"]["A"], 11.0)


def test_knee_is_the_smallest_fraction_meeting_the_rule_with_no_miss():
    fr = {"S10": 0.1, "S20": 0.2, "S30": 0.3, "S50": 0.5}

    def one(r, miss=0):
        return {t: dict(ratio_of_means=r, misses=miss) for t in L.TARGETS}
    assert L.knee({"S10": one(0.9), "S20": one(0.8), "S30": one(0.6), "S50": one(0.7)}, fr) == 0.2
    assert L.knee({"S10": one(0.5, miss=1), "S20": one(0.9), "S30": one(0.7)}, fr) == 0.3        # a missed seed disqualifies
    assert L.knee({"S10": one(0.9), "S20": one(0.95)}, fr) is None
    assert L.frac_name(0.1) == "S10" and L.frac_name(0.3) == "S30" and L.frac_name(0.5) == "S50"


# ---------------------------------------------------------------- the small-batch pipeline with a fake runner
def fake_runner(calls):
    def run(cfg, device):
        calls.append((cfg["stage"], cfg["seed"]))
        b, lr = cfg["batch_size"], cfg["lr"]
        fps = 1000.0 * b * (0.5 if cfg["arm"] == "selective" else 1.0)
        out, step = [], 0
        while step + cfg["eval_every"] <= cfg["max_steps"]:
            step += cfg["eval_every"]
            out.append(dict(step=step, flops=fps * step, val_acc=0.985 * (1 - math.exp(-step * b * lr / 4.0))))
        return out
    return run


def test_batch_control_pipeline_end_to_end(tmp_path, capsys):
    mod = load_script("run_batch_control")
    calls = []
    argv = ["--tag", "t", "--smoke", "--n-trials", "2", "--max-steps", "600", "--device", "cpu",
            "--out-dir", str(tmp_path / "raw"), "--cfg-dir", str(tmp_path / "cfg"), "--csv", str(tmp_path / "out.csv")]
    ver = mod.main(argv, runner=fake_runner(calls))
    assert len(calls) == 3 * 2 * 2 + 5 * 3                       # tuning: 3 batch sizes x 2 trials x 2 seeds; evaluation: 5 variants x 3 seeds
    assert ver["verdict"] in ("BASELINE PROBLEM", "A WAS FINE", "PARTIAL") and ver["best"] in ("b32", "b48", "b64")
    best = json.loads((tmp_path / "cfg" / "best.json").read_text())
    assert set(best) == {"b32", "b48", "b64"}
    for b in (32, 48, 64):                                       # in this fake world a higher learning rate always wins
        lrs = mod.L.sample_lrs("t", b, 2)
        assert best[f"b{b}"]["lr"] == max(lrs)
    assert (tmp_path / "out.csv").read_text().splitlines()[0].startswith("variant,target")
    n = len(calls)
    mod.main(argv, runner=fake_runner(calls))                    # resumable: nothing is run twice
    assert len(calls) == n


def test_design_is_fixed_outside_smoke_mode(tmp_path):
    mod = load_script("run_batch_control")
    with pytest.raises(SystemExit):
        mod.main(["--tag", "x", "--n-trials", "3", "--dry-run"])
    with pytest.raises(SystemExit):
        mod.main(["--tag", "x", "--smoke"])                      # smoke needs its own directories
    mod.main(["--tag", "x", "--dry-run", "--cfg-dir", str(tmp_path / "c"), "--out-dir", str(tmp_path / "o")])   # dry run writes the trial configs only
    assert len(list((tmp_path / "c").glob("*.yaml"))) == 3 * L.N_TRIALS


def test_changed_trial_config_is_refused(tmp_path):
    mod = load_script("run_batch_control")
    path = tmp_path / "x.yaml"
    mod.write_config(path, {"a": 1})
    mod.write_config(path, {"a": 1})
    with pytest.raises(SystemExit):
        mod.write_config(path, {"a": 2})
