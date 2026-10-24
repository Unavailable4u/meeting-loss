"""Tests for the selective-backward pilot (scripts/pilot_selective.py)."""
import json

from scripts_loader import load_script

mod = load_script("pilot_selective")


def stats(ratio, misses=0):
    return dict(flops_mean=1e9 * ratio, ratio_of_means=ratio, paired_ratio=ratio, misses=misses, n=3,
                steps_mean=1000.0, steps_ratio=1.0, final_val=0.98)


def summary(s30, s50, t97=None):
    mk = lambda r: {0.97: stats(*(r if isinstance(r, tuple) else (r,))), 0.975: stats(*(r if isinstance(r, tuple) else (r,)))}
    return {"A": mk(1.0), "S30": mk(s30), "S50": mk(s50)}


def test_verdict_go_stop_tune():
    assert mod.pilot_verdict(summary(0.70, 1.20)) == "GO"                 # one variant clearly cheaper at both targets
    assert mod.pilot_verdict(summary(0.84, 0.90)) == "GO"                 # boundary: 0.85 is the limit
    assert mod.pilot_verdict(summary((0.70, 1), 0.95)) == "TUNE"          # cheap but a seed missed the target
    assert mod.pilot_verdict(summary(0.90, 0.95)) == "TUNE"               # saving too small to call
    assert mod.pilot_verdict(summary(1.10, 1.30)) == "STOP"
    assert mod.pilot_verdict(summary((0.80, 2), 1.05)) == "STOP"          # two missed seeds / no saving
    assert mod.pilot_verdict({"A": summary(1, 1)["A"]}) == "TUNE"         # nothing to judge


def test_control_never_decides():
    s = summary(1.10, 1.30)
    s["R30"] = {0.97: stats(0.5), 0.975: stats(0.5)}
    assert mod.pilot_verdict(s) == "STOP"


def test_crossing_cost_uses_the_shared_rule_and_charges_full_budget_on_a_miss():
    rows = [dict(step=50 * (i + 1), flops=100 * (i + 1), val_acc=a) for i, a in enumerate([0.9, 0.976, 0.97, 0.976, 0.977, 0.9])]
    assert mod.crossing_cost(rows, 0.975) == (400, 200, True)       # two consecutive evaluations, index 3
    assert mod.crossing_cost(rows, 0.99) == (600, None, False)      # never reached: full-run FLOPs


def test_summarize_pairs_seeds_against_the_baseline():
    def run(flops_per_eval, accs):
        return [dict(step=50 * (i + 1), flops=flops_per_eval * (i + 1), val_acc=a) for i, a in enumerate(accs)]
    accs = [0.9, 0.98, 0.98, 0.98]
    runs = {"A": {0: run(100, accs), 1: run(100, accs)}, "S30": {0: run(60, accs), 1: run(60, accs)}}
    s = mod.summarize(runs)
    assert abs(s["S30"][0.975]["ratio_of_means"] - 0.6) < 1e-9 and s["S30"][0.975]["misses"] == 0


def test_script_smoke_run_synthetic(tmp_path):
    out, csv_path = tmp_path / "pilot", tmp_path / "p.csv"
    verdict = mod.main(["--synthetic", "--seeds", "0", "--max-steps", "200", "--variants", "S30", "R30",
                        "--tag", "smoke", "--out-dir", str(out), "--csv", str(csv_path)])
    assert verdict in ("GO", "STOP", "TUNE")
    files = sorted(p.name for p in out.glob("*.jsonl"))
    assert files == ["A_s0.jsonl", "R30_s0.jsonl", "S30_s0.jsonl"]
    header = json.loads(open(out / "S30_s0.jsonl").readline())
    assert header["config"]["arm"] == "selective" and header["pilot"] and "git_commit" in header
    # re-running skips existing raw files (raw logs are never overwritten)
    mod.main(["--synthetic", "--seeds", "0", "--max-steps", "200", "--variants", "S30", "R30",
              "--tag", "smoke", "--out-dir", str(out), "--csv", str(csv_path)])
    assert csv_path.read_text().splitlines()[0].startswith("variant,target")
