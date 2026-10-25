"""Tests for the regression-check script (scripts/verify_regression.py); no torch needed."""
import json

from scripts_loader import load_script

vr = load_script("verify_regression")


def rows(accs, step0=50, every=50, fps=1000):
    return [{"step": step0 + i * every, "flops": fps * (step0 + i * every), "val_acc": a} for i, a in enumerate(accs)]


ACCS = [0.90, 0.95, 0.968, 0.971, 0.972, 0.976, 0.976, 0.978]


def test_identical_logs_pass_bit_identical():
    r = vr.compare_logs(rows(ACCS), rows(ACCS))
    assert r["verdict"] == "PASS (bit-identical)" and r["max_abs_acc_diff"] == 0.0


def test_small_noise_passes_within_tolerance():
    noisy = [a + (0.0005 if i % 2 else -0.0005) for i, a in enumerate(ACCS)]
    r = vr.compare_logs(rows(ACCS), rows(noisy))
    assert r["verdict"] == "PASS (within tolerance)"


def test_large_difference_fails():
    bad = list(ACCS)
    bad[3] = 0.90
    r = vr.compare_logs(rows(ACCS), rows(bad))
    assert r["verdict"] == "FAIL" and any("exceeds" in x for x in r["reasons"])


def test_flops_difference_fails_even_if_accuracy_identical():
    new = rows(ACCS)
    new[4]["flops"] += 1
    r = vr.compare_logs(rows(ACCS), new)
    assert r["verdict"] == "FAIL" and any("FLOP" in x for x in r["reasons"])


def test_different_steps_fail():
    r = vr.compare_logs(rows(ACCS), rows(ACCS[:-1]))
    assert r["verdict"] == "FAIL" and not r["same_steps"]


def test_steps_to_target_use_two_consecutive_rule():
    # 0.971 at index 3 and 0.972 at index 4: crosses 0.97 at index 3 (step 200); 0.975 at indices 5 and 6 (step 300)
    r = vr.compare_logs(rows(ACCS), rows(ACCS))
    assert r["steps_to_target"][0.97] == (200, 200)
    assert r["steps_to_target"][0.975] == (300, 300)


def test_tolerance_pass_requires_same_steps_to_target():
    # a tiny change that moves the crossing from step 200 to later must fail even inside the accuracy tolerance
    new = list(ACCS)
    new[3] = 0.9695
    r = vr.compare_logs(rows(ACCS), rows(new))
    assert r["verdict"] == "FAIL" and any("steps-to-target" in x for x in r["reasons"])


def test_load_log_skips_header_and_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"_header": True, "config": {"a": 1}}) + "\n\n" +
                 json.dumps({"step": 50, "flops": 5, "val_acc": 0.9}) + "\n")
    h, rs = vr.load_log(p)
    assert h["config"] == {"a": 1} and len(rs) == 1


def test_config_differences_ignores_stage_and_new_keys():
    old = {"config": {"stage": "a", "lr": 0.1, "arm": "base"}}
    new = {"config": {"stage": "b", "lr": 0.2, "arm": "base", "sel_frac": 0.3}}
    assert vr.config_differences(old, new) == {"lr": (0.1, 0.2)}


def test_stage_name_includes_the_thread_count():
    assert vr.stage_name(1) == "verify-regression-t1"
    assert vr.stage_name(0) != vr.stage_name(1)


def test_train_command_fixes_threads_before_importing_train():
    cmd = vr.train_command("cfg.yaml", 0, 1)
    assert cmd[1] == "-c" and cmd[3:] == ["cfg.yaml", "0", "1"]
    code = cmd[2]
    assert code.index("set_num_threads") < code.index("from src.train import main")
    assert "--seed" in code and "main()" in code
