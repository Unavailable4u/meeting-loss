"""The analysis of the Improve-track headroom study (tracks/improve/headroom_analysis.py): crossing costs, paired changes and
the reading rules, on synthetic numbers."""
import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("headroom_analysis", ROOT / "tracks" / "improve" / "headroom_analysis.py")
ha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ha)

IDX = np.random.default_rng(0).integers(0, 10, (2000, 10))


def rows(accs, per_eval=100, key="flops"):
    return [dict(step=50 * (i + 1), val_acc=a, **{key: per_eval * (i + 1)}) for i, a in enumerate(accs)]


def test_cost_uses_two_consecutive_evaluations_and_charges_a_miss_in_full():
    r = rows([0.5, 0.96, 0.98, 0.96, 0.98, 0.98])                    # a single 0.98 at index 2 is not enough
    assert ha.cost_to_target(r, 0.975) == (500, 250, True)         # first of the pair at index 4
    assert ha.cost_to_target(r, 0.999) == (600, 300, False)         # never: the whole run
    assert ha.cost_to_target(rows([0.9, 0.96, 0.98, 0.98]), 0.97, key="flops") == (300, 150, True)


def test_paired_change_and_the_improvement_helpers():
    a, b = [90.0] * 10, [100.0] * 10
    pt, lo, hi = ha.paired_change(a, b, IDX)
    np.testing.assert_allclose([pt, lo, hi], [-10.0] * 3)
    assert ha.below((pt, lo, hi), 5.0) and not ha.below((-3.0, -8.0, 1.0), 5.0) and not ha.below((-6.0, -9.0, 2.0), 5.0)
    assert ha.above((12.0, 4.0, 20.0), 10.0) and not ha.above((12.0, -1.0, 20.0), 10.0)
    assert ha.excludes_zero((-3.0, -6.0, -0.5)) and not ha.excludes_zero((-3.0, -6.0, 0.5))


def vec(flops975, steps975=None, f97=None, never=0):
    steps975 = steps975 or [1000] * 10
    f97 = f97 or [x * 0.7 for x in flops975]
    return {0.97: dict(flops=f97, steps=[s * 0.7 for s in steps975], never=never), 0.975: dict(flops=flops975, steps=steps975, never=never)}


def test_readings_follow_the_rules_of_the_plan_note():
    rng = np.random.default_rng(1)
    noise = lambda: list(1 + rng.normal(0, 0.005, 10))
    base = [100.0 * x for x in noise()]
    v = {"A1": vec([x * 2 for x in base]), "S30": vec(base), "S25": vec([x * 0.97 for x in base]),
         "S20": vec([x * 0.90 for x in base]), "S15": vec([x * 0.93 for x in base], steps975=[1200] * 10),
         "S10": vec([x * 1.2 for x in base], steps975=[1500] * 10),
         "S20w0": vec([x * 0.90 * 0.95 for x in base]), "S20w25": vec([x * 0.90 for x in base]),
         "SB2": vec([x * 1.5 for x in base]), "SB2_cached": vec([x * 0.8 for x in base]),
         "S20gn": vec([x * 0.90 * 0.99 for x in base]), "SOSGD": vec([x * 0.9 * 0.97 for x in base])}
    out = "\n".join(ha.readings(v, IDX))
    assert ha.best_constant(v) == "S20"
    assert "R1 best constant fraction: S20 (0.2)" in out and "HEADROOM CONFIRMED" in out
    assert "seen at kept fractions [0.15, 0.1]" in out                    # steps +20% and +50% against S30
    assert "R3 warm-up S20w0" in out and "PAYS" in out and "S20w25" in out and "NO CLEAR EFFECT" in out
    assert "R4 SB2 against S30" in out and "SB LOSES" in out
    assert "R4 SB2_cached against S30" in out and "SB WINS" in out
    assert "R5 S20gn against S20" in out and "UNCLEAR" in out
    assert "R5 SOSGD against S20" in out
    assert "R6 leads for a pre-registered test on fresh seeds (prereg-v3): S20, S20w0, SB2_cached" in out


def test_no_gain_reading_when_s30_is_already_best():
    base = [100.0] * 10
    v = {"S30": vec(base), "S20": vec([x * 1.1 for x in base]), "A1": vec([x * 2 for x in base])}
    assert "R1 best constant fraction: S30 (0.3) -> NO FURTHER GAIN" in "\n".join(ha.readings(v, IDX))


def test_pipeline_reads_fabricated_logs_in_the_run_format_for_every_group(tmp_path):
    import json
    import yaml
    plan = ROOT / "tracks" / "improve" / "headroom_plan.yaml"
    names = [g["name"] for g in yaml.safe_load(open(plan))["groups"]]
    raw = tmp_path / "raw"
    raw.mkdir()
    for k, name in enumerate(names):
        for seed in ha.SEEDS:
            sb = name.startswith("SB")
            speed = 0.001 * (1 + 0.02 * k) * (1 + 0.01 * (seed - 2100))
            lines = [json.dumps({"_header": True, "run_id": "x", "config": {"seed": seed, "arm": "sb" if sb else "base", "stage": f"hr-{name}"}})]
            for i in range(1, 61):
                step = 50 * i
                row = dict(step=step, flops=step * 1e8 * (1 + 0.1 * k), time=0.1, val_acc=min(0.99, 0.9 + speed * step))
                if sb:
                    row.update(flops_cached=row["flops"] * 0.9, n_fwd=3 * step, rows_seen=3 * step * 128, sel_rate=0.33, queue=1)
                lines.append(json.dumps(row))
            (raw / f"20261101_hr-{name}_{'sb' if sb else 'base'}_s{seed}.jsonl").write_text("\n".join(lines) + "\n")
    logs = ha.load_logs(plan, raw)
    assert sorted(logs) == sorted(names) and all(len(r) == 10 for r in logs.values())
    v = ha.vectors(logs)
    assert "SB2_cached" in v and "S30_cached" not in v and set(v["A1"]) == {0.97, 0.975}
    lines = ha.readings(v, IDX)
    assert lines[-1].startswith("R6 leads") and any(l.startswith("R4 SB2_cached") for l in lines)
    # a group with missing runs is skipped, not crashed on
    next(raw.glob("*_hr-S10_*_s2100.jsonl")).unlink()
    assert "S10" not in ha.load_logs(plan, raw)
