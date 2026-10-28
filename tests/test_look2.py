"""Look 2 of the Improve track (tracks/improve/look2_plan.yaml and look2_analysis.py): plan, configs, code paths, reading rules."""
import importlib.util
import json
import pathlib

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src import final
from src.model import ARMS
from src.train import run

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN = ROOT / "tracks" / "improve" / "look2_plan.yaml"
A1 = yaml.safe_load(open(ROOT / "configs" / "tuned" / "t1_base_two.yaml"))
ALLOWED = {"stage", "arm", "sel_frac", "sel_warmup", "sel_mode", "sel_scale"}
spec = importlib.util.spec_from_file_location("look2_analysis", ROOT / "tracks" / "improve" / "look2_analysis.py")
la = importlib.util.module_from_spec(spec)
spec.loader.exec_module(la)
IDX = np.random.default_rng(0).integers(0, 10, (2000, 10))


def _loaders(seed=0):
    g = torch.Generator().manual_seed(7)
    means = torch.randn(10, 784, generator=g)
    def make(n, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.randint(0, 10, (n,), generator=gg)
        return means[y] + torch.randn(n, 784, generator=gg) * 3.0, y
    x, y = make(1024, 1); xv, yv = make(256, 2)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(xv, yv), batch_size=128))


def test_plan_has_twelve_groups_of_ten_new_seeds_and_configs_that_differ_from_a1_only_as_declared():
    groups = final.load_plan(PLAN)
    assert len(groups) == 12 and len(final.plan_jobs(groups, "x")) == 120
    assert all(g["seeds"] == list(range(2110, 2120)) for g in groups)
    assert len({(g["cfg"]["stage"], g["cfg"]["arm"]) for g in groups}) == 12
    assert final.plan_meta(PLAN)["out_dir"] == "results/raw/improve_look2"
    by = {g["name"]: g["cfg"] for g in groups}
    for g in groups:
        cfg = g["cfg"]
        assert cfg["arm"] in ARMS and cfg["stage"] == f"l2-{g['name']}" and cfg["max_steps"] == 3000
        assert {k for k in set(cfg) | set(A1) if cfg.get(k) != A1.get(k)} <= ALLOWED, g["name"]
        assert cfg["lr"] == A1["lr"] and cfg["wd"] == A1["wd"]
    assert by["A1"] == {**A1, "stage": "l2-A1"}
    assert [by[n]["sel_warmup"] for n in ("S20w0", "S20w5", "S20w10", "S20w25", "S20")] == [0, 5, 10, 25, 100]
    assert [by[n]["sel_frac"] for n in ("S15w0", "S10w0", "S20w0")] == [0.15, 0.1, 0.2]
    assert by["S20k"]["sel_scale"] == "k" and by["S20k"]["sel_warmup"] == 100
    assert by["S20w0k"]["sel_scale"] == "k" and by["S20w0k"]["sel_warmup"] == 0
    assert "sel_scale" not in by["S20"] and "sel_scale" not in by["S20w0"]


def test_every_group_runs_a_few_steps_through_its_own_code_path():
    for g in final.load_plan(PLAN):
        cfg = {**g["cfg"], "batch_size": 32, "max_steps": 24, "eval_every": 12, "seed": 2110}
        log = run(cfg, device="cpu", data=_loaders())
        assert [r["step"] for r in log] == [12, 24] and all(0.0 <= r["val_acc"] <= 1.0 and r["flops"] > 0 for r in log), g["name"]


def vec(f975, steps=None):
    steps = steps or [1000] * 10
    return {0.97: dict(flops=[x * 0.7 for x in f975], steps=[s * 0.7 for s in steps], never=0), 0.975: dict(flops=f975, steps=steps, never=0)}


def scenario(**mult):
    rng = np.random.default_rng(3)
    base = [100.0 * (1 + rng.normal(0, 0.004)) for _ in range(10)]
    m = dict(S30=1.1, S20=1.0, S15=0.99, S20w0=0.8, S15w0=0.8, S10w0=0.8, S20w5=0.85, S20w10=0.9, S20w25=0.95, S20k=0.85, S20w0k=0.8)
    m.update(mult)
    v = {g: vec([b * f for b in base]) for g, f in m.items()}
    v["A1"] = vec([b * 2 for b in base])
    return v


def lines(v):
    return "\n".join(la.readings(v, IDX))


def test_graded_dose_and_scale_effect_read_as_the_optimizer_state_hypothesis_predicts():
    out = lines(scenario(S15=0.9 * 1.1, S15w0=0.8 * 0.95, S20w0k=0.8 * 1.01))
    assert "R1 S20w0 against S20" in out and "REPLICATED" in out and "NOT REPLICATED" not in out
    assert "R1 S15 against S30" in out and out.count("-> REPLICATED") == 2
    assert "R2 S15w0 against S20w0" in out and "COMBINATION PAYS" in out
    assert "R2 S10w0 against S20w0" in out and "-> NO GAIN" in out
    assert "-> GRADED" in out
    assert "R4 S20k against S20" in out and "-> BETTER" in out and "OPTIMIZER-STATE HYPOTHESIS SUPPORTED" in out
    assert "R6 leads for a pre-registered test on fresh seeds (prereg-v3): S20w0, S15, S15w0, S20k" in out
    assert "R5 cheapest S arm by mean FLOPs over the two targets: S15w0" in out


def test_a_step_at_zero_and_no_scale_effect_read_as_the_other_hypothesis():
    out = lines(scenario(S20w5=1.0, S20w10=1.0, S20w25=1.0, S20k=1.0, S20w0k=0.8))
    assert "-> STEP AT ZERO" in out and "-> GRADED" not in out
    assert "-> SCALE DOES NOT MATTER" in out and "OPTIMIZER-STATE" not in out


def test_no_replication_when_the_effect_is_gone_and_mixed_when_neither_pattern_fits():
    out = lines(scenario(S20w0=0.99, S20w0k=0.99, S20w5=0.99, S20w10=0.99, S20w25=0.99))
    assert "R1 S20w0 against S20" in out and "-> NOT REPLICATED" in out
    assert "R6 leads for a pre-registered test on fresh seeds (prereg-v3): S15" in out
    out2 = lines(scenario(S20w5=0.80, S20w10=0.97, S20w25=0.90))                   # not monotone and not flat
    assert "-> MIXED" in out2
    out3 = lines(scenario(S20k=1.2, S20w0k=0.8))                                    # scale hurts S20: an 'other pattern'
    assert "SCALE MATTERS (other pattern)" in out3


def test_pipeline_reads_fabricated_logs_in_the_run_format_for_every_group(tmp_path):
    names = [g["name"] for g in yaml.safe_load(open(PLAN))["groups"]]
    raw = tmp_path / "raw"; raw.mkdir()
    for k, name in enumerate(names):
        for seed in la.SEEDS:
            speed = 0.001 * (1 + 0.02 * k) * (1 + 0.01 * (seed - 2110))
            lines_ = [json.dumps({"config": {"seed": seed, "arm": "base", "stage": f"l2-{name}"}})]
            for i in range(1, 61):
                step = 50 * i
                lines_.append(json.dumps(dict(step=step, flops=step * 1e8 * (1 + 0.1 * k), time=0.1, val_acc=min(0.99, 0.9 + speed * step))))
            (raw / f"20261101_l2-{name}_base_s{seed}.jsonl").write_text("\n".join(lines_) + "\n")
    logs = la.load_logs(PLAN, raw)
    assert sorted(logs) == sorted(names) and all(len(r) == 10 for r in logs.values())
    out = la.readings(la.vectors(logs), IDX)
    assert out[-1].startswith("R6 leads") and any(l.startswith("R3") for l in out) and any(l.startswith("R4 reading") for l in out)
    next(raw.glob("*_l2-S15w0_*_s2110.jsonl")).unlink()
    assert "S15w0" not in la.load_logs(PLAN, raw)
