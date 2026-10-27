"""The Improve-track headroom plan (tracks/improve/headroom_plan.yaml): valid groups, configs that differ from A1 only in
their declared keys, and every group runs through the code path it names (a few steps on synthetic data)."""
import json
import pathlib

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src import final
from src.model import ARMS
from src.train import run

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN = ROOT / "tracks" / "improve" / "headroom_plan.yaml"
A1 = yaml.safe_load(open(ROOT / "configs" / "tuned" / "t1_base_two.yaml"))
ALLOWED = {"stage", "arm", "sel_frac", "sel_warmup", "sel_mode", "sel_score", "sel_schedule", "sel_frac_start",
           "sel_frac_end", "sb_beta", "sb_history", "sb_prime_steps"}


def _loaders(seed=0):
    g = torch.Generator().manual_seed(7)
    means = torch.randn(10, 784, generator=g) * 1.0
    def make(n, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.randint(0, 10, (n,), generator=gg)
        return means[y] + torch.randn(n, 784, generator=gg) * 3.0, y
    x, y = make(1024, 1); xv, yv = make(256, 2)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(xv, yv), batch_size=128))


def test_plan_has_fifteen_groups_of_ten_fresh_seeds_in_the_exploration_block():
    groups = final.load_plan(PLAN)
    assert len(groups) == 15
    assert [g["name"] for g in groups][:2] == ["A1", "S30"]
    assert all(g["seeds"] == list(range(2100, 2110)) for g in groups)
    assert len({(g["cfg"]["stage"], g["cfg"]["arm"]) for g in groups}) == 15
    meta = final.plan_meta(PLAN)
    assert meta["seed_block"] == [2100, 2199] and meta["out_dir"] == "results/raw/improve_headroom"
    assert len(final.plan_jobs(groups, "x")) == 150


def test_every_config_is_a1_plus_its_declared_keys_and_names_a_known_arm():
    for g in final.load_plan(PLAN):
        cfg = g["cfg"]
        assert cfg["arm"] in ARMS and cfg["stage"] == f"hr-{g['name']}" and cfg["max_steps"] == 3000
        changed = {k for k in set(cfg) | set(A1) if cfg.get(k) != A1.get(k)}
        assert changed <= ALLOWED, (g["name"], changed - ALLOWED)
        assert cfg["lr"] == A1["lr"] and cfg["wd"] == A1["wd"] and cfg["batch_size"] == 128      # inherited hyperparameters
    by = {g["name"]: g["cfg"] for g in final.load_plan(PLAN)}
    assert by["A1"] == {**A1, "stage": "hr-A1"}
    assert by["S30"]["sel_frac"] == 0.3 and by["S30"]["sel_warmup"] == 100 and by["S30"]["sel_mode"] == "top"
    assert [by[n]["sel_frac"] for n in ("S25", "S20", "S15", "S10")] == [0.25, 0.2, 0.15, 0.1]
    assert [by[n]["sel_warmup"] for n in ("S20w0", "S20w25", "S20w50")] == [0, 25, 50]
    assert by["S20gn"]["sel_score"] == "gradnorm" and by["SOSGD"]["sel_schedule"] == "osgd"
    assert (by["Sup"]["sel_frac_start"], by["Sup"]["sel_frac_end"]) == (0.1, 0.5)
    assert (by["SB2"]["sb_beta"], by["SB3"]["sb_beta"], by["SB2p100"]["sb_prime_steps"]) == (2.0, 3.0, 100)
    assert "sb_prime_steps" not in by["SB2"] and by["SB2"]["sb_history"] == 1024


def test_every_group_runs_a_few_steps_through_its_own_code_path():
    for g in final.load_plan(PLAN):
        cfg = {**g["cfg"], "batch_size": 32, "max_steps": 24, "eval_every": 12, "seed": 2100}
        if cfg["arm"] == "sb":
            cfg["sb_prime_steps"] = 4                           # reach the selection phase within 24 updates
        log = run(cfg, device="cpu", data=_loaders())
        assert [r["step"] for r in log] == [12, 24], g["name"]
        assert all(0.0 <= r["val_acc"] <= 1.0 and r["flops"] > 0 for r in log), g["name"]
        if cfg["arm"] == "sb":
            assert log[-1]["n_fwd"] > 24 and log[-1]["flops_cached"] < log[-1]["flops"]


def test_a_job_of_an_sb_group_writes_a_log_with_its_group_name(tmp_path):
    groups = [g for g in final.load_plan(PLAN) if g["name"] == "SB2"]
    job = final.plan_jobs(groups, tmp_path)[0]
    fake = lambda cfg, device: [dict(step=50, flops=1, flops_cached=1, val_acc=0.5)]
    assert final.run_final_job(job, "cpu", runner=fake) == "done"
    (f,) = list(tmp_path.glob("*_hr-SB2_sb_s2100.jsonl"))
    head = json.loads(f.read_text().splitlines()[0])
    assert head["plan_group"] == "SB2" and head["config"]["arm"] == "sb"
    assert final.run_final_job(job, "cpu", runner=fake) == "skipped"
