"""Sweep arms of the Improve track (src/sweep.py, DR-016): Selective-Backprop (tag t11) and S without warm-up (tag t12)."""
import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from src import sweep
from src.train import run


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


def test_new_arms_are_registered_with_arm_a_search_space_and_the_declared_fixed_keys():
    arm, fixed, extra = sweep.SWEEP_ARMS["sb"]
    assert arm == "sb" and fixed == {"sb_beta": 2.0, "sb_history": 1024} and extra == {}
    arm, fixed, extra = sweep.SWEEP_ARMS["selective20w0"]
    assert arm == "selective" and fixed == {"sel_frac": 0.2, "sel_warmup": 0, "sel_mode": "top"} and extra == {}
    assert sweep.expected_steps(["sb"]) == 3000 and sweep.expected_steps(["selective20w0"]) == 3000


def test_t11_has_25_trials_over_arm_as_four_dimensions_and_reproducible_draws():
    trials = sweep.sample_trials("sb", "t11")
    assert len(trials) == 25 and sweep.sample_trials("sb", "t11") == trials
    t0 = trials[0]
    assert set(t0) >= {"lr", "wd", "act", "lr_schedule"} and "sb_beta" not in t0
    assert all(1e-4 <= t["lr"] <= 1e-2 and 1e-6 <= t["wd"] <= 1e-2 and t["act"] in ("relu", "tanh") for t in trials)
    assert trials != sweep.sample_trials("base", "t11")                      # its own draws, not arm A's


def test_trial_config_runs_selective_backprop_with_fixed_beta_and_the_trial_hyperparameters():
    hp = sweep.sample_trials("sb", "t11")[3]
    cfg = sweep.build_cfg("sb", hp, 9200, "t11", 3000)
    assert cfg["arm"] == "sb" and cfg["stage"] == "sweept11" and cfg["max_steps"] == 3000 and cfg["seed"] == 9200
    assert cfg["sb_beta"] == 2.0 and cfg["sb_history"] == 1024 and cfg["lr"] == hp["lr"] and cfg["wd"] == hp["wd"]
    assert cfg["batch_size"] == 128


def test_a_job_writes_a_log_whose_score_is_the_uncached_flops_to_the_target(tmp_path):
    hp = sweep.sample_trials("sb", "t11")[0]
    job = sweep.plan_jobs("t11", ["sb"], [9200, 9201], root=tmp_path)[0]
    small = lambda cfg, device: run({**cfg, "batch_size": 32, "max_steps": 40, "eval_every": 10, "sb_prime_steps": 8}, device, data=_loaders())
    assert sweep.run_job(job, "cpu", runner=small) == "done" and sweep.run_job(job, "cpu", runner=small) == "skipped"
    head, rows = sweep.load_run(job.path)
    assert rows[-1]["flops"] > rows[-1]["flops_cached"] > 0
    first = rows[0]                                            # a target met at the first evaluations: the score is the logged `flops`
    s = sweep.score_run([{**r, "val_acc": 1.0} for r in rows], target=0.5)
    assert s["reached"] and s["score"] == first["flops"] and s["score"] != first["flops_cached"]
    assert head["sweep"]["arm"] == "sb" and head["config"]["arm"] == "sb" and head["config"]["stage"] == "sweept11"


def test_t11_plan_is_fifty_runs_for_two_selection_seeds():
    jobs = sweep.plan_jobs("t11", ["sb"], [9200, 9201])
    assert len(jobs) == 50 and {j.seed for j in jobs} == {9200, 9201} and {j.trial for j in jobs} == set(range(25))
