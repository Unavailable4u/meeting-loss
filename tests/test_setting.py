"""Setting study (docs/notes/setting-fmnist.md, D-031, DR-024): src/datasets.py, the `dataset` key, the sweep arms `f_*`, the target probe,
exploratory/setting.py (plan, derived configs, rules) and the runners scripts/probe_dataset.py and scripts/run_setting.py.

The tests the note asks for first:
1. a config without `dataset` behaves exactly as before; `dataset` reaches `get_data` only when it is set (test_dataset_key_*)
2. the target rule reproduces the MNIST targets from arm A's MNIST plateau (test_target_rule_*)
3. a dataset other than MNIST has no targets until its probe has been run (test_targets_need_probe)
4. the `f_*` sweep arms equal their MNIST arms plus `dataset: fmnist`; every other sweep arm is unchanged (test_fmnist_sweep_arms)
5. `analyze.py select` takes its target from the arms' dataset (test_select_target_follows_dataset)
6. the plan, derived configs and rules of exploratory/setting.py; the runners (test_group_specs, test_build_config, test_decide_margin, test_runner_*)
"""
import json
import pathlib
import sys

import pytest
import yaml
from torch.utils.data import DataLoader, TensorDataset

from exploratory import setting as st
from src import datasets, sweep
from src.train import run

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scripts_loader import load_script  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def _loaders(n=512, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(10, 784, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    x = proto[y] + 2.0 * torch.randn(n, 784, generator=g)
    return (DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(TensorDataset(x[:128], y[:128]), batch_size=128))


# ------------------------------------------------------------------ datasets and targets
def test_target_rule_reproduces_mnist():
    assert datasets.targets_from_plateau(0.9804) == (0.97, 0.975)       # arm A's MNIST plateau (E-035)
    assert datasets.targets_from_plateau(0.9809) == (0.97, 0.975)
    assert datasets.targets_from_plateau(0.8912) == (0.88, 0.885)
    assert datasets.targets_from_plateau(0.885) == (0.875, 0.88)         # on the boundary: floor((p - 0.005) / 0.005) * 0.005
    assert datasets.dataset_targets("mnist") == (0.97, 0.975)


def test_targets_need_probe(tmp_path):
    with pytest.raises(FileNotFoundError, match="probe_dataset"):
        datasets.dataset_targets("fmnist", tmp_path)
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    (tmp_path / "configs" / "datasets" / "fmnist_targets.json").write_text(json.dumps({"targets": [0.88, 0.885]}))
    assert datasets.dataset_targets("fmnist", tmp_path) == (0.88, 0.885)
    with pytest.raises(ValueError):
        datasets.dataset_targets("cifar", tmp_path)


def test_normalisation_constants_are_the_standard_ones():
    assert datasets.DATASETS["mnist"]["mean"] == 0.1307 and datasets.DATASETS["mnist"]["std"] == 0.3081
    assert datasets.DATASETS["fmnist"]["tv"] == "FashionMNIST" and datasets.DATASETS["fmnist"]["mean"] == 0.2860


def test_get_data_builds_the_chosen_torchvision_dataset(monkeypatch, tmp_path):
    """No download: a stub class stands in for torchvision's dataset; the split and the normalisation come from the registry."""
    import torch
    from torchvision import datasets as tvd
    from src import data
    seen = {}

    class Stub(torch.utils.data.Dataset):
        def __init__(self, root, train, download, transform):
            seen["tfm"], seen["root"] = transform, root

        def __len__(self):
            return 60000

        def __getitem__(self, i):
            return torch.zeros(784), 0

    monkeypatch.setattr(tvd, "FashionMNIST", Stub, raising=False)
    tr, va = data.get_data(128, 0, root=str(tmp_path), dataset="fmnist")
    assert len(tr.dataset) == 55000 and len(va.dataset) == 5000
    assert "0.286" in repr(seen["tfm"]) and "0.353" in repr(seen["tfm"])
    with pytest.raises(ValueError):
        data.get_data(128, 0, dataset="nope")


def test_dataset_key_reaches_get_data_only_when_set(monkeypatch):
    import src.data as data
    calls = []

    def fake(batch_size, seed, **kw):
        calls.append(kw)
        return _loaders()

    monkeypatch.setattr(data, "get_data", fake)
    base = dict(seed=0, batch_size=32, max_steps=10, eval_every=5, lr=1e-3, wd=1e-4, arm="base", act="relu")
    run(base)
    run({**base, "dataset": "fmnist"})
    assert calls == [{}, {"dataset": "fmnist"}]
    from src import sb
    calls.clear()
    sb.run_sb({**base, "arm": "sb", "dataset": "fmnist", "sb_beta": 2.0, "sb_history": 64, "sb_prime_steps": 5})
    assert calls == [{"dataset": "fmnist"}]


# ------------------------------------------------------------------ sweep arms and selection target
def test_fmnist_sweep_arms():
    assert set(sweep.FMNIST_SWEEP_ARMS) <= set(sweep.SWEEP_ARMS)
    for new, old in sweep.FMNIST_SWEEP_ARMS.items():
        arm, fixed, extra = sweep.SWEEP_ARMS[old]
        arm2, fixed2, extra2 = sweep.SWEEP_ARMS[new]
        assert (arm2, extra2) == (arm, extra) and fixed2 == {**fixed, "dataset": "fmnist"}
        assert sweep.arm_dataset(new) == "fmnist" and sweep.arm_dataset(old) == "mnist"
        cfg = sweep.build_cfg(new, sweep.sample_trials(new, "t17", 1)[0], 1000, "t17")
        assert cfg["dataset"] == "fmnist" and cfg["max_steps"] == sweep.SWEEP_STEPS
        assert sweep.sample_trials(new, "t17", 3) != sweep.sample_trials(old, "t17", 3)       # its own random stream
    assert sweep.expected_steps(list(sweep.FMNIST_SWEEP_ARMS)) == 3000
    assert set(sweep.FMNIST_SWEEP_ARMS) >= {"f_base", "f_sb", "f_meet", "f_deepsup", "f_shallow", "f_mn_match", "f_mn_fwd"}


def test_select_target_follows_dataset(tmp_path, monkeypatch):
    an = load_script("analyze")
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    (tmp_path / "configs" / "datasets" / "fmnist_targets.json").write_text(json.dumps({"targets": [0.88, 0.885]}))
    monkeypatch.setattr(datasets, "REPO", tmp_path)
    monkeypatch.setattr(datasets, "dataset_targets", lambda name, repo=tmp_path: (0.88, 0.885) if name == "fmnist" else datasets.MNIST_TARGETS)

    class A:
        tag, arms, root, target = "x", ["f_base"], str(tmp_path), None

    assert an._selection_target(A) == 0.885
    A.arms = ["base"]
    assert an._selection_target(A) == sweep.SELECTION_TARGET
    A.arms = ["base", "f_base"]
    with pytest.raises(SystemExit):
        an._selection_target(A)
    A.arms, A.target = ["f_base"], 0.9
    assert an._selection_target(A) == 0.9


# ------------------------------------------------------------------ plan and configs
@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Tuned configs for the groups of f1 (stand-ins for the t17 and t18 outputs) and the dataset's targets."""
    (tmp_path / "configs" / "tuned").mkdir(parents=True)
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    (tmp_path / "configs" / "datasets" / "fmnist_targets.json").write_text(json.dumps({"targets": [0.88, 0.885]}))
    for name, (kind, arm, _) in st.GROUPS.items():
        t = st.tag_spec("f1")["t_second" if kind == "tuned2" else "t_main"]
        hp = dict(lr=1e-3, wd=1e-4, act="relu", lr_schedule="cosine", **({} if arm in ("f_base", "f_shallow", "f_sb") else {"mn_scale": 12.0}))
        if arm in ("f_meet",):
            hp.update(lam_meet=0.1, lam_inv=1.0, warmup_steps=200)
        if arm == "f_deepsup":
            hp.update(lam_aux=0.3, aux_gamma=0.0)
        cfg = sweep.build_cfg(arm, hp, 1000, t)
        cfg.update(seed=0, stage=f"final-{t}-{arm}-two")
        (tmp_path / "configs" / "tuned" / f"{t}_{arm}_two.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(st, "ROOT", tmp_path)
    return tmp_path


def test_group_specs(fake_root):
    specs = st.group_specs("f1")
    assert [s["name"] for s in specs][:3] == ["A", "A2", "S"] and len(specs) == 13
    assert {s["seeds"] for s in specs} == {tuple(range(2500, 2510))}
    assert {s["name"] for s in specs if s["role"] == "reference"} == {"A", "A2", "S"}


def test_seeds_are_disjoint_from_the_other_studies():
    from exploratory import meetnet, oracle
    used = set(meetnet.TAGS["mn1"]["seeds"])
    for t in oracle.TAGS.values():
        for seeds in t["seeds"].values():
            used |= set(seeds)
    mine = set(st.TAGS["f1"]["seeds"])
    assert not used & mine and not mine & set(range(0, 100)) and not mine & set(range(1300, 1320)) and not mine & set(range(2300, 2320))
    from scripts_loader import load_script as _l
    probe = _l("probe_dataset")
    assert not mine & set(probe.PROBE_SEEDS) and not used & set(probe.PROBE_SEEDS)


def test_build_config(fake_root):
    specs = {s["name"]: s for s in st.group_specs("f1")}
    a, s20 = st.build_config(specs["A"], "f1"), st.build_config(specs["S20"], "f1")
    assert a["arm"] == "base" and a["dataset"] == "fmnist" and a["stage"] == "set-f1-A"
    assert s20["arm"] == "selective" and (s20["sel_frac"], s20["sel_warmup"], s20["sel_mode"]) == (0.2, 0, "top")
    assert {k: v for k, v in s20.items() if k not in ("arm", "stage", "sel_frac", "sel_warmup", "sel_mode")} == \
           {k: v for k, v in a.items() if k not in ("arm", "stage", "sel_frac", "sel_warmup", "sel_mode")}     # A's hyperparameters, nothing else
    assert st.build_config(specs["A2"], "f1")["stage"] == "set-f1-A2"
    assert st.build_config(specs["M3f"], "f1")["mn_k"] == 3 and st.build_config(specs["SBT"], "f1")["arm"] == "sb"
    stages = [st.build_config(s, "f1")["stage"] for s in specs.values()]
    assert len(set(stages)) == len(stages)
    assert st.tuned_path("f1", "A2") == "configs/tuned/t18_f_base_two.yaml" and st.tuned_path("f1", "D") == "configs/tuned/t17_f_meet_two.yaml"
    assert st.targets_of("f1") == (0.88, 0.885)


def test_build_config_rejects_a_wrong_dataset_or_k(fake_root):
    specs = {s["name"]: s for s in st.group_specs("f1")}
    p = fake_root / "configs" / "tuned" / "t17_f_base_two.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["dataset"] = "mnist"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="dataset"):
        st.build_config(specs["A"], "f1")
    p = fake_root / "configs" / "tuned" / "t17_f_mn_match_k2_two.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["mn_k"] = 1
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="mn_k"):
        st.build_config(specs["M2m"], "f1")


def test_missing_tuned_config_has_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="tuning pass"):
        st.build_config({"name": "A", "seeds": (1,)}, "f1")


# ------------------------------------------------------------------ rules
def _ch(point, lo, hi, targets=(0.88, 0.885)):
    return {t: (point, lo, hi) for t in targets}


def test_decide_margin():
    assert st.decide_margin("S20", _ch(-40, -50, -30), {0.88: 5.0, 0.885: 5.0}) == "HOLDS"
    assert st.decide_margin("S20", _ch(-40, -50, -30), {0.88: 45.0, 0.885: 45.0}) == "INCONCLUSIVE"      # gain below the re-tuning shift
    assert st.decide_margin("D", _ch(-40, -50, -30), None) == "HELPS" and st.decide_margin("B", _ch(-40, -50, -30), None) == "HELPS"
    assert st.decide_margin("D", _ch(+30, +20, +45), None) == "NO HELP"
    assert st.decide_margin("S20", _ch(-10, -20, +5), None) == "INCONCLUSIVE"
    mixed = {0.88: (-40, -50, -30), 0.885: (+5, -4, +15)}
    assert st.decide_margin("S20", mixed, None) == "INCONCLUSIVE"                                          # both targets must agree


def _score(rng, mean_flops, targets, acc=0.89, n=10, reached=True, readouts=None):
    f = rng.normal(mean_flops, mean_flops * 0.01, n)
    return dict(seeds=list(range(n)), flops={t: list(f) for t in targets}, steps={t: [1000] * n for t in targets},
                reached={t: [reached] * n for t in targets}, final=[acc] * n, readouts=readouts or {})


def test_analyze_and_report_on_a_full_set():
    import numpy as np
    rng = np.random.default_rng(0)
    tg = (0.88, 0.885)
    means = dict(A=100, A2=95, S=80, S20=50, SBT=60, D=120, B=105, M1m=110, M1f=130, M1t=105, M2m=100, M3m=100, M3f=110)
    sc = {n: _score(rng, m, tg) for n, m in means.items()}
    res = st.analyze(sc, tg, n_boot=500)
    assert res["margin"]["S20"]["verdict"] == "HOLDS" and res["margin"]["D"]["verdict"] == "NO HELP"
    assert res["margin"]["B"]["verdict"] in ("INCONCLUSIVE", "NO HELP")
    assert res["s20_vs_sbt"]["verdict"] in ("INCONCLUSIVE", "HOLDS")
    assert set(res["mn"]["groups"]) == set(st.MN_GROUPS)
    text = st.report(sc, res, "f1")
    assert "S20 against A:" in text and "Study 1b groups" in text and "dataset fmnist" in text


# ------------------------------------------------------------------ the runners
def test_probe_runner_writes_targets_once(tmp_path, monkeypatch):
    import src.data as data
    pr = load_script("probe_dataset")
    monkeypatch.setattr(pr, "REPO", tmp_path)
    (tmp_path / "configs" / "tuned").mkdir(parents=True)
    (tmp_path / "configs" / "tuned" / "t1_base_two.yaml").write_text((REPO / "configs" / "tuned" / "t1_base_two.yaml").read_text())
    monkeypatch.setattr(pr, "targets_path", lambda ds: tmp_path / "configs" / "datasets" / f"{ds}_targets.json")
    monkeypatch.setattr(data, "get_data", lambda batch_size, seed, **kw: _loaders())
    out = tmp_path / "probe_out"
    monkeypatch.setattr(sys, "argv", ["probe_dataset.py", "--dataset", "fmnist", "--seeds", "3000", "3001", "--out-dir", str(out)])
    pr.main()
    t = json.loads((tmp_path / "configs" / "datasets" / "fmnist_targets.json").read_text())
    assert t["targets"][1] - t["targets"][0] == pytest.approx(0.005) and t["targets"] == list(datasets.targets_from_plateau(t["plateau"]))
    assert (tmp_path / "configs" / "datasets" / "probe_fmnist_A.yaml").exists()
    probe_cfg = yaml.safe_load((tmp_path / "configs" / "datasets" / "probe_fmnist_A.yaml").read_text())
    assert probe_cfg["dataset"] == "fmnist" and probe_cfg["arm"] == "base"
    with pytest.raises(SystemExit):                                  # fixed once
        pr.main()
    monkeypatch.setattr(sys, "argv", ["probe_dataset.py", "--dataset", "mnist"])
    with pytest.raises(SystemExit):
        pr.main()


def test_runner_writes_configs_and_refuses_edits(fake_root, monkeypatch):
    rs = load_script("run_setting")
    monkeypatch.setattr(rs.st, "ROOT", fake_root)
    specs = st.group_specs("f1")
    with pytest.raises(SystemExit):
        rs.ensure_configs("f1", specs)
    rs.ensure_configs("f1", specs, write=True)
    rs.ensure_configs("f1", specs)
    p = st.config_path("f1", "S20")
    p.write_text(p.read_text().replace("sel_frac: 0.2", "sel_frac: 0.3"))
    with pytest.raises(SystemExit):
        rs.ensure_configs("f1", specs)


def test_runner_job_order_is_seed_major(fake_root):
    rs = load_script("run_setting")
    jobs = rs.make_jobs("f1", st.group_specs("f1"), "out")
    assert len(jobs) == 13 * 10 and [j.seed for j in jobs[:13]] == [2500] * 13 and jobs[0].group == "A"


def test_runner_end_to_end_on_a_tiny_run(fake_root, tmp_path, monkeypatch):
    """All 13 groups, 2 seeds, a few steps on synthetic loaders: run files exist, load_group reads them, the analysis and the CSV run."""
    import src.data as data
    rs = load_script("run_setting")
    monkeypatch.setattr(rs.st, "ROOT", fake_root)
    monkeypatch.setattr(data, "get_data", lambda batch_size, seed, **kw: _loaders())
    from src.final import run_final_job
    specs = st.group_specs("f1", seeds=(1, 2))
    rs.ensure_configs("f1", specs, write=True)
    out = tmp_path / "out"
    for job in rs.make_jobs("f1", specs, out):
        assert run_final_job(job, "cpu", runner=lambda cfg, dev: run({**cfg, "max_steps": 100, "eval_every": 10, "batch_size": 32, "sb_history": 64,
                                                                       "sb_prime_steps": 10}, dev)) == "done"
    tg = st.targets_of("f1")
    sc = {s["name"]: st.load_group(out, "f1", s) for s in specs}
    assert set(sc["A"]["flops"]) == set(tg) and len(sc["M1f"]["final"]) == 2
    res = st.analyze(sc, tg, n_boot=100)
    assert set(res["margin"]) == {"S20", "D", "B"}
    monkeypatch.chdir(tmp_path)
    rs.write_csv(sc, res, tg, tmp_path / "t.csv")
    assert (tmp_path / "t.csv").read_text().startswith("group,")
