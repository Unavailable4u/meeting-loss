"""Random-search tuning and analysis helpers (decisions D-014 and D-017).

Everything that is not command-line plumbing lives here so that it can be unit-tested:
the search spaces, the trial sampler, the job runner, the selection rule, and the bootstrap
statistics. The command-line entry points are scripts/run_sweep.py and scripts/analyze.py.

Selection rule (D-017): a trial's score is the mean over its selection seeds of the FLOPs needed
to first reach SELECTION_TARGET validation accuracy. A run that never reaches the target is scored
with its full budget (the FLOPs of the whole run, so an arm's extra cost is counted). Lower is
better. Ties are broken by the higher mean validation accuracy over the last LATE_K evaluations,
then by the lower trial index.
"""
import datetime
import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "configs" / "template_all_options.yaml"
SWEEP_ROOT = pathlib.Path("results/raw/sweeps")

# --- protocol constants (D-014, D-017) -------------------------------------------------------
SELECTION_TARGET = 0.975        # D-017: FLOPs-to-0.975 is the selection score
N_TRIALS = 25                   # D-014: trials per arm
SWEEP_STEPS = 3000              # D-017: sweeps are limited to 3,000 steps
N_SELECTION_SEEDS = 2           # D-014: 2 seeds per trial
DEFAULT_SEEDS = (1000, 1001)    # first tuning pass; use e.g. 2000 2001 for the re-tuning pass
MIN_SELECTION_SEED = 100        # seeds below this are reserved: 0-9 final runs, E-001, diagnostics
LATE_K = 5                      # tie-break: mean val_acc over the last LATE_K evaluations
AUX_GAMMA_SMALL = 0.05          # PLACEHOLDER for D-014's "small" aux_gamma; confirm before the lock

# --- search spaces (D-014). ("log", lo, hi) | ("uniform", lo, hi) | ("choice", [...]) ---------
COMMON = {
    "lr": ("log", 1e-4, 1e-2),
    "wd": ("log", 1e-6, 1e-2),
    "act": ("choice", ["relu", "tanh"]),
    "lr_schedule": ("choice", ["constant", "cosine"]),
}
_MEET = {
    "lam_meet": ("log", 1e-3, 10.0),
    "lam_inv": ("log", 0.1, 10.0),
    "warmup_steps": ("choice", [200, 500, 1000]),
}
_MEET_NO_INV = {k: v for k, v in _MEET.items() if k != "lam_inv"}      # frozen G, exact-gradient F
_AUX = {
    "lam_aux": ("log", 1e-2, 1.0),
    "aux_gamma": ("choice", [0.0, AUX_GAMMA_SMALL]),
}

# sweep arm -> (config `arm`, fixed config overrides, extra search dimensions)
SWEEP_ARMS = {
    "base":         ("base", {}, {}),                                                  # A
    "deepsup":      ("deepsup", {"aux_sched": "linear"}, _AUX),                        # B-paper
    "deepsup_code": ("deepsup", {"aux_sched": "const", "aux_lr_decay": "power_inv"}, _AUX),  # B-code
    "ae":           ("ae", {}, {"lam_aux": _AUX["lam_aux"]}),                          # C
    "meet":         ("meet", {}, _MEET),                                               # D
    "base_lowlr":   ("base", {}, {"low_lr_mult": ("log", 0.5, 8.0)}),                  # E
    "meet_grad":    ("meet_grad", {}, _MEET_NO_INV),                                   # F (eta_hat fixed)
    "meet_noise":   ("meet", {}, {**_MEET, "sigma_inv": ("log", 0.01, 0.3)}),          # V1
    "meet_ao":      ("meet_ao", {}, _MEET),                                            # V2
    "meet_frozen":  ("meet_frozen", {}, _MEET_NO_INV),                                 # V3
}
MAIN_ARMS = ("base", "deepsup", "ae", "meet", "base_lowlr", "meet_grad")   # A to F


# --- sampling --------------------------------------------------------------------------------
def _rng(tag, arm):
    digest = hashlib.sha1(f"{tag}|{arm}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _draw(rng, spec):
    kind = spec[0]
    if kind == "log":
        v = math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2])))
        return float(f"{v:.6g}")
    if kind == "uniform":
        return float(f"{rng.uniform(spec[1], spec[2]):.6g}")
    if kind == "choice":
        return rng.choice(spec[1])
    raise ValueError(f"unknown search-space kind: {kind}")


def sample_trials(arm, tag, n=N_TRIALS):
    """Deterministic list of n hyperparameter dicts for this arm; depends only on (tag, arm)."""
    if arm not in SWEEP_ARMS:
        raise ValueError(f"unknown sweep arm {arm!r}; choose from {sorted(SWEEP_ARMS)}")
    dims = {**COMMON, **SWEEP_ARMS[arm][2]}
    rng = _rng(tag, arm)
    return [{k: _draw(rng, spec) for k, spec in dims.items()} for _ in range(n)]


def build_cfg(arm, hp, seed, tag, max_steps=SWEEP_STEPS, template=TEMPLATE):
    """Full run config for one (arm, trial, seed): template defaults + arm settings + trial hp."""
    with open(template) as f:
        cfg = yaml.safe_load(f)
    cfg_arm, fixed, _ = SWEEP_ARMS[arm]
    cfg.update(fixed)
    cfg.update(hp)
    cfg.update(arm=cfg_arm, seed=int(seed), stage=f"sweep{tag}", max_steps=int(max_steps))
    return cfg


# --- jobs ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Job:
    tag: str
    arm: str
    trial: int
    seed: int
    hp: dict = field(compare=False)
    max_steps: int = SWEEP_STEPS
    root: str = str(SWEEP_ROOT)

    @property
    def path(self):
        return pathlib.Path(self.root) / self.tag / f"{self.arm}_t{self.trial:02d}_s{self.seed}.jsonl"


def check_tag(tag):
    if not tag or not tag.replace("-", "").isalnum():
        raise ValueError("tag must be letters, digits and hyphens only (it becomes part of file names)")


def check_protocol(tag, n_trials, max_steps, seeds):
    """Non-protocol sweeps (wrong trial count, horizon or seed count) must be labelled `smoke...`
    so that they cannot be mistaken for the equal-budget tuning runs (D-014, D-017)."""
    if any(s < MIN_SELECTION_SEED for s in seeds):
        raise ValueError(f"selection seeds must be >= {MIN_SELECTION_SEED} (0-99 are reserved for "
                         f"final runs and earlier experiments, so selection and final seeds stay disjoint)")
    off = (n_trials != N_TRIALS or max_steps != SWEEP_STEPS or len(seeds) != N_SELECTION_SEEDS)
    if off and not tag.startswith("smoke"):
        raise ValueError(f"non-protocol settings (trials={n_trials}, max_steps={max_steps}, "
                         f"seeds={list(seeds)}); protocol is {N_TRIALS} trials, {SWEEP_STEPS} steps, "
                         f"{N_SELECTION_SEEDS} seeds. Start the tag with 'smoke' to run it anyway.")
    return off


def plan_jobs(tag, arms, seeds, n_trials=N_TRIALS, max_steps=SWEEP_STEPS, root=SWEEP_ROOT):
    """All jobs, ordered trial-major then arm then seed, so a stopped sweep stays balanced."""
    trials = {a: sample_trials(a, tag, n_trials) for a in arms}
    return [Job(tag, a, i, int(s), trials[a][i], max_steps, str(root))
            for i in range(n_trials) for a in arms for s in seeds]


def write_manifest(tag, arm, seeds, n_trials, max_steps, root=SWEEP_ROOT):
    """Record the sampled trials once. On resume, the sampled trials must be identical."""
    trials = sample_trials(arm, tag, n_trials)
    path = pathlib.Path(root) / tag / f"{arm}_trials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(tag=tag, arm=arm, n_trials=n_trials, max_steps=max_steps, seeds=list(seeds),
               selection_target=SELECTION_TARGET, trials=trials)
    if path.exists():
        old = json.loads(path.read_text())
        if old["trials"] != trials or old["max_steps"] != max_steps:
            raise RuntimeError(f"{path} exists with different trials or max_steps; "
                               f"the search space changed since this sweep started. Use a new tag.")
        merged = sorted(set(old["seeds"]) | set(seeds))
        if merged != old["seeds"]:
            doc["seeds"] = merged
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=1) + "\n")
            os.replace(tmp, path)
        return path
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, path)
    return path


@lru_cache(maxsize=1)
def _git_info():
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def _default_runner(cfg, device):
    from src import train
    return train.run(cfg, device)


def run_job(job, device="auto", runner=None):
    """Run one (arm, trial, seed) and write its raw log. Returns 'skipped' if the file exists.
    Raw files are never overwritten: the log is written to a temp file and renamed into place."""
    out = job.path
    if out.exists():
        return "skipped"
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_cfg(job.arm, job.hp, job.seed, job.tag, job.max_steps)
    log = (runner or _default_runner)(cfg, device)
    commit, dirty = _git_info()
    header = {"_header": True,
              "run_id": f"{datetime.date.today():%Y%m%d}_sweep{job.tag}_{job.arm}_t{job.trial:02d}_s{job.seed}",
              "config": cfg, "config_hash": config_hash(cfg), "git_commit": commit, "git_dirty": dirty,
              "sweep": {"tag": job.tag, "arm": job.arm, "trial": job.trial, "hp": job.hp}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(header) + "\n")
        for row in log:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, out)
    return "done"


# --- reading logs and scoring -----------------------------------------------------------------
def load_run(path):
    """(header, rows) of one raw log."""
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[0], lines[1:]


def score_run(rows, target=SELECTION_TARGET, late_k=LATE_K):
    """Selection score of one run. `score` is the FLOPs at the first evaluation with
    val_acc >= target, or the full-run FLOPs if the target is never reached."""
    if not rows:
        raise ValueError("empty run")
    budget = rows[-1]["flops"]
    late = float(np.mean([r["val_acc"] for r in rows[-late_k:]]))
    for r in rows:
        if r["val_acc"] >= target:
            return dict(reached=True, steps=r["step"], flops=r["flops"], score=r["flops"],
                        budget=budget, final_val=rows[-1]["val_acc"], late_val=late)
    return dict(reached=False, steps=None, flops=None, score=budget,
                budget=budget, final_val=rows[-1]["val_acc"], late_val=late)


def summarize_trial(per_seed):
    """per_seed: {seed: score_run(...)} -> trial-level summary used for selection."""
    scores = [s["score"] for s in per_seed.values()]
    return dict(score=float(np.mean(scores)),
                n_reached=sum(s["reached"] for s in per_seed.values()),
                n_seeds=len(per_seed),
                late_val=float(np.mean([s["late_val"] for s in per_seed.values()])))


def select_best(trials):
    """trials: list of dicts with keys trial, score, late_val. Lower score wins; ties go to the
    higher late_val, then the lower trial index (D-017)."""
    return min(trials, key=lambda t: (t["score"], -t["late_val"], t["trial"]))


def collect_sweep(tag, arms=None, root=SWEEP_ROOT, target=SELECTION_TARGET):
    """Read a sweep directory. Returns {arm: {"trials": [...], "incomplete": [...], "meta": {...}}}.
    A trial is complete when every seed listed in the arm's manifest has a log."""
    base = pathlib.Path(root) / tag
    if not base.exists():
        raise FileNotFoundError(base)
    found = sorted(p.name[:-len("_trials.json")] for p in base.glob("*_trials.json"))
    out = {}
    for arm in (arms or found):
        mpath = base / f"{arm}_trials.json"
        if not mpath.exists():
            raise FileNotFoundError(f"no manifest for arm {arm!r} in {base}")
        meta = json.loads(mpath.read_text())
        runs = {}
        for p in sorted(base.glob(f"{arm}_t*_s*.jsonl")):
            hdr, rows = load_run(p)
            sw = hdr["sweep"]
            if sw["arm"] != arm:        # e.g. "meet" must not swallow "meet_ao" files
                continue
            runs.setdefault(sw["trial"], {"hp": sw["hp"], "seeds": {}, "cfg": hdr["config"]})
            runs[sw["trial"]]["seeds"][hdr["config"]["seed"]] = dict(
                score_run(rows, target), path=str(p), complete=rows[-1]["step"] == hdr["config"]["max_steps"])
        want = set(meta["seeds"])
        trials, incomplete = [], []
        for i in range(meta["n_trials"]):
            r = runs.get(i)
            ok = r is not None and set(r["seeds"]) >= want and all(r["seeds"][s]["complete"] for s in want)
            if not ok:
                incomplete.append(i)
                continue
            per_seed = {s: r["seeds"][s] for s in sorted(want)}
            trials.append(dict(trial=i, hp=r["hp"], cfg=r["cfg"], per_seed=per_seed,
                               **summarize_trial(per_seed)))
        out[arm] = dict(trials=trials, incomplete=incomplete, meta=meta)
    return out


def tuned_config(trial_cfg, tag, arm, source_note):
    """Config for the final runs of the selected trial: same hyperparameters, seed reset to 0
    (final seeds are passed with --seed), stage named after the tuning pass and sweep arm.
    The horizon (max_steps) is kept from the sweep: cosine schedules depend on it."""
    cfg = dict(trial_cfg)
    cfg.update(seed=0, stage=f"final-{tag}-{arm}")
    header = (f"# Selected by D-017 from sweep '{tag}', arm '{arm}'.\n# {source_note}\n"
              f"# Run final seeds with: python -m src.train --config <this file> --seed N\n")
    return header + yaml.safe_dump(cfg, sort_keys=False)


# --- statistics -------------------------------------------------------------------------------
def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_pct_change(ref, grp, n_boot=10000, alpha=0.05, seed=0):
    """Paired bootstrap over seeds of mean(grp)/mean(ref) - 1, in percent. ref and grp are aligned
    by seed. Negative = group needs less than the reference. Returns (point, lo, hi)."""
    ref, grp = np.asarray(ref, dtype=float), np.asarray(grp, dtype=float)
    if ref.shape != grp.shape:
        raise ValueError("paired arrays must have the same length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref), size=(n_boot, len(ref)))
    stat = 100.0 * (grp[idx].mean(axis=1) / ref[idx].mean(axis=1) - 1.0)
    point = 100.0 * (grp.mean() / ref.mean() - 1.0)
    return float(point), float(np.quantile(stat, alpha / 2)), float(np.quantile(stat, 1 - alpha / 2))


def group_scores(paths, targets):
    """{seed: {target: score_run(...)}} for a set of final-run logs. Duplicate seeds are an error
    (they usually mean the glob matched two different configurations)."""
    out = {}
    for p in paths:
        hdr, rows = load_run(p)
        seed = hdr["config"]["seed"]
        if seed in out:
            raise ValueError(f"seed {seed} appears twice in one group (check the glob); second file: {p}")
        out[seed] = {t: score_run(rows, t) for t in targets}
    return out
