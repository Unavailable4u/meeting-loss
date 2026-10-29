"""Random-search tuning and analysis helpers (decisions D-014, D-017 and D-018).

Everything that is not command-line plumbing lives here so that it can be unit-tested:
the search spaces, the trial sampler, the job runner, the selection rule, and the bootstrap
statistics. The command-line entry points are scripts/run_sweep.py and scripts/analyze.py.

Selection rule (D-017, with the crossing definition of D-018): a trial's score is the mean over its
selection seeds of the FLOPs needed to reach SELECTION_TARGET validation accuracy, where "reach" is
defined in src/crossing.py (default: two consecutive evaluations at or above the target). A run that
never reaches the target is scored with its full budget (the FLOPs of the whole run, so an arm's
extra cost is counted). Lower is better. Ties are broken by the higher mean validation accuracy over
the last LATE_K evaluations, then by the lower trial index. Scores are computed from the saved
curves at analysis time, so the crossing rule can be changed without rerunning anything.
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

from src.crossing import CROSSINGS, DEFAULT_CROSSING, crossing_index

REPO =pathlib.Path(__file__).resolve().parents[1]
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

# prereg-v2 (D-023, DR-011). t9: arm A at batch 48 with the horizon of equal samples (3,000 x 128 / 48 = 8,000 steps) and
# evaluations every 125 steps (6,000 samples; arm A evaluates every 6,400; 125 divides 8,000 so that the last evaluation is at the last
# step, which the completeness check of collect_sweep needs: the 133 steps of the small-batch look E-017 would stop at 7,980); the same
# search space as arm A.
# t10: the selective backward pass at sel_frac 0.3 (E-010's S30), the same search space, tuned once as a secondary arm.
SWEEP_ARMS["base_b48"] = ("base", {"batch_size": 48, "eval_every": 125}, {})
SWEEP_ARMS["selective30"] = ("selective", {"sel_frac": 0.3, "sel_warmup": 100, "sel_mode": "top"}, {})
ARM_STEPS = {"base_b48": 8000}  # protocol horizon per sweep arm; every other arm uses SWEEP_STEPS

# Improve track (D-025, DR-016). t11: Selective-Backprop as released (src/sb.py) with beta 2, history 1,024 and one-epoch priming
# fixed and the same four-dimensional search space as arm A (lr, wd, act, lr_schedule), 25 trials, 2 selection seeds; the score is
# the FLOPs to 0.975 as counted for the released algorithm (`flops`: every selection forward pass and the second forward pass
# charged). beta is not searched: it is the method's own knob and E-024 found beta 2 and 3 alike. t12 (optional, descriptive): S
# with kept 0.2 and no warm-up, the same search space, tuned once as S30 was in t10.
SWEEP_ARMS["sb"] = ("sb", {"sb_beta": 2.0, "sb_history": 1024}, {})
SWEEP_ARMS["selective20w0"] = ("selective", {"sel_frac": 0.2, "sel_warmup": 0, "sel_mode": "top"}, {})


def expected_steps(arms):
    """Protocol horizon of a sweep over `arms`. Arms with different horizons need separate tags."""
    steps = {ARM_STEPS.get(a, SWEEP_STEPS) for a in arms}
    if len(steps) != 1:
        raise ValueError(f"arms {list(arms)} have different horizons {sorted(steps)}; use separate tags")
    return steps.pop()


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


def check_protocol(tag, n_trials, max_steps, seeds, arms=None):
    """Non-protocol sweeps (wrong trial count, horizon or seed count) must be labelled `smoke...`
    so that they cannot be mistaken for the equal-budget tuning runs (D-014, D-017)."""
    if any(s < MIN_SELECTION_SEED for s in seeds):
        raise ValueError(f"selection seeds must be >= {MIN_SELECTION_SEED} (0-99 are reserved for "
                         f"final runs and earlier experiments, so selection and final seeds stay disjoint)")
    horizon = expected_steps(arms) if arms else SWEEP_STEPS
    off = (n_trials != N_TRIALS or max_steps != horizon or len(seeds) != N_SELECTION_SEEDS)
    if off and not tag.startswith("smoke"):
        raise ValueError(f"non-protocol settings (trials={n_trials}, max_steps={max_steps}, "
                         f"seeds={list(seeds)}); protocol is {N_TRIALS} trials, {horizon} steps, "
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


def score_run(rows, target=SELECTION_TARGET, late_k=LATE_K, crossing=DEFAULT_CROSSING):
    """Selection score of one run. `score` is the FLOPs at the evaluation where the run reaches the
    target under `crossing` (src/crossing.py), or the full-run FLOPs if it never does."""
    if not rows:
        raise ValueError("empty run")
    budget = rows[-1]["flops"]
    late = float(np.mean([r["val_acc"] for r in rows[-late_k:]]))
    i = crossing_index([r["val_acc"] for r in rows], target, crossing)
    if i is not None:
        r = rows[i]
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


def collect_sweep(tag, arms=None, root=SWEEP_ROOT, target=SELECTION_TARGET, crossing=DEFAULT_CROSSING):
    """Read a sweep directory. Returns {arm: {"trials": [...], "incomplete": [...], "meta": {...}}}.
    A trial is complete when every seed listed in the arm's manifest has a log. Scores use `crossing`."""
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
                score_run(rows, target, crossing=crossing), path=str(p),
                complete=rows[-1]["step"] == hdr["config"]["max_steps"])
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


def tuned_config(trial_cfg, tag, arm, source_note, crossing=DEFAULT_CROSSING):
    """Config for the final runs of the selected trial: same hyperparameters, seed reset to 0
    (final seeds are passed with --seed), stage named after the tuning pass, sweep arm and crossing
    rule (so that configs selected under different rules cannot be mixed up in run ids).
    The horizon (max_steps) is kept from the sweep: cosine schedules depend on it."""
    cfg = dict(trial_cfg)
    cfg.update(seed=0, stage=f"final-{tag}-{arm}-{crossing}")
    header = (f"# Selected by D-017/D-018 from sweep '{tag}', arm '{arm}', crossing rule '{crossing}'.\n"
              f"# {source_note}\n"
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


def group_scores(paths, targets, crossing=DEFAULT_CROSSING):
    """{seed: {target: score_run(...)}} for a set of final-run logs. Duplicate seeds are an error
    (they usually mean the glob matched two different configurations)."""
    out = {}
    for p in paths:
        hdr, rows = load_run(p)
        seed = hdr["config"]["seed"]
        if seed in out:
            raise ValueError(f"seed {seed} appears twice in one group (check the glob); second file: {p}")
        out[seed] = {t: score_run(rows, t, crossing=crossing) for t in targets}
    return out


def cross_seed_check(trials, seeds):
    """Selection-bias check for ONE arm: pick the best trial using only seed a, then score that trial
    on seed b; average over the two directions (for exactly two selection seeds). Returns
    (score_selected_on_one_seed_tested_on_the_other, [(selected_trial, held_out_score), ...]).
    The in-sample winning score is optimistic because the winner was picked on the same seeds."""
    if len(seeds) != 2:
        raise ValueError("cross_seed_check needs exactly two selection seeds")
    out = []
    for a, b in ((seeds[0], seeds[1]), (seeds[1], seeds[0])):
        best = min(trials, key=lambda t: (t["per_seed"][a]["score"], t["trial"]))
        out.append((best["trial"], best["per_seed"][b]["score"]))
    return float(np.mean([s for _, s in out])), out


# --- pre-registered three-way outcome (D-019) ----------------------------------------------------
def margin_verdict(point, lo, hi, margin=15.0, shift=None):
    """Three-way outcome for one target. `point`, `lo`, `hi` are the paired percentage change of mean
    FLOPs-to-target against the reference and its confidence interval (negative = fewer FLOPs).
      supported     point change <= -margin and the interval excludes zero (hi < 0); if `shift` (the
                    percentage change between two independent tunings of the reference) is given, the
                    gain must also exceed it in size, otherwise the outcome is inconclusive
      refuted       the interval excludes every gain of `margin` percent or more (lo > -margin)
      inconclusive  anything else"""
    if point <= -margin and hi < 0:
        if shift is not None and abs(point) <= abs(shift):
            return "inconclusive"
        return "supported"
    if lo > -margin:
        return "refuted"
    return "inconclusive"


def shift_value(changes, rule):
    """Re-tuning shift of the reference (D-019 point 3, extended in D-023) from the paired percentage changes of the
    further tunings of the reference against the reference. A "supported" verdict needs the gain to exceed it.
      max_abs     the largest absolute change (a worse tuning counts as much as a better one; D-019's form)
      max_gain    the largest improvement of any further tuning over the reference (0 if none is better)
      median_abs  the median absolute change"""
    changes = [float(c) for c in changes]
    if rule == "max_abs":
        return max(abs(c) for c in changes)
    if rule == "max_gain":
        return max(0.0, max(-c for c in changes))
    if rule == "median_abs":
        return float(np.median([abs(c) for c in changes]))
    raise ValueError(f"unknown shift rule {rule!r}; choose max_abs, max_gain or median_abs")


def overall_verdict(verdicts):
    """Outcome over the primary targets: supported or refuted only if every target agrees."""
    verdicts = list(verdicts)
    if verdicts and all(v == "supported" for v in verdicts):
        return "supported"
    if verdicts and all(v == "refuted" for v in verdicts):
        return "refuted"
    return "inconclusive"
