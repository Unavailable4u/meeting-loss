"""Pure logic of the three exploratory looks of track S (docs/notes/selective-backward-prereg-prep.md).

Look 1  wall-clock time per step of the selective arm against the baseline  (scripts/time_selective.py)
Look 2  the small-batch control: is arm A compute-optimal in batch size?     (scripts/run_batch_control.py)
Look 3  the sel_frac curve and its knee                                      (scripts/pilot_selective.py --sel-fracs)

Everything that is not training or timing lives here, without torch, so that it can be unit tested anywhere.
The rules were written before any run (the prep note); the dated clarifications are in the same note.
"""
import hashlib
import math

import numpy as np

from src.crossing import crossing_index
from src.sweep import score_run, select_best, summarize_trial

TARGETS = (0.97, 0.975)

# ---- Look 1: timing rule -------------------------------------------------------------------------
TIME_TRANSLATES = 0.85          # S30's time per step at most 0.85 of A's: the FLOPs saving translates
TIME_NOT = 1.00                 # 1.00 or more: it does not translate

# ---- Look 2: design and rule -----------------------------------------------------------------------
BASE_BATCH = 128
BASE_STEPS = 3000
SMALL_BATCHES = (32, 48, 64)
N_TRIALS = 8
LR_RANGE = (3e-4, 5e-3)
TUNE_SEEDS = (1200, 1201)
EVAL_SEEDS = (1202, 1203, 1204)
TUNE_TARGET = 0.97
SPACING_SAMPLES = 50 * BASE_BATCH     # arm A evaluates every 50 steps of 128 samples; keep the spacing in samples
PROBLEM_RATIO = 0.85            # BASELINE PROBLEM: best small-batch ratio <= 0.85 at both targets
FINE_RATIO = 1.00               # A WAS FINE: every small-batch ratio >= 1.00 at both targets
S_BEATS_RATIO = 0.85            # S30 / best small batch <= 0.85

# ---- Look 3 -------------------------------------------------------------------------------------
KNEE_RATIO = 0.85               # knee: smallest sel_frac with ratio <= 0.85 at both targets and no missed seed


# ================================================================== Look 1
def summarize_timing(times):
    """times: {variant: [seconds per step, one per repeat]} (must contain "A" and "S30").
    Returns medians (ms per step), the ratio S30 / A and the verdict of the rule."""
    med = {v: float(np.median(t)) for v, t in times.items()}
    ratio = med["S30"] / med["A"]
    return dict(median_ms={v: 1000.0 * m for v, m in med.items()}, ratio=ratio, verdict=timing_verdict(ratio))


def timing_verdict(ratio):
    if ratio <= TIME_TRANSLATES:
        return "FLOPS SAVING TRANSLATES"
    if ratio < TIME_NOT:
        return "PARTIAL"
    return "DOES NOT TRANSLATE"


# ================================================================== Look 2
def horizon_steps(b):
    """Equal samples seen as arm A's 3,000 steps of 128: 12,000 steps at 32, 8,000 at 48, 6,000 at 64."""
    return int(round(BASE_STEPS * BASE_BATCH / b))


def eval_every(b):
    """Evaluation spacing kept at (about) the same number of samples as arm A: 200, 133 and 100 steps."""
    return int(round(SPACING_SAMPLES / b))


def sample_lrs(tag, b, n=N_TRIALS, lo=LR_RANGE[0], hi=LR_RANGE[1]):
    """n learning rates, log-uniform in [lo, hi], deterministic given the tag and the batch size."""
    seed = int(hashlib.sha1(f"{tag}:{b}".encode()).hexdigest()[:8], 16)
    u = np.random.default_rng(seed).uniform(math.log(lo), math.log(hi), size=n)
    return [float(f"{math.exp(x):.6g}") for x in u]


def trial_stage(tag, b, i):
    return f"{tag}-tune-b{b}-t{i:02d}"


def eval_stage(tag, name):
    return f"{tag}-eval-{name}"


def trial_config(base, b, lr, stage):
    """Arm A's tuned config with only the batch size, its horizon and evaluation spacing, and lr changed."""
    cfg = dict(base)
    cfg.update(stage=stage, batch_size=b, lr=lr, max_steps=horizon_steps(b), eval_every=eval_every(b))
    return cfg


def s30_config(base, stage):
    """The selective arm with arm A's tuned hyperparameters (the pilot's S30)."""
    cfg = dict(base)
    cfg.update(stage=stage, arm="selective", sel_frac=0.3, sel_warmup=100, sel_mode="top")
    return cfg


def eval_config(base, name, stage, best_trial_cfgs=None):
    """name: "A", "S30" or "b32"/"b48"/"b64" (the best tuned trial of that batch size)."""
    if name == "A":
        return {**base, "stage": stage}
    if name == "S30":
        return s30_config(base, stage)
    cfg = dict(best_trial_cfgs[name])
    cfg["stage"] = stage
    return cfg


def crossing_cost(rows, target):
    """(flops, steps, reached) at the crossing; a run that never reaches the target costs its full budget."""
    i = crossing_index([r["val_acc"] for r in rows], target)
    if i is None:
        return rows[-1]["flops"], None, False
    return rows[i]["flops"], rows[i]["step"], True


def tune_table(trial_runs, target=TUNE_TARGET):
    """trial_runs: {trial: {seed: rows}} -> (list of dicts trial/score/late_val/n_reached, the best one).
    Score = mean over seeds of the FLOPs to reach `target` (D-017/D-018; a miss is scored at its full budget);
    ties go to the higher late validation accuracy, then the lower trial index."""
    table = []
    for trial, per_seed in sorted(trial_runs.items()):
        s = summarize_trial({seed: score_run(rows, target=target) for seed, rows in per_seed.items()})
        table.append(dict(trial=trial, **s))
    return table, select_best(table)


def ratios(runs, ref="A", targets=TARGETS):
    """runs: {name: {seed: rows}} -> {name: {target: stats}} against `ref`, on the seeds both have.
    ratio_of_means is the mean FLOPs-to-target over the mean of the reference (a miss costs its full budget)."""
    out = {}
    for name, by_seed in runs.items():
        out[name] = {}
        for t in targets:
            cost = {s: crossing_cost(rows, t) for s, rows in by_seed.items()}
            refc = {s: crossing_cost(rows, t) for s, rows in runs[ref].items()}
            seeds = sorted(set(cost) & set(refc))
            f = np.array([cost[s][0] for s in seeds], float)
            fa = np.array([refc[s][0] for s in seeds], float)
            st = [cost[s][1] for s in seeds if cost[s][2]]
            sta = [refc[s][1] for s in seeds if refc[s][2]]
            out[name][t] = dict(
                flops_mean=float(f.mean()), ratio_of_means=float(f.mean() / fa.mean()), n=len(seeds),
                misses=sum(1 for s in seeds if not cost[s][2]),
                steps_mean=float(np.mean(st)) if st else float("nan"),
                steps_ratio=float(np.mean(st) / np.mean(sta)) if st and sta else float("nan"))
    return out


def best_small_batch(summary, names):
    """The small-batch arm with the lowest worst-case ratio (max over the two targets)."""
    return min(names, key=lambda n: max(summary[n][t]["ratio_of_means"] for t in TARGETS))


def batch_control_verdict(summary, small_names, s_name="S30"):
    """Pre-registered rule of Look 2 (ratios are FLOPs-to-target against arm A at batch 128):
    BASELINE PROBLEM  the best small-batch arm has ratio <= 0.85 at both targets (A is not compute-optimal in
                      batch size; the confirmatory test must use the best-batch A as its reference)
    A WAS FINE        every small-batch arm has ratio >= 1.00 at both targets
    PARTIAL           anything else (include the best small-batch arm as a reference arm in prereg-v2)
    Also: whether S30 still beats the best small-batch arm by at least 15% (S30 / best <= 0.85 at both targets)."""
    best = best_small_batch(summary, small_names)
    if all(summary[best][t]["ratio_of_means"] <= PROBLEM_RATIO for t in TARGETS):
        verdict = "BASELINE PROBLEM"
    elif all(summary[n][t]["ratio_of_means"] >= FINE_RATIO for n in small_names for t in TARGETS):
        verdict = "A WAS FINE"
    else:
        verdict = "PARTIAL"
    rel = {t: summary[s_name][t]["flops_mean"] / summary[best][t]["flops_mean"] for t in TARGETS}
    return dict(verdict=verdict, best=best, s_vs_best=rel, s_beats_best=all(r <= S_BEATS_RATIO for r in rel.values()))


# ================================================================== Look 3
def frac_name(frac):
    return f"S{int(round(100 * frac))}"


def knee(summary, fracs):
    """fracs: {variant name: sel_frac}. The knee is the smallest sel_frac whose ratio_of_means is <= 0.85 at both
    targets with no missed seed; None if there is none."""
    ok = [f for n, f in fracs.items() if n in summary and
          all(summary[n][t]["ratio_of_means"] <= KNEE_RATIO and summary[n][t]["misses"] == 0 for t in TARGETS)]
    return min(ok) if ok else None
