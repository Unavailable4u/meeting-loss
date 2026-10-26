"""prereg-v2 (decision D-023): the derived configs, the pre-registered analysis of H7a and H7b, and the noise floor.

H7a (primary): S30 against A1, margin 15%; H7b (control): S30 against the fully tuned batch-48 arm, margin 0%. Both use the
rule of D-019 (paired percentage change of the mean FLOPs-to-target, 97.5% percentile interval, both targets 0.97 and 0.975
must agree). H7a also requires the gain to exceed the re-tuning shift of A (`src.sweep.shift_value`, the rule is a key of the
plan). The noise floor (D-022) is descriptive: the ratio max / min of the mean FLOPs-to-target over the tunings of A.
Nothing here decides anything beyond what D-023 wrote; the groups are found by their role in the plan:
reference (A1), primary (S30), control (B48), shift (A2, A5 to A8).
"""
import copy

import numpy as np
import yaml

from src.sweep import REPO, margin_verdict, overall_verdict, paired_pct_change, shift_value

TARGETS = (0.97, 0.975)
ALPHA = 0.025                     # D-019: two confirmatory comparisons (here H7a and H7b) share the error budget
MARGIN_A = 15.0                   # H7a
MARGIN_B = 0.0                    # H7b
NOISE_LARGE, NOISE_SMALL = 1.15, 1.10
A1_CONFIG = "configs/tuned/t1_base_two.yaml"
DERIVED = {                       # name -> (sel_frac, sel_mode)
    "S30": (0.3, "top"),
    "S20": (0.2, "top"),
    "R30": (0.3, "random"),
}


def derive_config(name, repo=REPO):
    """A1's tuned config with only the selective-backward keys (E-010, E-017, E-018: sel_warmup 100) and the stage changed."""
    if name not in DERIVED:
        raise ValueError(f"unknown derived config {name!r}; choose from {sorted(DERIVED)}")
    frac, mode = DERIVED[name]
    cfg = copy.deepcopy(yaml.safe_load(open(repo / A1_CONFIG)))
    cfg.update(stage=f"prereg2-{name}", arm="selective", sel_frac=frac, sel_warmup=100, sel_mode=mode)
    return cfg


def _by_role(roles, role, exactly_one=False):
    names = [n for n, r in roles.items() if r == role]
    if exactly_one and len(names) != 1:
        raise ValueError(f"the plan needs exactly one group with role {role!r}, found {names}")
    return names[0] if exactly_one else names


def _check_paired(scores):
    seeds = {n: sorted(g) for n, g in scores.items()}
    first = next(iter(seeds.values()))
    if any(s != first for s in seeds.values()):
        raise ValueError(f"all groups must have the same seeds (paired); got { {n: (s[0], s[-1], len(s)) for n, s in seeds.items()} }")
    return first


def _flops(scores, name, seeds, target):
    return np.array([scores[name][s][target]["score"] for s in seeds], dtype=float)


def _steps(scores, name, seeds, target):
    return np.array([scores[name][s][target]["steps"] if scores[name][s][target]["reached"] else np.nan for s in seeds], dtype=float)


def _verdict_rows(ref, grp, margin, shifts, n_boot, seed):
    rows, verdicts = {}, []
    for t in TARGETS:
        p, lo, hi = paired_pct_change(ref[t], grp[t], n_boot=n_boot, alpha=ALPHA, seed=seed)
        v = margin_verdict(p, lo, hi, margin, None if shifts is None else shifts[t])
        rows[t] = dict(point=p, lo=lo, hi=hi, verdict=v, shift=None if shifts is None else shifts[t])
        verdicts.append(v)
    return dict(per_target=rows, overall=overall_verdict(verdicts))


def noise_floor(means):
    """means: {tuning: {target: mean FLOPs-to-target}}. LARGE if max / min >= 1.15 at both targets, SMALL if <= 1.10 at both,
    else MIXED (D-022)."""
    ratio = {t: max(m[t] for m in means.values()) / min(m[t] for m in means.values()) for t in TARGETS}
    if all(r >= NOISE_LARGE for r in ratio.values()):
        outcome = "LARGE"
    elif all(r <= NOISE_SMALL for r in ratio.values()):
        outcome = "SMALL"
    else:
        outcome = "MIXED"
    return dict(ratio=ratio, outcome=outcome, means=means)


def h7(scores, roles, shift_rule, n_boot=10000, boot_seed=0):
    """scores: {group: {seed: {target: score_run dict}}}; roles: {group: role}. Returns the verdicts of H7a and H7b, the
    shift per target, the noise floor and descriptive comparisons of every group against A1 and against the best tuning."""
    ref = _by_role(roles, "reference", True)
    s = _by_role(roles, "primary", True)
    ctrl = _by_role(roles, "control")
    shift_groups = _by_role(roles, "shift")
    seeds = _check_paired(scores)
    f = {n: {t: _flops(scores, n, seeds, t) for t in TARGETS} for n in scores}
    res = dict(seeds=list(seeds), shift_rule=shift_rule, groups={n: roles[n] for n in scores})
    # H7a: S30 against A1, with the shift of A
    changes = {t: [paired_pct_change(f[ref][t], f[g][t], n_boot=n_boot, alpha=ALPHA, seed=boot_seed)[0] for g in shift_groups]
               for t in TARGETS}
    shift = {t: (shift_value(changes[t], shift_rule) if shift_groups else None) for t in TARGETS}
    res["shift"], res["shift_changes"] = shift, {t: dict(zip(shift_groups, changes[t])) for t in TARGETS}
    res["H7a"] = _verdict_rows(f[ref], f[s], MARGIN_A, shift if shift_groups else None, n_boot, boot_seed)
    # H7b: S30 against the tuned small batch
    if ctrl:
        res["H7b"] = _verdict_rows(f[ctrl[0]], f[s], MARGIN_B, None, n_boot, boot_seed)
    # noise floor over A1 and the further tunings of A
    tunings = [ref] + shift_groups
    res["noise_floor"] = noise_floor({n: {t: float(f[n][t].mean()) for t in TARGETS} for n in tunings})
    # descriptive: every group against A1 and against the best tuning of A on these seeds (best is chosen on the same seeds,
    # so this is optimistic for the baseline)
    desc = {}
    for n in scores:
        d = {}
        for t in TARGETS:
            row = dict(mean_gflops=float(f[n][t].mean() / 1e9),
                       n_never=int(sum(not scores[n][x][t]["reached"] for x in seeds)),
                       mean_steps=float(np.nanmean(_steps(scores, n, seeds, t))))
            if n != ref:
                row["vs_A1"] = paired_pct_change(f[ref][t], f[n][t], n_boot=n_boot, alpha=ALPHA, seed=boot_seed)
            best = min(tunings, key=lambda g: f[g][t].mean())
            if n != best:
                row["best_tuning"] = best
                row["vs_best_tuning"] = paired_pct_change(f[best][t], f[n][t], n_boot=n_boot, alpha=ALPHA, seed=boot_seed)
            d[t] = row
        desc[n] = d
    res["descriptive"] = desc
    return res
