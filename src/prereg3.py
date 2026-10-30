"""prereg-v3 (decision D-026): the derived configs, the pre-registered analysis of H8, H9 and H10, and the secondary readings.

H8 (primary): S20w0 against A1, margin 15%, with the re-tuning shift of A (rule `max_gain`, as H7a).
H9: S20w0 against S30 (the arm confirmed in prereg-v2), margin 10%, no shift.
H10: S20w0 against the tuned Selective-Backprop SBT (tag t11, E-026), margin 15%, uncached count (the released algorithm), no shift.
All three use the rule of D-019 (paired percentage change of the mean FLOPs-to-target, percentile interval, both targets 0.97 and
0.975 must agree). Three confirmatory comparisons share the error budget of 0.05 (D-019 did the same for two: 0.05 / 2 = 0.025), so
ALPHA = 0.05 / 3 and the intervals are 98.33% intervals. The noise floor (D-022) and everything under `secondary` and `descriptive`
decide nothing. Groups are found by role (reference A1, primary S20w0, shift A2 and A5 to A8) and, for the two other references, by
name (REFERENCES): the plan has two groups with role `control`.

Secondary readings (reported, never decisive): the cached count of Selective-Backprop against S20w0; the warm-up decomposition
(S20w0 against S20 and S20w10, steps, and FLOPs with the warm-up steps charged like selective steps); the random-row control R20w0.
"""
import copy

import numpy as np
import yaml

from src.crossing import DEFAULT_CROSSING, crossing_index
from src.prereg2 import noise_floor  # noqa: F401  (re-exported: same rule as prereg-v2, D-022)
from src.sweep import REPO, margin_verdict, overall_verdict, paired_pct_change, shift_value

TARGETS = (0.97, 0.975)
FAMILY_ALPHA, N_HYPOTHESES = 0.05, 3
ALPHA = FAMILY_ALPHA / N_HYPOTHESES                  # per comparison; 98.33% two-sided percentile intervals
MARGINS = {"H8": 15.0, "H9": 10.0, "H10": 15.0}
PRIMARY = "S20w0"
REFERENCES = {"H8": "A1", "H9": "S30", "H10": "SBT"}
A1_CONFIG = "configs/tuned/t1_base_two.yaml"
DERIVED = {                       # name -> (sel_frac, sel_warmup, sel_mode)
    "S20w0": (0.2, 0, "top"),
    "S30": (0.3, 100, "top"),
    "S20": (0.2, 100, "top"),
    "S15": (0.15, 100, "top"),
    "S20w10": (0.2, 10, "top"),
    "R20w0": (0.2, 0, "random"),
}


def derive_config(name, repo=REPO):
    """A1's tuned config with only the selective-backward keys and the stage changed."""
    if name not in DERIVED:
        raise ValueError(f"unknown derived config {name!r}; choose from {sorted(DERIVED)}")
    frac, warm, mode = DERIVED[name]
    cfg = copy.deepcopy(yaml.safe_load(open(repo / A1_CONFIG)))
    cfg.update(stage=f"prereg3-{name}", arm="selective", sel_frac=frac, sel_warmup=warm, sel_mode=mode)
    return cfg


# ----------------------------------------------------------------------------- secondary quantities read from a run's rows
def score_run_cached(rows, target, crossing=DEFAULT_CROSSING):
    """FLOPs under the cached count (`flops_cached`) at the evaluation where the run crosses the target (the whole run if it never does);
    None if the log has no cached count (only Selective-Backprop logs it)."""
    if not rows or "flops_cached" not in rows[0]:
        return None
    i = crossing_index([r["val_acc"] for r in rows], target, crossing)
    return float(rows[i]["flops_cached"] if i is not None else rows[-1]["flops_cached"])


def full_step_flops(rows):
    """FLOPs of one full step (forward and complete backward) from a run in which every step is one: the first evaluation of arm A."""
    return rows[0]["flops"] / rows[0]["step"]


def selective_step_flops(rows, warmup):
    """FLOPs of one selective step (forward on all rows, backward on the kept rows) from the evaluations after the warm-up."""
    after = [r for r in rows if r["step"] >= warmup]
    a, b = after[0], rows[-1]
    if b["step"] <= a["step"]:
        raise ValueError("no evaluations after the warm-up")
    return (b["flops"] - a["flops"]) / (b["step"] - a["step"])


def warmup_credit(rows, warmup, target, full, crossing=DEFAULT_CROSSING):
    """What the warm-up steps cost beyond selective steps by the time the run crosses the target: min(warmup, crossing step) x
    (full - selective step). Subtracted from the FLOPs it gives the cost with the warm-up charged like selective steps."""
    if warmup <= 0:
        return 0.0
    i = crossing_index([r["val_acc"] for r in rows], target, crossing)
    step = rows[i]["step"] if i is not None else rows[-1]["step"]
    return min(warmup, step) * (full - selective_step_flops(rows, warmup))


# ----------------------------------------------------------------------------- the analysis
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


def _arr(scores, name, seeds, target, key="score"):
    vals = [scores[name][s][target].get(key) for s in seeds]
    if any(v is None for v in vals):
        return None
    return np.array(vals, dtype=float)


def _steps(scores, name, seeds, target):
    vals = [scores[name][s][target]["steps"] if scores[name][s][target]["reached"] else np.nan for s in seeds]
    return np.array(vals, dtype=float)


def _change(ref, grp, n_boot, seed):
    return paired_pct_change(ref, grp, n_boot=n_boot, alpha=ALPHA, seed=seed)


def _verdict_rows(ref, grp, margin, shifts, n_boot, seed):
    rows, verdicts = {}, []
    for t in TARGETS:
        p, lo, hi = _change(ref[t], grp[t], n_boot, seed)
        v = margin_verdict(p, lo, hi, margin, None if shifts is None else shifts[t])
        rows[t] = dict(point=p, lo=lo, hi=hi, verdict=v, shift=None if shifts is None else shifts[t],
                       n_better=int(np.sum(grp[t] < ref[t])), n_seeds=int(len(grp[t])))
        verdicts.append(v)
    return dict(per_target=rows, overall=overall_verdict(verdicts), margin=margin)


def h8_h10(scores, roles, shift_rule, n_boot=10000, boot_seed=0):
    """scores: {group: {seed: {target: score dict}}}; roles: {group: role}. Returns the verdicts of H8, H9 and H10, the shift per
    target, the noise floor, the secondary readings and descriptive comparisons of every group."""
    ref = _by_role(roles, "reference", True)
    prim = _by_role(roles, "primary", True)
    if ref != REFERENCES["H8"] or prim != PRIMARY:
        raise ValueError(f"the plan's reference must be {REFERENCES['H8']!r} and its primary group {PRIMARY!r}; got {ref!r} and {prim!r}")
    for h in ("H9", "H10"):
        if REFERENCES[h] not in scores or roles.get(REFERENCES[h]) != "control":
            raise ValueError(f"{h} needs the group {REFERENCES[h]!r} with role 'control'")
    shift_groups = _by_role(roles, "shift")
    seeds = _check_paired(scores)
    f = {n: {t: _arr(scores, n, seeds, t) for t in TARGETS} for n in scores}
    res = dict(seeds=list(seeds), shift_rule=shift_rule, alpha=ALPHA, groups={n: roles[n] for n in scores})
    # H8: the re-tuning shift of A, as in H7a
    changes = {t: [_change(f[ref][t], f[g][t], n_boot, boot_seed)[0] for g in shift_groups] for t in TARGETS}
    shift = {t: (shift_value(changes[t], shift_rule) if shift_groups else None) for t in TARGETS}
    res["shift"], res["shift_changes"] = shift, {t: dict(zip(shift_groups, changes[t])) for t in TARGETS}
    res["H8"] = _verdict_rows(f[ref], f[prim], MARGINS["H8"], shift if shift_groups else None, n_boot, boot_seed)
    res["H9"] = _verdict_rows(f[REFERENCES["H9"]], f[prim], MARGINS["H9"], None, n_boot, boot_seed)
    res["H10"] = _verdict_rows(f[REFERENCES["H10"]], f[prim], MARGINS["H10"], None, n_boot, boot_seed)
    tunings = [ref] + shift_groups
    res["noise_floor"] = noise_floor({n: {t: float(f[n][t].mean()) for t in TARGETS} for n in tunings})
    res["secondary"] = _secondary(scores, f, seeds, prim, n_boot, boot_seed)
    res["descriptive"] = _descriptive(scores, f, seeds, ref, tunings, n_boot, boot_seed)
    return res


def _secondary(scores, f, seeds, prim, n_boot, boot_seed):
    """Readings that decide nothing. Each entry: {target: (point, lo, hi)} of `group` against `reference` on `metric`."""
    out = {}
    sb = REFERENCES["H10"]
    cached = {t: _arr(scores, sb, seeds, t, "score_cached") for t in TARGETS}
    if all(c is not None for c in cached.values()):
        out["H10_cached"] = dict(what=f"{prim} against {sb} with the cached count of {sb}", metric="flops",
                                 change={t: _change(cached[t], f[prim][t], n_boot, boot_seed) for t in TARGETS})
    pairs = (("S20w0_vs_S20", prim, "S20", "flops", "S20w0 against S20 (the warm-up, FLOPs as logged)"),
             ("S20w0_vs_S20w10", prim, "S20w10", "flops", "S20w0 against S20w10"),
             ("control_R20w0", "R20w0", prim, "flops", "random rows without warm-up against S20w0 (selection by loss produces the saving)"))
    for key, g, r, metric, what in pairs:
        if g in f and r in f:
            out[key] = dict(what=what, metric=metric, change={t: _change(f[r][t], f[g][t], n_boot, boot_seed) for t in TARGETS})
            sg, sr = ({t: _steps(scores, n, seeds, t) for t in TARGETS} for n in (g, r))
            if not any(np.isnan(sg[t]).any() or np.isnan(sr[t]).any() for t in TARGETS):
                out[key + "_steps"] = dict(what=what + ", steps", metric="steps",
                                           change={t: _change(sr[t], sg[t], n_boot, boot_seed) for t in TARGETS})
    credit = {t: _arr(scores, "S20", seeds, t, "score_credited") for t in TARGETS}
    if all(c is not None for c in credit.values()) and prim in f:
        out["S20w0_vs_S20_credited"] = dict(what="S20w0 against S20 with its warm-up steps charged like selective steps", metric="flops",
                                            change={t: _change(credit[t], f[prim][t], n_boot, boot_seed) for t in TARGETS})
    return out


def _descriptive(scores, f, seeds, ref, tunings, n_boot, boot_seed):
    desc = {}
    for n in scores:
        d = {}
        for t in TARGETS:
            st = _steps(scores, n, seeds, t)
            row = dict(mean_gflops=float(f[n][t].mean() / 1e9), n_never=int(sum(not scores[n][x][t]["reached"] for x in seeds)),
                       mean_steps=float(np.nanmean(st)) if not np.isnan(st).all() else float("nan"))
            if n != ref:
                row["vs_A1"] = _change(f[ref][t], f[n][t], n_boot, boot_seed)
            if n != REFERENCES["H9"] and REFERENCES["H9"] in f:
                row["vs_S30"] = _change(f[REFERENCES["H9"]][t], f[n][t], n_boot, boot_seed)
            best = min(tunings, key=lambda g: f[g][t].mean())
            if n != best:
                row["best_tuning"] = best
                row["vs_best_tuning"] = _change(f[best][t], f[n][t], n_boot, boot_seed)
            d[t] = row
        desc[n] = d
    return desc
