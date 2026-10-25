"""Check G0 (docs/notes/mechanism-dose-response.md, D-020): configs, scoring and decision rule of the dose-response.

Everything except the dose rho is fixed: arm A's tuned configuration (configs/tuned/t1_base_two.yaml) plus the
learned inverse G of arm D (its tuned `lam_inv`), rho in {0, 0.1, 0.3, 1, 3} on seeds 1100 to 1109 (paired).
Metric: steps-to-target at 0.97 and 0.975 under the D-018 rule (two consecutive evaluations at or above the
target); a run that never reaches it is charged its full run. Paired percentage change of the mean against
rho = 0, 95% bootstrap interval (src/sweep.paired_pct_change). FLOPs-to-target is reported next to it.

Decision rule (written before any run, steps-to-target, both targets must agree):
  DOSE HELPS   some rho > 0 has a point estimate <= -10% at both targets and an interval that excludes 0 at both
  DOSE HARMS   rho = 1 and rho = 3 are >= +10% at both targets (point estimates) and the point estimates do not
               decrease with rho for rho in {0.3, 1, 3}
  FLAT         every rho > 0 is within +-5% at both targets (point estimates) and its interval includes 0
  MIXED        anything else; also, when both HELPS and HARMS hold, MIXED (report the whole curve)
"""
import copy
import glob
import pathlib

import numpy as np
import yaml

from src.crossing import DEFAULT_CROSSING, crossing_index
from src.sweep import paired_pct_change

ROOT = pathlib.Path(__file__).resolve().parents[1]
RHOS = (0.0, 0.1, 0.3, 1.0, 3.0)
TARGETS = (0.97, 0.975)
SEEDS = tuple(range(1100, 1110))
RHO_WARMUP_STEPS = 200                       # fixed in the note, not tuned
HELP_PCT, HARM_PCT, FLAT_PCT = 10.0, 10.0, 5.0
A_CONFIG = "configs/tuned/t1_base_two.yaml"
D_CONFIG = "configs/tuned/t1_meet_two.yaml"


def rho_tag(rho):
    """0.3 -> '0p3', 1.0 -> '1', 0.0 -> '0' (used in stage names and file names)."""
    return f"{float(rho):g}".replace(".", "p")


def stage_name(tag, rho):
    return f"dose-{tag}-rho{rho_tag(rho)}"


def config_path(tag, rho):
    return ROOT / "configs" / "dose" / f"{tag}_rho{rho_tag(rho)}.yaml"


def build_config(rho, tag):
    """Arm A's tuned config with arm meet_rho, the dose and arm D's tuned lam_inv. Nothing else is changed."""
    cfg = copy.deepcopy(yaml.safe_load(open(ROOT / A_CONFIG)))
    lam_inv = yaml.safe_load(open(ROOT / D_CONFIG))["lam_inv"]
    cfg.update(stage=stage_name(tag, rho), arm="meet_rho", rho=float(rho), rho_warmup_steps=RHO_WARMUP_STEPS,
               lam_inv=lam_inv, sigma_inv=0.0)
    return cfg


def _load(path):
    import json
    header, rows = None, []
    for line in open(path):
        line = line.strip()
        if line:
            rec = json.loads(line)
            if rec.get("_header"):
                header = rec
            elif "step" in rec and "val_acc" in rec:
                rows.append(rec)
    return header, rows


def find_run(out_dir, tag, rho, seed):
    hits = glob.glob(str(pathlib.Path(out_dir) / f"*_{stage_name(tag, rho)}_meet_rho_s{seed}.jsonl"))
    return hits[0] if hits else None


def score(rows, target, crossing=DEFAULT_CROSSING):
    """(steps, flops, reached). A run that never reaches the target is charged its full run (never dropped)."""
    i = crossing_index([r["val_acc"] for r in rows], target, crossing)
    if i is None:
        return rows[-1]["step"], rows[-1]["flops"], False
    return rows[i]["step"], rows[i]["flops"], True


def decide(table):
    """table[rho][target] = (point, lo, hi) of the paired change of mean steps against rho = 0, in percent, for the
    doses rho > 0. Returns 'DOSE HELPS', 'DOSE HARMS', 'FLAT' or 'MIXED' (module docstring)."""
    rhos = sorted(r for r in table if r > 0)
    t1, t2 = TARGETS
    helps = any(all(table[r][t][0] <= -HELP_PCT and table[r][t][2] < 0 for t in (t1, t2)) for r in rhos)
    harms = False
    if all(r in table for r in (0.3, 1.0, 3.0)):
        big = all(table[r][t][0] >= HARM_PCT for r in (1.0, 3.0) for t in (t1, t2))
        mono = all(table[0.3][t][0] <= table[1.0][t][0] <= table[3.0][t][0] for t in (t1, t2))
        harms = big and mono
    flat = bool(rhos) and all(abs(table[r][t][0]) <= FLAT_PCT and table[r][t][1] <= 0 <= table[r][t][2]
                              for r in rhos for t in (t1, t2))
    if helps and harms:
        return "MIXED"
    if helps:
        return "DOSE HELPS"
    if harms:
        return "DOSE HARMS"
    if flat:
        return "FLAT"
    return "MIXED"


def _diag(rows, after=1000):
    late = [r for r in rows if r["step"] > after]
    take = lambda k: float(np.median([r[k] for r in late])) if late and k in late[0] else float("nan")
    return dict(align=take("align"), h_var=take("h_var"), inv=take("inv"), final_acc=rows[-1]["val_acc"])


def analyze(out_dir, tag, seeds=SEEDS, rhos=RHOS, n_boot=10000, boot_seed=0, crossing=DEFAULT_CROSSING):
    """Read the runs, score them and apply the decision rule. Raises ValueError if any run is missing."""
    runs, missing = {}, []
    for rho in rhos:
        for s in seeds:
            p = find_run(out_dir, tag, rho, s)
            if p is None:
                missing.append((rho, s))
            else:
                runs[(rho, s)] = _load(p)[1]
    if missing:
        raise ValueError(f"missing runs (rho, seed): {missing[:6]}{' ...' if len(missing) > 6 else ''} ({len(missing)} in total)")
    sc = {(rho, t): [score(runs[(rho, s)], t, crossing) for s in seeds] for rho in rhos for t in TARGETS}
    res = dict(table={}, flops={}, n_never={}, mean_steps={}, mean_flops={}, diag={}, seeds=list(seeds), rhos=list(rhos))
    for rho in rhos:
        res["n_never"][rho] = {t: sum(not x[2] for x in sc[(rho, t)]) for t in TARGETS}
        res["mean_steps"][rho] = {t: float(np.mean([x[0] for x in sc[(rho, t)]])) for t in TARGETS}
        res["mean_flops"][rho] = {t: float(np.mean([x[1] for x in sc[(rho, t)]])) for t in TARGETS}
        res["diag"][rho] = {k: float(np.nanmedian([_diag(runs[(rho, s)])[k] for s in seeds])) for k in ("align", "h_var", "inv", "final_acc")}
    if 0.0 not in rhos:
        raise ValueError("rho = 0 is the reference and must be among the doses")
    for rho in rhos:
        if rho == 0:
            continue
        res["table"][rho], res["flops"][rho] = {}, {}
        for t in TARGETS:
            ref_s = [x[0] for x in sc[(0.0, t)]]
            res["table"][rho][t] = paired_pct_change(ref_s, [x[0] for x in sc[(rho, t)]], n_boot=n_boot, seed=boot_seed)
            res["flops"][rho][t] = paired_pct_change([x[1] for x in sc[(0.0, t)]], [x[1] for x in sc[(rho, t)]],
                                                     n_boot=n_boot, seed=boot_seed)
    res["verdict"] = decide(res["table"])
    return res


def format_report(res, tag):
    L = [f"check G0, tag {tag}: dose-response of the learned meeting loss, {len(res['seeds'])} paired seeds, "
         f"steps-to-target (D-018 rule); change versus rho = 0 [95% bootstrap interval]"]
    L.append(f"{'rho':>5} {'target':>7} {'mean steps':>11} {'never':>5} {'change in steps':>26} {'change in FLOPs':>26}")
    for rho in res["rhos"]:
        for t in TARGETS:
            ref = "(reference)" if rho == 0 else (lambda c: f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]")(res["table"][rho][t])
            fl = "" if rho == 0 else (lambda c: f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]")(res["flops"][rho][t])
            L.append(f"{rho:>5g} {t:>7} {res['mean_steps'][rho][t]:>11.0f} {res['n_never'][rho][t]:>5} {ref:>26} {fl:>26}")
    L.append("diagnostics (median over seeds, evaluations after step 1000): rho, alignment with the task gradient at F_low, h_var, inverse loss, final val acc")
    for rho in res["rhos"]:
        d = res["diag"][rho]
        L.append(f"  {rho:>5g}  align {d['align']:+.3f}  h_var {d['h_var']:.2f}  inv {d['inv']:.3f}  final acc {d['final_acc']:.4f}")
    L.append(f"VERDICT (rule fixed before the run): {res['verdict']}")
    return "\n".join(L)
