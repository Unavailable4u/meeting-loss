"""Study 1b, the meeting network (docs/notes/meeting-network.md, D-030, DR-023): groups, derived configs, scoring, decision rules.
Exploratory (D-015). The rules below are fixed BEFORE any run.

Tag mn1, seeds 1500-1509, one phase, every group at its tuned config (tuning tag t16: `run_sweep.py --tag t16 --arms <MN_SWEEP_ARMS>`;
`analyze.py select --tag t16 --write-configs`). Metric: FLOPs to 0.97 and 0.975 validation accuracy (rule `two` of D-018, misses charged
the whole run, FLOPs as counted by src/flops.py including the backward chain), paired change of the mean, 95% bootstrap interval;
final accuracy = mean validation accuracy over the last 5 evaluations (the readout the arm predicts with).
Groups (reference A = arm A tuned (relu), S = shallow net tuned; D = old auxiliary-loss arm D as tuned, descriptive):
  M1m mn_match k=1 (the equal-cost meeting node)   M1f mn_fwd k=1   M1n noise code   M1h multi-hot code (control)   M1t table ablation
  M2m M2f M3m M3f  the k sweep (k = 2, 3), each tuned on its own
GROUP VERDICT (against BOTH A and S, both targets, in this order):
  EXCELS                at least 8 of 10 seeds reach both targets, FLOPs change <= -10% against A and against S with the interval
                        excluding 0 at both targets, and final accuracy within 0.2 points of A (A minus the group <= 0.002)
  FASTER, LESS ACCURATE the FLOPs conditions of EXCELS hold but the accuracy gap is above 0.2 points
  FAILS                 fewer than 5 of 10 seeds reach 0.97, or the final accuracy is more than 1 point below A's
  COMPETITIVE           at least 8 of 10 seeds reach both targets, accuracy within 0.2 points of A and the FLOPs change is <= +15% against
                        the cheaper baseline at both targets
  SLOWER                at least 8 of 10 reach both targets and accuracy is within 0.2 points of A, but it is not COMPETITIVE
  UNCLEAR               anything else
READINGS (diagnose_study):
  chain vs table   M1t against M1m: CHAIN HELPS (the table needs >= +10% FLOPs, interval above 0, at both targets) | CHAIN NOT NEEDED
                   (interval's upper end below +10% at both targets) | UNCLEAR
  label richness   M1n and M1h each against M1m: RICHER LABEL HELPS (<= -10%, interval below 0, both targets) | NO EFFECT (interval inside
                   (-10%, +10%) at both targets) | UNCLEAR. M1h is a control: the multi-hot code carries the same information.
  shallow control  M1m against S: BETTER | WORSE | SAME AS SHALLOW NET (interval inside (-10%, +10%) at both targets) | UNCLEAR
  meeting point    for each readout family (match: M1m M2m M3m; fwd: M1f M2f M3f) the k with the lowest mean FLOPs to 0.975;
                   EQUAL-COST RULE CONFIRMED when that k is the equal-cost k or the equal-cost k is not distinguishable from it (the
                   paired interval of the change contains 0), otherwise NOT CONFIRMED
  readout          M1f: mean final accuracy of the forward readout minus the matching readout: FORWARD BETTER (>= +0.2 points) |
                   MATCH BETTER (<= -0.2 points) | SAME
"""
import copy
import glob
import json
import pathlib

import numpy as np
import yaml

from src.crossing import DEFAULT_CROSSING
from src.meetnet import equal_cost_k
from src.sweep import LATE_K, paired_pct_change, score_run

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = (0.97, 0.975)
TAGS = {"mn1": dict(seeds=tuple(range(1500, 1510)), tuning="t16")}
A_CONFIG = "configs/tuned/t1_base_two.yaml"
D_CONFIG = "configs/tuned/t1_meet_two.yaml"
EXCEL_PCT, COMP_PCT, ACC_TOL, FAIL_ACC = 10.0, 15.0, 0.002, 0.01
REACH_OK, REACH_BAD = 0.8, 0.5
# group -> sweep arm of the tuning tag (None: not tuned by t16)
GROUP_ARM = {"S": "shallow", "M1m": "mn_match", "M1f": "mn_fwd", "M1n": "mn_noise", "M1h": "mn_mh", "M1t": "mn_table",
             "M2m": "mn_match_k2", "M2f": "mn_fwd_k2", "M3m": "mn_match_k3", "M3f": "mn_fwd_k3"}
REFS = {"A": None, "S": None}


def stage_name(tag, name):
    return f"mn-{tag}-{name}"


def config_path(tag, name):
    return ROOT / "configs" / "mnet" / f"{tag}_{name}.yaml"


def tuned_path(tag, name):
    if name == "A":
        return A_CONFIG
    if name == "D":
        return D_CONFIG
    return f"configs/tuned/{TAGS[tag]['tuning']}_{GROUP_ARM[name]}_two.yaml"


def _load_yaml(rel):
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"{rel} is missing: it comes from the t16 tuning pass, which has to be run and selected first "
                                f"(docs/notes/meeting-network.md, commands in order)")
    return yaml.safe_load(open(path))


def group_specs(tag="mn1", seeds=None):
    """Groups: dicts with name, seeds, role. The references come first."""
    seeds = tuple(seeds) if seeds is not None else TAGS[tag]["seeds"]
    names = ["A", "S", "D"] + [n for n in GROUP_ARM if n != "S"]
    return [dict(name=n, seeds=seeds, role="reference" if n in ("A", "S") else "descriptive" if n == "D" else "test") for n in names]


def build_config(spec, tag="mn1"):
    """Derived config: the tuned config of the group, with the stage renamed (nothing is tuned here)."""
    cfg = copy.deepcopy(_load_yaml(tuned_path(tag, spec["name"])))
    cfg["stage"] = stage_name(tag, spec["name"])
    if spec["name"] in GROUP_ARM:
        want = {"M1": 1, "M2": 2, "M3": 3}.get(spec["name"][:2], 1)
        if cfg.get("mn_k", 1) != want:
            raise ValueError(f"group {spec['name']}: tuned config has mn_k {cfg.get('mn_k')}, expected {want}")
    return cfg


# ------------------------------------------------------------------ reading runs
def _load(path):
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


def find_run(out_dir, cfg, seed):
    hits = glob.glob(str(pathlib.Path(out_dir) / f"*_{cfg['stage']}_{cfg['arm']}_s{seed}.jsonl"))
    return hits[0] if hits else None


def load_group(out_dir, tag, spec):
    """Per seed: FLOPs and steps to each target (misses charged the full run), reached, final accuracy (mean of the last LATE_K
    evaluations), and the late accuracy of every logged readout."""
    cfg = build_config(spec, tag)
    g = dict(seeds=list(spec["seeds"]), flops={t: [] for t in TARGETS}, steps={t: [] for t in TARGETS}, reached={t: [] for t in TARGETS},
             final=[], readouts={}, name=spec["name"])
    missing = []
    for s in spec["seeds"]:
        p = find_run(out_dir, cfg, s)
        if p is None:
            missing.append(s)
            continue
        rows = _load(p)[1]
        for t in TARGETS:
            sc = score_run(rows, t, crossing=DEFAULT_CROSSING)
            g["flops"][t].append(float(sc["score"]))
            g["steps"][t].append(sc["steps"] if sc["reached"] else rows[-1]["step"])
            g["reached"][t].append(bool(sc["reached"]))
        g["final"].append(float(np.mean([r["val_acc"] for r in rows[-LATE_K:]])))
        for key in rows[-1]:
            if key.startswith("val_acc_"):
                g["readouts"].setdefault(key[len("val_acc_"):], []).append(float(np.mean([r[key] for r in rows[-LATE_K:]])))
    if missing:
        raise ValueError(f"group {spec['name']}: missing seeds {missing}")
    return g


def change(ref, grp, key="flops", n_boot=10000):
    """{target: (point, lo, hi)}: paired percent change of the mean `key` (flops or steps) of grp against ref (negative = less)."""
    return {t: paired_pct_change(ref[key][t], grp[key][t], n_boot=n_boot) for t in TARGETS}


# ------------------------------------------------------------------ decision rules
def decide_group(reached, vs_a, vs_s, acc_gap, n=10):
    """Group verdict (module docstring). reached[t] = seeds reaching the target; vs_a / vs_s [t] = (point, lo, hi) FLOPs change against
    A / S; acc_gap = A's final accuracy minus the group's (positive = worse)."""
    ok, bad = REACH_OK * n, REACH_BAD * n
    reach_ok = all(reached[t] >= ok for t in TARGETS)
    fast = all(c[t][0] <= -EXCEL_PCT and c[t][2] < 0 for c in (vs_a, vs_s) for t in TARGETS)
    if reach_ok and fast and acc_gap <= ACC_TOL:
        return "EXCELS"
    if reach_ok and fast:
        return "FASTER, LESS ACCURATE"
    if reached[TARGETS[0]] < bad or acc_gap > FAIL_ACC:
        return "FAILS"
    if reach_ok and acc_gap <= ACC_TOL:
        if all(max(vs_a[t][0], vs_s[t][0]) <= COMP_PCT for t in TARGETS):
            return "COMPETITIVE"
        return "SLOWER"
    return "UNCLEAR"


def _verdict3(ch, lim=EXCEL_PCT):
    """BETTER / WORSE / SAME / UNCLEAR for a change dict (negative = the group needs less)."""
    if all(ch[t][0] <= -lim and ch[t][2] < 0 for t in TARGETS):
        return "BETTER"
    if all(ch[t][0] >= lim and ch[t][1] > 0 for t in TARGETS):
        return "WORSE"
    if all(ch[t][1] > -lim and ch[t][2] < lim for t in TARGETS):
        return "SAME"
    return "UNCLEAR"


def diagnose_study(scores, n_boot=10000):
    """Pre-written readings (module docstring). scores = {group: load_group(...)}; returns a dict of short verdict strings and numbers."""
    out = {}
    if "M1m" not in scores:
        return out
    m = scores["M1m"]
    if "M1t" in scores:
        v = _verdict3(change(m, scores["M1t"], n_boot=n_boot))          # the table against the chain: WORSE = the chain helps
        out["chain_vs_table"] = {"WORSE": "CHAIN HELPS", "SAME": "CHAIN NOT NEEDED", "BETTER": "TABLE BETTER"}.get(v, "UNCLEAR")
    for name in ("M1n", "M1h"):
        if name in scores:
            v = _verdict3(change(m, scores[name], n_boot=n_boot))
            out[f"label_{name}"] = {"BETTER": "RICHER LABEL HELPS", "SAME": "NO EFFECT"}.get(v, "WORSE" if v == "WORSE" else "UNCLEAR")
    if "S" in scores:
        v = _verdict3(change(scores["S"], m, n_boot=n_boot))
        out["shallow"] = {"BETTER": "BETTER", "WORSE": "WORSE", "SAME": "SAME AS SHALLOW NET"}.get(v, "UNCLEAR")
    k_eq = equal_cost_k()
    for fam, names in (("match", ("M1m", "M2m", "M3m")), ("fwd", ("M1f", "M2f", "M3f"))):
        have = {i + 1: scores[n] for i, n in enumerate(names) if n in scores}
        if k_eq not in have or len(have) < 2:
            continue
        means = {k: float(np.mean(g["flops"][TARGETS[1]])) for k, g in have.items()}
        best = min(means, key=lambda k: (means[k], k))
        if best == k_eq:
            verdict = "EQUAL-COST RULE CONFIRMED"
        else:
            pt, lo, hi = paired_pct_change(have[best]["flops"][TARGETS[1]], have[k_eq]["flops"][TARGETS[1]], n_boot=n_boot)
            verdict = "EQUAL-COST RULE CONFIRMED" if lo <= 0 <= hi else "EQUAL-COST RULE NOT CONFIRMED"
        out[f"meeting_point_{fam}"] = dict(verdict=verdict, best_k=best, mean_flops_0975=means)
    if "fwd" in scores.get("M1f", {}).get("readouts", {}) and "match" in scores["M1f"]["readouts"]:
        d = float(np.mean(scores["M1f"]["readouts"]["fwd"]) - np.mean(scores["M1f"]["readouts"]["match"]))
        out["readout"] = dict(verdict="FORWARD BETTER" if d >= 0.002 else "MATCH BETTER" if d <= -0.002 else "SAME", fwd_minus_match=d)
    return out


def analyze(scores, n_boot=10000):
    """Everything the report needs: per-group changes against A and S, verdicts, readings."""
    res = {"groups": {}, "readings": diagnose_study(scores, n_boot)}
    for name, g in scores.items():
        if name in ("A", "S", "D"):
            continue
        vs_a, vs_s = change(scores["A"], g, n_boot=n_boot), change(scores["S"], g, n_boot=n_boot)
        gap = float(np.mean(scores["A"]["final"]) - np.mean(g["final"]))
        reached = {t: int(sum(g["reached"][t])) for t in TARGETS}
        res["groups"][name] = dict(verdict=decide_group(reached, vs_a, vs_s, gap, n=len(g["seeds"])), vs_a=vs_a, vs_s=vs_s, acc_gap=gap, reached=reached)
    return res


def report(scores, res):
    lines = ["Study 1b, the meeting network (tag mn1; FLOPs to target, misses charged the full run; change against A and against S)"]
    lines.append(f"{'group':6s} {'reach .97/.975':>14s} {'final acc':>9s}  {'FLOPs .97 vs A / vs S':>24s}  {'FLOPs .975 vs A / vs S':>24s}  verdict")
    for name, g in scores.items():
        r = f"{sum(g['reached'][TARGETS[0]])}/{len(g['seeds'])} {sum(g['reached'][TARGETS[1]])}/{len(g['seeds'])}"
        row = f"{name:6s} {r:>14s} {np.mean(g['final']):9.4f}  "
        if name in res["groups"]:
            q = res["groups"][name]
            f = lambda c: f"{c[0]:+6.1f}% [{c[1]:+.0f},{c[2]:+.0f}]"
            row += f"{f(q['vs_a'][TARGETS[0]]):>12s} / {f(q['vs_s'][TARGETS[0]]):>12s}  {f(q['vs_a'][TARGETS[1]]):>12s} / {f(q['vs_s'][TARGETS[1]]):>12s}  {q['verdict']}"
        else:
            row += "(reference)" if name in ("A", "S") else "(descriptive)"
        lines.append(row)
    lines.append("")
    for k, v in res["readings"].items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)
