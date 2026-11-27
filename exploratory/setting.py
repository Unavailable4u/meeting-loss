"""Setting study (docs/notes/setting-fmnist.md, D-031, DR-024): the three method families of Studies 1, 1b and 2 on another dataset.
Exploratory (D-015). The rules are fixed BEFORE any run on the dataset, and so are the targets (src/datasets.py: from a probe of arm A).

Tag f1: Fashion-MNIST, the 784-256-128-256-10 MLP, AdamW, 3,000 steps, seeds 2500-2509 (paired), every arm tuned on the dataset
(tuning tag t17: the arms of `src.sweep.FMNIST_SWEEP_ARMS`, 25 trials x 2 seeds, selection target = the dataset's high target; t18: a
second, independent tuning of arm A, the noise reference). Metric: FLOPs to the dataset's two targets (rule `two` of D-018, misses
charged the whole run, FLOPs as counted by src/flops.py and src/sb.py), paired change of the mean, 95% bootstrap interval.
Groups (reference A = arm A tuned on the dataset, A2 its second tuning, S the shallow net tuned):
  Study 2   S20 = selective backward pass at 20% of the rows by loss, no warm-up, A's hyperparameters (the arm confirmed on MNIST, E-027)
            SBT = Selective-Backprop tuned (t17)
  Study 1   D = meeting loss at the middle state (arm D tuned), B = deep supervision (tuned)
  Study 1b  M1m, M1f, M1t, M2m, M3m, M3f of exploratory/meetnet.py (tuned on the dataset)
RULES
  Study 2 and Study 1 (S20, D, B against A; S20 also against SBT): margin rule of D-019 per target (`margin_verdict`: point change <= -15%
    and the interval excludes 0 gives the positive verdict; the interval excluding every gain of 15% gives the negative one; else
    INCONCLUSIVE); the positive verdict also needs the gain to exceed the re-tuning shift of A (A2 against A, rule `max_gain`); both
    targets must agree. Labels: S20 HOLDS / DOES NOT HOLD / INCONCLUSIVE (the claim C-24 wording), D and B HELPS / NO HELP /
    INCONCLUSIVE. Against SBT no shift is applied (as H10).
  Study 1b: the group verdicts and the readings of exploratory/meetnet.py (EXCELS, ..., against A and against S, at the dataset's targets).
  A group that reaches the low target in fewer than 5 of 10 seeds is reported as FAILS TO REACH whatever else the rule says.
"""
import copy
import pathlib

import numpy as np
import yaml

from exploratory import meetnet as mn
from src.datasets import dataset_targets
from src.sweep import margin_verdict, paired_pct_change, shift_value

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARGIN = 15.0
SHIFT_RULE = "max_gain"
SELECTIVE = dict(arm="selective", sel_frac=0.2, sel_warmup=0, sel_mode="top")
# tag -> dataset, seeds, tuning tags (t_main: every arm; t_second: second tuning of A)
TAGS = {"f1": dict(dataset="fmnist", seeds=tuple(range(2500, 2510)), t_main="t17", t_second="t18")}
# group -> (kind, sweep arm of the tuning tag, role)
GROUPS = {
    "A": ("tuned", "f_base", "reference"), "A2": ("tuned2", "f_base", "reference"), "S": ("tuned", "f_shallow", "reference"),
    "S20": ("derived", "f_base", "test"), "SBT": ("tuned", "f_sb", "test"),
    "D": ("tuned", "f_meet", "test"), "B": ("tuned", "f_deepsup", "test"),
    "M1m": ("tuned", "f_mn_match", "test"), "M1f": ("tuned", "f_mn_fwd", "test"), "M1t": ("tuned", "f_mn_table", "test"),
    "M2m": ("tuned", "f_mn_match_k2", "test"), "M3m": ("tuned", "f_mn_match_k3", "test"), "M3f": ("tuned", "f_mn_fwd_k3", "test"),
}
MN_GROUPS = ("M1m", "M1f", "M1t", "M2m", "M3m", "M3f")


def tag_spec(tag):
    """Registered tag; an unregistered tag (smoke runs) behaves like f1."""
    return TAGS.get(tag, TAGS["f1"])


def stage_name(tag, name):
    return f"set-{tag}-{name}"


def config_path(tag, name):
    return ROOT / "configs" / "setting" / f"{tag}_{name}.yaml"


def tuned_path(tag, name):
    kind, arm, _ = GROUPS[name]
    t = tag_spec(tag)["t_second" if kind == "tuned2" else "t_main"]
    return f"configs/tuned/{t}_{arm}_two.yaml"


def _load_yaml(rel):
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"{rel} is missing: it comes from a tuning pass that has to be run and selected first "
                                f"(docs/notes/setting-fmnist.md, commands in order)")
    return yaml.safe_load(open(path))


def group_specs(tag="f1", seeds=None):
    seeds = tuple(seeds) if seeds is not None else tag_spec(tag)["seeds"]
    return [dict(name=n, seeds=seeds, role=r) for n, (_, _, r) in GROUPS.items()]


def build_config(spec, tag="f1"):
    """Derived config: the tuned config of the group (S20: arm A's with the selective keys), stage renamed, dataset checked."""
    name, kind = spec["name"], GROUPS[spec["name"]][0]
    cfg = copy.deepcopy(_load_yaml(tuned_path(tag, name)))
    if cfg.get("dataset") != tag_spec(tag)["dataset"]:
        raise ValueError(f"group {name}: the tuned config is for dataset {cfg.get('dataset')!r}, expected {tag_spec(tag)['dataset']!r}")
    if kind == "derived":
        cfg.update(SELECTIVE)
    if name in MN_GROUPS:
        want = {"M1": 1, "M2": 2, "M3": 3}[name[:2]]
        if cfg.get("mn_k", 1) != want:
            raise ValueError(f"group {name}: tuned config has mn_k {cfg.get('mn_k')}, expected {want}")
    cfg["stage"] = stage_name(tag, name)
    return cfg


def targets_of(tag):
    return dataset_targets(tag_spec(tag)["dataset"], ROOT)


def load_group(out_dir, tag, spec):
    return mn.load_group(out_dir, tag, spec, targets=targets_of(tag), cfg=build_config(spec, tag))


# ------------------------------------------------------------------ rules
_POS = {"S20": "HOLDS", "D": "HELPS", "B": "HELPS"}
_NEG = {"S20": "DOES NOT HOLD", "D": "NO HELP", "B": "NO HELP"}


def decide_margin(name, change, shift=None, margin=MARGIN, targets=None):
    """Margin rule of D-019 for S20, D or B against A: change[t] = (point, lo, hi) FLOPs change; shift[t] = the re-tuning shift of A (or
    None). Both targets must agree."""
    targets = targets or tuple(change)
    v = [margin_verdict(*change[t], margin=margin, shift=None if shift is None else shift[t]) for t in targets]
    if all(x == "supported" for x in v):
        return _POS[name]
    if all(x == "refuted" for x in v):
        return _NEG[name]
    return "INCONCLUSIVE"


def analyze(scores, targets, n_boot=10000):
    """Everything the report needs: shift of A, margin verdicts of S20, D and B, the Study 1b group verdicts and readings."""
    res = {"targets": targets, "margin": {}, "mn": None}
    a2 = mn.change(scores["A"], scores["A2"], n_boot=n_boot, targets=targets)
    shift = {t: shift_value([a2[t][0]], SHIFT_RULE) for t in targets}
    res["shift"], res["a2_vs_a"] = shift, a2
    for name in ("S20", "D", "B"):
        ch = mn.change(scores["A"], scores[name], n_boot=n_boot, targets=targets)
        res["margin"][name] = dict(change=ch, verdict=decide_margin(name, ch, shift, targets=targets),
                                   reached={t: int(sum(scores[name]["reached"][t])) for t in targets})
    if "SBT" in scores:
        ch = mn.change(scores["SBT"], scores["S20"], n_boot=n_boot, targets=targets)
        res["s20_vs_sbt"] = dict(change=ch, verdict=decide_margin("S20", ch, None, targets=targets))
    sub = {n: scores[n] for n in ("A", "S") + MN_GROUPS if n in scores}
    res["mn"] = mn.analyze(sub, n_boot=n_boot, targets=targets)
    return res


def report(scores, res, tag="f1"):
    targets = res["targets"]
    f = lambda c: f"{c[0]:+6.1f}% [{c[1]:+.0f},{c[2]:+.0f}]"
    lines = [f"Setting study, tag {tag} (dataset {tag_spec(tag)['dataset']}; targets {targets[0]} / {targets[1]}; FLOPs to target, misses charged the full run)",
             f"{'group':5s} {'reach lo/hi':>11s} {'final acc':>9s} {'GFLOPs lo/hi':>14s}"]
    for n, g in scores.items():
        gf = [np.mean(g["flops"][t]) / 1e9 for t in targets]
        lines.append(f"{n:5s} {sum(g['reached'][targets[0]])}/{len(g['seeds'])} {sum(g['reached'][targets[1]])}/{len(g['seeds'])}   {np.mean(g['final']):9.4f} {gf[0]:7.1f} /{gf[1]:6.1f}")
    lines.append("")
    lines.append("Re-tuning shift of A (A2 against A, max_gain): " + ", ".join(f"{t}: {res['shift'][t]:.1f}%" for t in targets)
                 + "   (A2 against A: " + ", ".join(f(res["a2_vs_a"][t]) for t in targets) + ")")
    for name, q in res["margin"].items():
        lines.append(f"{name} against A: " + " | ".join(f(q["change"][t]) for t in targets) + f"  reached {q['reached']}  ->  {q['verdict']}")
    if "s20_vs_sbt" in res:
        q = res["s20_vs_sbt"]
        lines.append("S20 against SBT: " + " | ".join(f(q["change"][t]) for t in targets) + f"  ->  {q['verdict']}")
    lines.append("")
    lines.append(mn.report({"A": scores["A"], "S": scores["S"], **{n: scores[n] for n in res["mn"]["groups"]}}, res["mn"], targets,
                           title="Study 1b groups (FLOPs change against A and against S)"))
    return "\n".join(lines)
