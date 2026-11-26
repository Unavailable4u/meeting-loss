"""Root-cause checks B and D (docs/notes/mechanism-root-cause.md, D-028, DR-019): plans, configs, scoring and decision rules.
Exploratory (D-015). Rules and thresholds are those of docs/notes/mechanism-b-oracle-arm.md and mechanism-d-replacement-test.md,
fixed BEFORE any run; deviations are listed in mechanism-root-cause.md.

Two settings, so that the activation is not a confound (the G2 check of E-014 ran tanh with arm A's relu-tuned lr and wd):
  relu  arm A as tuned (configs/tuned/t1_base_two.yaml)
  tanh  arm A tuned IN tanh (tag t13, configs/tuned/t13_base_tanh_two.yaml; `run_sweep.py --tag t13 --arms base_tanh`)

TAG mb1 (check B: oracle directions next to backprop, src/oracle.py; strength rho = gradient norm at h relative to the task gradient's)
  F   exact backprop direction, the REPAIRED control F' (the old arm F was inert, F-005)    P   pinv / Gauss-Newton oracle
  Q   damped logit-space oracle    Pr  P with the radial component removed (bounded scale, first order)
  phase 1 (screen)  seeds 700-702: rho in {0.3, 1, 3} for F, P, Q; pick for every (setting, type) the rho with the lowest mean
                    steps-to-0.97 (ties: the smaller rho) -> configs/oracle/mb1_chosen_rho.json
  phase 2 (test)    seeds 710-719: A (and A2 in relu), F, P, Q, Pr at the chosen rho (Pr uses P's rho); in tanh also arm D as
                    tuned (t1_meet_two), descriptive: is D's deficit a tanh effect or a meeting-loss effect?
  phase 3 (lr)      seeds 710-719: P and Q at lr x 0.5 and x 2; per (setting, type) the lr is chosen on seeds 710-714 among
                    {0.5, 1, 2} and evaluated on seeds 715-719
TAG md1 (check D: replacement, backprop cut at h)   seeds 900-909, one phase
  R1 learned G (tuned, tag t4)   R1w R1 with 300 steps of backprop first   R2 oracle P with rho 1   R4 frozen random G
  A per setting (and A2 in relu) are the references

Decision rules (steps-to-target at 0.97 and 0.975, rule `two` of D-018, misses charged the full run, paired change of the mean,
95% bootstrap interval; both targets must agree):
  B phase 2, per setting. ORACLE HELPS: for some type X in {P, Q, Pr}, at both targets the point change against A is <= -10%, the
    interval excludes 0, the point is <= -10% against A2 too (relu only) and at least 5 points below F'. NO HELP: for every X, at
    both targets the interval's lower end is above -10%. INCONCLUSIVE: anything else (no seeds are added to the same test).
  B phase 3: the same two outcomes for the lr-selected P and Q on seeds 715-719 (5 seeds: wide intervals).
  D. SUFFICIENT: a group reaches both targets in at least 8 of 10 seeds and its paired mean steps are within +15% of its reference at
    both targets. INSUFFICIENT: it reaches 0.97 in fewer than 5 of 10 seeds, or its final accuracy is more than 1 point below the
    reference's. PARTIAL: it reaches 0.97 in at least 8 seeds but is not SUFFICIENT. UNCLEAR: anything else. (Checked in this order.)
"""
import copy
import glob
import json
import pathlib

import numpy as np
import yaml

from src.crossing import DEFAULT_CROSSING, crossing_index
from src.sweep import paired_pct_change

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = (0.97, 0.975)
SETTINGS = ("relu", "tanh")
BASE_CONFIG = {"relu": "configs/tuned/t1_base_two.yaml", "tanh": "configs/tuned/t13_base_tanh_two.yaml"}
A2_CONFIG = "configs/tuned/t2_base_two.yaml"
D_CONFIG = "configs/tuned/t1_meet_two.yaml"          # arm D as tuned (tanh); descriptive group of phase 2
R1_CONFIG = "configs/tuned/t4_meet_replace_two.yaml"
CHOSEN_FILE = "configs/oracle/mb1_chosen_rho.json"
RHO_GRID = (0.3, 1.0, 3.0)
LR_MULTS = (0.5, 2.0)
R1W_START = 300
ORACLE_TYPE = {"F": "grad", "P": "pinv", "Q": "logit", "Pr": "pinv"}
HELP_PCT, BEAT_F_PTS = 10.0, 5.0                      # check B
SUFF_PCT, REACH_OK, REACH_BAD, ACC_GAP = 15.0, 0.8, 0.5, 0.01   # check D (reach thresholds are fractions of the seeds: 8 and 5 of 10)
TAGS = {
    "mb1": dict(check="B", seeds={1: tuple(range(700, 703)), 2: tuple(range(710, 720)), 3: tuple(range(710, 720))}),
    "md1": dict(check="D", seeds={1: tuple(range(900, 910))}),
}


def tag_spec(tag):
    """Registered tag; an unregistered tag (plumbing tests, smoke runs) behaves like mb1."""
    return TAGS.get(tag, TAGS["mb1"])


def _tag(x):
    return f"{float(x):g}".replace(".", "p")


def group_name(kind, setting=None, rho=None, lr_mult=1.0):
    parts = [kind] + ([setting] if setting else []) + ([f"rho{_tag(rho)}"] if rho is not None else []) \
        + ([f"lr{_tag(lr_mult)}"] if lr_mult != 1.0 else [])
    return "-".join(parts)


def stage_name(tag, name):
    return f"orc-{tag}-{name}"


def config_path(tag, name):
    return ROOT / "configs" / "oracle" / f"{tag}_{name}.yaml"


def _load_yaml(rel):
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"{rel} is missing: it comes from a tuning pass that has to be run and selected first "
                                f"(docs/notes/mechanism-root-cause.md, commands in order)")
    return yaml.safe_load(open(path))


def load_chosen():
    path = ROOT / CHOSEN_FILE
    if not path.exists():
        raise FileNotFoundError(f"{CHOSEN_FILE} is missing: run phase 1 and then `--phase 1 --analyze-only --write-chosen`")
    return json.load(open(path))


def _spec(kind, setting, ref, seeds, rho=None, lr_mult=1.0):
    return dict(name=group_name(kind, setting, rho, lr_mult), kind=kind, setting=setting, rho=rho, lr_mult=lr_mult,
                ref=ref, seeds=tuple(seeds))


def r1_setting():
    """The activation the tuned learned replacement arm (R1) ended up with: its reference is arm A in that activation."""
    return _load_yaml(R1_CONFIG)["act"]


def group_specs(tag, phase, settings=SETTINGS):
    """Groups of a phase: dicts with name, kind, setting, rho, lr_mult, ref (name of the reference group, None for references)
    and seeds."""
    seeds = tag_spec(tag)["seeds"][phase]
    specs = []
    if tag_spec(tag)["check"] == "B":
        chosen = load_chosen() if phase > 1 else None
        for st in settings:
            ref = group_name("A", st)
            if phase < 3:                                   # phase 3 reuses the A and oracle groups of phase 2
                specs.append(_spec("A", st, None, seeds))
            if phase == 1:
                specs += [_spec(k, st, ref, seeds, rho) for k in ("F", "P", "Q") for rho in RHO_GRID]
            elif phase == 2:
                if st == "relu":
                    specs.append(_spec("A2", st, None, seeds))
                specs += [_spec(k, st, ref, seeds, chosen[st][k[0]]) for k in ("F", "P", "Q", "Pr")]
                if st == "tanh":                            # descriptive: arm D as tuned (tanh) on the same seeds as the tanh baseline
                    specs.append(_spec("D", st, ref, seeds))
            else:
                specs += [_spec(k, st, ref, seeds, chosen[st][k], m) for k in ("P", "Q") for m in LR_MULTS]
        return specs
    r1 = r1_setting()
    for st in settings:
        specs.append(_spec("A", st, None, seeds))
        if st == "relu":
            specs.append(_spec("A2", st, None, seeds))
    specs += [_spec(k, None, group_name("A", r1), seeds) for k in ("R1", "R1w")]
    specs += [_spec(k, st, group_name("A", st), seeds, 1.0 if k == "R2" else None) for st in settings for k in ("R2", "R4")]
    return specs


def build_config(spec, tag):
    """Derived config of a group. Nothing is tuned here: the base configs are the tuned ones, the check's own keys are added."""
    kind = spec["kind"]
    rel = (A2_CONFIG if kind == "A2" else R1_CONFIG if kind in ("R1", "R1w") else D_CONFIG if kind == "D"
           else BASE_CONFIG[spec["setting"]])
    cfg = copy.deepcopy(_load_yaml(rel))
    cfg["stage"] = stage_name(tag, spec["name"])
    if kind in ("F", "P", "Q", "Pr"):
        cfg.update(arm="meet_oracle", oracle_type=ORACLE_TYPE[kind], oracle_rho=float(spec["rho"]), oracle_warmup_steps=0,
                   oracle_alpha_mult=1.0, oracle_bound="radial" if kind == "Pr" else "none")
        cfg["lr"] = float(f"{cfg['lr'] * spec['lr_mult']:.6g}")
    elif kind in ("R1", "R1w"):
        if cfg["arm"] != "meet_replace":
            raise ValueError(f"{R1_CONFIG} must be a meet_replace config, found arm {cfg['arm']!r}")
        cfg.update(replace_start=R1W_START if kind == "R1w" else 0, lam_meet=1.0)
    elif kind == "R2":
        cfg.update(arm="meet_replace_oracle", oracle_type="pinv", oracle_rho=1.0, oracle_warmup_steps=0, oracle_alpha_mult=1.0,
                   oracle_bound="none", replace_start=0)
    elif kind == "R4":
        cfg.update(arm="meet_replace_frozen", replace_start=0, lam_meet=1.0)
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


def score(rows, target, crossing=DEFAULT_CROSSING):
    """(steps, reached). A run that never reaches the target is charged its full run (never dropped)."""
    i = crossing_index([r["val_acc"] for r in rows], target, crossing)
    return (rows[-1]["step"], False) if i is None else (rows[i]["step"], True)


def _median_late(rows, key, after=1000):
    v = [r[key] for r in rows if r["step"] > after and key in r and r[key] == r[key]]
    return float(np.median(v)) if v else float("nan")


def _nanmedian(v):
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else float("nan")


def load_group(out_dir, tag, spec):
    """Scores of one group on its seeds: steps and reached per target, final accuracy, late h_var and alignment."""
    cfg = build_config(spec, tag)
    seeds, steps, reached, final, hvar, align, missing = list(spec["seeds"]), {t: [] for t in TARGETS}, {t: [] for t in TARGETS}, [], [], [], []
    for s in seeds:
        p = find_run(out_dir, cfg, s)
        if p is None:
            missing.append(s)
            continue
        rows = _load(p)[1]
        for t in TARGETS:
            st, ok = score(rows, t)
            steps[t].append(st)
            reached[t].append(ok)
        final.append(rows[-1]["val_acc"])
        hvar.append(_median_late(rows, "h_var"))
        align.append(_median_late(rows, "align"))
    if missing:
        raise ValueError(f"group {spec['name']}: missing seeds {missing}")
    return dict(seeds=seeds, steps=steps, reached=reached, final=final, h_var=_nanmedian(hvar), align=_nanmedian(align))


def load_phase(out_dir, tag, phase, settings=SETTINGS):
    specs = group_specs(tag, phase, settings)
    return specs, {s["name"]: load_group(out_dir, tag, s) for s in specs}


def change(ref, grp, targets=TARGETS, idx=None, n_boot=10000):
    """{target: (point, lo, hi)} paired change of mean steps (percent, negative = fewer steps) on the seeds `idx` (default all)."""
    sel = (lambda a: [a[i] for i in idx]) if idx is not None else (lambda a: a)
    return {t: paired_pct_change(sel(ref["steps"][t]), sel(grp["steps"][t]), n_boot=n_boot) for t in targets}


# ------------------------------------------------------------------ phase 1: choose rho
def choose_rho(scores, setting):
    """{'F': rho, 'P': rho, 'Q': rho}: the rho with the lowest mean steps-to-0.97 (ties: the smaller rho)."""
    out = {}
    for k in ("F", "P", "Q"):
        best = min(RHO_GRID, key=lambda r: (np.mean(scores[group_name(k, setting, r)]["steps"][TARGETS[0]]), r))
        out[k] = float(best)
    return out


# ------------------------------------------------------------------ decision rules
def decide_oracle(vs_a, vs_a2, vs_f, limit=HELP_PCT):
    """Rule of check B (module docstring). vs_a[X][t] = (point, lo, hi) against A for X in P, Q, Pr (or the lr-selected P, Q);
    vs_a2[X][t] = point against A2 (relu) or None; vs_f[t] = point of F' against A (None when not available).
    Returns 'ORACLE HELPS', 'NO HELP' or 'INCONCLUSIVE'."""
    def helps(x):
        for t in TARGETS:
            p, lo, hi = vs_a[x][t]
            if not (p <= -limit and hi < 0):
                return False
            if vs_a2 and vs_a2.get(x) and vs_a2[x][t] is not None and vs_a2[x][t] > -limit:
                return False
            if vs_f and vs_f.get(t) is not None and p > vs_f[t] - BEAT_F_PTS:
                return False
        return True
    if any(helps(x) for x in vs_a):
        return "ORACLE HELPS"
    if all(vs_a[x][t][1] > -limit for x in vs_a for t in TARGETS):
        return "NO HELP"
    return "INCONCLUSIVE"


def decide_replace(reached, point, acc_gap, n=10):
    """Rule of check D. reached[t] = number of seeds that reached the target (of n), point[t] = paired change of mean steps
    against the reference (percent), acc_gap = reference mean final accuracy minus the group's (positive = worse)."""
    ok, bad = REACH_OK * n, REACH_BAD * n
    if all(reached[t] >= ok for t in TARGETS) and all(point[t] <= SUFF_PCT for t in TARGETS):
        return "SUFFICIENT"
    if reached[TARGETS[0]] < bad or acc_gap > ACC_GAP:
        return "INSUFFICIENT"
    if reached[TARGETS[0]] >= ok:
        return "PARTIAL"
    return "UNCLEAR"


def diagnose_replace(verdicts):
    """Pre-written reading of check D (note): verdicts = {'R1': .., 'R2': .., 'R4': ..} in R1's activation."""
    r1, r2, r4 = verdicts["R1"], verdicts["R2"], verdicts["R4"]
    lines = []
    if r1 == "SUFFICIENT":
        lines.append("R1 SUFFICIENT: replacement is viable with a learned G (this shows sufficiency, not a speedup).")
    elif r2 == "SUFFICIENT":
        lines.append("R1 not sufficient but R2 SUFFICIENT: the quality of the learned G is the bottleneck.")
    elif r1 in ("PARTIAL", "INSUFFICIENT") and r2 in ("PARTIAL", "INSUFFICIENT"):
        lines.append("R1 and R2 both short of SUFFICIENT: the local-inverse signal itself does not carry enough to train on.")
    else:
        lines.append("R1 / R2 outcome unclear: report as it is.")
    lines.append("R4 has the same verdict as R1: G adds nothing over a random feedback matrix." if r4 == r1 else
                 f"R4 ({r4}) differs from R1 ({r1}).")
    return lines


# ------------------------------------------------------------------ analyses and reports
def analyze_phase1(scores, settings=SETTINGS):
    chosen = {st: choose_rho(scores, st) for st in settings}
    return dict(chosen=chosen)


def analyze_phase2(specs, scores, n_boot=10000):
    """Per setting: changes against A (and A2, F'), the verdict, and diagnostics."""
    out = {}
    for st in sorted({s["setting"] for s in specs}):
        names = {s["kind"]: s["name"] for s in specs if s["setting"] == st}
        a = scores[names["A"]]
        res = dict(groups={}, a2=None)
        for k in ("F", "P", "Q", "Pr"):
            if k in names:
                g = scores[names[k]]
                res["groups"][k] = dict(name=names[k], vs_a=change(a, g, n_boot=n_boot),
                                        vs_a2=change(scores[names["A2"]], g, n_boot=n_boot) if "A2" in names else None,
                                        h_var_ratio=g["h_var"] / a["h_var"] if a["h_var"] == a["h_var"] else float("nan"),
                                        align=g["align"], never={t: len(g["reached"][t]) - sum(g["reached"][t]) for t in TARGETS})
        if "A2" in names:
            res["a2"] = change(a, scores[names["A2"]], n_boot=n_boot)
        if "D" in names:                                   # descriptive: is arm D's deficit a tanh effect?
            d = scores[names["D"]]
            relu_a = group_name("A", "relu")
            res["D"] = dict(vs_a_tanh=change(a, d, n_boot=n_boot),
                            vs_a_relu=change(scores[relu_a], d, n_boot=n_boot) if relu_a in scores else None)
            if relu_a in scores:                           # and the tanh baseline against the relu baseline
                res["a_vs_relu"] = change(scores[relu_a], a, n_boot=n_boot)
        vs_a = {k: v["vs_a"] for k, v in res["groups"].items() if k in ("P", "Q", "Pr")}
        vs_a2 = {k: ({t: v["vs_a2"][t][0] for t in TARGETS} if v["vs_a2"] else None) for k, v in res["groups"].items() if k in vs_a}
        vs_f = {t: res["groups"]["F"]["vs_a"][t][0] for t in TARGETS} if "F" in res["groups"] else None
        res["verdict"] = decide_oracle(vs_a, vs_a2, vs_f)
        out[st] = res
    return out


def analyze_phase3(specs3, scores3, scores2, tag, n_boot=10000):
    """lr chosen on the first five seeds of the block among {0.5, 1, 2}, evaluated on the last five (note, phase 3)."""
    out = {}
    for st in sorted({s["setting"] for s in specs3}):
        a = scores2[group_name("A", st)]
        seeds = a["seeds"]
        first, last = list(range(len(seeds) // 2)), list(range(len(seeds) // 2, len(seeds)))
        vs_a, picked = {}, {}
        for k in ("P", "Q"):
            rho = next(s["rho"] for s in specs3 if s["setting"] == st and s["kind"] == k)
            cands = {1.0: scores2[group_name(k, st, rho)]}
            cands.update({m: scores3[group_name(k, st, rho, m)] for m in LR_MULTS})
            m_best = min(cands, key=lambda m: (np.mean([cands[m]["steps"][TARGETS[0]][i] for i in first]), abs(m - 1.0)))
            picked[k] = m_best
            vs_a[k] = change(a, cands[m_best], idx=last, n_boot=n_boot)
        out[st] = dict(picked=picked, vs_a=vs_a, verdict=decide_oracle(vs_a, None, None))
    return out


def analyze_replace(specs, scores, n_boot=10000):
    out = dict(groups={}, diagnosis={})
    for s in specs:
        if s["ref"] is None:
            continue
        g, ref = scores[s["name"]], scores[s["ref"]]
        ch = change(ref, g, n_boot=n_boot)
        reached = {t: int(sum(g["reached"][t])) for t in TARGETS}
        gap = float(np.mean(ref["final"]) - np.mean(g["final"]))
        out["groups"][s["name"]] = dict(ref=s["ref"], change=ch, reached=reached, n=len(g["seeds"]), acc_gap=gap,
                                        verdict=decide_replace(reached, {t: ch[t][0] for t in TARGETS}, gap, len(g["seeds"])),
                                        h_var=g["h_var"], align=g["align"])
    st = r1_setting()
    v = {"R1": out["groups"]["R1"]["verdict"], "R2": out["groups"][group_name("R2", st, 1.0)]["verdict"],
         "R4": out["groups"][group_name("R4", st)]["verdict"]}
    out["diagnosis"] = diagnose_replace(v)
    out["r1_setting"] = st
    return out


def _fmt(c):
    return f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]"


def report_phase1(scores, specs, res):
    L = ["check B, phase 1 (screen): mean steps to 0.97 / 0.975 per (setting, type, rho); chosen rho = lowest mean steps to 0.97"]
    for s in specs:
        g = scores[s["name"]]
        L.append(f"  {s['name']:<22} {np.mean(g['steps'][0.97]):7.0f} {np.mean(g['steps'][0.975]):7.0f}  "
                 f"never {len(g['reached'][0.97]) - sum(g['reached'][0.97])}/{len(g['reached'][0.975]) - sum(g['reached'][0.975])}  "
                 f"h_var {g['h_var']:.2f}")
    for st, c in res["chosen"].items():
        L.append(f"CHOSEN rho ({st}): F' {c['F']:g}  P {c['P']:g}  Q {c['Q']:g}")
    return "\n".join(L)


def report_phase2(res):
    L = ["check B, phase 2: paired change in mean steps-to-target against A [95% bootstrap interval], percent (negative = fewer steps)"]
    for st, r in res.items():
        L.append(f"--- setting {st}" + (f"   (A2 against A: {_fmt(r['a2'][0.97])} / {_fmt(r['a2'][0.975])})" if r["a2"] else ""))
        for k, g in r["groups"].items():
            a2 = "" if not g["vs_a2"] else f"   vs A2 {g['vs_a2'][0.97][0]:+.1f} / {g['vs_a2'][0.975][0]:+.1f}"
            L.append(f"  {k:<3} {g['name']:<20} 0.97 {_fmt(g['vs_a'][0.97]):>24}  0.975 {_fmt(g['vs_a'][0.975]):>24}"
                     f"  h_var x{g['h_var_ratio']:.2f}  align {g['align']:+.2f}{a2}")
        if "a_vs_relu" in r:
            L.append(f"  tanh-tuned baseline against relu baseline: {_fmt(r['a_vs_relu'][0.97])} / {_fmt(r['a_vs_relu'][0.975])}")
        if "D" in r:
            L.append(f"  D (as tuned, descriptive) against A in tanh: {_fmt(r['D']['vs_a_tanh'][0.97])} / {_fmt(r['D']['vs_a_tanh'][0.975])}"
                     + ("" if not r["D"]["vs_a_relu"] else f";  against A in relu: {_fmt(r['D']['vs_a_relu'][0.97])} / {_fmt(r['D']['vs_a_relu'][0.975])}"))
        L.append(f"  VERDICT ({st}): {r['verdict']}")
    return "\n".join(L)


def report_phase3(res):
    L = ["check B, phase 3 (lr x 0.5 / 1 / 2 chosen on seeds 710-714, evaluated on 715-719, 5 seeds)"]
    for st, r in res.items():
        for k in ("P", "Q"):
            L.append(f"  {st} {k}: lr x{r['picked'][k]:g}  0.97 {_fmt(r['vs_a'][k][0.97])}  0.975 {_fmt(r['vs_a'][k][0.975])}")
        L.append(f"  VERDICT ({st}, phase 3): {r['verdict']}")
    return "\n".join(L)


def report_replace(res):
    L = [f"check D: replacement (backprop cut at h); reference = arm A in the same activation; R1 tuned activation: {res['r1_setting']}"]
    for name, g in res["groups"].items():
        L.append(f"  {name:<14} ref {g['ref']:<8} reached {g['reached'][0.97]}/{g['n']} and {g['reached'][0.975]}/{g['n']}  "
                 f"0.97 {_fmt(g['change'][0.97])}  0.975 {_fmt(g['change'][0.975])}  acc gap {g['acc_gap']:+.4f}  "
                 f"h_var {g['h_var']:.1f}  align {g['align']:+.2f}  -> {g['verdict']}")
    L += ["READING: " + x for x in res["diagnosis"]]
    return "\n".join(L)
