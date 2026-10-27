"""Analysis of the Improve-track headroom study (docs/notes/improve-headroom-plan.md). Exploratory; reads logs only.

  python tracks/improve/headroom_analysis.py            # prints the table and the readings, writes results/processed/improve_headroom.csv

Quantities (written in the plan note before any run): FLOPs-to-target and steps-to-target at 0.97 and 0.975 (D-018 rule:
the first of two consecutive evaluations at or above the target); a run that never reaches a target is charged the FLOPs
and steps of its whole run and counted as 'never'. Paired percentage change of the mean (group against reference, same
seeds), 97.5% paired-bootstrap interval (10,000 resamples, seed 0). Selective-Backprop is reported twice: `flops`
(the released algorithm, second forward pass counted) and `flops_cached` (name suffix _cached).
"""
import csv
import glob
import json
import pathlib
import re
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.crossing import crossing_index  # noqa: E402

PLAN = ROOT / "tracks" / "improve" / "headroom_plan.yaml"
OUT = ROOT / "results" / "processed" / "improve_headroom.csv"
SEEDS = list(range(2100, 2110))
TARGETS = (0.97, 0.975)
N_BOOT, BOOT_SEED, LO, HI = 10000, 0, 1.25, 98.75
FRACTIONS = {"S30": 0.3, "S25": 0.25, "S20": 0.2, "S15": 0.15, "S10": 0.1}
WIN, PENALTY, WARM_GAIN = 5.0, 10.0, 3.0          # percent thresholds of the reading rules


def cost_to_target(rows, target, key="flops"):
    """(cost, steps, reached). A run that never reaches the target is charged its whole run."""
    i = crossing_index([r["val_acc"] for r in rows], target)
    if i is None:
        return rows[-1][key], rows[-1]["step"], False
    return rows[i][key], rows[i]["step"], True


def paired_change(a, b, idx):
    """Percentage change of mean(a) against mean(b) and its percentile interval over the resampled seed indices."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pt = (a.mean() / b.mean() - 1) * 100
    bs = (a[idx].mean(1) / b[idx].mean(1) - 1) * 100
    return pt, np.percentile(bs, LO), np.percentile(bs, HI)


def excludes_zero(ch):
    return ch[2] < 0 or ch[1] > 0


def below(ch, thr):
    """True if the change is at most -thr percent and its interval excludes zero (an improvement)."""
    return ch[0] <= -thr and ch[2] < 0


def above(ch, thr):
    return ch[0] >= thr and ch[1] > 0


def load_logs(plan=PLAN, raw=ROOT / "results" / "raw" / "improve_headroom"):
    groups = yaml.safe_load(open(plan))["groups"]
    logs = {}
    for g in groups:
        stage = f"hr-{g['name']}"
        runs = {}
        for f in glob.glob(str(raw / f"*_{stage}_*_s*.jsonl")):
            lines = [json.loads(x) for x in open(f)]
            runs[lines[0]["config"]["seed"]] = [r for r in lines[1:] if "val_acc" in r]
        if sorted(runs) == SEEDS:
            logs[g["name"]] = runs
        else:
            print(f"group {g['name']}: {len(runs)} of {len(SEEDS)} runs finished, skipped")
    return logs


def vectors(logs):
    """name -> {target: dict(flops, steps, never)}; Selective-Backprop groups also appear as <name>_cached."""
    out = {}
    for name, runs in logs.items():
        keys = [("", "flops")] + ([("_cached", "flops_cached")] if "flops_cached" in runs[SEEDS[0]][0] else [])
        for suffix, key in keys:
            out[name + suffix] = {}
            for t in TARGETS:
                c = [cost_to_target(runs[s], t, key) for s in SEEDS]
                out[name + suffix][t] = dict(flops=[x[0] for x in c], steps=[x[1] for x in c], never=sum(not x[2] for x in c))
    return out


def compare(v, group, ref, idx):
    """{(target, metric): (point, lo, hi)} of group against ref."""
    return {(t, m): paired_change(v[group][t][m], v[ref][t][m], idx) for t in TARGETS for m in ("flops", "steps")}


def best_constant(v):
    """The constant kept fraction with the lowest mean FLOPs-to-target averaged over the two targets."""
    score = {g: np.mean([np.mean(v[g][t]["flops"]) for t in TARGETS]) for g in FRACTIONS if g in v}
    return min(score, key=score.get) if score else None


def readings(v, idx):
    """The pre-registered readings (plan note section 'Reading rules'); returns a list of strings."""
    out, leads = [], []
    if "S30" in v:
        best = best_constant(v)
        if best and best != "S30":
            ch = compare(v, best, "S30", idx)
            fl = [ch[(t, "flops")] for t in TARGETS]
            r = "HEADROOM CONFIRMED" if all(below(c, WIN) for c in fl) else ("NO FURTHER GAIN" if not any(excludes_zero(c) for c in fl) else "PARTIAL")
            if r == "HEADROOM CONFIRMED":
                leads.append(best)
            out.append(f"R1 best constant fraction: {best} ({FRACTIONS[best]}); against S30 FLOPs " + ", ".join(f"{c[0]:+.1f}% [{c[1]:+.1f}, {c[2]:+.1f}]" for c in fl) + f" -> {r}")
        elif best == "S30":
            out.append("R1 best constant fraction: S30 (0.3) -> NO FURTHER GAIN")
        pen = []
        for g in ("S25", "S20", "S15", "S10"):
            if g in v:
                ch = compare(v, g, "S30", idx)
                if any(above(ch[(t, "steps")], PENALTY) for t in TARGETS):
                    pen.append(FRACTIONS[g])
        out.append("R2 step penalty (steps > +10% against S30 with the interval above 0 at a target): " + (f"seen at kept fractions {pen}" if pen else "none seen"))
    if "S20" in v:
        for w in ("S20w0", "S20w25", "S20w50"):
            if w in v:
                ch = compare(v, w, "S20", idx)
                fl, st = [ch[(t, "flops")] for t in TARGETS], [ch[(t, "steps")] for t in TARGETS]
                r = "PAYS" if all(below(c, WARM_GAIN) for c in fl) else ("COSTS STEPS" if any(above(c, PENALTY) for c in st) else "NO CLEAR EFFECT")
                if r == "PAYS":
                    leads.append(w)
                out.append(f"R3 warm-up {w}: FLOPs vs S20 " + ", ".join(f"{c[0]:+.1f}%" for c in fl) + ", steps " + ", ".join(f"{c[0]:+.1f}%" for c in st) + f" -> {r}")
    if "S30" in v:
        for sb in ("SB2", "SB2p100", "SB3"):
            for suffix in ("", "_cached"):
                g = sb + suffix
                if g in v:
                    ch = compare(v, g, "S30", idx)
                    fl = [ch[(t, "flops")] for t in TARGETS]
                    r = "SB WINS" if all(below(c, WIN) for c in fl) else ("SB LOSES" if all(above(c, WIN) for c in fl) else "UNCLEAR")
                    if r == "SB WINS":
                        leads.append(g)
                    out.append(f"R4 {g} against S30, FLOPs " + ", ".join(f"{c[0]:+.1f}% [{c[1]:+.1f}, {c[2]:+.1f}]" for c in fl) + f" -> {r}")
    best = best_constant(v)
    for g, ref in (("S20gn", "S20"), ("SOSGD", best), ("Sup", best)):
        if g in v and ref in v:
            ch = compare(v, g, ref, idx)
            fl = [ch[(t, "flops")] for t in TARGETS]
            r = "BETTER" if all(below(c, WIN) for c in fl) else ("WORSE" if all(above(c, WIN) for c in fl) else "UNCLEAR")
            if r == "BETTER":
                leads.append(g)
            out.append(f"R5 {g} against {ref}, FLOPs " + ", ".join(f"{c[0]:+.1f}% [{c[1]:+.1f}, {c[2]:+.1f}]" for c in fl) + f" -> {r}")
    out.append("R6 leads for a pre-registered test on fresh seeds (prereg-v3): " + (", ".join(leads) if leads else "none"))
    return out


def main():
    logs = load_logs()
    if "A1" not in logs:
        sys.exit("A1 (the reference) is not finished; nothing to analyse")
    v = vectors(logs)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(SEEDS), (N_BOOT, len(SEEDS)))
    rows = []
    print(f"{'group':>11} | GFLOPs@0.97 @0.975 | steps@0.975 | vs A1 FLOPs (0.97, 0.975) | vs S30 FLOPs (0.97, 0.975) | never")
    for g in v:
        f97, f975 = (np.mean(v[g][t]["flops"]) / 1e9 for t in TARGETS)
        s975 = np.mean(v[g][0.975]["steps"])
        c1 = compare(v, g, "A1", idx) if g != "A1" else None
        c2 = compare(v, g, "S30", idx) if "S30" in v and g != "S30" else None
        fmt = lambda c: "   (reference)    " if c is None else " ".join(f"{c[(t, 'flops')][0]:+6.1f}%" for t in TARGETS)
        nv = " ".join(str(v[g][t]["never"]) for t in TARGETS)
        print(f"{g:>11} | {f97:11.1f} {f975:6.1f} | {s975:11.0f} | {fmt(c1)}          | {fmt(c2)}          | {nv}")
        for ref, c in (("A1", c1), ("S30", c2)):
            if c is not None:
                for (t, m), (pt, lo, hi) in c.items():
                    rows.append([g, ref, t, m, round(pt, 2), round(lo, 2), round(hi, 2), v[g][t]["never"]])
    print()
    for line in readings(v, idx):
        print(line)
    for g, runs in logs.items():
        if g.startswith("SB"):
            last = [r[-1] for r in runs.values()]
            print(f"{g}: forward batches per update {np.mean([r['n_fwd'] / r['step'] for r in last]):.2f}, kept rate {np.mean([r['sel_rate'] for r in last]):.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "reference", "target", "metric", "paired_change_pct", "lo", "hi", "n_never"])
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
