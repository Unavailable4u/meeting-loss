"""Analysis of look 2 of the Improve track (docs/notes/improve-look2-plan.md). Exploratory; reads logs only.

  python tracks/improve/look2_analysis.py     # prints the table and the readings, writes results/processed/improve_look2.csv

Quantities, crossing rule, interval and thresholds as in tracks/improve/headroom_analysis.py (imported); seeds 2110 to 2119.
"""
import csv
import glob
import json
import pathlib
import sys

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import headroom_analysis as ha  # noqa: E402

ROOT = HERE.parents[1]
PLAN = HERE / "look2_plan.yaml"
OUT = ROOT / "results" / "processed" / "improve_look2.csv"
SEEDS = list(range(2110, 2120))
TARGETS = ha.TARGETS
REPLICATE_W0, REPLICATE_FRAC, COMBO, DOSE_STEP, SCALE = 10.0, 5.0, 3.0, 5.0, 5.0      # percent thresholds of the rules
STEPS_TABLE = (50, 100, 150, 200, 300, 400, 600, 800, 1000)
S_GROUPS = ("S30", "S20", "S15", "S20w0", "S15w0", "S10w0", "S20w5", "S20w10", "S20w25", "S20k", "S20w0k")


def load_logs(plan=PLAN, raw=ROOT / "results" / "raw" / "improve_look2"):
    logs = {}
    for g in yaml.safe_load(open(plan))["groups"]:
        runs = {}
        for f in glob.glob(str(raw / f"*_l2-{g['name']}_*_s*.jsonl")):
            lines = [json.loads(x) for x in open(f)]
            runs[lines[0]["config"]["seed"]] = [r for r in lines[1:] if "val_acc" in r]
        if sorted(runs) == SEEDS:
            logs[g["name"]] = runs
        else:
            print(f"group {g['name']}: {len(runs)} of {len(SEEDS)} runs finished, skipped")
    return logs


def vectors(logs):
    out = {}
    for name, runs in logs.items():
        out[name] = {}
        for t in TARGETS:
            c = [ha.cost_to_target(runs[s], t) for s in SEEDS]
            out[name][t] = dict(flops=[x[0] for x in c], steps=[x[1] for x in c], never=sum(not x[2] for x in c))
    return out


def verdict_scale(c, thr=SCALE):
    fl = [c[(t, "flops")] for t in TARGETS]
    return "BETTER" if all(ha.below(x, thr) for x in fl) else ("WORSE" if all(ha.above(x, thr) for x in fl) else "NO CLEAR EFFECT")


def fmt(c, key="flops"):
    return ", ".join(f"{c[(t, key)][0]:+.1f}% [{c[(t, key)][1]:+.1f}, {c[(t, key)][2]:+.1f}]" for t in TARGETS)


def readings(v, idx):
    """The pre-registered readings (plan note, 'Reading rules')."""
    out, leads = [], []
    cmp_ = lambda g, ref: ha.compare(v, g, ref, idx)
    # R1 replication
    if "S20w0" in v and "S20" in v:
        c = cmp_("S20w0", "S20"); fl = [c[(t, "flops")] for t in TARGETS]
        r = ("REPLICATED" if all(ha.below(x, REPLICATE_W0) for x in fl)
             else ("NOT REPLICATED" if (not any(ha.excludes_zero(x) for x in fl)) or all(x[0] > -5 for x in fl) else "PARTIAL"))
        out.append(f"R1 S20w0 against S20, FLOPs {fmt(c)}; steps {fmt(c, 'steps')} -> {r}")
        if r == "REPLICATED":
            leads.append("S20w0")
    if "S15" in v and "S30" in v:
        c = cmp_("S15", "S30"); fl = [c[(t, "flops")] for t in TARGETS]
        r = ("REPLICATED" if all(ha.below(x, REPLICATE_FRAC) for x in fl)
             else ("NOT REPLICATED" if (not any(ha.excludes_zero(x) for x in fl)) or all(x[0] > -3 for x in fl) else "PARTIAL"))
        out.append(f"R1 S15 against S30, FLOPs {fmt(c)} -> {r}")
        if r == "REPLICATED":
            leads.append("S15")
    # R2 combinations
    for g in ("S15w0", "S10w0"):
        if g in v and "S20w0" in v:
            c = cmp_(g, "S20w0"); fl = [c[(t, "flops")] for t in TARGETS]
            r = "COMBINATION PAYS" if all(ha.below(x, COMBO) for x in fl) else ("WORSE" if all(ha.above(x, COMBO) for x in fl) else "NO GAIN")
            out.append(f"R2 {g} against S20w0, FLOPs {fmt(c)}; steps {fmt(c, 'steps')} -> {r}")
            if r == "COMBINATION PAYS":
                leads.append(g)
    # R3 warm-up dose-response against S20 (warm-up 100)
    doses = ("S20w0", "S20w5", "S20w10", "S20w25")
    if "S20" in v and all(g in v for g in doses):
        ch = {g: cmp_(g, "S20") for g in doses}
        pts = {g: [ch[g][(t, "flops")][0] for t in TARGETS] for g in doses}
        w0_ok = all(ha.below(ch["S20w0"][(t, "flops")], REPLICATE_W0) for t in TARGETS)
        graded = w0_ok and all(pts[a][i] < pts[b][i] for a, b in zip(doses, doses[1:]) for i in range(2)) and all(pts["S20w25"][i] < 0 for i in range(2))
        step = w0_ok and all(abs(pts[g][i]) < DOSE_STEP for g in doses[1:] for i in range(2))
        r = "GRADED" if graded else ("STEP AT ZERO" if step else "MIXED")
        out.append("R3 warm-up dose against S20, FLOPs " + "; ".join(f"{g[3:]} {pts[g][0]:+.1f}% / {pts[g][1]:+.1f}%" for g in doses) + f" -> {r}")
    # R4 gradient scale
    if all(g in v for g in ("S20k", "S20", "S20w0k", "S20w0")):
        a, b = cmp_("S20k", "S20"), cmp_("S20w0k", "S20w0")
        ra, rb = verdict_scale(a), verdict_scale(b)
        out.append(f"R4 S20k against S20, FLOPs {fmt(a)} -> {ra}")
        out.append(f"R4 S20w0k against S20w0, FLOPs {fmt(b)} -> {rb}")
        if ra == "BETTER" and rb == "NO CLEAR EFFECT":
            r = "OPTIMIZER-STATE HYPOTHESIS SUPPORTED"
        elif ra == "NO CLEAR EFFECT" and rb == "NO CLEAR EFFECT":
            r = "SCALE DOES NOT MATTER"
        else:
            r = "SCALE MATTERS (other pattern)"
        out.append(f"R4 reading -> {r}")
        for g, x in (("S20k", ra), ("S20w0k", rb)):
            if x == "BETTER":
                leads.append(g)
    # R5 best arm
    present = [g for g in S_GROUPS if g in v]
    if present:
        score = {g: np.mean([np.mean(v[g][t]["flops"]) for t in TARGETS]) for g in present}
        best = min(score, key=score.get)
        s = f"R5 cheapest S arm by mean FLOPs over the two targets: {best}"
        if "A1" in v:
            c = cmp_(best, "A1"); s += f"; against A1 {fmt(c)}"
        if "S30" in v and best != "S30":
            c = cmp_(best, "S30"); s += f"; against S30 {fmt(c)}"
        out.append(s)
    out.append("R6 leads for a pre-registered test on fresh seeds (prereg-v3): " + (", ".join(leads) if leads else "none"))
    return out


def main():
    logs = load_logs()
    if "A1" not in logs:
        sys.exit("A1 (the reference) is not finished; nothing to analyse")
    v = vectors(logs)
    idx = np.random.default_rng(ha.BOOT_SEED).integers(0, len(SEEDS), (ha.N_BOOT, len(SEEDS)))
    rows = []
    print(f"{'group':>8} | GFLOPs@0.97 @0.975 | steps@0.975 | vs A1 FLOPs (0.97, 0.975) | vs S30 FLOPs (0.97, 0.975) | never")
    for g in v:
        f97, f975 = (np.mean(v[g][t]["flops"]) / 1e9 for t in TARGETS)
        c1 = ha.compare(v, g, "A1", idx) if g != "A1" else None
        c2 = ha.compare(v, g, "S30", idx) if "S30" in v and g != "S30" else None
        f = lambda c: "   (reference)   " if c is None else " ".join(f"{c[(t, 'flops')][0]:+6.1f}%" for t in TARGETS)
        print(f"{g:>8} | {f97:11.1f} {f975:6.1f} | {np.mean(v[g][0.975]['steps']):11.0f} | {f(c1)}          | {f(c2)}          | " + " ".join(str(v[g][t]['never']) for t in TARGETS))
        for ref, c in (("A1", c1), ("S30", c2)):
            if c is not None:
                for (t, m), (pt, lo, hi) in c.items():
                    rows.append([g, ref, t, m, round(pt, 2), round(lo, 2), round(hi, 2), v[g][t]["never"]])
    print("\nmean validation accuracy at equal updates")
    print(f"{'group':>8} " + " ".join(f"{s:>7d}" for s in STEPS_TABLE))
    for g, runs in logs.items():
        vals = [np.mean([next(r["val_acc"] for r in runs[s] if r["step"] == st) for s in SEEDS]) for st in STEPS_TABLE]
        print(f"{g:>8} " + " ".join(f"{x:7.4f}" for x in vals))
    print()
    for line in readings(v, idx):
        print(line)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "reference", "target", "metric", "paired_change_pct", "lo", "hi", "n_never"])
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
