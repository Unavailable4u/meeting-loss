"""Independent recomputation and extra looks at the prereg-v3 results (E-027). Exploratory; none of the extra looks decides a hypothesis.

  python tracks/improve/prereg3_extra.py     # prints; reads the logs and results/processed/prereg_v3.csv; writes nothing

Part 1 recomputes H8, H9 and H10 from the raw logs with its own loader, crossing rule and bootstrap (written separately from
src/prereg3.py: groups are found by the stage name in each header, not through the plan) and compares them with the committed table.
The remaining parts are post hoc looks: other interval conventions, per-seed ratios, the tuned Selective-Backprop against A1, the
non-arithmetic part of the no-warm-up saving, and the three groups of paired seeds of the Improve track pooled (looks 1 and 2 and prereg-v3).
"""
import csv
import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tracks" / "improve"))
RAW = ROOT / "results" / "raw" / "prereg_v3"
TARGETS = (0.97, 0.975)
STAGES = {"A1": "final-t1-base-two", "A2": "final-t2-base-two", "A5": "final-t5-base-two", "A6": "final-t6-base-two",
          "A7": "final-t7-base-two", "A8": "final-t8-base-two", "SBT": "final-t11-sb-two", "S20w0": "prereg3-S20w0", "S30": "prereg3-S30",
          "S20": "prereg3-S20", "S15": "prereg3-S15", "S20w10": "prereg3-S20w10", "R20w0": "prereg3-R20w0"}
ALPHA3 = 0.05 / 3


def reach(accs, t):
    """First evaluation i with accuracy >= t at i and at i + 1 (decision D-018); None if there is none."""
    for i in range(len(accs) - 1):
        if accs[i] >= t and accs[i + 1] >= t:
            return i
    return None


def cost(rows, t, key="flops"):
    i = reach([r["val_acc"] for r in rows], t)
    j = len(rows) - 1 if i is None else i
    return rows[j][key], rows[j]["step"], i is not None


def load():
    runs = {g: {} for g in STAGES}
    names = {v: k for k, v in STAGES.items()}
    for f in glob.glob(str(RAW / "*.jsonl")):
        lines = [json.loads(x) for x in open(f)]
        cfg = lines[0]["config"]
        runs[names[cfg["stage"]]][cfg["seed"]] = [r for r in lines[1:] if "val_acc" in r]
    seeds = sorted(runs["A1"])
    assert all(sorted(r) == seeds for r in runs.values()) and seeds == list(range(2300, 2320))
    return runs, seeds


def vec(runs, seeds, g, t, what=0, key="flops"):
    return np.array([cost(runs[g][s], t, key)[what] for s in seeds], dtype=float)


def pct(ref, grp, alpha, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref), size=(n_boot, len(ref)))
    stat = 100 * (grp[idx].mean(1) / ref[idx].mean(1) - 1)
    return 100 * (grp.mean() / ref.mean() - 1), np.quantile(stat, alpha / 2), np.quantile(stat, 1 - alpha / 2)


def verdict(p, lo, hi, margin):
    return "supported" if (p <= -margin and hi < 0) else ("refuted" if lo > -margin else "inconclusive")


def fmt(c):
    return f"{c[0]:+.1f}% [{c[1]:+.1f}, {c[2]:+.1f}]"


def committed_table():
    rows = list(csv.DictReader(open(ROOT / "results" / "processed" / "prereg_v3.csv")))
    return {(r["group"], float(r["target"])): r for r in rows if r["role"] == "hypothesis"}


def main():
    runs, seeds = load()
    tab = committed_table()
    margins = {"H8": ("A1", 15), "H9": ("S30", 10), "H10": ("SBT", 15)}
    print("1. independent recomputation (own loader, crossing rule and bootstrap) against results/processed/prereg_v3.csv")
    worst = 0.0
    for h, (ref, m) in margins.items():
        for t in TARGETS:
            c = pct(vec(runs, seeds, ref, t), vec(runs, seeds, "S20w0", t), ALPHA3)
            r = tab[(h, t)]
            d = max(abs(c[0] - float(r["pct"])), abs(c[1] - float(r["lo"])), abs(c[2] - float(r["hi"])))
            worst = max(worst, d)
            print(f"  {h} target {t}: {fmt(c)} -> {verdict(*c, m)}; committed {float(r['pct']):+.1f} [{float(r['lo']):+.1f}, {float(r['hi']):+.1f}] {r['verdict']}; max abs difference {d:.2e}")
    shifts = {t: max(0.0, max(-pct(vec(runs, seeds, "A1", t), vec(runs, seeds, g, t), 0.05)[0] for g in ("A2", "A5", "A6", "A7", "A8"))) for t in TARGETS}
    print(f"  max_gain shift: {shifts[0.97]:.1f}% / {shifts[0.975]:.1f}% (committed {tab[('H8', 0.97)]['shift']} / {tab[('H8', 0.975)]['shift']}); worst difference of any number above {worst:.2e}")

    print("\n2. other interval conventions (verdict of each hypothesis; the pre-registered one is alpha 0.05 / 3)")
    for name, alpha in (("98.33% (pre-registered)", ALPHA3), ("97.5% (D-019, two comparisons)", 0.025), ("95%", 0.05)):
        out = []
        for h, (ref, m) in margins.items():
            vs = [verdict(*pct(vec(runs, seeds, ref, t), vec(runs, seeds, "S20w0", t), alpha), m) for t in TARGETS]
            out.append(f"{h} {vs[0]}/{vs[1]}")
        print(f"  {name}: " + "; ".join(out))

    print("\n3. per-seed ratio S20w0 / reference in FLOPs (min, median, max over the 20 seeds) and seeds in which S20w0 is cheaper")
    for ref in ("A1", "S30", "SBT", "A5"):
        for t in TARGETS:
            r = vec(runs, seeds, "S20w0", t) / vec(runs, seeds, ref, t)
            print(f"  against {ref:>3} at {t}: {r.min():.2f}, {np.median(r):.2f}, {r.max():.2f}; cheaper in {int((r < 1).sum())} of 20")

    print("\n4. the tuned Selective-Backprop (SBT) against A1 (paired change of SBT, 98.33% intervals)")
    for t in TARGETS:
        up, ca = vec(runs, seeds, "SBT", t), vec(runs, seeds, "SBT", t, 0, "flops_cached")
        a1 = vec(runs, seeds, "A1", t)
        st = pct(vec(runs, seeds, "A1", t, 1), vec(runs, seeds, "SBT", t, 1), ALPHA3)
        print(f"  target {t}: FLOPs uncached {fmt(pct(a1, up, ALPHA3))}, cached {fmt(pct(a1, ca, ALPHA3))}, updates {fmt(st)}; "
              f"mean GFLOPs uncached {up.mean() / 1e9:.1f}, cached {ca.mean() / 1e9:.1f}; in-sample at selection (E-026) 209.9 and 182.7 at 0.975")

    print("\n5. descriptive contrasts on these seeds (paired change of the first group against the second, 98.33% intervals)")
    for g, ref in (("S15", "S20"), ("S20w10", "S20"), ("S20", "S30"), ("S15", "S30"), ("S20w0", "S15")):
        for what, label in ((0, "FLOPs"), (1, "steps")):
            print(f"  {g} against {ref}, {label}: " + "; ".join(f"{t}: {fmt(pct(vec(runs, seeds, ref, t, what), vec(runs, seeds, g, t, what), ALPHA3))}" for t in TARGETS))
    for t in TARGETS:
        n = sum(not cost(runs["R20w0"][s], t)[2] for s in seeds)
        print(f"  R20w0 did not reach {t} in {n} of 20 seeds (charged its whole run, so its change against A1 and S20w0 is a lower bound)")

    print("\n6. the three groups of paired seeds pooled: look 1 (2100-2109, E-024), look 2 (2110-2119, E-025), prereg-v3 (2300-2319, E-027); 98.33% intervals")
    import headroom_analysis as ha
    import look2_analysis as la
    l1, l2 = ha.load_logs(), la.load_logs()
    def pooled(g, what, t):
        return np.concatenate([np.array([ha.cost_to_target(l1[g][s], t)[what] for s in ha.SEEDS], dtype=float),
                               np.array([ha.cost_to_target(l2[g][s], t)[what] for s in la.SEEDS], dtype=float), vec(runs, seeds, g, t, what)])
    for g, ref in (("S20w0", "A1"), ("S20w0", "S30"), ("S20w0", "S20"), ("S20", "S30")):
        for what, label in ((0, "FLOPs"), (1, "steps")):
            print(f"  {g} against {ref}, {label}, 40 seeds: " + "; ".join(f"{t}: {fmt(pct(pooled(ref, what, t), pooled(g, what, t), ALPHA3))}" for t in TARGETS))


if __name__ == "__main__":
    main()
