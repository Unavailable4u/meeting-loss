"""Extra looks at look 2 of the Improve track (E-025). Exploratory and post hoc: none of these was a pre-fixed reading rule.

  python tracks/improve/look2_extra.py     # prints the extra looks; reads the logs only, writes nothing

Uses the same crossing rule, targets, bootstrap (10,000 resamples, seed 0, 97.5% percentile interval) and logs as
look2_analysis.py. The numbers of the pre-fixed rules R1 to R6 come from look2_analysis.py, not from here.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import headroom_analysis as ha  # noqa: E402
import look2_analysis as la  # noqa: E402

SEEDS, TARGETS = la.SEEDS, la.TARGETS

# (group, reference) pairs whose per-seed direction and, where not already printed by look2_analysis.py, interval are shown
PAIRS = [("S20w0", "S20"), ("S15", "S30"), ("S20w0", "S30"), ("S20w5", "S20"), ("S20w10", "S20"), ("S20w25", "S20"),
         ("S15w0", "S20w0"), ("S10w0", "S20w0"), ("S20k", "S20"), ("S20w0k", "S20w0"), ("S20", "S30")]


def fmt(c):
    return ", ".join(f"{c[0]:+.1f}% [{c[1]:+.1f}, {c[2]:+.1f}]" for c in c)


def main():
    logs = la.load_logs()
    v = la.vectors(logs)
    idx = np.random.default_rng(ha.BOOT_SEED).integers(0, len(SEEDS), (ha.N_BOOT, len(SEEDS)))

    print("1. paired changes with intervals and the number of seeds (of 10) where the group is cheaper / needs fewer steps")
    print(f"{'group':>8} vs {'ref':<6} | FLOPs 0.97, 0.975 | cheaper (0.97, 0.975) | steps 0.97, 0.975 | fewer steps")
    for g, ref in PAIRS:
        cf = [ha.paired_change(v[g][t]["flops"], v[ref][t]["flops"], idx) for t in TARGETS]
        cs = [ha.paired_change(v[g][t]["steps"], v[ref][t]["steps"], idx) for t in TARGETS]
        nf = [int(np.sum(np.asarray(v[g][t]["flops"]) < np.asarray(v[ref][t]["flops"]))) for t in TARGETS]
        ns = [int(np.sum(np.asarray(v[g][t]["steps"]) < np.asarray(v[ref][t]["steps"]))) for t in TARGETS]
        print(f"{g:>8} vs {ref:<6} | {fmt(cf)} | {nf[0]:2d}, {nf[1]:2d} | {fmt(cs)} | {ns[0]:2d}, {ns[1]:2d}")

    print("\n2. mean steps to 0.97 / 0.975 and mean GFLOPs to 0.97 / 0.975 (never-reached runs: 0 in every group)")
    for g in v:
        print(f"{g:>8} | steps {np.mean(v[g][0.97]['steps']):7.1f} {np.mean(v[g][0.975]['steps']):7.1f} | "
              f"GFLOPs {np.mean(v[g][0.97]['flops']) / 1e9:6.2f} {np.mean(v[g][0.975]['flops']) / 1e9:6.2f}")

    print("\n3. split of the S20 minus S20w0 gap into the warm-up's own cost and the change in steps (derived from the logs)")
    full = np.mean([logs["S20"][s][0]["flops"] / logs["S20"][s][0]["step"] for s in SEEDS])           # FLOPs per full step (step 50, in warm-up)
    sel = np.mean([(logs["S20w0"][s][1]["flops"] - logs["S20w0"][s][0]["flops"]) /
                   (logs["S20w0"][s][1]["step"] - logs["S20w0"][s][0]["step"]) for s in SEEDS])         # FLOPs per selective step
    print(f"FLOPs per full step {full / 1e6:.2f} M; per selective step (kept 0.2) {sel / 1e6:.2f} M")
    for t in TARGETS:
        gap = (np.mean(v["S20"][t]["flops"]) - np.mean(v["S20w0"][t]["flops"])) / 1e9
        dsteps = np.mean(v["S20"][t]["steps"]) - np.mean(v["S20w0"][t]["steps"])
        warm_part = 100 * (full - sel) / 1e9
        step_part = dsteps * sel / 1e9
        print(f"target {t}: gap {gap:.2f} GFLOPs = warm-up's extra cost {warm_part:.2f} + fewer steps {dsteps:.0f} x selective step {step_part:.2f}"
              f" (sum {warm_part + step_part:.2f})")
    for g, w in (("S20w5", 5), ("S20w10", 10), ("S20w25", 25)):
        for t in TARGETS:
            gap = (np.mean(v["S20"][t]["flops"]) - np.mean(v[g][t]["flops"])) / 1e9
            dsteps = np.mean(v["S20"][t]["steps"]) - np.mean(v[g][t]["steps"])
            pred_warm = (100 - w) * (full - sel) / 1e9
            print(f"{g} target {t}: gap to S20 {gap:.2f} GFLOPs; warm-up saving alone would be {pred_warm:.2f}; steps differ by {dsteps:+.0f}")

    print("\n4. validation accuracy at update 50 and 100, mean over seeds, and the number of seeds below A1 at update 100")
    for g in logs:
        a50 = np.mean([next(r["val_acc"] for r in logs[g][s] if r["step"] == 50) for s in SEEDS])
        a100 = [next(r["val_acc"] for r in logs[g][s] if r["step"] == 100) for s in SEEDS]
        a100_ref = [next(r["val_acc"] for r in logs["A1"][s] if r["step"] == 100) for s in SEEDS]
        print(f"{g:>8} | update 50 {a50:.4f} | update 100 {np.mean(a100):.4f} | seeds below A1 at 100: {sum(x < y for x, y in zip(a100, a100_ref))}")


if __name__ == "__main__":
    main()
