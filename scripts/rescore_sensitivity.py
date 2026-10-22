"""How much do the tuning results depend on the crossing rule? (decision D-018; exploratory analysis)

Re-scores every run of a sweep under each crossing rule from the saved curves (nothing is rerun) and
reports, per rule and arm: the winning trial, its in-sample selection score, and its ratio to arm A.
It also reports a selection-bias check: pick each arm's best trial on one selection seed, score that
trial on the other seed, and average the two directions (the in-sample score is optimistic because
the winner was chosen on the same seeds).

Usage (from the repo root):
  python scripts/rescore_sensitivity.py --tag t1
  python scripts/rescore_sensitivity.py --tag t1 --arms base deepsup meet --ref base

Writes results/processed/sensitivity_<tag>.csv. Reading guide: the rule that was adopted in advance
(D-018) is the one used for selection; the others only show whether a conclusion would change.
"""
import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src.crossing import CROSSINGS, DEFAULT_CROSSING
from src.sweep import SELECTION_TARGET, SWEEP_ROOT, collect_sweep, cross_seed_check, select_best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", nargs="+", default=None, help="default: every arm in the sweep")
    ap.add_argument("--ref", default="base", help="reference arm for the ratios (default: base = arm A)")
    ap.add_argument("--target", type=float, default=SELECTION_TARGET)
    ap.add_argument("--root", default=str(SWEEP_ROOT))
    ap.add_argument("--out-dir", default="results/processed")
    args = ap.parse_args()

    rows = []
    for rule in CROSSINGS:
        data = collect_sweep(args.tag, args.arms, args.root, args.target, rule)
        if args.ref not in data:
            sys.exit(f"reference arm {args.ref!r} is not in the sweep (arms: {sorted(data)})")
        best = {a: select_best(d["trials"]) for a, d in data.items() if d["trials"]}
        held = {a: cross_seed_check(d["trials"], d["meta"]["seeds"])[0] for a, d in data.items() if d["trials"]}
        ref_in, ref_out = best[args.ref]["score"], held[args.ref]
        for a in best:
            n_cross = sum(t["n_reached"] > 0 for t in data[a]["trials"])
            rows.append(dict(
                rule=rule, arm=a, winner_trial=best[a]["trial"], in_sample_gflops=best[a]["score"] / 1e9,
                ratio_to_ref=best[a]["score"] / ref_in, held_out_gflops=held[a] / 1e9,
                held_out_ratio_to_ref=held[a] / ref_out, n_seeds_reached=f'{best[a]["n_reached"]}/{best[a]["n_seeds"]}',
                trials_reaching_in_any_seed=n_cross, n_trials=len(data[a]["trials"])))

    for rule in CROSSINGS:
        mark = "  (adopted, D-018)" if rule == DEFAULT_CROSSING else "  (sensitivity only)"
        print(f"\ncrossing rule: {rule}{mark}")
        print(f"  {'arm':>11} {'winner':>7} {'in-sample GFLOPs':>17} {'x ' + args.ref:>8} "
              f"{'held-out GFLOPs':>16} {'x ' + args.ref:>8} {'reached':>8}")
        for r in (r for r in rows if r["rule"] == rule):
            print(f"  {r['arm']:>11} {'t%02d' % r['winner_trial']:>7} {r['in_sample_gflops']:>17.1f} "
                  f"{r['ratio_to_ref']:>8.2f} {r['held_out_gflops']:>16.1f} {r['held_out_ratio_to_ref']:>8.2f} "
                  f"{r['n_seeds_reached']:>8}")

    out = pathlib.Path(args.out_dir) / f"sensitivity_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
