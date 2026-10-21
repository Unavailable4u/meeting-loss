"""Analysis of sweeps and final runs (decisions D-014 and D-017). Reads results/raw/, writes
results/processed/. Nothing here edits a raw log.

1) Pick the winner of each arm in a tuning sweep (selection rule D-017):
     python scripts/analyze.py select --tag t1
     python scripts/analyze.py select --tag t1 --arms base meet --write-configs
   Score = mean over the selection seeds of FLOPs to first reach 0.975 validation accuracy; a run
   that never reaches it is scored with its full-run FLOPs. Lower is better. With --write-configs
   the winner's config is written to configs/tuned/<tag>_<arm>.yaml, ready for the final seeds.

2) Compare final runs (steps and FLOPs to target, bootstrap CIs, paired difference vs a reference).
   Quote the globs so the shell does not expand them:
     python scripts/analyze.py compare --ref A \\
         --group A='results/raw/*_final-t1-base_base_s*.jsonl' \\
         --group D='results/raw/*_final-t1-meet_meet_s*.jsonl'
   Runs that never reach a target are scored with their full-run FLOPs and counted in `n_never`
   (the mean is then a lower bound on the true mean, so a gain is not overstated). The paired
   difference resamples seeds jointly, so reference and group must be run on the same seeds.
"""
import argparse
import csv
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src.sweep import (SELECTION_TARGET, SWEEP_ROOT, bootstrap_ci, collect_sweep, group_scores,
                       paired_pct_change, select_best, tuned_config)


def _write_csv(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def cmd_select(args):
    data = collect_sweep(args.tag, args.arms, args.root, args.target)
    counts = {a: len(d["trials"]) for a, d in data.items()}
    if len(set(counts.values())) > 1 or any(d["incomplete"] for d in data.values()):
        print("WARNING: arms do not all have the same number of complete trials; the equal-budget "
              "rule (D-014) is violated until the missing runs are finished.")
    for arm, d in data.items():
        trials = d["trials"]
        if d["incomplete"]:
            print(f"\n{arm}: incomplete trials (not scored): {d['incomplete']}")
        if not trials:
            print(f"{arm}: no complete trials")
            continue
        seeds = d["meta"]["seeds"]
        rows = []
        for t in sorted(trials, key=lambda t: (t["score"], -t["late_val"], t["trial"])):
            row = dict(trial=t["trial"], score_flops=t["score"], n_reached=f'{t["n_reached"]}/{t["n_seeds"]}',
                       late_val_acc=round(t["late_val"], 5), **t["hp"])
            for s in seeds:
                ps = t["per_seed"][s]
                row[f"s{s}_steps"] = ps["steps"]
                row[f"s{s}_flops"] = ps["flops"]
                row[f"s{s}_final_val"] = ps["final_val"]
            rows.append(row)
        _write_csv(pathlib.Path(args.out_dir) / f"sweep_{args.tag}_{arm}.csv", rows)
        best = select_best(trials)
        n_ok = sum(t["n_reached"] > 0 for t in trials)
        print(f"{arm}: {len(trials)} trials, {n_ok} reached {args.target} in at least one seed")
        print(f"  best trial t{best['trial']:02d}: score {best['score'] / 1e9:.1f} GFLOPs, "
              f"reached in {best['n_reached']}/{best['n_seeds']} seeds, late val {best['late_val']:.4f}")
        print(f"  hp: {best['hp']}")
        if best["n_reached"] < best["n_seeds"]:
            print(f"  NOTE: the winner did not reach {args.target} in every seed (a missed seed is scored as "
                  f"the full budget); check the CSV before trusting this arm's tuning.")
        if args.write_configs:
            note = (f"trial t{best['trial']:02d}, score {best['score']:.0f} FLOPs, "
                    f"reached {best['n_reached']}/{best['n_seeds']} seeds, sweep max_steps "
                    f"{best['cfg']['max_steps']}")
            out = pathlib.Path("configs/tuned") / f"{args.tag}_{arm}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                print(f"  {out} exists; not overwritten")
            else:
                out.write_text(tuned_config(best["cfg"], args.tag, arm, note))
                print(f"  wrote {out}")


def cmd_compare(args):
    groups = {}
    for spec in args.group:
        name, _, pattern = spec.partition("=")
        paths = sorted(glob.glob(pattern))
        if not paths:
            sys.exit(f"group {name!r}: no files match {pattern!r}")
        groups[name] = group_scores(paths, args.targets)
    if args.ref not in groups:
        sys.exit(f"--ref {args.ref!r} is not one of the groups {sorted(groups)}")

    rows = []
    for target in args.targets:
        ref = groups[args.ref]
        for name, g in groups.items():
            seeds = sorted(g)
            flops = np.array([g[s][target]["score"] for s in seeds])
            steps_reached = [g[s][target]["steps"] for s in seeds if g[s][target]["reached"]]
            lo, hi = bootstrap_ci(flops, args.n_boot, seed=args.seed)
            row = dict(target=target, group=name, n=len(seeds), n_never=sum(not g[s][target]["reached"] for s in seeds),
                       mean_gflops=flops.mean() / 1e9, ci_lo=lo / 1e9, ci_hi=hi / 1e9,
                       median_gflops=float(np.median(flops)) / 1e9,
                       mean_steps_reached=float(np.mean(steps_reached)) if steps_reached else None)
            if name != args.ref:
                common = sorted(set(seeds) & set(ref))
                if len(common) >= 2:
                    r = np.array([ref[s][target]["score"] for s in common])
                    x = np.array([g[s][target]["score"] for s in common])
                    p, plo, phi = paired_pct_change(r, x, args.n_boot, seed=args.seed)
                    row.update(n_paired=len(common), pct_change_vs_ref=p, pct_lo=plo, pct_hi=phi)
                else:
                    row.update(n_paired=len(common))
            rows.append(row)

    print(f"{'target':>7} {'group':>10} {'n':>3} {'never':>5} {'mean GFLOPs [95% CI]':>28} "
          f"{'median':>8} {'steps':>7} {'vs ' + args.ref + ' (paired %, 95% CI)':>30}")
    for r in rows:
        ci = f"{r['mean_gflops']:.1f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
        steps = f"{r['mean_steps_reached']:.0f}" if r["mean_steps_reached"] is not None else "-"
        d = (f"{r['pct_change_vs_ref']:+.1f} [{r['pct_lo']:+.1f}, {r['pct_hi']:+.1f}]"
             if r.get("pct_change_vs_ref") is not None else ("(reference)" if r["group"] == args.ref else "n/a"))
        print(f"{r['target']:>7} {r['group']:>10} {r['n']:>3} {r['n_never']:>5} {ci:>28} "
              f"{r['median_gflops']:>8.1f} {steps:>7} {d:>30}")
    if args.out:
        _write_csv(args.out, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="pick the best trial per arm from a sweep (D-017)")
    s.add_argument("--tag", required=True)
    s.add_argument("--arms", nargs="+", default=None, help="default: every arm found in the sweep")
    s.add_argument("--target", type=float, default=SELECTION_TARGET)
    s.add_argument("--root", default=str(SWEEP_ROOT))
    s.add_argument("--out-dir", default="results/processed")
    s.add_argument("--write-configs", action="store_true", help="write configs/tuned/<tag>_<arm>.yaml")
    s.set_defaults(func=cmd_select)

    c = sub.add_parser("compare", help="steps/FLOPs-to-target with bootstrap CIs across groups of final runs")
    c.add_argument("--group", action="append", required=True, metavar="NAME=GLOB")
    c.add_argument("--ref", required=True, help="group that the others are compared against")
    c.add_argument("--targets", nargs="+", type=float, default=[0.97, 0.975, 0.98])
    c.add_argument("--n-boot", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    c.add_argument("--out", default=None, help="also write the table as CSV")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
