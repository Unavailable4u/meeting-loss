"""Pre-registered analysis of prereg-v2 (D-023): H7a, H7b, the noise floor and the descriptive groups.

  python scripts/analyze_prereg2.py                          # reads configs/prereg_v2_plan.yaml and its out_dir
  python scripts/analyze_prereg2.py --shift-rule max_gain    # only to see what another rule would give (descriptive)

Refuses to run if any group lacks any of its seeds or any run did not finish its horizon. The rules are those of
src/prereg2.py; the shift rule is the plan's unless --shift-rule is given (then the output says so and the verdict is
not the pre-registered one). Writes results/processed/prereg_v2.csv.
"""
import argparse
import csv
import glob
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src import prereg2
from src.final import load_plan, plan_meta
from src.sweep import REPO, load_run, score_run


def load_scores(plan_path, repo=REPO):
    groups = load_plan(plan_path, repo=repo)
    meta = plan_meta(plan_path)
    scores, roles, problems = {}, {}, []
    for g in groups:
        stage, arm = g["cfg"]["stage"], g["cfg"]["arm"]
        files = glob.glob(str(pathlib.Path(meta["out_dir"]) / f"*_{stage}_{arm}_s*.jsonl"))
        by_seed = {}
        for p in files:
            hdr, rows = load_run(p)
            seed = hdr["config"]["seed"]
            if seed in by_seed:
                raise ValueError(f"group {g['name']}: seed {seed} appears twice ({p})")
            if rows[-1]["step"] != hdr["config"]["max_steps"]:
                problems.append(f"unfinished: {g['name']} seed {seed} stopped at step {rows[-1]['step']} of {hdr['config']['max_steps']}")
            by_seed[seed] = {t: score_run(rows, t) for t in prereg2.TARGETS}
        miss = sorted(set(g["seeds"]) - set(by_seed))
        if miss:
            problems.append(f"missing: group {g['name']} lacks seeds {miss}")
        scores[g["name"]] = {s: v for s, v in by_seed.items() if s in g["seeds"]}
        roles[g["name"]] = g["role"]
    if problems:
        kind = "missing runs" if any(p.startswith("missing") for p in problems) else "unfinished runs"
        raise ValueError(f"{kind}; the pre-registered analysis needs every group complete:\n  " + "\n  ".join(problems[:12]))
    return scores, roles, meta


def analyze(plan_path, repo=REPO, n_boot=10000, shift_rule=None):
    scores, roles, meta = load_scores(plan_path, repo)
    rule = shift_rule or meta["shift_rule"]
    res = prereg2.h7(scores, roles, rule, n_boot=n_boot)
    res["pre_registered_rule"] = shift_rule is None
    return res


def _fmt(c):
    return f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]"


def report(res):
    L = [f"prereg-v2 (D-023): {len(res['seeds'])} paired seeds {res['seeds'][0]} to {res['seeds'][-1]}; 97.5% paired-bootstrap intervals; "
         f"shift rule {res['shift_rule']}" + ("" if res["pre_registered_rule"] else "  (NOT the pre-registered rule: descriptive)")]
    for h, margin, what in (("H7a", prereg2.MARGIN_A, "S30 versus A1, margin 15%, shift rule"), ("H7b", prereg2.MARGIN_B, "S30 versus the tuned batch-48 arm, margin 0%")):
        if h not in res:
            continue
        L.append(f"\n{h}: {what}")
        for t, r in res[h]["per_target"].items():
            sh = "" if r["shift"] is None else f", shift {r['shift']:.1f}%"
            L.append(f"  target {t}: {_fmt((r['point'], r['lo'], r['hi']))}{sh} -> {r['verdict']}")
        L.append(f"  VERDICT {h}: {res[h]['overall']}")
    L.append("\nre-tuning shift of A (paired change of each further tuning against A1, %): " + "; ".join(
        f"{t}: " + ", ".join(f"{g} {c:+.1f}" for g, c in res["shift_changes"][t].items()) for t in prereg2.TARGETS))
    nf = res["noise_floor"]
    L.append(f"noise floor (max / min of the mean FLOPs-to-target over A1 and the further tunings): " +
             ", ".join(f"{t}: {r:.3f}" for t, r in nf["ratio"].items()) + f" -> {nf['outcome']}")
    L.append("\ndescriptive (mean GFLOPs; never = runs that did not reach; change against A1 and against the best tuning of A on these seeds)")
    for n, d in res["descriptive"].items():
        for t, r in d.items():
            a1 = _fmt(r["vs_A1"]) if "vs_A1" in r else "(reference)"
            bt = f"{_fmt(r['vs_best_tuning'])} vs {r['best_tuning']}" if "vs_best_tuning" in r else "(best tuning)"
            L.append(f"  {n:>5} {res['groups'][n]:>11} {t}: {r['mean_gflops']:7.1f} GFLOPs, never {r['n_never']:>2}, vs A1 {a1:>26}, {bt}")
    return "\n".join(L)


def write_csv(res, path):
    rows = []
    for n, d in res["descriptive"].items():
        for t, r in d.items():
            row = dict(group=n, role=res["groups"][n], target=t, mean_gflops=r["mean_gflops"], n_never=r["n_never"], mean_steps_reached=r["mean_steps"])
            if "vs_A1" in r:
                row.update(vs_A1_pct=r["vs_A1"][0], vs_A1_lo=r["vs_A1"][1], vs_A1_hi=r["vs_A1"][2])
            if "vs_best_tuning" in r:
                row.update(best_tuning=r["best_tuning"], vs_best_pct=r["vs_best_tuning"][0], vs_best_lo=r["vs_best_tuning"][1], vs_best_hi=r["vs_best_tuning"][2])
            rows.append(row)
    for h in ("H7a", "H7b"):
        for t, r in res.get(h, {}).get("per_target", {}).items():
            rows.append(dict(group=h, role="hypothesis", target=t, vs_A1_pct=r["point"], vs_A1_lo=r["lo"], vs_A1_hi=r["hi"], verdict=r["verdict"], shift=r["shift"]))
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="configs/prereg_v2_plan.yaml")
    ap.add_argument("--shift-rule", choices=("max_abs", "max_gain", "median_abs"), default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="results/processed/prereg_v2.csv")
    args = ap.parse_args()
    res = analyze(args.plan, n_boot=args.n_boot, shift_rule=args.shift_rule)
    print(report(res))
    if args.shift_rule is None:
        write_csv(res, args.out)


if __name__ == "__main__":
    main()
