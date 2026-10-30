"""Pre-registered analysis of prereg-v3 (D-026): H8, H9, H10, the re-tuning shift, the noise floor, the secondary readings and the
descriptive groups.

  python scripts/analyze_prereg3.py                          # reads configs/prereg_v3_plan.yaml and its out_dir
  python scripts/analyze_prereg3.py --shift-rule max_abs     # only to see what another rule would give (descriptive)

Refuses to run if any group lacks any of its seeds or any run did not finish its horizon. The rules are those of src/prereg3.py
(intervals at alpha = 0.05 / 3, i.e. 98.33%); the shift rule is the plan's unless --shift-rule is given (then the output says so and
the verdict is not the pre-registered one). The report also lists the git commits and `git_dirty` flags found in the run headers: more
than one commit, or a dirty tree, is printed as a warning and must be explained in the experiment entry. Writes
results/processed/prereg_v3.csv.
"""
import argparse
import csv
import glob
import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src import prereg3
from src.final import load_plan, plan_meta
from src.sweep import REPO, load_run, score_run


def load_scores(plan_path, repo=REPO):
    groups = load_plan(plan_path, repo=repo)
    meta = plan_meta(plan_path)
    runs, roles, problems, prov = {}, {}, [], {}
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
            by_seed[seed] = rows
            prov[(hdr.get("git_commit"), hdr.get("git_dirty"))] = prov.get((hdr.get("git_commit"), hdr.get("git_dirty")), 0) + 1
        miss = sorted(set(g["seeds"]) - set(by_seed))
        if miss:
            problems.append(f"missing: group {g['name']} lacks seeds {miss}")
        runs[g["name"]] = {s: r for s, r in by_seed.items() if s in g["seeds"]}
        roles[g["name"]] = g["role"]
    if problems:
        kind = "missing runs" if any(p.startswith("missing") for p in problems) else "unfinished runs"
        raise ValueError(f"{kind}; the pre-registered analysis needs every group complete:\n  " + "\n  ".join(problems[:12]))
    # per-step cost of a full step: arm A's logs (every step is a full step); identical in every seed
    ref = prereg3.REFERENCES["H8"]
    fulls = {prereg3.full_step_flops(r) for r in runs[ref].values()}
    if max(fulls) - min(fulls) > 1e-6 * max(fulls):
        raise ValueError(f"the FLOPs of a full step differ between seeds of {ref}: {sorted(fulls)}")
    full = fulls.pop()
    scores = {}
    for g in groups:
        warm = g["cfg"].get("sel_warmup", 0) if g["cfg"]["arm"] == "selective" else 0
        scores[g["name"]] = {}
        for s, rows in runs[g["name"]].items():
            per_target = {}
            for t in prereg3.TARGETS:
                d = score_run(rows, t)
                cached = prereg3.score_run_cached(rows, t)
                if cached is not None:
                    d["score_cached"] = cached
                if warm > 0:
                    d["score_credited"] = d["score"] - prereg3.warmup_credit(rows, warm, t, full)
                per_target[t] = d
            scores[g["name"]][s] = per_target
    return scores, roles, meta, prov


def analyze(plan_path, repo=REPO, n_boot=10000, shift_rule=None):
    scores, roles, meta, prov = load_scores(plan_path, repo)
    rule = shift_rule or meta["shift_rule"]
    res = prereg3.h8_h10(scores, roles, rule, n_boot=n_boot)
    res["pre_registered_rule"] = shift_rule is None
    res["provenance"] = {f"{c}|dirty={d}": n for (c, d), n in sorted(prov.items(), key=lambda kv: str(kv[0]))}
    res["provenance_ok"] = len(prov) == 1 and all(d is False for (_, d) in prov)
    return res


def _fmt(c):
    return f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]"


def report(res):
    L = [f"prereg-v3 (D-026): {len(res['seeds'])} paired seeds {res['seeds'][0]} to {res['seeds'][-1]}; "
         f"{100 * (1 - res['alpha']):.2f}% paired-bootstrap intervals (alpha {res['alpha']:.4f} = 0.05 / 3); shift rule {res['shift_rule']}"
         + ("" if res["pre_registered_rule"] else "  (NOT the pre-registered rule: descriptive)")]
    if not res["provenance_ok"]:
        L.append("WARNING: run headers do not show one clean commit: " + "; ".join(f"{k} x{n}" for k, n in res["provenance"].items()))
    else:
        L.append("run headers: " + "; ".join(f"{k} x{n}" for k, n in res["provenance"].items()))
    what = {"H8": "S20w0 versus A1, margin 15%, shift rule", "H9": "S20w0 versus S30, margin 10%",
            "H10": "S20w0 versus the tuned Selective-Backprop SBT (uncached count), margin 15%"}
    for h in ("H8", "H9", "H10"):
        L.append(f"\n{h}: {what[h]}")
        for t, r in res[h]["per_target"].items():
            sh = "" if r["shift"] is None else f", shift {r['shift']:.1f}%"
            L.append(f"  target {t}: {_fmt((r['point'], r['lo'], r['hi']))}{sh}, cheaper in {r['n_better']} of {r['n_seeds']} seeds -> {r['verdict']}")
        L.append(f"  VERDICT {h}: {res[h]['overall']}")
    L.append("\nre-tuning shift of A (paired change of each further tuning against A1, %): " + "; ".join(
        f"{t}: " + ", ".join(f"{g} {c:+.1f}" for g, c in res["shift_changes"][t].items()) for t in prereg3.TARGETS))
    nf = res["noise_floor"]
    L.append("noise floor (max / min of the mean FLOPs-to-target over A1 and the further tunings): " +
             ", ".join(f"{t}: {r:.3f}" for t, r in nf["ratio"].items()) + f" -> {nf['outcome']}")
    L.append("\nsecondary (reported, decides nothing; paired change of the first group against the reference named)")
    for key, s in res["secondary"].items():
        L.append(f"  {s['what']}: " + "; ".join(f"{t}: {_fmt(c)}" for t, c in s["change"].items()))
    L.append("\ndescriptive (mean GFLOPs; never = runs that did not reach; change against A1, S30 and the best tuning of A on these seeds)")
    for n, d in res["descriptive"].items():
        for t, r in d.items():
            a1 = _fmt(r["vs_A1"]) if "vs_A1" in r else "(reference)"
            s30 = _fmt(r["vs_S30"]) if "vs_S30" in r else "(S30)"
            bt = f"{_fmt(r['vs_best_tuning'])} vs {r['best_tuning']}" if "vs_best_tuning" in r else "(best tuning)"
            L.append(f"  {n:>6} {res['groups'][n]:>11} {t}: {r['mean_gflops']:7.1f} GFLOPs, steps {r['mean_steps']:6.0f}, never {r['n_never']:>2}, "
                     f"vs A1 {a1:>24}, vs S30 {s30:>24}, {bt}")
    return "\n".join(L)


def write_csv(res, path):
    rows = []
    for n, d in res["descriptive"].items():
        for t, r in d.items():
            row = dict(group=n, role=res["groups"][n], target=t, mean_gflops=r["mean_gflops"], n_never=r["n_never"], mean_steps_reached=r["mean_steps"])
            for key, tag in (("vs_A1", "vs_A1"), ("vs_S30", "vs_S30")):
                if key in r:
                    row.update({f"{tag}_pct": r[key][0], f"{tag}_lo": r[key][1], f"{tag}_hi": r[key][2]})
            if "vs_best_tuning" in r:
                row.update(best_tuning=r["best_tuning"], vs_best_pct=r["vs_best_tuning"][0], vs_best_lo=r["vs_best_tuning"][1], vs_best_hi=r["vs_best_tuning"][2])
            rows.append(row)
    for h in ("H8", "H9", "H10"):
        for t, r in res[h]["per_target"].items():
            rows.append(dict(group=h, role="hypothesis", target=t, pct=r["point"], lo=r["lo"], hi=r["hi"], verdict=r["verdict"], shift=r["shift"],
                             margin=res[h]["margin"], n_better=r["n_better"]))
    for key, s in res["secondary"].items():
        for t, c in s["change"].items():
            rows.append(dict(group=key, role="secondary", target=t, pct=c[0], lo=c[1], hi=c[2]))
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
    ap.add_argument("--plan", default="configs/prereg_v3_plan.yaml")
    ap.add_argument("--shift-rule", choices=("max_abs", "max_gain", "median_abs"), default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="results/processed/prereg_v3.csv")
    args = ap.parse_args()
    res = analyze(args.plan, n_boot=args.n_boot, shift_rule=args.shift_rule)
    print(report(res))
    if args.shift_rule is None:
        write_csv(res, args.out)


if __name__ == "__main__":
    main()
