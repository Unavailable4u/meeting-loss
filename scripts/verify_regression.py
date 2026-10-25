"""Regression check (E-002 style) after changes under src/ (DR-007): rerun one Stage 3 arm-A seed on the
current code and compare the new run log with the original raw log.

Rule (fixed in docs/notes/selective-backward-prereg-prep.md, Step 0, before the first use):
  FAIL                      the evaluation steps differ, or the logged FLOPs differ at any evaluation
                            (FLOP counts are deterministic, so any difference means the accounting changed),
                            or the validation accuracy differs by more than --tol at any evaluation, or the
                            steps-to-target (two consecutive evaluations, D-018) differ at 0.97 or 0.975
  PASS (bit-identical)      the validation accuracy is identical at every evaluation (expected on the same
                            machine and library versions)
  PASS (within tolerance)   anything else that passes (expected across hardware or library versions)

The rerun uses a temporary copy of the config with stage "verify-regression", so its log can never be
mixed up with the Stage 3 final logs (analysis scripts select logs by stage name).

Usage:
  python scripts/verify_regression.py                       # rerun arm A seed 0 and compare
  python scripts/verify_regression.py --new-log PATH        # compare an existing new log, no rerun
Exit status 0 on PASS, 1 on FAIL.
"""
import argparse
import glob
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crossing import DEFAULT_CROSSING, crossing_index  # noqa: E402  (pure Python, no torch)

TARGETS = (0.97, 0.975)
TOL_ACC = 0.002


def load_log(path):
    """Return (header dict, list of evaluation rows). Evaluation rows have 'step' and 'val_acc'."""
    header, rows = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_header"):
                header = rec
            elif "step" in rec and "val_acc" in rec:
                rows.append(rec)
    return header, rows


def steps_to_target(rows, target, rule=DEFAULT_CROSSING):
    idx = crossing_index([r["val_acc"] for r in rows], target, rule)
    return None if idx is None else rows[idx]["step"]


def compare_logs(old_rows, new_rows, targets=TARGETS, tol=TOL_ACC):
    """Compare two evaluation series. Returns a dict with the verdict and the reasons."""
    out = {"n_old": len(old_rows), "n_new": len(new_rows), "reasons": []}
    old_steps = [r["step"] for r in old_rows]
    new_steps = [r["step"] for r in new_rows]
    out["same_steps"] = old_steps == new_steps
    if not out["same_steps"]:
        out["reasons"].append("evaluation steps differ")
        out["verdict"] = "FAIL"
        return out
    out["flops_identical"] = all(a.get("flops") == b.get("flops") for a, b in zip(old_rows, new_rows))
    if not out["flops_identical"]:
        out["reasons"].append("logged FLOPs differ at some evaluation (the FLOP accounting changed)")
    diffs = [abs(a["val_acc"] - b["val_acc"]) for a, b in zip(old_rows, new_rows)]
    out["max_abs_acc_diff"] = max(diffs) if diffs else 0.0
    out["steps_to_target"] = {}
    same_targets = True
    for t in targets:
        o, n = steps_to_target(old_rows, t), steps_to_target(new_rows, t)
        out["steps_to_target"][t] = (o, n)
        same_targets = same_targets and (o == n)
    if out["max_abs_acc_diff"] > tol:
        out["reasons"].append(f"max |val_acc difference| {out['max_abs_acc_diff']:.4f} exceeds {tol}")
    if not same_targets:
        out["reasons"].append("steps-to-target differ")
    if out["reasons"]:
        out["verdict"] = "FAIL"
    elif out["max_abs_acc_diff"] == 0.0:
        out["verdict"] = "PASS (bit-identical)"
    else:
        out["verdict"] = "PASS (within tolerance)"
    return out


def config_differences(old_header, new_header, ignore=("stage",)):
    """Keys present in the original config whose values differ in the new config (new-only keys are ignored)."""
    oc, nc = old_header["config"], new_header["config"]
    return {k: (oc[k], nc.get(k)) for k in oc if k not in ignore and oc[k] != nc.get(k)}


def rerun(config_path, seed):
    import yaml  # imported here so the comparison functions work without it
    cfg = yaml.safe_load(open(config_path))
    cfg["stage"] = "verify-regression"
    cfg["seed"] = seed
    with tempfile.TemporaryDirectory() as tmp:
        tmp_cfg = pathlib.Path(tmp) / "verify_regression.yaml"
        tmp_cfg.write_text(yaml.safe_dump(cfg))
        subprocess.run([sys.executable, "-m", "src.train", "--config", str(tmp_cfg), "--seed", str(seed)],
                       cwd=ROOT, check=True)
    logs = sorted(glob.glob(str(ROOT / "results" / "raw" / f"*verify-regression_{cfg['arm']}_s{seed}.jsonl")),
                  key=lambda p: pathlib.Path(p).stat().st_mtime)
    if not logs:
        raise SystemExit("rerun finished but no verify-regression log was found in results/raw")
    return logs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuned/t1_base_two.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--old-log", default=None, help="default: the Stage 3 final log of this arm and seed")
    ap.add_argument("--new-log", default=None, help="compare this existing log instead of rerunning")
    ap.add_argument("--tol", type=float, default=TOL_ACC)
    args = ap.parse_args()

    old_path = args.old_log
    if old_path is None:
        found = sorted(glob.glob(str(ROOT / "results" / "raw" / f"*final-t1-base-two_base_s{args.seed}.jsonl")))
        if not found:
            raise SystemExit(f"no Stage 3 log for arm A seed {args.seed}; pass --old-log")
        old_path = found[0]
    new_path = args.new_log or rerun(args.config, args.seed)

    old_h, old_rows = load_log(old_path)
    new_h, new_rows = load_log(new_path)
    res = compare_logs(old_rows, new_rows, tol=args.tol)

    print(f"old log : {old_path}")
    print(f"new log : {new_path}")
    if old_h and new_h:
        print(f"commits : old {str(old_h.get('git_commit'))[:8]}  new {str(new_h.get('git_commit'))[:8]}"
              f"  (new git_dirty={new_h.get('git_dirty')})")
        diff = config_differences(old_h, new_h)
        print("config differences (old, new):", diff if diff else "none")
    print(f"evaluations: old {res['n_old']}, new {res['n_new']}; same steps: {res['same_steps']}")
    if res["same_steps"]:
        print(f"FLOPs identical at every evaluation: {res['flops_identical']}")
        print(f"max |val_acc difference|: {res['max_abs_acc_diff']:.6f} (tolerance {args.tol})")
        for t, (o, n) in res["steps_to_target"].items():
            print(f"steps to {t}: old {o}, new {n}")
    for r in res["reasons"]:
        print("reason:", r)
    print("VERDICT:", res["verdict"])
    sys.exit(0 if res["verdict"].startswith("PASS") else 1)


if __name__ == "__main__":
    main()
