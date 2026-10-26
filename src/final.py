"""Stage 3 final runs (decision D-019): the plan, the lock check, and the job runner.

The plan lives in configs/stage3_plan.yaml and is locked with the git tag prereg-v1. scripts/run_final.py
refuses to run before the tag exists, and refuses to run if the locked files (plan, tuned configs,
src/, HYPOTHESES.md, METHOD.md) differ from the tag, so that the confirmatory runs use exactly the
code and configurations that were registered.
"""
import datetime
import glob
import json
import os
import pathlib
import subprocess
from dataclasses import dataclass

import yaml

from src.sweep import REPO, _git_info, config_hash

LOCK_TAG = "prereg-v1"
LOCKED_PATHS = ["configs/stage3_plan.yaml", "configs/tuned", "src", "docs/HYPOTHESES.md", "docs/METHOD.md"]
RESERVED_SEED_LIMIT = 100           # seeds below this are for final runs; selection seeds are 1000 and up
ROLES = ("primary", "secondary", "gated", "descriptive", "reference", "control", "shift")  # the last three: prereg-v2 (D-023)


def parse_seeds(spec):
    """'0-19' -> [0, ..., 19]; '3' -> [3]; '0-4,7' -> [0, 1, 2, 3, 4, 7]."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate seeds in {spec!r}")
    return out


@dataclass(frozen=True)
class FinalJob:
    group: str
    seed: int
    cfg_path: str
    out_dir: str


def plan_meta(path=REPO / "configs" / "stage3_plan.yaml"):
    """Plan-level settings with the Stage 3 defaults: the lock tag, the output directory, the lock paths, the seed
    block ([first, last] or None for the reserved range 0 to 99) and the shift rule of prereg-v2."""
    doc = yaml.safe_load(open(path))
    block = doc.get("seed_block")
    return dict(lock_tag=doc.get("lock_tag", LOCK_TAG), out_dir=doc.get("out_dir", "results/raw"),
                locked_paths=doc.get("locked_paths", LOCKED_PATHS),
                seed_block=None if block is None else [parse_seeds(block)[0], parse_seeds(block)[-1]],
                shift_rule=doc.get("shift_rule"), horizon_steps=doc["horizon_steps"])


def load_plan(path=REPO / "configs" / "stage3_plan.yaml", repo=REPO, allow_missing=False):
    """Validated plan: a list of dicts with name, config (path), cfg (loaded), seeds, role, horizon, note.
    Optional plan keys: seed_block ("1300-1319": the seeds of every group must lie in it; without it they must be below
    RESERVED_SEED_LIMIT) and, per group, horizon_steps (default: the plan's). With allow_missing a config file that
    does not exist yet gives cfg = None (the plan can be written before the tunings it names are finished)."""
    doc = yaml.safe_load(open(path))
    meta = plan_meta(path)
    groups = []
    names, ids = set(), set()
    for g in doc["groups"]:
        if g["name"] in names:
            raise ValueError(f"duplicate group name {g['name']!r}")
        names.add(g["name"])
        if g["role"] not in ROLES:
            raise ValueError(f"group {g['name']}: role must be one of {ROLES}")
        cfg_path = pathlib.Path(repo) / g["config"]
        if not cfg_path.exists() and allow_missing:
            cfg = None
        else:
            cfg = yaml.safe_load(open(cfg_path))
        seeds = parse_seeds(g["seeds"])
        if meta["seed_block"] is not None:
            lo, hi = meta["seed_block"]
            if min(seeds) < lo or max(seeds) > hi:
                raise ValueError(f"group {g['name']}: seeds must be in the plan's block [{lo}, {hi}]")
        elif min(seeds) < 0 or max(seeds) >= RESERVED_SEED_LIMIT:
            raise ValueError(f"group {g['name']}: final seeds must be in [0, {RESERVED_SEED_LIMIT})")
        horizon = g.get("horizon_steps", doc["horizon_steps"])
        if cfg is not None:
            key = (cfg["stage"], cfg["arm"])
            if key in ids:
                raise ValueError(f"groups share the same (stage, arm) {key}; their run files would collide")
            ids.add(key)
            if cfg["max_steps"] != horizon:
                raise ValueError(f"group {g['name']}: max_steps {cfg['max_steps']} != plan horizon {horizon}")
        groups.append(dict(name=g["name"], config=g["config"], cfg=cfg, seeds=seeds, role=g["role"], horizon=horizon,
                           note=g.get("note", "")))
    return groups


def plan_jobs(groups, out_dir="results/raw", repo=REPO):
    """Seed-major order across groups, so that a stopped run leaves every group with the same seeds done."""
    jobs = []
    for seed in sorted({s for g in groups for s in g["seeds"]}):
        for g in groups:
            if seed in g["seeds"]:
                jobs.append(FinalJob(g["name"], seed, str(pathlib.Path(repo) / g["config"]), str(out_dir)))
    return jobs


def existing_run(out_dir, stage, arm, seed):
    """Path of an existing run with this (stage, arm, seed), whatever date is in its name, else None."""
    hits = glob.glob(str(pathlib.Path(out_dir) / f"*_{stage}_{arm}_s{seed}.jsonl"))
    return hits[0] if hits else None


def run_final_job(job, device="auto", runner=None, max_steps=None):
    """Run one final run and write its raw log (same format as src.train.main). Skips a finished run."""
    cfg = yaml.safe_load(open(job.cfg_path))
    cfg["seed"] = job.seed
    if max_steps is not None:                          # smoke tests only (scripts/run_final.py --smoke)
        cfg["max_steps"] = int(max_steps)
    if existing_run(job.out_dir, cfg["stage"], cfg["arm"], job.seed):
        return "skipped"
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if runner is None:
        from src import train
        runner = train.run
    log = runner(cfg, device)
    commit, dirty = _git_info()
    run_id = f"{datetime.date.today():%Y%m%d}_{cfg['stage']}_{cfg['arm']}_s{cfg['seed']}"
    out = pathlib.Path(job.out_dir) / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps({"_header": True, "run_id": run_id, "config": cfg, "config_hash": config_hash(cfg),
                            "git_commit": commit, "git_dirty": dirty, "plan_group": job.group}) + "\n")
        for row in log:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, out)
    return "done"


def lock_status(repo=REPO, tag=LOCK_TAG, paths=LOCKED_PATHS):
    """(ok, message). ok only if the tag exists, is an ancestor of HEAD, and none of the locked paths
    differ from the tag (committed, uncommitted, or new untracked files)."""
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=repo)
    if git("rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode != 0:
        return False, f"the git tag {tag} does not exist; create it on the lock commit first (D-019)"
    if git("merge-base", "--is-ancestor", tag, "HEAD").returncode != 0:
        return False, f"{tag} is not an ancestor of HEAD"
    changed = git("diff", "--name-only", tag, "--", *paths).stdout.split()
    new = git("ls-files", "--others", "--exclude-standard", "--", *paths).stdout.split()
    if changed or new:
        return False, f"locked files differ from {tag}: {sorted(set(changed + new))}"
    return True, "ok"
