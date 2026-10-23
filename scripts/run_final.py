"""Run the Stage 3 final runs of configs/stage3_plan.yaml (decision D-019). Resumable and parallel.

  python scripts/run_final.py --dry-run               # what would run
  python scripts/run_final.py --workers 4             # all 150 runs (about 2.7 hours on 4 workers)
  python scripts/run_final.py --workers 4 --groups A D B A2

Refuses to run unless the git tag prereg-v1 exists, is an ancestor of HEAD, and the locked files (plan,
tuned configs, src/, HYPOTHESES.md, METHOD.md) are identical to the tag. A change to those files after the
lock is post-hoc: record a drift entry, then use --allow-changed to run anyway (the tag must still exist).
Raw logs go to results/raw/ as YYYYMMDD_<stage>_<arm>_s<seed>.jsonl; finished runs are skipped, never
overwritten. --smoke runs a few steps into a different directory without the lock check (pipeline test).
"""
import argparse
import multiprocessing as mp
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src.final import existing_run, load_plan, lock_status, plan_jobs, run_final_job


def _init_worker(threads):
    import torch
    torch.set_num_threads(threads)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="configs/stage3_plan.yaml")
    ap.add_argument("--groups", nargs="+", default=None, help="subset of plan groups (default: all)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker when --workers > 1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default="results/raw")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many of the remaining jobs")
    ap.add_argument("--allow-changed", action="store_true", help="skip the locked-files check (needs a drift entry)")
    ap.add_argument("--smoke", action="store_true", help="pipeline test: no lock check, needs --out-dir other than results/raw")
    ap.add_argument("--max-steps", type=int, default=None, help="only with --smoke")
    args = ap.parse_args()

    if args.smoke:
        if pathlib.Path(args.out_dir).resolve() == pathlib.Path("results/raw").resolve():
            sys.exit("--smoke must write somewhere other than results/raw")
    else:
        if args.max_steps is not None:
            sys.exit("--max-steps is only for --smoke")
        ok, msg = lock_status()
        if not ok and args.dry_run:
            print(f"(lock check would fail now: {msg})")
            ok = True
        if not ok and args.allow_changed and "does not exist" not in msg and "not an ancestor" not in msg:
            print(f"WARNING: {msg}\n(--allow-changed: continuing; record a drift entry for this)")
            ok = True
        if not ok:
            sys.exit(f"LOCK CHECK FAILED: {msg}")

    groups = load_plan(args.plan)
    if args.groups:
        unknown = set(args.groups) - {g["name"] for g in groups}
        if unknown:
            sys.exit(f"unknown groups {sorted(unknown)}; plan has {[g['name'] for g in groups]}")
        groups = [g for g in groups if g["name"] in args.groups]
    jobs = plan_jobs(groups, args.out_dir)
    stage_arm = {g["name"]: (g["cfg"]["stage"], g["cfg"]["arm"]) for g in groups}
    todo = [j for j in jobs if not existing_run(args.out_dir, *stage_arm[j.group], j.seed)]
    print(f"{len(groups)} groups, {len(jobs)} runs; {len(jobs) - len(todo)} already done, {len(todo)} to run")
    for g in groups:
        print(f"  {g['name']:>6} {g['role']:>11} seeds {g['seeds'][0]}-{g['seeds'][-1]}  {g['config']}")
    if args.dry_run:
        return
    if args.limit is not None:
        todo = todo[:args.limit]

    failed, t0 = [], time.time()

    def report(i, job, status):
        print(f"[{i}/{len(todo)}] {status:7s} {job.group}_s{job.seed}  (elapsed {time.time() - t0:7.0f}s)", flush=True)

    if args.workers <= 1:
        for i, job in enumerate(todo, 1):
            try:
                status = run_final_job(job, args.device, max_steps=args.max_steps)
            except Exception as e:
                failed.append((job, repr(e)))
                status = "FAILED"
                print(f"   {job.group}_s{job.seed}: {e!r}", file=sys.stderr)
            report(i, job, status)
    else:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(args.threads,)) as ex:
            futs = {ex.submit(run_final_job, j, args.device, None, args.max_steps): j for j in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                job = futs[fut]
                try:
                    status = fut.result()
                except Exception as e:
                    failed.append((job, repr(e)))
                    status = "FAILED"
                    print(f"   {job.group}_s{job.seed}: {e!r}", file=sys.stderr)
                report(i, job, status)
    if failed:
        print(f"\n{len(failed)} run(s) failed; rerun the same command to retry them:")
        for job, err in failed:
            print(f"  {job.group}_s{job.seed}: {err}")
        sys.exit(1)
    print("\nfinished. Next: see the analysis commands in docs/DECISIONS.md D-019 and scripts/README.md")


if __name__ == "__main__":
    main()
