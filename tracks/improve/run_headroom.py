"""Run the Improve-track headroom plan (exploratory, no lock tag). Resumable and parallel, like scripts/run_final.py.

  python tracks/improve/run_headroom.py --dry-run
  python tracks/improve/run_headroom.py --workers 4 --threads 1
  python tracks/improve/run_headroom.py --workers 4 --threads 1 --groups A1 S30 SB2

Uses src.final's plan loader and job runner (same raw-log format, finished runs are skipped, never overwritten) but does
NOT check a lock tag: these runs are exploratory (D-024). Logs go to the plan's out_dir (results/raw/improve_headroom).
"""
import argparse
import multiprocessing as mp
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.final import existing_run, load_plan, plan_jobs, plan_meta, run_final_job  # noqa: E402


def _init_worker(threads):
    import torch
    torch.set_num_threads(threads)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="tracks/improve/headroom_plan.yaml")
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    plan = ROOT / args.plan
    meta = plan_meta(plan)
    groups = load_plan(plan, repo=ROOT)
    if args.groups:
        unknown = set(args.groups) - {g["name"] for g in groups}
        if unknown:
            sys.exit(f"unknown groups {sorted(unknown)}; plan has {[g['name'] for g in groups]}")
        groups = [g for g in groups if g["name"] in args.groups]
    out_dir = ROOT / meta["out_dir"]
    jobs = plan_jobs(groups, out_dir, repo=ROOT)
    stage_arm = {g["name"]: (g["cfg"]["stage"], g["cfg"]["arm"]) for g in groups}
    todo = [j for j in jobs if not existing_run(out_dir, *stage_arm[j.group], j.seed)]
    print(f"plan {args.plan}: {len(groups)} groups, {len(jobs)} runs; {len(jobs) - len(todo)} done, {len(todo)} to run; logs in {out_dir}")
    for g in groups:
        print(f"  {g['name']:>8} {g['role']:>11} seeds {g['seeds'][0]}-{g['seeds'][-1]}  {g['note'][:90]}")
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
                status = run_final_job(job, args.device)
            except Exception as e:
                failed.append((job, repr(e))); status = "FAILED"
                print(f"   {job.group}_s{job.seed}: {e!r}", file=sys.stderr)
            report(i, job, status)
    else:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(args.threads,)) as ex:
            futs = {ex.submit(run_final_job, j, args.device): j for j in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                job = futs[fut]
                try:
                    status = fut.result()
                except Exception as e:
                    failed.append((job, repr(e))); status = "FAILED"
                    print(f"   {job.group}_s{job.seed}: {e!r}", file=sys.stderr)
                report(i, job, status)
    if failed:
        print(f"\n{len(failed)} run(s) failed; rerun the same command to retry them:")
        for job, err in failed:
            print(f"  {job.group}_s{job.seed}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
