"""Random-search tuning sweep (decisions D-014 and D-017). Resumable and parallel.

Examples (run from the repo root):
  # what would run, without training anything
  python scripts/run_sweep.py --tag t1 --arms base meet --dry-run

  # first tuning pass for arm A and arm D, 4 worker processes
  python scripts/run_sweep.py --tag t1 --arms base meet --workers 4

  # re-tuning noise floor for arm A: new tag AND new selection seeds (D-014)
  python scripts/run_sweep.py --tag t2 --arms base --seeds 2000 2001 --workers 4

  # quick pipeline check (non-protocol settings need a tag that starts with "smoke")
  python scripts/run_sweep.py --tag smoke1 --arms base meet --trials 2 --max-steps 100 --seeds 100 101

Protocol (D-014, D-017): 25 trials per arm, 2 selection seeds per trial, 3,000 steps per run (8,000 for the
batch-48 arm base_b48, equal samples; D-023), search spaces in src/sweep.py. prereg-v2 tunings: t9 = base_b48 on
seeds 9000 9001, t10 = selective30 on seeds 9100 9101. Every arm must get the same number of trials; the same tag with the same
arms resumes where it stopped (finished runs are skipped, never overwritten). Raw logs go to
results/raw/sweeps/<tag>/; pick the winners with scripts/analyze.py select.
"""
import argparse
import multiprocessing as mp
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src.sweep import (DEFAULT_SEEDS, MAIN_ARMS, N_TRIALS, SWEEP_ARMS, SWEEP_ROOT,
                       check_protocol, check_tag, expected_steps, plan_jobs, run_job, write_manifest)


def _init_worker(threads):
    import torch
    torch.set_num_threads(threads)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="name of this tuning pass, e.g. t1 (letters, digits, hyphens)")
    ap.add_argument("--arms", nargs="+", required=True, choices=sorted(SWEEP_ARMS) + ["main"],
                    help="sweep arms; 'main' expands to arms A to F: " + " ".join(MAIN_ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                    help="selection seeds (>= 100). Use different seeds for the re-tuning pass")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="default: the protocol horizon of the arms (3,000 steps; 8,000 for base_b48, D-023)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (1 = run inline)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker when --workers > 1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--root", default=str(SWEEP_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the sampled trials, run nothing")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many of the remaining jobs")
    args = ap.parse_args()

    arms = list(dict.fromkeys(x for a in args.arms for x in (MAIN_ARMS if a == "main" else (a,))))
    check_tag(args.tag)
    if args.max_steps is None:
        args.max_steps = expected_steps(arms)
    off = check_protocol(args.tag, args.trials, args.max_steps, args.seeds, arms)
    if off:
        print("NON-PROTOCOL RUN (smoke tag): results must not be used for selection.")

    jobs = plan_jobs(args.tag, arms, args.seeds, args.trials, args.max_steps, args.root)
    todo = [j for j in jobs if not j.path.exists()]
    print(f"tag {args.tag}: {len(arms)} arms x {args.trials} trials x {len(args.seeds)} seeds = "
          f"{len(jobs)} runs of {args.max_steps} steps; {len(jobs) - len(todo)} already done, {len(todo)} to run")

    if args.dry_run:
        for a in arms:
            print(f"\n{a}: first 3 sampled trials")
            for i in range(min(3, args.trials)):
                print(f"  t{i:02d}", next(j for j in jobs if j.arm == a and j.trial == i).hp)
        return

    for a in arms:
        write_manifest(args.tag, a, args.seeds, args.trials, args.max_steps, args.root)
    if args.limit is not None:
        todo = todo[:args.limit]

    failed, t0 = [], time.time()

    def report(i, job, status):
        print(f"[{i}/{len(todo)}] {status:7s} {job.arm}_t{job.trial:02d}_s{job.seed}  "
              f"(elapsed {time.time() - t0:7.0f}s)", flush=True)

    if args.workers <= 1:
        for i, job in enumerate(todo, 1):
            try:
                status = run_job(job, args.device)
            except Exception as e:                      # keep sweeping; report at the end
                failed.append((job, repr(e)))
                status = "FAILED"
                print(f"   {job.arm}_t{job.trial:02d}_s{job.seed}: {e!r}", file=sys.stderr)
            report(i, job, status)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(args.threads,)) as ex:
            futs = {ex.submit(run_job, j, args.device): j for j in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                job = futs[fut]
                try:
                    status = fut.result()
                except Exception as e:
                    failed.append((job, repr(e)))
                    status = "FAILED"
                    print(f"   {job.arm}_t{job.trial:02d}_s{job.seed}: {e!r}", file=sys.stderr)
                report(i, job, status)

    if failed:
        print(f"\n{len(failed)} run(s) failed; rerun the same command to retry them:")
        for job, err in failed:
            print(f"  {job.arm}_t{job.trial:02d}_s{job.seed}: {err}")
        sys.exit(1)
    print(f"\nfinished. Next: python scripts/analyze.py select --tag {args.tag}")


if __name__ == "__main__":
    main()
