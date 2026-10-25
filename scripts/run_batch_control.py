"""Look 2 of track S: is arm A compute-optimal in batch size? (docs/notes/selective-backward-prereg-prep.md)

Design (fixed before any run): arm A with batch size b in {32, 48, 64}; horizon 3000 x 128 / b steps (equal samples seen
as A's 3,000 steps); evaluation spacing kept at about the same number of samples as A; for each b, 8 random trials over the
learning rate (log-uniform 3e-4 to 5e-3), all other settings as arm A's tuned config, 2 seeds (1200, 1201), score = mean
FLOPs to reach 0.97 (two consecutive evaluations, D-018; a miss costs its full budget). The best trial per b is then run
on 3 fresh seeds (1202 to 1204), together with arm A (batch 128, tuned) and S30 (the pilot's selective arm with A's
hyperparameters). Exploratory (D-015). Threads: one per worker, as the Stage 3 runs (E-011).

  python scripts/run_batch_control.py --tag sb1 --dry-run
  python scripts/run_batch_control.py --tag sb1 --workers 4            # tune, select, evaluate, analyze (resumable)
  python scripts/run_batch_control.py --tag sb1 --phase analyze        # only recompute the table and the verdict
  python scripts/run_batch_control.py --tag smoke --smoke --n-trials 1 --max-steps 60 --device cpu \
         --out-dir /tmp/sb/raw --cfg-dir /tmp/sb/cfg --csv /tmp/sb/out.csv      # pipeline test, needs MNIST

Trial configs are written to configs/looks/<tag>/ (commit them); raw logs go to results/raw/looks/<tag>/ (finished runs
are skipped, never overwritten).
"""
import argparse
import csv
import json
import multiprocessing as mp
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from exploratory import looks as L  # noqa: E402
from src.final import FinalJob, existing_run, run_final_job  # noqa: E402

ARM = {"A": "base", "S30": "selective", **{f"b{b}": "base" for b in L.SMALL_BATCHES}}


def load_rows(path):
    rows = []
    for line in open(path):
        if line.strip():
            rec = json.loads(line)
            if not rec.get("_header"):
                rows.append(rec)
    return rows


def stage_rows(out_dir, stage, arm, seed):
    path = existing_run(out_dir, stage, arm, seed)
    return load_rows(path) if path else None


def write_config(path, cfg):
    path = pathlib.Path(path)
    if path.exists():
        if yaml.safe_load(path.read_text()) != cfg:
            sys.exit(f"{path} exists with different content; the design is fixed, use a new --tag")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))


def tune_jobs(tag, base, cfg_dir, out_dir, n_trials=L.N_TRIALS):
    jobs = []
    for b in L.SMALL_BATCHES:
        for i, lr in enumerate(L.sample_lrs(tag, b, n_trials)):
            stage = L.trial_stage(tag, b, i)
            path = pathlib.Path(cfg_dir) / f"{stage}.yaml"
            write_config(path, L.trial_config(base, b, lr, stage))
            jobs += [FinalJob(stage, seed, str(path), str(out_dir)) for seed in L.TUNE_SEEDS]
    return jobs


def select_best_trials(tag, out_dir, cfg_dir, n_trials=L.N_TRIALS):
    """Best tuning trial per batch size from the finished tuning runs. Returns {"b32": dict(...), ...} with the config."""
    best = {}
    for b in L.SMALL_BATCHES:
        trial_runs = {}
        for i in range(n_trials):
            stage = L.trial_stage(tag, b, i)
            per_seed = {}
            for seed in L.TUNE_SEEDS:
                rows = stage_rows(out_dir, stage, "base", seed)
                if rows is None:
                    sys.exit(f"missing tuning run {stage} seed {seed}; run --phase tune first")
                per_seed[seed] = rows
            trial_runs[i] = per_seed
        table, win = L.tune_table(trial_runs)
        stage = L.trial_stage(tag, b, win["trial"])
        cfg = yaml.safe_load((pathlib.Path(cfg_dir) / f"{stage}.yaml").read_text())
        best[f"b{b}"] = dict(trial=win["trial"], lr=cfg["lr"], score=win["score"], n_reached=win["n_reached"],
                             table=table, cfg=cfg)
    record = {k: {kk: v for kk, v in d.items() if kk not in ("cfg", "table")} for k, d in best.items()}
    (pathlib.Path(cfg_dir) / "best.json").write_text(json.dumps(record, indent=2))
    return best


def eval_jobs(tag, base, best, cfg_dir, out_dir):
    jobs = []
    for name in ["A", "S30"] + [f"b{b}" for b in L.SMALL_BATCHES]:
        stage = L.eval_stage(tag, name)
        cfg = L.eval_config(base, name, stage, {k: v["cfg"] for k, v in best.items()})
        path = pathlib.Path(cfg_dir) / f"{stage}.yaml"
        write_config(path, cfg)
        jobs += [FinalJob(stage, seed, str(path), str(out_dir)) for seed in L.EVAL_SEEDS]
    return jobs


def _init_worker(threads):
    import torch
    torch.set_num_threads(threads)


def run_jobs(jobs, device, workers, threads, runner=None, max_steps=None):
    failed = []
    if workers <= 1:
        if runner is None:
            _init_worker(threads)
        for i, job in enumerate(jobs, 1):
            try:
                status = run_final_job(job, device, runner, max_steps)
            except Exception as e:                      # noqa: BLE001
                failed.append((job, repr(e)))
                status = "FAILED"
            print(f"[{i}/{len(jobs)}] {status:7s} {job.group}_s{job.seed}", flush=True)
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(threads,)) as ex:
            futs = {ex.submit(run_final_job, j, device, None, max_steps): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                job = futs[fut]
                try:
                    status = fut.result()
                except Exception as e:                  # noqa: BLE001
                    failed.append((job, repr(e)))
                    status = "FAILED"
                print(f"[{i}/{len(jobs)}] {status:7s} {job.group}_s{job.seed}", flush=True)
    if failed:
        for job, err in failed:
            print(f"  FAILED {job.group}_s{job.seed}: {err}", file=sys.stderr)
        sys.exit(f"{len(failed)} run(s) failed; rerun the same command to retry them")


def analyze(tag, out_dir, csv_path, best=None):
    runs = {}
    for name in ["A", "S30"] + [f"b{b}" for b in L.SMALL_BATCHES]:
        stage = L.eval_stage(tag, name)
        per_seed = {s: stage_rows(out_dir, stage, ARM[name], s) for s in L.EVAL_SEEDS}
        per_seed = {s: r for s, r in per_seed.items() if r is not None}
        if per_seed:
            runs[name] = per_seed
    if "A" not in runs or "S30" not in runs or not any(n.startswith("b") for n in runs):
        sys.exit("evaluation runs are missing (need A, S30 and at least one small batch); run the eval phase first")
    summary = L.ratios(runs, "A")
    small = [n for n in runs if n.startswith("b")]
    ver = L.batch_control_verdict(summary, small)
    print("\n=== Look 2: FLOPs-to-target against arm A at batch 128 (ratio < 1 means cheaper); rule: two consecutive evaluations ===")
    if best:
        for k, d in best.items():
            print(f"  best trial {k}: trial {d['trial']}  lr {d['lr']:.6g}  tuning score {d['score'] / 1e9:.1f} GFLOPs "
                  f"({d['n_reached']}/{len(L.TUNE_SEEDS)} seeds reached 0.97)")
    print("variant  target | GFLOPs  ratio | steps ratio | missed seeds")
    for name in ["A", "S30"] + [n for n in small]:
        if name not in summary:
            continue
        for t in L.TARGETS:
            s = summary[name][t]
            print(f"{name:7s}  {t:.3f}  | {s['flops_mean'] / 1e9:6.1f}  {s['ratio_of_means']:6.3f} |  {s['steps_ratio']:6.3f}     |  {s['misses']}/{s['n']}")
    rel = ver["s_vs_best"]
    print(f"\nBest small-batch arm: {ver['best']}.  S30 / best small batch: " +
          ", ".join(f"{t}: {rel[t]:.3f}" for t in L.TARGETS) +
          f"  ({'S30 still beats it by at least 15%' if ver['s_beats_best'] else 'S30 does NOT beat it by 15% at both targets'})")
    print(f"\nPre-registered rule: BASELINE PROBLEM if the best small batch has ratio <= {L.PROBLEM_RATIO} at both targets; "
          f"A WAS FINE if every small batch has ratio >= {L.FINE_RATIO} at both; otherwise PARTIAL.\nVERDICT: {ver['verdict']}\n")
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["variant", "target", "flops_mean", "ratio_of_means", "steps_ratio", "misses", "n"])
        for name, per in summary.items():
            for t, s in per.items():
                w.writerow([name, t, int(s["flops_mean"]), round(s["ratio_of_means"], 4), round(s["steps_ratio"], 4),
                            s["misses"], s["n"]])
    print(f"wrote {csv_path}")
    return ver


def main(argv=None, runner=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="sb1")
    ap.add_argument("--phase", choices=["all", "tune", "eval", "analyze"], default="all")
    ap.add_argument("--config", default="configs/tuned/t1_base_two.yaml")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker (1 as the Stage 3 runs, E-011)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cfg-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="pipeline test: allows --n-trials and --max-steps, needs --out-dir and --cfg-dir")
    ap.add_argument("--n-trials", type=int, default=L.N_TRIALS, help="only with --smoke")
    ap.add_argument("--max-steps", type=int, default=None, help="only with --smoke")
    args = ap.parse_args(argv)

    if args.smoke:
        if not (args.out_dir and args.cfg_dir):
            sys.exit("--smoke needs --out-dir and --cfg-dir other than the defaults")
    elif args.n_trials != L.N_TRIALS or args.max_steps is not None:
        sys.exit("--n-trials and --max-steps are only for --smoke (the design is fixed)")
    cfg_dir = pathlib.Path(args.cfg_dir or REPO / "configs" / "looks" / args.tag)
    out_dir = pathlib.Path(args.out_dir or REPO / "results" / "raw" / "looks" / args.tag)
    csv_path = args.csv or REPO / "results" / "processed" / f"batch_control_{args.tag}.csv"
    cfg_path = pathlib.Path(args.config)
    base = yaml.safe_load(open(cfg_path if cfg_path.exists() else REPO / args.config))

    if args.phase in ("all", "tune"):
        jobs = tune_jobs(args.tag, base, cfg_dir, out_dir, args.n_trials)
        print(f"tuning: {len(jobs)} runs ({len(L.SMALL_BATCHES)} batch sizes x {args.n_trials} trials x {len(L.TUNE_SEEDS)} seeds)")
        if not args.dry_run:
            run_jobs(jobs, args.device, args.workers, args.threads, runner, args.max_steps)
    best = None
    if args.phase in ("all", "eval", "analyze") and not args.dry_run:
        best = select_best_trials(args.tag, out_dir, cfg_dir, args.n_trials)
        for k, d in best.items():
            print(f"best {k}: trial {d['trial']} lr {d['lr']:.6g}")
    if args.phase in ("all", "eval"):
        if args.dry_run:
            print(f"evaluation: {len(L.EVAL_SEEDS)} seeds x (A, S30, {', '.join(f'b{b}' for b in L.SMALL_BATCHES)}) = "
                  f"{len(L.EVAL_SEEDS) * (2 + len(L.SMALL_BATCHES))} runs, after the tuning finishes")
        else:
            jobs = eval_jobs(args.tag, base, best, cfg_dir, out_dir)
            run_jobs(jobs, args.device, args.workers, args.threads, runner, args.max_steps)
    if args.phase in ("all", "analyze") and not args.dry_run:
        return analyze(args.tag, out_dir, csv_path, best)


if __name__ == "__main__":
    main()
