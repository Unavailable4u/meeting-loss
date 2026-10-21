# Scripts

Written (shared logic lives in `src/sweep.py`, tested in `tests/test_sweep.py`; protocol in decisions D-014 and D-017):
- `run_sweep.py`: random-search tuning with an equal trial budget per arm, 2 selection seeds per trial, 3,000-step runs. Resumable (finished runs are skipped, never overwritten), parallel (`--workers`), and safe against mixing protocol and non-protocol runs (a sweep with other settings needs a tag starting with `smoke`). Raw logs go to `results/raw/sweeps/<tag>/`, one file per (arm, trial, seed), plus `<arm>_trials.json` recording the sampled trials. Start with `--dry-run`.
- `analyze.py select`: picks the best trial per arm by the D-017 rule (mean FLOPs to first reach 0.975 over the selection seeds, misses scored at full budget), writes `results/processed/sweep_<tag>_<arm>.csv`, and with `--write-configs` writes `configs/tuned/<tag>_<arm>.yaml` for the final seeds.
- `analyze.py compare`: steps-to-target and FLOPs-to-target for groups of final runs, with bootstrap confidence intervals and a paired-bootstrap percentage difference against a reference group (the same seeds must be run for both).
- `flops_overhead.py`: per-step FLOP overhead of every arm (F-003).

Typical flow:

```
python scripts/run_sweep.py --tag t1 --arms main --dry-run
python scripts/run_sweep.py --tag t1 --arms main --workers 4
python scripts/analyze.py select --tag t1 --write-configs
for s in 0 1 2 3 4 5 6 7 8 9; do python -m src.train --config configs/tuned/t1_base.yaml --seed $s; done
python scripts/analyze.py compare --ref A --group A='results/raw/*_final-t1-base_base_s*.jsonl' --group D='results/raw/*_final-t1-meet_meet_s*.jsonl'
```

Still to write:
- `make_figures.py`: regenerate every figure in `paper/FIGURES.md` from `results/raw/`

Figures and tables are always regenerated from raw results, never edited by hand.
