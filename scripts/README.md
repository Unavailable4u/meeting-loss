# Scripts

Written (shared logic lives in `src/sweep.py`, tested in `tests/test_sweep.py`; protocol in decisions D-014 and D-017):
- `run_sweep.py`: random-search tuning with an equal trial budget per arm, 2 selection seeds per trial, 3,000-step runs. Resumable (finished runs are skipped, never overwritten), parallel (`--workers`), and safe against mixing protocol and non-protocol runs (a sweep with other settings needs a tag starting with `smoke`). Raw logs go to `results/raw/sweeps/<tag>/`, one file per (arm, trial, seed), plus `<arm>_trials.json` recording the sampled trials. Start with `--dry-run`.
- `analyze.py select`: picks the best trial per arm by the D-017 rule (mean FLOPs to reach 0.975 over the selection seeds, misses scored at full budget), where reaching means two consecutive evaluations at or above the target (D-018, `src/crossing.py`; `--crossing first|mean3` for sensitivity only). Writes `results/processed/sweep_<tag>_<arm>_<crossing>.csv`, and with `--write-configs` writes `configs/tuned/<tag>_<arm>_<crossing>.yaml` for the final seeds. Files without the crossing suffix are from the superseded first-crossing rule.
- `rescore_sensitivity.py`: re-scores a sweep under all three crossing rules and checks selection bias by picking on one seed and scoring on the other; writes `results/processed/sensitivity_<tag>.csv`.
- `analyze.py compare`: steps-to-target and FLOPs-to-target for groups of final runs (same crossing rule, `--crossing`), with bootstrap confidence intervals and a paired-bootstrap percentage difference against a reference group (the same seeds must be run for both). The pre-registered outcome (D-019) is printed with `--alpha 0.025 --margin 15 --primary-targets 0.97 0.975 --shift-group A2`: a per-target verdict (supported / refuted / inconclusive) and an overall verdict per group.
- `run_final.py`: Stage 3 final runs from `configs/stage3_plan.yaml` (D-019): 150 runs, resumable, parallel (`--workers`, `--threads`), never overwrites a raw log, writes `results/raw/` with the plan group, config hash and commit in the header. **Refuses to run** unless the git tag `prereg-v1` exists, is an ancestor of HEAD, and the locked files (plan, tuned configs, `src/`, `docs/HYPOTHESES.md`, `docs/METHOD.md`) are unchanged since the tag. `--dry-run` lists the jobs, `--groups` selects groups, `--smoke --out-dir <dir> --max-steps N` is a plumbing test that skips the lock and must not write to `results/raw/`. Logic in `src/final.py`, tests in `tests/test_final.py`.
- `power_simulation.py`: seed noise of steps-to-target in the sweeps and the power of the D-019 outcome rule (E-007); writes `results/processed/power_stage3.csv`.
- `flops_overhead.py`: per-step FLOP overhead of every arm (F-003).
- `gradient_concentration.py`: exploratory check of how much of the gradient the highest-loss examples carry (plan and decision rule: `docs/notes/gradient-concentration-check.md`; logic in `exploratory/gradconc.py`, outside the locked `src/`). `--synthetic` is a plumbing test.

Typical flow:

```
python scripts/run_sweep.py --tag t1 --arms main --dry-run
python scripts/run_sweep.py --tag t1 --arms main --workers 4
python scripts/analyze.py select --tag t1 --write-configs
python scripts/rescore_sensitivity.py --tag t1
python scripts/run_final.py --dry-run                      # list the 150 Stage 3 jobs
python scripts/run_final.py --workers 4                      # only after the prereg-v1 tag (it refuses otherwise)
python scripts/analyze.py compare --ref A \
    --group A='results/raw/*_final-t1-base-two_base_s*.jsonl' \
    --group D='results/raw/*_final-t1-meet-two_meet_s*.jsonl' \
    --group A2='results/raw/*_final-t2-base-two_base_s*.jsonl' \
    --alpha 0.025 --margin 15 --targets 0.97 0.975 0.98 --primary-targets 0.97 0.975 --shift-group A2   # H1
# H6: same command with --group B='results/raw/*_final-t1-deepsup-two_deepsup_s*.jsonl' in place of D
# Run-file patterns of the other groups (the arm name in a file is not always the group name):
#   C      *_final-t1-ae-two_ae_s*              E      *_final-t1-base_lowlr-two_base_s*
#   Bcode  *_final-t1-deepsup_code-two_deepsup_s*   F  *_final-t1-meet_grad-two_meet_grad_s*
#   V1     *_final-t1-meet_noise-two_meet_s*    V2     *_final-t1-meet_ao-two_meet_ao_s*
#   V3     *_final-t1-meet_frozen-two_meet_frozen_s*
```

Still to write:
- `make_figures.py`: regenerate every figure in `paper/FIGURES.md` from `results/raw/`

Figures and tables are always regenerated from raw results, never edited by hand.
