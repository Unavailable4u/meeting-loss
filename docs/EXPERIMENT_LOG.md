# Experiment Log

One entry per run group, in chronological order (newest at the bottom). Raw results live in `results/raw/` and are never edited by hand. Template: `docs/templates/experiment-entry.md`.

Rules:
- Log every run group, including failures and sweeps.
- Each entry records the git commit, config files, seeds, and the exact command to reproduce.
- If a later bug or decision invalidates an entry, mark it `Valid: invalidated by <ID>`; do not delete it.

---

## E-001: Stage 1 untuned baseline (arm A), 5 seeds
- Date / Stage / Hypothesis tested (H-ID): 2026-10-21 / Stage 1 / none (baseline characterisation)
- Type: exploratory (baseline characterisation; no hypothesis tested; added by DR-002 / D-015)
- Git commit / config file(s) / seeds: see `git_commit` in each run header / configs/stage1_baseline.yaml / seeds 0-4
- Reproduce with: `for s in 0 1 2 3 4; do python -m src.train --config configs/stage1_baseline.yaml --seed $s; done`
- Result summary: final val_acc 0.9730-0.9798 (mean 0.9769, sd ~0.0025); steps to 0.97: 850-1200 (median 900); steps to 0.975: 1400-1750 (mean 1600); 155,058,176 FLOPs/step; ~460 s per run on CPU (raw: results/raw/20261021_stage1_base_s{0..4}.jsonl)
- Valid: yes
- Interpretation: Untuned baseline is reproducible across seeds and sits just under the ~98% gate. Curves are noisy around 97.5-98%, so 0.97 and 0.975 are more stable steps-to-target candidates than 0.98. Validation sampling error (~0.2 points) is comparable to seed spread.
- Follow-ups: run the equal-budget tuning protocol (D-005) for arm A; finalize D-003 targets after tuning; check whether all seeds reach 0.98 at any point.
- Note (2026-10-21, DR-002): the tuning protocol is now D-014, which supersedes D-005. E-001 remains valid as an untuned baseline characterisation.

## E-002: Regression check of the v1.1 code against E-001 (arm A, seed 0)
- Type: verification
- Date / Stage / Hypothesis tested (H-ID): 2026-10-22 / Stage 1 / none
- Git commit / config file(s) / seeds: see `git_commit` in the run header / configs/stage1_regression.yaml (copy of stage1_baseline.yaml with `stage: stage1reg`) / seed 0
- Reproduce with: `python -m src.train --config configs/stage1_regression.yaml --seed 0`
- Result summary: bit-identical to E-001 seed 0 at all 120 evaluation points (val_acc, task loss and FLOPs identical, max difference 0.0); steps to 0.97 / 0.975 / 0.98 = 900 / 1700 / 4300 in both; wall time 472 s vs 463 s (raw: results/raw/20261022_stage1reg_base_s0.jsonl)
- Valid: yes
- Interpretation: the v1.1 optimizer groups, scheduler and lazy data loading reproduce the original training exactly at default settings, so E-001 and later arm A runs remain comparable.
- Follow-ups: none.
