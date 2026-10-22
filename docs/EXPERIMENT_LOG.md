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

## E-003: Stage 2 diagnostic runs of the meeting-loss arms (meet, meet_ao, meet_frozen), seed 0, 2000 steps
- Type: exploratory (collapse and dead-signal check; no hypothesis tested; not a result)
- Date / Stage / Hypothesis tested (H-ID): 2026-10-22 / Stage 2 / none (informs H3 and risk R-03 only)
- Git commit / config file(s) / seeds: 4c3d4ee (see `git_commit` in each run header) / configs/stage2diag_meet.yaml, configs/stage2diag_meet_ao.yaml, configs/stage2diag_meet_frozen.yaml (copies of template_all_options.yaml with `stage: stage2diag`, the arm changed, `max_steps: 2000`; so relu, constant learning rate 1e-3, wd 1e-4, `lam_meet` 0.1, `lam_inv` 1.0, `warmup_steps` 500, untuned) / seed 0
- Reproduce with: `for arm in meet meet_ao meet_frozen; do python -m src.train --config configs/stage2diag_$arm.yaml --seed 0; done`
- Result summary (raw: results/raw/20261022_stage2diag_{meet,meet_ao,meet_frozen}_s0.jsonl; 40 evaluations each, all values finite):
  - Validation accuracy at steps 1000 / 2000: arm A seed 0 (E-001) 0.9680 / 0.9758; meet 0.9672 / 0.9750; meet_ao 0.9706 / 0.9716; meet_frozen 0.9700 / 0.9752. Mean of the last five evaluations (steps 1800 to 2000): A 0.9755, meet 0.9740, meet_ao 0.9703, meet_frozen 0.9757. All within about 0.4 points of arm A, comparable to the validation sampling error (about 0.2 points, F-002).
  - FLOPs per step: meet 190,578,688 (+22.9%), meet_ao 224,133,120 (+44.5%), meet_frozen 173,146,112 (+11.7%) versus 155,058,176 for arm A, matching F-003.
  - Gradient alignment `align`, after warm-up (30 evaluations, steps 550 to 2000): meet positive at 100% of evaluations (median 0.50, range 0.34 to 0.66, falling from about 0.65 at step 1000 to 0.34 at step 2000); meet_ao positive at 100% but small and falling (median 0.22, range 0.004 to 0.58, 0.14 at step 2000); meet_frozen positive at only 33% (median -0.02, range -0.18 to +0.18).
  - Variance of the middle state `h_var`: no collapse in any run, but it grows: meet 4.3 to 13.2 (peak 16.2), meet_ao 4.0 to 58.2, meet_frozen 4.1 to 22.9. Arm A's `h_var` was not logged, so how much of this growth is ordinary training is unknown.
  - Inverse loss `inv`: meet peaks at 62 early and ends at 8.3; meet_ao rises from 1.6 to 41.3 and is still rising at step 2000.
  - Wall time about 155 s per run on the laptop CPU.
- Valid: yes
- Interpretation: Nothing is broken: no NaN, no collapse, accuracy tracks the baseline. Arm D has a clearly positive alignment with the task gradient, the variant V2 a weaker and decaying one, and the frozen random G has essentially none, as expected for a random feedback network. Its accuracy is nevertheless the same as arm A's, which fits a meeting term with little effect. At equal steps none of the three is better than arm A, while each costs 12% to 45% more FLOPs per step, so at these untuned settings none would beat arm A on FLOPs-to-target (risk R-02). This is one seed with untuned loss weights, so it neither supports nor refutes H1 or H3. Two things to watch: the `h_var` growth, and the V2 inverse loss rising together with it (its backward network `G` may not be keeping up with the growing scale of `h`).
- Follow-ups: sweeps with the D-017 protocol (`scripts/run_sweep.py`); log `h_var` for arm A so that the growth can be compared (needs a `src/` change, so ask first); recheck V2 over a longer run before deciding whether it stays among the three pre-listed variants.

## E-004: First tuning pass t1, arms A to F (25 trials x 2 seeds x 3,000 steps each)
- Type: exploratory (selection runs under D-014 and D-017; no hypothesis tested)
- Date / Stage / Hypothesis tested (H-ID): 2026-10-22 / Stage 1 (tuning) / none
- Git commit / config file(s) / seeds: see `git_commit` and `git_dirty` in each run header / sampled trials in results/raw/sweeps/t1/<arm>_trials.json, winners in configs/tuned/t1_<arm>.yaml / selection seeds 1000 and 1001
- Reproduce with: `python scripts/run_sweep.py --tag t1 --arms main --workers 4`, then `python scripts/analyze.py select --tag t1 --write-configs` (300 runs, about 4.2 h on 4 workers)
- Result summary: all 6 arms completed 25 trials; every winner reached 0.975 in both seeds. Winning scores in GFLOPs to 0.975 (mean over 2 seeds): A 186.1 (t19, mean 1200 steps), B deepsup 191.2 (t10), F meet_grad 205.1 (t15), D meet 243.0 (t17, 1275 steps), E base_lowlr 255.8 (t06), C ae 384.5 (t22). Overhead vs A: +2.7%, +10.2%, +30.6%, +37.5%, +106.6%. Trials never reaching 0.975 in either seed: A 8, B 8, C 7, D 13, E 8, F 7. Winners are in results/processed/sweep_t1_<arm>.csv.
- Valid: yes
- Interpretation: tuned arm A needs about 25% fewer steps than the untuned E-001 baseline. At these selection seeds arm D takes about as many steps as arm A (1275 vs 1200) and pays about 23% more FLOPs per step, so it ranks fourth; its best trials use small `lam_meet` (at most 0.35, mostly below 0.02) and `lam_inv` at the range floor (0.106), and arm D trials with `lam_meet` above 0.5 mostly failed. Differences between the top arms are within the noise: E contains A as a special case but scored 37% worse, A's winner leads its runner-up by 4%, and its two seeds differ by 8%. E's winner has late validation accuracy 0.9735, below the target (R-18). Selection on 2 seeds is optimistic for every arm.
- Follow-ups: re-tuning noise floor for A (tag t2, seeds 2000 and 2001); variant sweeps under tag t1 (meet_noise, meet_ao, meet_frozen, deepsup_code); settle the D-017 provisional items before the `prereg-v1` lock.
- Note (2026-10-22, D-018, DR-004): the scores above use the first-crossing definition and are superseded as the selection result. Under the adopted rule (two consecutive evaluations at or above 0.975) the winners and in-sample scores in GFLOPs are: A t09 232.6, B t10 191.2, C t22 418.0, D t17 276.3, E t23 275.2, F t15 229.7 (ratio to A: 1.00, 0.82, 1.80, 1.19, 1.18, 0.99); the winner changes only for A and E. Reproduce with `python scripts/analyze.py select --tag t1 --write-configs` (files with the `_two` suffix) and `python scripts/rescore_sensitivity.py --tag t1` (F-006). The raw runs are unchanged. Validity of the original entry is unchanged: it is correct for the first-crossing rule.
