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
- `time_selective.py`: Look 1 of track S: wall-clock time per training step of arm A and of S30 (one-thread verdict, default threads as a second reading); rule in `docs/notes/selective-backward-prereg-prep.md`; logic in `exploratory/looks.py`.
- `run_batch_control.py`: Look 2: the small-batch control (tune the learning rate at batch 32, 48 and 64, evaluate the best trials against A and S30 on fresh seeds); phases `tune`, `eval`, `analyze`, resumable, trial configs in `configs/looks/<tag>/`.
- `pilot_selective.py`: pilot of the selective backward pass (arm `selective`, code in `src/selective.py`); plan and decision rule: `docs/notes/selective-backward-pilot.md`. `--synthetic` is a plumbing test. `--sel-fracs 0.1 0.2 0.3 0.5` runs Look 3 of track S (the `sel_frac` curve) and reports the knee.
- `run_dose_response.py`: checks G0, G1, G2 (tags `mr1`, `mr2`, `mr3`; doses, seeds, activation and rule per tag in `exploratory/dose.py`; plan in `docs/notes/mechanism-dose-follow-up.md`), dose-response of the learned meeting loss (arm `meet_rho`, `docs/notes/mechanism-dose-response.md`, D-020, DR-010). Arm A's tuned config with rho in {0, 0.1, 0.3, 1, 3}, seeds 1100 to 1109, 50 runs, exploratory. Resumable, parallel (`--workers`, `--threads 1` as the Stage 3 runs), never overwrites a log; raw logs in `results/raw/dose/<tag>/`; prints the verdict and writes `results/processed/dose_response_<tag>.csv`. `--dry-run`, `--analyze-only`, `--write-configs` (configs must be committed before a run; they are checked against their derivation), `--smoke --out-dir <dir> --max-steps N` for a plumbing test. Logic in `exploratory/dose.py`, tests in `tests/test_dose.py`. Run `scripts/verify_regression.py` on real MNIST first.
- `run_root_cause.py`: checks B and D and the repaired control F' (tags `mb1` = check B in phases 1 to 3, `md1` = check D; plan, rules and seeds in `exploratory/oracle.py`, note `docs/notes/mechanism-root-cause.md`, D-028, DR-019). Oracle directions next to backprop (`meet_oracle`: F' exact gradient, P pinv, Q damped logit, Pr = P with a radial bound) and replacement arms (backprop cut at h), in relu and in tanh (tuned tanh baseline from `run_sweep.py --tag t13 --arms base_tanh`; learned replacement from `--tag t4 --arms meet_replace`). Resumable, parallel (`--workers`, `--threads 1`), never overwrites a log; raw logs in `results/raw/oracle/<tag>/`; prints the verdicts and writes `results/processed/oracle_<tag>_p<phase>.csv`. `--phase`, `--settings`, `--dry-run`, `--analyze-only`, `--write-configs` (configs go to `configs/oracle/`, are checked against their derivation, and must be committed before a run), `--write-chosen` (phase 1 analysis), `--smoke --seeds ... --out-dir <dir> --max-steps N`. Logic in `exploratory/oracle.py` and `src/oracle.py`, tests in `tests/test_oracle.py`. Run `scripts/verify_regression.py` on real MNIST first. Follow-ups (D-029, `docs/notes/mechanism-root-cause-2.md`): tag `mb2` (check B in tanh with an lr-matched control and a second tuning, t14) and tag `md2` (relu replacement with the radially bounded meeting gradient, learned G tuned as t15).

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

prereg-v2 (D-023, DR-011; plan and order in `docs/notes/prereg-v2-plan.md`):
- `run_sweep.py --tag t9 --arms base_b48 --seeds 9000 9001` (batch 48, 8,000 steps) and `--tag t10 --arms selective30 --seeds 9100 9101` (tuned S): the protocol horizon is chosen per arm; other settings need a `smoke` tag.
- `run_final.py --plan configs/prereg_v2_plan.yaml`: lock tag, output directory (`results/raw/prereg_v2`) and locked paths come from the plan; `--dry-run` names configs that do not exist yet.
- `make_prereg2_configs.py`: writes `configs/prereg_v2/{S30,S20,R30}.yaml` (A1 plus the selective keys).
- `analyze_prereg2.py`: the pre-registered analysis (H7a, H7b, re-tuning shift, noise floor, descriptive groups) from the plan; refuses incomplete groups; `--shift-rule` shows another rule descriptively.

prereg-v3 (D-026, DR-017; plan and order in `docs/notes/prereg-v3-plan.md`):
- `run_final.py --plan configs/prereg_v3_plan.yaml`: 13 groups, 260 runs, lock tag `prereg-v3`, logs in `results/raw/prereg_v3`.
- `make_prereg3_configs.py`: writes `configs/prereg_v3/{S20w0,S30,S20,S15,S20w10,R20w0}.yaml` (A1 plus the selective keys).
- `analyze_prereg3.py`: the pre-registered analysis (H8, H9, H10, re-tuning shift, noise floor, secondary readings, descriptive groups, run-header provenance); refuses incomplete groups; `--shift-rule` shows another rule descriptively.
- `run_meeting_net.py`: Study 1b, the meeting network (tag `mn1`: groups A, S, D, M1m, M1f, M1n, M1h, M1t, M2m, M2f, M3m, M3f on seeds 1500 to 1509, each at its tuned config of tag t16; plan, rules and seeds in `exploratory/meetnet.py`, note `docs/notes/meeting-network.md`, D-030, DR-023). Needs the tuned configs from `run_sweep.py --tag t16 --arms mn_match mn_fwd mn_noise mn_mh mn_table shallow mn_match_k2 mn_fwd_k2 mn_match_k3 mn_fwd_k3 --seeds 16000 16001` and `analyze.py select --tag t16 --write-configs`. Resumable, parallel (`--workers`, `--threads 1`), never overwrites a log; raw logs in `results/raw/mnet/<tag>/`; prints the group verdicts (EXCELS, COMPETITIVE, ...) and readings and writes `results/processed/mnet_<tag>.csv`. `--dry-run`, `--analyze-only`, `--write-configs` (to `configs/mnet/`, checked against the derivation, commit before a run), `--smoke --seeds ... --out-dir <dir> --max-steps N`. Tests in `tests/test_meetnet.py`.
- `probe_dataset.py`: fixes the accuracy targets of a dataset other than MNIST (D-031, `src/datasets.py`): runs arm A as tuned on MNIST for 3,000 steps on probe seeds 3000 to 3002 and writes `configs/datasets/<dataset>_targets.json` once (refuses to overwrite). `--dataset fmnist`, `--workers`, `--dry-run`.
- `run_setting.py`: the setting study (tag `f1` = Fashion-MNIST: groups A, A2, S, S20, SBT, D, B, M1m, M1f, M1t, M2m, M3m, M3f on seeds 2500 to 2509, each at the tuned config of tags t17 and t18; plan, rules and seeds in `exploratory/setting.py`, note `docs/notes/setting-fmnist.md`, D-031, DR-024). Needs `probe_dataset.py --dataset fmnist`, `run_sweep.py --tag t17 --arms <f_* arms> --seeds 17000 17001`, `run_sweep.py --tag t18 --arms f_base --seeds 18000 18001` and `analyze.py select ... --write-configs`. Resumable, parallel, never overwrites a log; raw logs in `results/raw/setting/<tag>/`; prints the verdicts (S20 HOLDS / ..., D and B HELPS / ..., Study 1b group verdicts) and writes `results/processed/setting_<tag>.csv`. Same options as `run_meeting_net.py`. Tests in `tests/test_setting.py`.

