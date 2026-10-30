# prereg-v3: the plan, the machinery and what is still open

**Status:** plan and machinery written BEFORE any `prereg-v3` run (2026-10-30). Design: D-026 (accepted 2026-10-30; the proposal it answers is `prereg-v3-proposal.md`), drift entry DR-017. Nothing is locked until the researcher creates the tag `prereg-v3` on the lock commit (steps below). Type of the final runs: confirmatory (H8, H9, H10), everything else descriptive or secondary.

## The hypotheses (they are in `docs/HYPOTHESES.md` from the lock commit on)

**H8** S20w0 against A1, margin 15%, re-tuning shift `max_gain` (A2, A5 to A8 re-run on the new seeds); **H9** S20w0 against S30, margin 10%; **H10** S20w0 against the tuned Selective-Backprop SBT, margin 15%, uncached FLOP count decisive. 20 paired seeds (2300 to 2319) in every group, targets 0.97 and 0.975 (both must agree), the crossing rule of D-018, intervals with alpha 0.05 / 3 (98.33%), 10,000 resamples, seed 0.

## What exists now

| Piece | File |
|---|---|
| Plan (13 groups, 260 runs, all configs exist) | `configs/prereg_v3_plan.yaml` |
| Derived configs S20w0, S30, S20, S15, S20w10, R20w0 (A1 plus selective keys; equal to the look 2 configs apart from the stage) | `configs/prereg_v3/*.yaml`, written by `scripts/make_prereg3_configs.py`, checked against `src/prereg3.py` |
| Tuned Selective-Backprop (reference of H10) | `configs/tuned/t11_sb_two.yaml` (E-026) |
| Pre-registered analysis: H8, H9, H10, shift, noise floor, secondary readings, descriptive tables, provenance of the run headers | `src/prereg3.py`, `scripts/analyze_prereg3.py` |
| Tests (14) | `tests/test_prereg3.py` |
| Runner, lock check, plan loader | `scripts/run_final.py --plan configs/prereg_v3_plan.yaml`, `src/final.py` (unchanged) |

## Order of work (the researcher's steps)

1. `git pull`, `git am --3way` the patch, `python -m pytest -q` (all tests pass; the sandbox that wrote this ran only `tests/test_prereg3.py` and `tests/test_prereg2.py`, which need no torch), `git push`.
2. `python scripts/run_final.py --plan configs/prereg_v3_plan.yaml --dry-run` must list 13 groups and 260 runs (it says the lock check would fail until the tag exists).
3. **Lock:** `git tag prereg-v3 && git push origin prereg-v3` on the commit that holds this patch. The plan, `configs/prereg_v3`, `configs/tuned`, `src`, `docs/HYPOTHESES.md` and `docs/METHOD.md` must not change afterwards (`run_final.py` checks this against the tag). If you want a different margin, interval or group before the tag, change it first and tell the assistant so that the documents stay in step.
4. **Final runs:** `nohup python scripts/run_final.py --plan configs/prereg_v3_plan.yaml --workers 4 --threads 1 > prereg3.log 2>&1 &` (resumable; logs in `results/raw/prereg_v3/`). Laptop idle. My estimate is 4 to 4.5 hours: by the effective rates of E-008 (about 66 s per run for arm A on four workers) and E-025 (about 38 s per run for selective arms), 120 A-type runs, 120 selective runs and 20 Selective-Backprop runs at about 2.3 times A's cost come to about 15,000 s. This is derived, not measured; the proposal's 3 hours was too low.
5. **Analysis:** `python scripts/analyze_prereg3.py` (refuses incomplete groups or unfinished runs; writes `results/processed/prereg_v3.csv`; prints the commits found in the run headers, which must be one clean commit). Send the full output; do not look at partial results first. The assistant then logs one confirmatory entry (E-027), a finding, the status lines of H8 to H10 (a drift entry, as DR-013) and the claims.

## Implementation clarifications (written before any run)

- **Decisive count of H10.** `flops` of the `sb` logs, the uncached count of the released algorithm (D-025). The cached count is the same crossing evaluation read from `flops_cached` and is reported as `H10_cached`; it does not enter the verdict.
- **Never-reached runs** are charged their whole-run FLOPs and counted (D-019); steps of such runs are left out of the step comparisons, which are then not printed.
- **Warm-up credit.** For S30, S20, S15 and S20w10 the FLOPs at the crossing minus min(W, crossing step) times (a full step minus a selective step); the full step is read from the A1 logs (every step is full) and checked equal across seeds, the selective step from the run's own evaluations after the warm-up. It feeds only the secondary `S20w0_vs_S20_credited`.
- **Per-seed counts** ("cheaper in n of 20 seeds") are printed for H8, H9 and H10; they decide nothing.
- **Provenance.** The analysis prints the pairs (git commit, `git_dirty`) of all 260 headers; more than one commit or a dirty tree is a warning, to be explained in E-027 (in E-022 this was checked by hand).
- **Roles.** The plan has two groups with role `control` (S30 and SBT); which hypothesis each belongs to is fixed by name in `src/prereg3.py` (`REFERENCES`), and the analysis refuses a plan whose reference or primary group has another name.
- **The wall-clock secondary** is not built. `scripts/time_selective.py` times arm A and S30 only; a timing of A, S20w0, S30 and SBT at one and eight threads needs its own script and a rule written before it runs (as in `selective-backward-prereg-prep.md`). It is not part of the lock, and no wall-clock sentence may appear in a claim until it exists (F-016).
- **Not in this plan:** a tuned S20w0 (t12), a batch-48 control, a second setting, any group at a kept fraction below 0.15 or with another warm-up than 0 and 10, and the optimizer-state or step-at-zero explanations (F-024).
