# Plan

**Last updated:** 2026-10-22 (sweep and analysis scripts, D-017, DR-003; E-002 and E-003 logged)
**Project start (week 1):** 2026-10-21. Set real calendar dates for each stage below once you know your availability. Weeks are counted from the start date and are rough (part-time pace).
**Current stage:** Stage 1 (baseline tuning) with Stage 2 code ready (method v1.1)

## Current focus

Tune arm A under the D-014 protocol, measure the re-tuning noise floor, and finalize the D-003 targets, while running short exploratory diagnostic runs of the meeting-loss arms.

## Next three actions

1. Apply the sweep patch (`git am`), run `pytest` (46 tests expected to pass), and run the first tuning pass for arms A to F on a lab machine or GPU, in parallel: `python scripts/run_sweep.py --tag t1 --arms main --workers N`; then `python scripts/analyze.py select --tag t1 --write-configs`. All arms must finish the same 25 trials (D-017).
2. Re-tuning noise floor: `python scripts/run_sweep.py --tag t2 --arms base --seeds 2000 2001 --workers N`; run the 10 final seeds for both tuned arm A configs and compare them with `scripts/analyze.py compare`; then finalize and record D-003 (targets 0.97 and 0.975; 0.98 secondary) and decide whether to enlarge the validation split (F-002).
3. Before the `prereg-v1` lock, settle the provisional items of D-017 (final-run horizon, the "small" `aux_gamma`, control F's `eta_hat`, whether the first-crossing score needs smoothing, R-18), then sweep the pre-listed variants (V1 `meet_noise`, V2 `meet_ao`, V3 `meet_frozen`, B-code `deepsup_code`) with the same budget.

## Stage table

| Stage | Name | Target weeks | Status | Exit criteria (short) |
|---|---|---|---|---|
| 0 | Project setup and documents | 1 | **Done** (2026-10-21) | All living docs exist; decisions recorded; repo pushed |
| 1 | Baseline reproduction and tuning | 1 | **In progress** (E-001 and E-002 done; tuning scripts ready, tuning pending) | About 98% MNIST validation accuracy; reproducible across seeds; targets chosen (D-003); FLOP counter verified |
| 2 | Implement the meeting loss and test it | 1 to 2 | **Code ready** (method v1.1, 46 tests pass); diagnostic runs done (E-003, exploratory: no collapse) | All unit tests pass; no collapse in diagnostic runs; METHOD matches code |
| 3 | Main comparison (arms A to D) | 2 to 3 | Not started | H1 and H2 evaluated against pre-registered thresholds; gate decision recorded |
| 4 | Diagnostics and ablations | 3 to 4 | Not started | Components that matter identified; H3 evaluated |
| 5 | Generalization (CIFAR-10, modular arithmetic) | 4 to 6 | Not started | H4 evaluated on at least one setting beyond MNIST |
| 6 | Robustness and scale | 6 to 7 | Not started | Table of where the method helps, does not help, and hurts |
| 7 | Write-up and release | 7 to 10 | Not started | Paper claims all supported or reworded; code reproduces one headline result on a clean machine |

Full task lists and gates for each stage are in `docs/GUIDE.md`, Section 6.

## Reading sprint: DONE (2026-10-21)

All five papers were read in full; notes are in `docs/LITERATURE.md` and `docs/notes/dsn-reference-notes.md`. Outcome: the loss is the SDTP/DTP target, so the contribution was reframed (D-012), and arms, tuning protocol and method were updated (D-013, D-014; DR-002). Still to do before the `prereg-v1` lock: skim the first three "found by search" papers in `docs/LITERATURE.md` (Meulemans et al. 2020, the difference-reconstruction-loss work, fixed-weight DTP) and repeat a literature search.

## Limits to prevent forking paths

- Maximum pre-listed variants of the meeting loss per stage: **3** (V1 noise in the inverse loss, V2 augmented output, V3 frozen random G; D-013). B-code is a baseline-implementation variant and does not count. Log every variant tried, including failures.
- The pre-registration lock (git tag `prereg-v1`) happens **before** the Stage 3 main runs.

## What would make us stop or pivot

- The reading sprint shows the same auxiliary loss is already published in essentially the same form: reframe the contribution or pick the backup.
- Stage 3 gate outcomes (see `docs/GUIDE.md`, Section 6): no gain, or the gain is matched by arm B or C.
- Gradient alignment is near zero or negative after warm-up in Stage 2 or 3: diagnose before continuing.
- Compute overhead of the backward network cancels any gain at equal FLOPs.

**Designated backup plan:** the optimizer ideas 1 to 3 from `docs/reference/02-brain-vs-model-comparison.md`, Section 8 (surprise-gated learning rate, homeostatic update normalization, metaplastic per-parameter rates). Also a candidate for a companion study.

## Parked levers (from the proposal's lever map)

Optimizer, architecture (as primary focus), loss-function variants beyond the meeting loss, gradient computation (replacing backpropagation), and data. Revisit when a lead appears. Record leads in `docs/FINDINGS.md` as `tentative` or in the leads section of the experiment log.

## Status log

| Date | Note |
|---|---|
| 2026-10-21 | Repo created. Three guides added under `docs/`. D-001 to D-011 and DR-001 recorded. |
| 2026-10-21 | Stage 1 untuned baseline logged (E-001, 5 seeds). v1.1 patch: D-012 to D-016, DR-002, F-001 to F-003, method v1.1 (arms E, F, variants V1 to V3), 17 tests. |
| 2026-10-22 | v1.1 code reproduces E-001 exactly (E-002, verification). Short diagnostics of meet, meet_ao and meet_frozen (E-003, exploratory, F-004): no collapse, no gain at equal steps. Sweep selection rule agreed (D-017, DR-003); `scripts/run_sweep.py`, `scripts/analyze.py`, `src/sweep.py` and `tests/test_sweep.py` written (46 tests); R-18 added. No tuning sweep has been run yet. |
