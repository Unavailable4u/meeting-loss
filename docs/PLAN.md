# Plan

**Last updated:** 2026-10-25 (checks A and L done, E-012, F-013; D-020 accepted; Step 0 regression check passed, E-011 attempt 2; D-022 and DR-009 proposed; selective-backward pilot E-010, F-012, DR-007, D-021 proposed; before that: Stage 3 final runs E-008; F-009 to F-011; D-019, DR-005, DR-006)
**Project start (week 1):** 2026-10-21. Set real calendar dates for each stage below once you know your availability. Weeks are counted from the start date and are rough (part-time pace).
**Current stage:** Stage 3 finished; the gate decision is open (see below)

## Current focus

Stage 3 is done (E-008, confirmatory). **H1 is refuted:** arm D needs about 36% more FLOPs than tuned arm A (97.5% intervals +26% to +47% and +27% to +44% at 0.97 and 0.975). **H6 is refuted:** deep supervision equals A at those targets (+2%). H2 and H5 were not tested (gated on H1). Gradient alignment of D is high (+0.75, H3 threshold met), so the null is not explained by a misaligned target. The gate row "No gain or worse" of `docs/GUIDE.md` Section 6 applies.

**Side line (exploratory, 2026-10-25):** the selective backward pass (arm S) pilot, E-010, met its GO rule: S30 reached 0.97 and 0.975 at 0.63 and 0.61 of tuned A's FLOPs (3 seeds, random control worse than A; F-012). This is a lead, not a claim. D-021 (proposed) lists what must be decided before a pre-registered test; it does not replace the open gate decision D-020 below. **Update (2026-10-25):** Step 0 of the selective-backward preparation is settled: the regression check passed bit for bit under the Stage 3 thread setting (E-011 attempt 2; attempt 1 failed only because of the thread count), so the other runs of that note are no longer blocked; every new run uses one torch thread per process. D-022 (proposed, with the plan in `docs/notes/baseline-tunings-plan.md`) would run four further independent tunings of arm A (t5 to t8) to measure the baseline's tuning noise and fix the reference set for `prereg-v2`; its open items (number of tunings, the reference rule, whether B and S are tuned as often) are the researcher's.

## Next three actions

1. **Check G0** (fixed-hyperparameter dose-response of the learned loss, `docs/notes/mechanism-dose-response.md`): needs the `meet_rho` arm and `scripts/run_dose_response.py`, a change under `src/` that needs a drift entry stating that no Stage 3 arm changes, and the regression check (`scripts/verify_regression.py`) afterwards. D-020 was accepted on 2026-10-25, so the order of `mechanism-overview.md` applies: G0, then the three looks of track S (wall-clock, small batch, `sel_frac` curve; not while a sweep is running), then checks B and D.
2. **Decide D-022** (further tunings of arm A, t5 to t8; open items: number of tunings, the reference baseline of H7, whether B and S are tuned as often) and D-021's open items before `prereg-v2` is written.
3. Keep the figures and the paper draft in line with the evidence: `make_figures.py` is still to be written; C-15 and C-16 now rest on E-012 (exploratory, status weak). Checks A and L ended Q1 NOT REDUNDANT (probe caveat), Q2 NO, Q3 GN DIFFERS, L ALREADY SELECTIVE, so the hybrid arm is not added to `prereg-v2`.

## Stage table

| Stage | Name | Target weeks | Status | Exit criteria (short) |
|---|---|---|---|---|
| 0 | Project setup and documents | 1 | **Done** (2026-10-21) | All living docs exist; decisions recorded; repo pushed |
| 1 | Baseline reproduction and tuning | 1 | **In progress** (E-001 to E-006 done: all tuning passes complete; D-019 written, the lock tag pending) | About 98% MNIST validation accuracy; reproducible across seeds; targets chosen (D-003); FLOP counter verified |
| 2 | Implement the meeting loss and test it | 1 to 2 | **Code ready** (method v1.1, 46 tests pass); diagnostic runs done (E-003, exploratory: no collapse) | All unit tests pass; no collapse in diagnostic runs; METHOD matches code |
| 3 | Main comparison (arms A to D) | 2 to 3 | **Done** (2026-10-23, E-008): H1 and H6 refuted, H2 and H5 not tested; gate decision pending | H1 and H2 evaluated against pre-registered thresholds; gate decision recorded |
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
| 2026-10-22 | First tuning pass t1 for arms A to F (E-004, 300 runs, exploratory). Under the adopted crossing rule (D-018, DR-004) winners in GFLOPs to 0.975: A 232.6, B 191.2, F 229.7, D 276.3, E 275.2, C 418.0; D is behind A on every rule (F-006). Control F measured as nearly inert (F-005, R-19). `src/crossing.py`, `scripts/rescore_sensitivity.py`, 55 tests. |
| 2026-10-22 | Re-tuning pass t2 for arm A (E-005, exploratory): same winning score as t1 under the adopted rule (232.6 GFLOPs, 0.0% shift) with very different hyperparameters (F-007); B's t1 winner is 18% below both. Variant sweeps (200 runs) finished; analysis pending. |
| 2026-10-22 | Variant sweeps analyzed (E-006, exploratory, F-008): no meeting-loss variant beats tuned A on FLOPs beyond noise; V3 ties A with about zero alignment; B-paper is ahead of B-code. All tuning passes complete; D-003 and the `prereg-v1` lock are next. |
| 2026-10-23 | D-019 written (final D-003, DR-005): targets, seeds (20 for A, D, B, A2; 10 for the rest), three-way outcome rule with 97.5% intervals for H1 and new H6, H2 and H5 gated on H1, E refinement pre-registered, F a negative control. Lock-guarded runner `scripts/run_final.py` and `configs/stage3_plan.yaml` added. Noise and power: E-007. |
| 2026-10-23 | Tag `prereg-v1` set on 3378d36; Stage 3 final runs finished (150 runs, about 2.75 h) and analyzed as pre-registered (E-008): H1 refuted (D +36% FLOPs), H6 refuted (B +2%), H2 and H5 not tested, H3 alignment threshold met. F-009 to F-011; R-21. Gate decision is next. |
| 2026-10-25 | Arm S (selective backward pass) added to `src/` (DR-007) and piloted on seeds 600 to 602 (E-010, exploratory, rule written before the run): verdict GO, S30 about 0.61 to 0.63 of A's FLOPs-to-target, random control R30 worse than A (F-012). 87 tests pass. D-021 proposed; four design items open (S tuning, baseline, seeds and threshold, second setting). D-020 (gate) still open. |
