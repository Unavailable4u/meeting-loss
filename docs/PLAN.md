# Plan

**Last updated:** 2026-10-29 (look 2 run and analysed: E-025, F-023, F-024; regression check E-011 attempt 6 PASS; before that 2026-10-28: look 2 built and tested, DR-015, plan fixed; headroom study done: E-024, F-021, F-022, regression check E-011 attempt 5 PASS; before that 2026-10-27: Improve arms implemented, DR-014, headroom plan fixed, D-025 accepted; literature read for the Improve track, D-025 proposed, reference-arm spec; Improve look 1: S20 against S30, E-023, F-020; program of three tracks on `main`, D-024, component ledger; `prereg-v2` final runs done: H7a and H7b supported, E-022, F-019, DR-013; prereg-v2 machinery built, DR-011; E-019; track S looks done, E-016 to E-018; design of `prereg-v2` accepted, D-023, DR-011; checks G0, G1, G2 done: E-013 to E-015, F-014, F-015; DR-010; checks A and L done, E-012, F-013; D-020 accepted; Step 0 regression check passed, E-011 attempt 2; D-022 and DR-009 proposed; selective-backward pilot E-010, F-012, DR-007, D-021 proposed; before that: Stage 3 final runs E-008; F-009 to F-011; D-019, DR-005, DR-006; 2026-10-27: t9 and t10 logged (E-020, E-021), regression check E-011 attempt 4 PASS, `prereg-v2` lock commit prepared: H7a, H7b, DR-012)
**Project start (week 1):** 2026-10-21. Set real calendar dates for each stage below once you know your availability. Weeks are counted from the start date and are rough (part-time pace).
**Current stage:** Stage 3 finished; the gate decision is open (see below)

## Current focus

Stage 3 is done (E-008, confirmatory). **H1 is refuted:** arm D needs about 36% more FLOPs than tuned arm A (97.5% intervals +26% to +47% and +27% to +44% at 0.97 and 0.975). **H6 is refuted:** deep supervision equals A at those targets (+2%). H2 and H5 were not tested (gated on H1). Gradient alignment of D is high (+0.75, H3 threshold met), so the null is not explained by a misaligned target. The gate row "No gain or worse" of `docs/GUIDE.md` Section 6 applies.

**Side line (exploratory, 2026-10-25):** the selective backward pass (arm S) pilot, E-010, met its GO rule: S30 reached 0.97 and 0.975 at 0.63 and 0.61 of tuned A's FLOPs (3 seeds, random control worse than A; F-012). This is a lead, not a claim. D-021 (proposed) lists what must be decided before a pre-registered test; it does not replace the open gate decision D-020 below. **Update (2026-10-25):** Step 0 of the selective-backward preparation is settled: the regression check passed bit for bit under the Stage 3 thread setting (E-011 attempt 2; attempt 1 failed only because of the thread count), so the other runs of that note are no longer blocked; every new run uses one torch thread per process. D-022 (proposed, with the plan in `docs/notes/baseline-tunings-plan.md`) would run four further independent tunings of arm A (t5 to t8) to measure the baseline's tuning noise and fix the reference set for `prereg-v2`; its open items (number of tunings, the reference rule, whether B and S are tuned as often) are the researcher's.

## Next three actions

One branch (`main`), three tracks (D-024, `tracks/README.md`). Earlier text of this list, including the closed items about `prereg-v2` and checks G0 to G2, is in the git history; the state it described is in E-013 to E-015 and E-022.

1. **Paper track:** (a) wall-clock secondary on the E-022 final runs at one and eight threads (reported secondary of D-023); (b) verify the related work (Jiang et al. 2019, online hard example mining, importance sampling, online batch selection) and write the entries in `docs/LITERATURE.md` before any novelty sentence; (c) decide the second setting (Fashion-MNIST MLP with its own tuned baseline, C-24) as a separate pre-registered plan; (d) figures (`make_figures.py`) and the paper draft in line with the claims register.
2. **Mechanism track:** check D (can the meeting signal replace part of backpropagation at lower cost?), then check B (oracle targets), with check C alongside; Gate 3 stays closed (`docs/notes/mechanism-overview.md`). Each needs its plan note and a drift entry for any change to `src/`.
3. **Improve track:** the headroom study (E-024; F-021, F-022) and look 2 (E-025; F-023, F-024) are done. Look 2 replicated both leads on new seeds (S20w0 against S20 -13.8% / -12.0%; S15 against S30 -12.2% / -10.1%), found that the two do not add (S15w0, S10w0: no gain), that the gradient scale does not matter, and that the warm-up dose is not graded (0, 5, 10 steps alike, 25 not); no mechanism is supported. About half of the S20w0 saving against S20 is the saved cost of the warm-up's full steps (F-023). Next: (b) a tuning of Selective-Backprop (D-014 protocol, 25 trials) so that a pre-registered comparison has a fair reference; (c) the researcher decides the content of `prereg-v3` on fresh seeds (2300 to 2399): which of the leads S20w0 and S15 (rule R6) go in and as how many hypotheses, whether warm-up 0 and 10 are both arms, the margin (the interval bound nearest zero for S20w0 against S30 at 0.975 was -6.5% in look 2 and -16.3% in look 1, and the arithmetic share of the effect does not depend on learning), and the reference set (A1, further tunings of A, the tuned batch-48 arm, tuned Selective-Backprop). The assistant writes hypotheses and margins into `docs/HYPOTHESES.md` and a plan file only after those decisions. Selective-Backprop and both schedules are not leads under FLOPs.

## Stage table

| Stage | Name | Target weeks | Status | Exit criteria (short) |
|---|---|---|---|---|
| 0 | Project setup and documents | 1 | **Done** (2026-10-21) | All living docs exist; decisions recorded; repo pushed |
| 1 | Baseline reproduction and tuning | 1 | **In progress** (E-001 to E-006 done: all tuning passes complete; D-019 written, the lock tag pending) | About 98% MNIST validation accuracy; reproducible across seeds; targets chosen (D-003); FLOP counter verified |
| 2 | Implement the meeting loss and test it | 1 to 2 | **Code ready** (method v1.1, 46 tests pass); diagnostic runs done (E-003, exploratory: no collapse) | All unit tests pass; no collapse in diagnostic runs; METHOD matches code |
| 3 | Main comparison (arms A to D) | 2 to 3 | **Done** (2026-10-23, E-008): H1 and H6 refuted, H2 and H5 not tested; gate decision pending | H1 and H2 evaluated against pre-registered thresholds; gate decision recorded |
| 4 | Diagnostics and ablations | 3 to 4 | **In progress** (checks A, L, G0 to G2 done; B, C, D open; Gate 3 closed) | Components that matter identified; H3 evaluated |
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
| 2026-10-27 | Tunings t9 (batch 48, 8,000 steps; 207.1 GFLOPs in-sample) and t10 (S tuned once; 129.6) done and logged (E-020, E-021); regression check of the machinery PASS (E-011 attempt 4); 180 tests pass. Lock commit prepared: H7a and H7b in `docs/HYPOTHESES.md`, arm S in `docs/METHOD.md`, DR-012. The tag `prereg-v2` is created by the researcher on that commit; then the 220 final runs on seeds 1300 to 1319. |
| 2026-10-27 | `prereg-v2` final runs finished (220 runs, seeds 1300 to 1319, tag cc2830c, E-022): **H7a supported** (S30 against A1 -35.1% at 0.97 and -38.4% at 0.975; shift 5.5% and 3.9%) and **H7b supported** (S30 against the tuned batch-48 arm -30.2% and -35.5%); random control worse than A1; noise floor LARGE. F-019 and claims C-21, C-27 updated. Open: wall-clock secondary on the final runs, literature check, second setting (C-24), paper. |
| 2026-10-27 | Program decided (D-024): evaluation paper, mechanism checks B, D, C, and a method to beat Selective-Backprop, all on `main` with folders and lock tags instead of branches. Component ledger written (`docs/notes/component-ledger.md`); `tracks/README.md` added. No run, no `src/` change. |
| 2026-10-27 | Improve look 1 (E-023, exploratory, no new runs): S20 against S30 on the `prereg-v2` logs, -7.9% / -7.2% FLOPs at 0.97 / 0.975 with the same steps; rule outcome CONSTANT FRACTION HEADROOM, schedule lead not met (F-020). Warm-up is 11% to 12% of the cost. Component ledger updated. |
| 2026-10-27 | Literature for the Improve track read (Selective-Backprop paper and code, Ordered SGD, Katharopoulos and Fleuret, Loshchilov and Hutter, OHEM): the top-k rule and a shrinking kept fraction are prior art; added to `docs/LITERATURE.md`, novelty statement draft for Improve; D-025 proposed (reference arms and what may be claimed); spec `docs/notes/improve-reference-arms.md`; ledger section E. No run, no `src/` change. |
| 2026-10-27 | D-025 accepted. Selective-Backprop arm (`src/sb.py`, released algorithm, both FLOP counts), gradient-norm score and two kept-fraction schedules added to `src/` (additive, DR-014); 11 existing arm configurations give bit-identical logs on synthetic data; 210 tests pass in the sandbox. Headroom plan fixed (`docs/notes/improve-headroom-plan.md`: 15 groups, seeds 2100 to 2109, reading rules R1 to R6) with its analysis script. Real-MNIST regression check and the runs are the researcher's next steps. |
| 2026-10-28 | Real-MNIST regression check PASS (E-011 attempt 5). Headroom study finished (E-024, exploratory, 150 runs, seeds 2100 to 2109): S30 replicates (-35.7% / -35.3% against A1); constant fractions 0.1 to 0.2 are 6% to 13% cheaper than 0.3; S20 without warm-up is the cheapest arm (-22.7% / -23.9% against S30, 10 of 10 seeds; a lead to replicate); Selective-Backprop as released needs 44% to 84% more FLOPs than S30 (F-021); both schedules worse (F-022). Leads by the pre-fixed rule R6: S15 and S20w0. |
| 2026-10-28 | Look 2 of the Improve track built (exploratory plan fixed before any run): replication of the no-warm-up and kept-fraction leads, combinations (S15w0, S10w0), warm-up dose (5, 10, 25) and a gradient-scale test (`sel_scale`, DR-015); 12 groups, seeds 2110 to 2119, reading rules R1 to R6 and the predictions of two mechanism hypotheses written down in advance. No run yet. |
| 2026-10-29 | Real-MNIST regression check PASS (E-011 attempt 6). Look 2 finished (E-025, exploratory, 120 runs, seeds 2110 to 2119, 4,542 s): R1 both leads REPLICATED (S20w0 against S20 -13.8% / -12.0%, shrunk from look 1; S15 against S30 -12.2% / -10.1%), R2 combinations NO GAIN, R3 warm-up dose MIXED (w5 and w10 as good as w0, w25 not), R4 gradient scale NO CLEAR EFFECT; leads for `prereg-v3`: S20w0 and S15 (F-023, F-024). Extra look: 6.87 of the 12.05 and 14.64 GFLOPs that S20w0 saves against S20 is the saved warm-up cost. Neither mechanism hypothesis is supported. Ledger and DR-015 updated; no claim changes. Open: tuned Selective-Backprop, content and margins of `prereg-v3`. |
