# Drift Log

Every time the work moves away from what the plan, hypotheses, or paper say, record it here (see `docs/GUIDE.md`, Section 4 for the trigger list, procedure, and change-propagation matrix). Template: `docs/templates/drift-entry.md`. Never delete entries.

---

## DR-001: Phase 1 lever changed from optimizer/architecture to the meeting loss (loss and gradient-signal lever); timeline replaced
- Date: 2026-10-21
- Trigger: 9 (pivot to a different idea) and 4 (scope change), applied at project setup
- What changed: before: Phase 1 = optimizer and architecture levers, 12-week timeline (proposal, `reference/01`). After: Phase 1 = the meeting loss added next to backpropagation (loss/gradient-signal lever), 10-week staged timeline (`GUIDE.md`, `PLAN.md`). Optimizer ideas 1 to 3 from `reference/02` become the designated backup plan.
- Decision recorded: D-009, D-010, D-011
- Docs updated: [x] PLAN  [x] HYPOTHESES  [x] METHOD  [x] FINDINGS (none yet)  [x] LITERATURE  [x] RISKS  [x] CLAIMS  [x] PAPER_DRAFT  [x] FIGURES
- Other: status banner added to the top of `docs/reference/01-proposal-and-starting-guide.md` (content otherwise frozen); `GUIDE.md` changelog entry v1.1.
- Experiments invalidated or needing rerun: none (no experiments yet)

## DR-002: Reading sprint changed the contribution framing, arm set, and tuning protocol; exploratory/confirmatory rule added
- Date: 2026-10-21
- Trigger: 7 (related work overlaps the claim), 2 (method change), 3 (baselines and tuning protocol change), 8 (variants to explore)
- What changed: before: "meeting loss" presented as the idea, arms A to D, 20 trials per arm, constant learning rate, no noise-floor check. After: contribution reframed as a compute-matched study of a TP-style auxiliary target (D-012); arms A to F plus three pre-listed variants of D and one baseline-implementation variant of B (D-013); 25 trials, schedules, activation as hyperparameters, re-tuning noise floor, 10 final seeds (D-014); exploratory versus confirmatory labeling (D-015); DSN used as specification only (D-016).
- Decision recorded: D-012, D-013, D-014, D-015, D-016 (D-004 partly and D-005 superseded)
- Docs updated: [x] PLAN  [x] HYPOTHESES  [x] METHOD (v1.1)  [x] FINDINGS (F-001, F-002 from E-001)  [x] LITERATURE  [x] RISKS  [x] CLAIMS  [x] PAPER_DRAFT  [x] FIGURES (unchanged)  [x] GUIDE (v1.2)  [x] DOCUMENT_MAP and README (docs/notes/)
- Experiments invalidated or needing rerun: none. E-001 (untuned arm A) stays valid as baseline characterization; it will be superseded as the baseline by the tuned arm A from the D-014 protocol.

## DR-003: Tuning protocol made executable: selection score, sweep horizon, seed sets, tie-break
- Date: 2026-10-22
- Trigger: 3 (tuning protocol), as a refinement of D-014; no number in D-014 changed
- What changed: before: D-014 listed 25 trials, 2 selection seeds and the search ranges, but not the selection score, the run length, the seed sets or a tie-break. After: D-017 fixes them (mean FLOPs-to-0.975 over 2 seeds with misses scored at full budget, 3,000-step sweeps, selection seeds of at least 100, tie-break on late accuracy). The script also fills gaps in the ranges (D-017 point 6), provisionally.
- Decision recorded: D-017
- Docs updated: [x] PLAN  [ ] HYPOTHESES (no change)  [ ] METHOD (no change)  [x] DECISIONS  [x] EXPERIMENT_LOG (E-003 added; no comparison invalidated)  [x] FINDINGS (F-004)  [x] RISKS (R-10 count, R-18)  [ ] CLAIMS (no change)  [ ] PAPER_DRAFT (the setup section needs D-017 once results exist)
- Experiments invalidated or needing rerun: none. No sweep had been run before D-017.

## DR-004: Definition of "reaching a target" changed from first crossing to two consecutive evaluations; control F found nearly inert
- Date: 2026-10-22
- Trigger: 3 (tuning protocol, refinement of D-017), 6 (the F measurement contradicts the D-013 remark that F is informative under SGD); change-propagation row "Metric definition"
- What changed: before: a run reached a target at the first evaluation at or above it (D-017 point 1, `steps_to_target`, E-001 and F-002 step counts); control F was expected to be informative under SGD with momentum (D-013). After: two consecutive evaluations at or above the target, in selection, in `steps_to_target` and in the H1/H2/H5 metrics (D-018); control F is a negative control (F-005, R-19).
- Decision recorded: D-018
- Docs updated: [x] PLAN  [x] HYPOTHESES (H1 metric wording; before the lock, not post-hoc)  [ ] METHOD (no change: it already says an exact-gradient component only rescales the gradient; the measured scale is in F-005)  [x] DECISIONS  [x] EXPERIMENT_LOG (note on E-004; no run invalidated)  [x] FINDINGS (F-005, F-006)  [x] RISKS (R-18, R-19)  [ ] CLAIMS (none)  [ ] FIGURES (none)  [ ] PAPER_DRAFT (the setup section must state the rule once results exist)
- Experiments invalidated or needing rerun: none. Under the new rule the E-004 winners change for arm A (t19 to t09) and control E (t06 to t23); the other four are unchanged. The old selection files are marked superseded.

## DR-005: Stage 3 pre-registration finalized: three-way outcome rule, 97.5% intervals, 20 seeds for the decisive groups, new hypothesis H6, gating of H2 and H5
- Date: 2026-10-23
- Trigger: 1 (hypothesis wording and thresholds), 3 (final-seed counts of D-014; evaluation protocol), 8 (new hypothesis from an exploratory finding, F-007)
- What changed: before: H1 was "supported" at a 15% gain with a paired 95% interval excluding zero at 2 targets, and had no stated rule for a null result; 10 final seeds per group (D-014); no H6; H2 and H5 ungated. After (D-019): a three-way outcome (supported / refuted / inconclusive) with a 97.5% interval and a 15% margin at 0.97 and 0.975, both targets agreeing, and the gain exceeding the A re-tuning shift; 20 seeds for A, D, B and A2, 10 for the other groups; H6 (B vs A) added and labeled as generated from F-007; H2 and H5 gated on H1; V1 to V3, B-code, C, E, F and A2 descriptive; the E refinement pre-registered conditional on H1; F pre-registered as a negative control. All before any final seed was run, so none of it is post-hoc.
- Decision recorded: D-019 (D-003 superseded; D-007 and D-014 noted)
- Docs updated: [x] PLAN  [x] HYPOTHESES  [ ] METHOD (no change)  [x] DECISIONS  [x] EXPERIMENT_LOG (E-007)  [ ] FINDINGS (no new finding)  [x] RISKS (R-20)  [x] CLAIMS (C-01, C-02, C-06 wording; C-08 added)  [ ] FIGURES (none)  [ ] PAPER_DRAFT (the setup section must state D-018, D-019 and the origin of H6 once results exist)  [x] GUIDE (changelog 1.3, Sections 4 and 5, Stage 3 gate row)  [x] scripts/README  [x] CLAUDE.md (test count)
- Experiments invalidated or needing rerun: none. The tuning results are unchanged.

## DR-006: Status lines of the locked HYPOTHESES.md updated after the Stage 3 results; analyses not pre-registered
- Date: 2026-10-23
- Trigger: change to a file under the `prereg-v1` lock (`docs/HYPOTHESES.md`), made to record outcomes (not a change of any pre-registered hypothesis, threshold, rule or seed set); analyses added after seeing the results
- What changed: before: Status lines of H1 to H3, H5 and H6 read "open". After: H1 refuted, H6 refuted, H2 and H5 not tested (gated), H3 supported on the alignment threshold, each with the finding IDs. The statements, metrics, thresholds, gating and roles of D-019 are untouched (the diff against the tag is limited to Status and Linked-findings lines). Not pre-registered and therefore exploratory: comparisons of D and B against the second tuning of A (A2), the 0.98 analysis for B, the selection-to-final comparison, the Spearman correlation of alignment with speedup, and the two readings of "after warm-up" for H3 (the threshold was pre-registered, the warm-up length was not). The CLI also prints a verdict label for descriptive groups; they decide nothing (D-019).
- Decision recorded: none (D-019 stands; the gate decision will be D-020)
- Docs updated: [x] PLAN  [x] HYPOTHESES (status lines only)  [ ] METHOD (no change)  [ ] DECISIONS  [x] EXPERIMENT_LOG (E-008; note on E-007)  [x] FINDINGS (F-009 to F-011)  [x] RISKS (R-20 outcome, R-21)  [x] CLAIMS (status of C-01, C-03 and C-08 changed; C-02 and C-06 annotated as not tested; C-09 added)  [ ] FIGURES (none yet)  [ ] PAPER_DRAFT (not started for results)
- Experiments invalidated or needing rerun: none.

## DR-007: New arm `selective` added to the locked `src/` after `prereg-v1`; pilot E-010 run outside the Stage 3 plan
- Date: 2026-10-25
- Trigger: change to files under the `prereg-v1` lock (`src/flops.py`, `src/model.py`, `src/train.py`, new `src/selective.py`); GUIDE Section 4 trigger 8 (a variant or arm not in the plan) and, if the line is continued, trigger 9 (pivot)
- What changed: before: arms base, deepsup, ae, meet, meet_ao, meet_frozen and meet_grad; `flops_per_step` and `run` took `step * fps` as cumulative FLOPs. After: an eighth arm `selective` (hand-written backward on the highest-loss or random rows, `src/selective.py`); `flops_per_step` takes an optional `step` and handles the new arm; `run` logs cumulative FLOPs per step (identical to `step * fps` for every other arm because there is no warm-up phase for them). No Stage 3 hypothesis, threshold, rule, seed set or tuned config was changed. `scripts/run_final.py` would now refuse to start under its lock check (it also fails because of the status lines changed in DR-006); a later final run needs `--allow-changed` or a new tag.
- Decision recorded: D-021 (proposed; D-020 stays reserved for the gate / Stage 4 decision)
- Docs updated: [x] PLAN  [ ] HYPOTHESES (no hypothesis yet; a test needs a new one before `prereg-v2`)  [ ] METHOD (not changed: the new arm is described in `docs/notes/selective-backward-pilot.md`)  [x] FINDINGS (F-012)  [ ] LITERATURE (citations still to verify)  [ ] RISKS  [ ] CLAIMS (none)  [ ] PAPER_DRAFT  [ ] FIGURES
- Experiments invalidated or needing rerun: none. Evidence that the old arms are unchanged: the full test suite passes (87 tests, of which 66 predate the change); the pilot's arm A logged 465.2 GFLOPs over 3,000 steps, which equals 3,000 times the pre-existing analytic count of 155,058,176 per step. Not yet done: rerun one Stage 3 arm-A seed on the current code and compare its validation curve with its raw log (a verification entry, E-002 style). **Update 2026-10-25:** done, see E-011 (attempt 2 PASS, bit-identical; attempt 1 had failed because of a different torch thread count).

## DR-008: Paper reframed from a positive-claim study to a protocol, a negative result with its mechanism, and a conditional selective-backward result; claims register extended
- Date: 2026-10-25
- Trigger: GUIDE Section 4 trigger 9 (pivot) for the paper: the Stage 3 outcome (H1 and H6 refuted) removed the premise of the original draft skeleton
- What changed: before: `paper/PAPER_DRAFT.md` was a skeleton for a positive efficiency claim (claims C-01, C-02, C-04, C-06). After: one living draft whose headline does not depend on a positive result, `paper/PAPER_PLAN.md` (one paper and the split triggers), claims C-10 to C-25 added to `paper/CLAIMS.md` (existing rows C-01 to C-09 unchanged), figures FIG-07 to FIG-10 planned, bibliography entries added (each marked for verification). No experiment, hypothesis, threshold, seed set or tuned config was changed.
- Decision recorded: D-020 (proposed)
- Docs updated: [ ] PLAN  [ ] HYPOTHESES  [ ] METHOD  [ ] FINDINGS  [ ] LITERATURE (citations still to verify)  [ ] RISKS  [x] CLAIMS  [x] PAPER_DRAFT  [x] FIGURES
- Experiments invalidated or needing rerun: none.
## DR-009: Four further independent tunings of arm A proposed as the reference set for `prereg-v2` (proposed with D-022)
- Date: 2026-10-25
- Trigger: GUIDE Section 4 trigger 3 (baselines and tuning protocol: additional tuning passes), as a refinement of D-014 point 3 (re-tuning noise floor); not yet in force
- What changed: before: two tunings of arm A (t1, t2); one fresh tuning (t5) planned for `prereg-v2` (prep note, decision 2). After, if D-022 is accepted: four further tunings (t5 to t8) selected under the unchanged D-017 and D-018 rules, a noise-floor outcome fixed in advance, and a reference-baseline choice for H7 left OPEN. No number of D-014, D-017, D-018 or D-019 changes, no Stage 3 result or config is touched.
- Decision recorded: D-022 (proposed)
- Docs updated: [x] PLAN  [ ] HYPOTHESES (H7 is not written yet)  [ ] METHOD  [x] DECISIONS  [x] EXPERIMENT_LOG (E-011 attempt 2, verification)  [ ] FINDINGS  [x] RISKS (R-21 mitigation)  [ ] CLAIMS  [ ] PAPER_DRAFT  [ ] FIGURES  [x] DOCUMENT_MAP, `docs/notes/baseline-tunings-plan.md`, CLAUDE.md (test count)
- Experiments invalidated or needing rerun: none.

## DR-010: New arm `meet_rho` added to `src/model.py` for check G0 (dose-response); Stage 3 arms untouched
- Date: 2026-10-26
- Trigger: GUIDE Section 4 trigger on changes to locked code paths (D-019 locks `src/` as of `prereg-v1`); the change implements check G0 of `docs/notes/mechanism-dose-response.md`, part of the Stage 4 plan accepted as D-020
- What changed: before: arms A to F, V1 to V3, B-code and S. After: arm `meet_rho` is added to `ARMS`, with its own `G` in `MeetNet.__init__` and its own branch in `compute_losses` (placed before `meet_grad`); `src/train.py` logs alignment and `h_var` for it. No existing branch was edited, `MEET_ARMS` is not extended, and the FLOP counter is unchanged (it counts what autograd runs; a unit test checks the analytic overhead). New files: `exploratory/dose.py`, `scripts/run_dose_response.py`, `configs/dose/mr1_rho*.yaml`, `tests/test_dose.py`; `tests/test_variants.py` gets `rho` and `rho_warmup_steps` in its shared config so that the loops over `ARMS` cover the new arm. No hypothesis, threshold, seed set, tuned config or tag is changed.
- Decision recorded: D-020 (accepted)
- Docs updated: [x] PLAN  [ ] HYPOTHESES  [ ] METHOD (the arm is described in `docs/notes/mechanism-dose-response.md`)  [ ] FINDINGS (no result yet)  [ ] LITERATURE  [ ] RISKS  [ ] CLAIMS  [ ] PAPER_DRAFT  [ ] FIGURES  [x] `scripts/README.md`, `CLAUDE.md` (test count)
- Experiments invalidated or needing rerun: none. Evidence that the old arms are unchanged: the full suite passes (133 tests, 115 of which predate the change), and the new test shows that `meet_rho` with `rho = 0` reproduces arm A's validation and task-loss curves exactly on synthetic data. **Update 2026-10-26:** the regression check was run on the patched code (commit 2127dea) and passed bit for bit (E-011 attempt 3); the G0 run followed (E-013). **Follow-up checks G1 and G2 (2026-10-26):** new tags in `exploratory/dose.py` and new configs only; nothing under `src/` changes, so no further drift entry.

## DR-011: Design of `prereg-v2` fixed in D-023 (two-tier hypothesis, extra baseline tunings, a fully tuned small-batch arm, tuned S as a secondary)
- Date: 2026-10-26
- Trigger: GUIDE Section 4 triggers 2 and 3 (hypotheses and baselines, tuning protocol): a new confirmatory test with additional tuning passes and a tuning horizon (8,000 steps for batch 48) that differs from D-014's 3,000
- What changed: before: D-021 and D-022 proposed, with the reference baseline, the margin, the number of tunings and S's settings OPEN. After: D-023 accepted: H7a (S30 versus A1, 15%, D-019 rule with the shift rule) and H7b (S30 versus the fully tuned best small batch, margin 0%), K = 4 further tunings of A, a 25-trial tuning of batch 48 at 8,000 steps, S30 primary with A1's hyperparameters, S tuned once as a secondary, fractions 0.3 (primary) and 0.2. No number of D-014, D-017, D-018 or D-019 changes for the arms already tuned; the horizon of 8,000 steps applies only to the batch-48 sweep and its final runs, and equals 3,000 steps in samples. No Stage 3 result, tuned config, hypothesis or tag changes. `docs/HYPOTHESES.md` is not edited yet (locked by `prereg-v1`); the hypotheses go into it in the `prereg-v2` lock commit.
- Decision recorded: D-023 (accepted); D-021 and D-022 accepted with it
- Docs updated: [x] PLAN  [ ] HYPOTHESES (at the lock)  [ ] METHOD  [ ] FINDINGS  [ ] LITERATURE  [x] RISKS (R-21 unchanged: addressed by the design)  [ ] CLAIMS  [ ] PAPER_DRAFT  [ ] FIGURES  [x] DECISIONS, `docs/notes/selective-backward-prereg-prep.md`, `docs/notes/baseline-tunings-plan.md`
- Experiments invalidated or needing rerun: none. Code under `src/` is not changed by this entry; the code changes it requires will get their own drift line or entry when made.

- Update (2026-10-27, code for DR-011): changes under `src/`: `src/sweep.py` (sweep arms `base_b48` and `selective30`, per-arm horizon `ARM_STEPS`, `expected_steps`, `check_protocol(..., arms)`, `shift_value`), `src/final.py` (`plan_meta`, `load_plan` with a seed block, group horizons, `allow_missing`, new roles `reference`, `control`, `shift`), new `src/prereg2.py` (derived configs, `h7`, `noise_floor`). New files: `configs/prereg_v2_plan.yaml`, `configs/prereg_v2/*.yaml`, `scripts/analyze_prereg2.py`, `scripts/make_prereg2_configs.py`, `tests/test_prereg2.py`; `scripts/run_sweep.py` (default horizon per arm) and `scripts/run_final.py` (lock tag, output directory and locked paths from the plan) changed. No Stage 3 arm, loss, FLOP counter, config or result changes; the Stage 3 plan loads and runs as before (its tests are unchanged and pass: 180 tests in total, 39 of which are new since the track S looks). The sweeps already run (t1 to t8) use code paths whose behaviour is unchanged (their trial draws depend on (tag, arm) only). **Still to do by the researcher (needs MNIST):** `python scripts/verify_regression.py` on the patched code; log it as a verification entry.

## DR-012: Lock commit of `prereg-v2`: hypotheses H7a and H7b added to `docs/HYPOTHESES.md`, arm S and the new groups described in `docs/METHOD.md`
- Date: 2026-10-27
- Trigger: change to two files locked by `prereg-v1` (`docs/HYPOTHESES.md`, `docs/METHOD.md`), by addition only, before any seed of the `prereg-v2` plan; the same procedure as H6 (D-019, 2026-10-23)
- What changed: before: H1 to H6 and the Stage 3 method; after: H7a (S30 against A1, margin 15%, shift rule `max_gain`) and H7b (S30 against the tuned batch-48 arm, margin 0%) with the roles of the groups, and a section "v1.2 additions" in `docs/METHOD.md` describing arm S, its FLOP counting and the small-batch arm. The header comment of `configs/prereg_v2_plan.yaml` now says locked. No existing hypothesis, rule, threshold, config, seed or result of `prereg-v1` was changed; `src/` was not touched by this commit.
- Decision recorded: D-023 (design), `docs/notes/prereg-v2-plan.md` (shift rule `max_gain`, decided 2026-10-27)
- Docs updated: [x] PLAN  [x] HYPOTHESES  [x] METHOD  [ ] FINDINGS (no result yet)  [ ] LITERATURE (citations of selective backprop still to verify before any claim)  [ ] RISKS  [ ] CLAIMS  [ ] PAPER_DRAFT  [ ] FIGURES
- Experiments invalidated or needing rerun: none. Regression check of the code that will be locked: E-011 attempt 4 PASS (bit-identical); 180 tests pass.


## DR-013: Status lines of the locked HYPOTHESES.md updated after the `prereg-v2` results; claims C-21 and C-27 restated to the locked rules
- Date: 2026-10-27
- Trigger: change to a file under the `prereg-v2` lock (`docs/HYPOTHESES.md`), made to record outcomes (not a change of any pre-registered hypothesis, threshold, rule, shift rule, group or seed set); same procedure as DR-006
- What changed: before: the Status lines of H7a and H7b read "open". After: both read "supported" with the numbers and the finding IDs, and the Linked-findings lines name F-019 and E-022. The statements, metrics, margins, shift rule and roles of D-023 are untouched (the diff against the tag is limited to Status and Linked-findings lines). `paper/CLAIMS.md`: C-21 was written before the test as "against the best of the tuned baselines" and is restated to what H7a tests (S30 against A1, with the re-tuning shift); C-27 is new (H7b); C-13 and C-22 are annotated. Not pre-registered and therefore exploratory: the per-seed counts (20 of 20), the per-seed ratios, the independent recomputation with separate code, the comparison of S30 against the best further tuning of A on these seeds, and the 95% interval check.
- Decision recorded: none (D-023 stands)
- Docs updated: [x] PLAN  [x] HYPOTHESES (status lines only)  [ ] METHOD (no change)  [ ] DECISIONS  [x] EXPERIMENT_LOG (E-022)  [x] FINDINGS (F-019)  [ ] RISKS (R-21 outcome not yet written)  [x] CLAIMS (C-21, C-27; C-13, C-22 annotated)  [ ] LITERATURE (still to verify)  [ ] FIGURES  [ ] PAPER_DRAFT
- Experiments invalidated or needing rerun: none.

## DR-014: Additive change to the locked `src/` for the Improve track: Selective-Backprop arm, gradient-norm score and kept-fraction schedules
- Date: 2026-10-27
- Trigger: change to files under the `prereg-v2` lock (`src/model.py`, `src/flops.py`, `src/selective.py`, `src/train.py`, new `src/sb.py`), made under D-024 (additive only) for D-025; `scripts/run_final.py` on the `prereg-v2` plan will now report changed locked files, by design (D-024)
- What changed: before: arms base, deepsup, ae, meet family, meet_grad, meet_rho and selective; the selective step ranked rows by loss, had a fixed kept fraction and no row-count argument. After: an arm `sb` (new network identical to arm A's; `src/sb.py`, its own loop called from the first lines of `run`), `selective_step` takes optional `k`, `frac` and `stats` and the config keys `sel_score` and `sel_schedule` (osgd, linear), `flops_per_step` takes an optional `k`, and `run` prices each step by the rows actually kept when a schedule is set. Every default reproduces the old behaviour: no existing config sets the new keys. No existing arm, loss, FLOP count, config or result was changed.
- Decision recorded: D-024 (rules for `src/` changes), D-025 (accepted; arms and reference)
- Docs updated: [x] PLAN  [ ] HYPOTHESES (none yet; the plan is exploratory)  [ ] METHOD (the arms are described in `docs/notes/improve-reference-arms.md` and the headroom plan; `docs/METHOD.md` is locked and gets them at the lock commit of `prereg-v3`)  [ ] FINDINGS  [x] LITERATURE  [x] DECISIONS (D-025)  [ ] RISKS  [ ] CLAIMS  [x] ledger
- Experiments invalidated or needing rerun: none. Evidence that the old arms are unchanged: on synthetic data the logs (steps, FLOPs, validation accuracy, loss, k, alignment, mid-state variance) of 11 arm configurations (base, deepsup, ae, meet, meet_ao, meet_frozen, meet_grad, meet_rho, selective top with warm-up, selective random, selective with all rows) are identical before and after the change, compared exactly; the full test suite passes (210 tests, of which 180 predate the change); deliberately breaking three details of `src/sb.py` (an uncharged selection forward pass, a non-strict percentile, an update on the whole queue) makes the new tests fail. Still to do by the researcher: the real-MNIST regression check `python scripts/verify_regression.py` (E-011 attempt 5), which must pass before any headroom run is trusted.
- Update 2026-10-28: the real-MNIST regression check passed (E-011 attempt 5, bit-identical to the Stage 3 log of arm A seed 0); the 150 runs of E-024 were made on the checked code (commit 15e4277c).

## DR-015: Additive option `sel_scale` in `src/selective.py` for look 2 of the Improve track
- Date: 2026-10-28
- Trigger: change to a file under the `prereg-v2` lock (`src/selective.py`), made under D-024 (additive only) for the gradient-scale question of `docs/notes/improve-look2-plan.md`
- What changed: before: the kept rows' gradient was always divided by the batch size B. After: `sel_scale` ("B", the default, unchanged; or "k", divided by the number of kept rows, as in Ordered SGD); the two are equal when all rows are kept. One line of `selective_step` and its docstring; nothing else in `src/`. No existing config sets the key.
- Decision recorded: D-024 (rules for `src/` changes), D-025
- Docs updated: [x] PLAN  [ ] HYPOTHESES (none; look 2 is exploratory)  [ ] METHOD (locked; the option is described in the look 2 plan and gets into METHOD at the lock commit of `prereg-v3` if used)  [ ] FINDINGS  [ ] LITERATURE  [ ] DECISIONS  [ ] RISKS  [ ] CLAIMS  [x] ledger
- Experiments invalidated or needing rerun: none. Evidence that existing behaviour is unchanged: on synthetic data the logs of 16 configurations (the 11 of DR-014 plus selective with the gradient-norm score, the Ordered-SGD schedule, the linear schedule, and Selective-Backprop with and without priming) are identical before and after the change, compared exactly; the new tests pin `sel_scale` "k" to the autograd gradient of the mean over the kept rows and "B" to the sum over the kept rows divided by the batch size, and show that the two coincide when all rows are kept. Still to do by the researcher: `python scripts/verify_regression.py` on real MNIST (E-011 attempt 6) before the look 2 runs are trusted; the path of arm A does not use the changed line, so a pass is expected.
- Update 2026-10-29: the real-MNIST regression check passed (E-011 attempt 6, bit-identical to the Stage 3 log of arm A seed 0, run header `git_commit` a2d1499e); the 120 runs of E-025 were made on the checked code (all headers a2d1499e, `git_dirty` false). `sel_scale: k` was used in two groups (S20k, S20w0k) and showed no effect by rule R4; METHOD gets no description of it unless a later plan uses it.
