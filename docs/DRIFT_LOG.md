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
