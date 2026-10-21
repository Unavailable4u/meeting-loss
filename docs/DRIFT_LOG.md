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
