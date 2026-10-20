# Decision Log

Newest decisions are added at the bottom. Never delete an entry: mark it `superseded by D-0XX`. Bugs are logged here too, as decisions titled "BUG: ...". Template: `docs/templates/decision.md`.

---

## D-001: Implement in PyTorch, Python 3.10 or newer
- Date: 2026-10-21
- Status: accepted
- Context: Need a mainstream framework with custom-gradient control and a FLOP counter.
- Decision: PyTorch 2.1 or newer (for `torch.utils.flop_counter`), Python 3.10 or newer.
- Alternatives considered: JAX (good for research, but a steeper start for this project).
- Consequences: `requirements.txt` pins versions; FLOP counting uses `FlopCounterMode`.
- Evidence: `docs/GUIDE.md`, Section 5.

## D-002: Version 1 uses the meeting loss as an auxiliary signal next to backpropagation (Design B)
- Date: 2026-10-21
- Status: accepted
- Context: Replacing backpropagation entirely is a separate, much harder research problem, and alternatives to backprop rarely match it.
- Decision: Keep backprop for the task loss and add the meeting loss as an extra signal for the lower layers.
- Alternatives considered: Weakening reliance on backprop (future work); local-only learning.
- Consequences: `docs/METHOD.md` describes Design B only. Replacing backprop is parked.
- Evidence: `docs/GUIDE.md`, Sections 1 and 2.

## D-003: Metrics and success thresholds (PROPOSED, to be finalized after Stage 1)
- Date: 2026-10-21
- Status: proposed (finalize and lock before Stage 3; tag `prereg-v1`)
- Context: Thresholds must be fixed before seeing main results to avoid fooling ourselves.
- Decision (proposed): Headline metrics are FLOPs-to-target and steps-to-target. Target accuracies are chosen so the tuned baseline reaches them in every seed (starting guess for an MNIST MLP: 97.0%, 97.5%, 98.0% validation accuracy). Success thresholds are as listed in `docs/HYPOTHESES.md`.
- Alternatives considered: Final accuracy at a fixed budget only (kept as a secondary metric).
- Consequences: Stage 1 results decide the final target values. After locking, any change is post-hoc.
- Evidence: `docs/GUIDE.md`, Sections 1 and 5.

## D-004: Four arms for the main comparison (A to D)
- Date: 2026-10-21
- Status: accepted
- Context: To show the backward-derived target is specific (H2), we need other auxiliary losses of similar cost.
- Decision: A = baseline (AdamW); B = deep supervision (auxiliary head on the middle state); C = autoencoder reconstruction auxiliary; D = meeting loss.
- Alternatives considered: Baseline only (cannot support H2).
- Consequences: Code supports all four arms via the `arm` option; tuning budgets are equal across arms.
- Evidence: `docs/GUIDE.md`, Section 6 (Stage 3).

## D-005: Tuning protocol: equal trial budget per arm, selection on validation
- Date: 2026-10-21
- Status: accepted
- Context: Untuned baselines are the most common reason claimed improvements fail to replicate.
- Decision: 20 random trials per arm (learning rate log-uniform in [1e-4, 1e-2], weight decay log-uniform in [1e-6, 1e-2], plus each arm's own loss weights), 2 seeds per trial for selection on validation, then fresh seeds for final results.
- Alternatives considered: Fixed hyperparameters for all arms (unfair to the baseline).
- Consequences: Sweeps must be logged in `EXPERIMENT_LOG.md`; the number of trials is reported in the paper.
- Evidence: `docs/GUIDE.md`, Section 5.

## D-006: Stage order: MNIST, then CIFAR-10, then modular arithmetic
- Date: 2026-10-21
- Status: accepted
- Context: Small experiments iterate fast. Difficulty should increase only after passing a gate.
- Decision: Follow the stage order in `docs/GUIDE.md`, Section 6.
- Alternatives considered: Starting with a transformer task (slower iteration, harder debugging).
- Consequences: See `docs/PLAN.md`.
- Evidence: `docs/GUIDE.md`.

## D-007: Seeds: at least 5 for exploration, at least 10 for headline results
- Date: 2026-10-21
- Status: accepted
- Context: Training is noisy; MNIST-scale runs are cheap.
- Decision: 5 seeds minimum for early comparisons, 10 or more for headline numbers, with paired bootstrap confidence intervals.
- Alternatives considered: 3 seeds (too little power for a 15% effect).
- Consequences: Budget compute for 10 seeds per arm at Stage 3.
- Evidence: `docs/GUIDE.md`, Section 5.

## D-008: Data split: fixed train/validation split from the training set; test set used once
- Date: 2026-10-21
- Status: accepted
- Context: Prevents tuning on test data.
- Decision: MNIST split 55,000 train / 5,000 validation with a fixed generator seed shared by all runs. The test set is evaluated exactly once, at the end of Stage 7.
- Alternatives considered: Cross-validation (unnecessary at this scale).
- Consequences: Analysis scripts must never read test results before Stage 7.
- Evidence: `docs/GUIDE.md`, Section 5.

## D-009: Phase 1 focus changes from optimizer/architecture to the loss and gradient-signal lever (meeting loss)
- Date: 2026-10-21
- Status: accepted
- Context: The original proposal planned to start with the optimizer and architecture levers. A new idea (forward calculation plus backward calculation from the expected result, meeting in the middle) was developed and chosen for the **training** setting (as a learning signal), not inference-time verification. It is a loss and gradient-signal idea, with a small extra backward network.
- Decision: Phase 1 is the meeting loss (Design B). The optimizer ideas 1 to 3 from `docs/reference/02-brain-vs-model-comparison.md`, Section 8 (surprise-gated learning rate, homeostatic update normalization, metaplastic per-parameter rates) become the **designated backup plan** and a possible companion study. The other levers stay in the backlog.
- Alternatives considered: Keeping the optimizer-first plan (rejected: the meeting-loss idea was the one the researcher wanted to pursue, and it has a clear, testable mechanism).
- Consequences: Proposal Phase 1 text is historical. `docs/PLAN.md`, `docs/HYPOTHESES.md`, and `docs/METHOD.md` describe the meeting loss. Logged in `DR-001`.
- Evidence: Project discussion on 2026-10-20 and 2026-10-21; `docs/GUIDE.md`, Section 1.

## D-010: Use the 10-week timeline in the guide instead of the proposal's 12-week timeline
- Date: 2026-10-21
- Status: accepted
- Context: The proposal's timeline was written for the optimizer-first plan.
- Decision: Follow the stage timeline in `docs/PLAN.md` (rough weeks; set real dates). Cut Stage 5b (modular arithmetic) or Stage 6 first if time runs short.
- Alternatives considered: Keeping 12 weeks (no reason now).
- Consequences: Supersedes the timeline table in the proposal.
- Evidence: `docs/GUIDE.md`, Section 6; risk R-08.

## D-011: Repository conventions: document hierarchy, frozen reference docs, precedence, assistant instructions
- Date: 2026-10-21
- Status: accepted
- Context: The project must be understandable in later sessions and by other people or assistants without relying on memory of conversations.
- Decision: (1) The three guides are stored in the repo: `docs/GUIDE.md` (authoritative execution guide) and `docs/reference/01-...` and `02-...` (background, frozen). (2) `docs/DOCUMENT_MAP.md` explains how all documents connect and defines precedence: DECISIONS, then HYPOTHESES/METHOD, then PLAN, then GUIDE, then reference. (3) `CLAUDE.md` holds standing instructions for AI assistants. (4) Repo working name: `meeting-loss`.
- Alternatives considered: Keeping the guides outside the repo (they would drift away from the project record).
- Consequences: Reference docs get a status banner but their content is not edited. `GUIDE.md` changes go through its changelog.
- Evidence: `docs/DOCUMENT_MAP.md`.
