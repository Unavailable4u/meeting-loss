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
- Status: accepted; superseded in part by D-013 (arm set expanded with controls and variants)
- Context: To show the backward-derived target is specific (H2), we need other auxiliary losses of similar cost.
- Decision: A = baseline (AdamW); B = deep supervision (auxiliary head on the middle state); C = autoencoder reconstruction auxiliary; D = meeting loss.
- Alternatives considered: Baseline only (cannot support H2).
- Consequences: Code supports all four arms via the `arm` option; tuning budgets are equal across arms.
- Evidence: `docs/GUIDE.md`, Section 6 (Stage 3).

## D-005: Tuning protocol: equal trial budget per arm, selection on validation
- Date: 2026-10-21
- Status: superseded by D-014
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

## D-012: Contribution reframed: a compute-matched study of a target-propagation-style auxiliary loss, not a new loss
- Date: 2026-10-21
- Status: accepted
- Context: The reading sprint (all five papers read) showed that our target `b = h + G(e_y) - G(p)` is the SDTP/DTP target (Bartunov et al. 2018, the SDTP target for the penultimate layer; Lee et al. 2015, Eq. 15), applied at a middle layer with `G` inverting the whole top half. DSN and Ladder Networks show auxiliary losses next to backpropagation. None of the five measures compute (FLOPs) to a target under tuned baselines. Later work also overlaps (Meulemans et al. 2020 on DTP and Gauss-Newton targets; a later difference-reconstruction loss for more backprop-aligned targets; fixed-weight DTP). See `docs/LITERATURE.md`.
- Decision: The paper will **not** claim a new loss. The claim is a question and its evidence: does a TP-style target used as an auxiliary signal next to backpropagation improve compute-matched training efficiency against tuned baselines, when does it, and why (mechanism: gradient alignment, target diversity)? Controls address confounds (lower-layer learning-rate effect, DSN-style deep supervision, autoencoder auxiliary). A clean negative or mixed result with a diagnosis is an acceptable outcome.
- Alternatives considered: Claiming novelty for the loss itself (rejected: not defensible given SDTP/DTP).
- Consequences: Update the novelty statement (`docs/LITERATURE.md`), `paper/CLAIMS.md` (C-05, C-06, C-07), the Introduction and Related Work of the draft, hypotheses H2 and H5, and method v1.1. Risk R-01 is now confirmed overlap with a reframed mitigation.
- Evidence: Bartunov 2018 (SDTP, Table 1), Lee 2015 (Eq. 15, Theorems 1 and 2), DSN 2014, Ladder 2015, Schmidt 2021; DR-002.

## D-013: Arms, controls, and variants (supersedes D-004 in part)
- Date: 2026-10-21
- Status: accepted
- Context: H2 needs fair alternatives, and H5 needs controls for confounds. The reading sprint also showed that implementations differ between papers and code.
- Decision: Main comparison (Stage 3): **A** baseline; **B** DSN-style deep supervision (primary implementation "B-paper": cross-entropy companion head on the middle state with the companion weight decaying linearly to zero and an optional hinge threshold `aux_gamma`); **C** autoencoder-reconstruction auxiliary; **D** meeting loss (default configuration). Controls: **E** = arm A plus a learning-rate multiplier `low_lr_mult` on the lower layers; **F** = exact-gradient target (`meet_grad`, DTP-style target `h - eta*dL/dh`). Pre-listed variants of D (maximum 3, see `docs/PLAN.md`): **V1** noise in the inverse loss (`sigma_inv > 0`, as in DTP and SDTP); **V2** AO-D, augmented output: `G` also receives fixed random features `z` of `h` (Bartunov's AO-SDTP idea); **V3** frozen random `G` (no inverse loss, inspired by fixed-weight DTP). A secondary implementation of B, **B-code** (constant companion weight with a decaying learning rate on the companion head, following the DSN reference code), is logged as a baseline-implementation variant and does not count against the D-variant limit. Activation function `act` in {relu, tanh} is a categorical hyperparameter of every arm (Lee et al. found ReLU hurt TP while backprop prefers ReLU, so fixing one would be unfair).
- Alternatives considered: Only arms A to D (cannot rule out a pure learning-rate effect); running every variant (forking paths).
- Consequences: Code supports all arms via `arm` and config keys (`src/model.py`, `src/train.py`). Note: with AdamW, control F rescales the lower-layer gradient, which Adam largely normalizes away, so F is most informative under SGD with momentum (H4); under AdamW control E is the one that changes the effective step size.
- Evidence: `docs/notes/dsn-reference-notes.md`, `docs/LITERATURE.md`, Bartunov 2018 (AO-SDTP), Lee 2015 (tanh vs ReLU).
- Note (2026-10-22, D-018): the remark that control F is most informative under SGD with momentum does not hold at the D-014 loss weights: F's extra gradient is only 2*lam_meet*eta_hat/d times the task gradient (F-005). F is reported as a negative control.

## D-014: Tuning and evaluation protocol updates (supersedes D-005)
- Date: 2026-10-21
- Status: accepted
- Context: Crowded Valley (Schmidt et al. 2021) shows learning-rate schedules add several percent on median, tuning budget changes rankings, and re-tuning the same optimizer shifts results by about as much as swapping optimizers.
- Decision: (1) **25** random-search trials per arm (parity with Crowded Valley's small budget), 2 seeds per trial for selection on validation, then **10** fresh seeds for final results. (2) Search space for every arm: learning rate log-uniform in [1e-4, 1e-2], weight decay log-uniform in [1e-6, 1e-2], `act` in {relu, tanh}, learning-rate schedule in {constant, cosine}; arm-specific: `lam_meet` log-uniform in [1e-3, 10], `lam_inv` log-uniform in [0.1, 10], `warmup_steps` in {200, 500, 1000}, `lam_aux` log-uniform in [1e-2, 1], `aux_gamma` in {0, small}, `sigma_inv` in [0.01, 0.3], `low_lr_mult` log-uniform in [0.5, 8]. (3) **Re-tuning noise floor:** tune arm A twice with fully independent seeds; the gain of any arm must exceed the shift between the two tunings, and both are reported. (4) Wall-clock reported separately from FLOPs.
- Alternatives considered: Keeping 20 trials and a constant learning rate (weaker baselines, and Crowded Valley suggests gains could be an artifact).
- Consequences: `docs/GUIDE.md` Section 5 updated (v1.2); `scripts/run_sweep.py` to be written with these ranges; the search-space numbers are starting values and are finalized before the `prereg-v1` lock.
- Evidence: Schmidt et al. 2021 (Sections 2.3, 3, Appendix D); DR-002.

## D-015: Exploratory versus confirmatory findings
- Date: 2026-10-21
- Status: accepted
- Context: Exploring many variants and settings finds real discoveries but also noise. We expect to discover things we did not plan for and want to keep them without letting them leak into claims.
- Decision: Every entry in `EXPERIMENT_LOG.md` and `FINDINGS.md` carries a **Type**: `confirmatory` (tests a pre-registered hypothesis from the `prereg-v1` tag, with fresh seeds), `exploratory`, or `verification` (a deterministic check that tests no hypothesis, such as a FLOP count or a unit test). Paper claims may only rest on confirmatory or verification evidence. An exploratory discovery becomes a claim only after a pre-planned confirmatory run with fresh seeds, planned and logged before that run.
- Alternatives considered: No distinction (risks reporting noise as findings).
- Consequences: Templates updated (`docs/templates/`); `GUIDE.md` Section 4 gains a subsection; `CLAIMS.md` evidence entries must be confirmatory or verification.
- Evidence: `docs/GUIDE.md` Section 4 (pre-registration lock and forking paths).

## D-016: Third-party reference code (DSN) is used as a specification only; licensing noted
- Date: 2026-10-21
- Status: accepted
- Context: The DSN reference repository is an old Caffe fork. Its license permits educational, research, and non-profit use; commercial use requires contacting UC San Diego's technology transfer office. The paper's first page also notes a patent disclosure.
- Decision: Reimplement arm B in PyTorch from the paper and from `docs/notes/dsn-reference-notes.md`; do not copy the code. Keep the license note visible. Before any commercial use of DSN-style code or ideas, check with the university.
- Alternatives considered: Running or vendoring the Caffe code (impractical and carries license obligations).
- Consequences: Risk R-11 added.
- Evidence: DSN repository LICENSE and README (read 2026-10-21).

## D-017: Sweep selection rule, 3,000-step sweep horizon, seed sets and tie-break (refines D-014)
- Date: 2026-10-22
- Status: accepted (rule and horizon accepted by the researcher on 2026-10-22; items marked provisional in point 6 are confirmed before the `prereg-v1` lock). Refines D-014; supersedes nothing.
- Context: D-014 fixed 25 trials and 2 selection seeds per trial, but not the score that decides which trial wins, the run length, or the seed sets. Compute is the constraint (about 230 s per 3,000-step run on the laptop, from E-001).
- Decision: (1) The selection score of a trial is the mean, over its 2 selection seeds, of the FLOPs at the first evaluation where validation accuracy reaches 0.975. A run that never reaches 0.975 is scored with the FLOPs of its full run. Lower is better. This matches H1 and counts each arm's extra compute (F-003). (2) Every sweep run is 3,000 steps, evaluated every 50 steps. (3) Ties are broken by the higher mean validation accuracy over the last 5 evaluations, then by the lower trial index. (4) Selection seeds are at least 100, so they never overlap the final seeds 0 to 9, E-001 or the diagnostics. The first tuning pass uses seeds 1000 and 1001; the re-tuning noise-floor pass uses a different tag (so different sampled trials) and seeds 2000 and 2001. (5) Trials are sampled once per (tag, arm), depend only on the tag and the arm name, and are recorded in `results/raw/sweeps/<tag>/<arm>_trials.json`; a sweep refuses to resume if its trials or horizon changed. (6) Provisional values where D-014 left a gap: "small" `aux_gamma` is 0.05; `sigma_inv` is log-uniform in [0.01, 0.3]; control F searches `lam_meet` and `warmup_steps` only (`eta_hat` stays at 0.1, `lam_inv` is unused); B-code is swept as its own sweep arm (`deepsup_code`) and V1 as `meet_noise`.
- Alternatives considered: steps-to-target as the score (ignores the cost of the backward network); final accuracy at a fixed budget (kept as a secondary metric); smoothing the first crossing to resist a single lucky evaluation (more robust, not agreed; see R-18).
- Consequences: `src/sweep.py`, `scripts/run_sweep.py`, `scripts/analyze.py` and `tests/test_sweep.py` implement it. Cosine schedules depend on `max_steps`, so the final runs of a tuned arm should use the same horizon as the sweep; the horizon of the final runs is still to be decided. Selection is only valid when every arm has completed the same number of trials (the analysis script warns otherwise). Logged in DR-003.
- Evidence: D-014; H1; F-003; R-16, R-18.
- Note (2026-10-22, D-018): the definition of "reaches 0.975" in point 1 is superseded by D-018 (two consecutive evaluations). Points 2 to 6 stand; the provisional items of point 6 were accepted in D-018 point 4.

## D-018: "Reaching a target" means two consecutive evaluations at or above it; control F is read as a negative control (refines D-017 point 1, corrects a remark in D-013)
- Date: 2026-10-22
- Status: accepted by the researcher on 2026-10-22, before the `prereg-v1` lock and before any final run. Supersedes the first-crossing definition in D-017 point 1. The step counts in E-001 and F-002 are first-crossing numbers and stay valid as labelled.
- Context: D-017 scored a run at the first evaluation with validation accuracy of at least 0.975, and R-18 flagged that as noisy before any sweep ran. In the raw runs of E-004, 118 of the 182 runs that crossed 0.975 were back below it at the next evaluation (65%). Re-scoring all runs from the saved curves (`scripts/rescore_sensitivity.py`, F-006) shows the rule changes which trial wins for arm A (t19 to t09) and for control E (t06 to t23), and the order of A and B (B/A 1.03 under first crossing, 0.82 under two consecutive, 0.89 under a 3-evaluation mean). The standing of the meeting loss does not depend on the rule: D/A is 1.31, 1.19 and 1.25 in-sample. In the same review, control F's gradient scale was measured (F-005).
- Decision: (1) A run reaches a target at the first evaluation i with validation accuracy of at least the target at i and at i + 1; the reported step and FLOPs are those of evaluation i. A run whose only evaluation at or above the target is the last one has not reached it. A run that never reaches the target is scored with its full-run FLOPs (as in D-017). (2) One definition everywhere: the selection score (D-017), `steps_to_target` in `src/train.py`, and the steps-to-target and FLOPs-to-target metrics of H1, H2 and H5 (D-003 will cite this decision). Implemented in `src/crossing.py`. (3) The first-crossing rule and a 3-evaluation mean stay available (`--crossing`) for sensitivity checks only and are always reported next to the adopted rule; the adopted rule is not chosen again after final runs exist. (4) The other provisional items of D-017 point 6 stand as they are: final-run horizon 3,000 steps (the sweep horizon, because the cosine schedules were tuned for it), "small" `aux_gamma` 0.05, `sigma_inv` range, control F's `eta_hat` fixed at 0.1. (5) Control F is kept as it is and reported as a negative control (arm A plus 5.8% FLOPs overhead), not as an upper bound; it is not rerun. A rescaled F is a separate decision before the SGD experiments (R-19). (6) Outputs selected under the first-crossing rule are superseded: the unsuffixed `results/processed/sweep_t1_<arm>.csv` and `configs/tuned/t1_<arm>.yaml` (marked in their first line). New outputs carry the rule in their names (`_two`), and the stage of a tuned config includes it (`final-t1-base-two`).
- Alternatives considered: keeping first crossing (noisy; the A-versus-B order then depends on single lucky evaluations); a 3-evaluation mean (similar rankings, harder to explain); requiring three or more consecutive evaluations (not examined).
- Consequences: raw logs are unaffected (scores are computed from the saved curves). The rule was chosen after the t1 sweep results were seen. The reasons are noise-based and the same for every arm, and the re-scoring under all three rules is part of the record (F-006), but a reader can still object, so the paper's setup section must state it. Under the new rule the untuned arm A of E-001 needs on average 1120 and 2060 steps to reach 0.97 and 0.975 (first crossing: 960 and 1600) and misses 0.98 in 3 of 5 seeds (first crossing: 1 of 5), so 0.98 stays a secondary target (F-002). Logged in DR-004.
- Evidence: E-004; F-005; F-006; `results/processed/sensitivity_t1.csv`; R-18; R-19.
