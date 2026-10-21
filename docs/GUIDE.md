# Implementation Guide: A Bidirectional "Meeting Loss" for Faster Neural Network Training

**Guide version:** 1.0 (created 2026-10-21)
**Status:** starting point. If the plan changes, follow the drift protocol in Section 4 and add a line to the changelog at the bottom.

---

## 0. How to use this guide

**Audience.** You, plus any collaborator or AI coding assistant (for example Claude Code) that helps you build and write. Put this file in the project repo as `docs/GUIDE.md`. Appendix B has a ready-to-paste kickoff prompt for an AI assistant.

**Order of work.**

1. Create the living documents (Section 3) *before* writing any experiment code.
2. Record the starting decisions (Section 9).
3. Work through the stages (Section 6), one at a time, passing each gate.
4. Keep the paper draft (Section 7) alive from day one, so it is updated as evidence arrives instead of written at the end.

**The eight project rules.**

1. **Docs before code.** The living docs exist before the first experiment.
2. **One source of truth per kind of information.** Decisions live in `DECISIONS.md`, results in `EXPERIMENT_LOG.md`, conclusions in `FINDINGS.md`, claims in `paper/CLAIMS.md`. Other files link to these and never restate them.
3. **Never delete, always supersede.** Mark old entries `superseded by <ID>`. The history is how you detect and explain drift.
4. **Pre-register before you look.** Hypotheses, metrics, and success thresholds are written down and tagged in git *before* the main runs. Changes made after seeing results are labeled **post-hoc**.
5. **Every number in the paper traces to a logged run.** The path is: paper claim → `CLAIMS.md` → `FINDINGS.md` → `EXPERIMENT_LOG.md` → raw results → code commit.
6. **Negative and inconclusive results are recorded** in the same detail as positive ones.
7. **Docs change in the same commit as the work** that made them change.
8. **Run the drift check** (Section 4) at every stage gate and once a week.

---

## 1. The research idea on one page

**Problem.** Standard training sends the learning signal from the output back to every layer through one long chain (backpropagation). Lower layers receive credit only after it has passed through every layer above them.

**Idea.** Compute forward from the input to a middle layer, and compute *backward* from the known target (the label) to the same middle layer. The two middle states should agree, so their disagreement becomes an extra learning signal for the lower layers: a "meeting loss." This is the training-time version of solving an equation by simplifying both sides until they meet.

**Why the label alone is not enough.** A label (such as "7") carries much less information than an input (a particular drawing of a 7). So the backward path cannot produce a per-example target from the label alone. The design below uses the **difference target propagation** trick: learn a backward network `G` that approximately inverts the top half of the network, then build each example's target from the difference between what `G` does with the true label and with the model's current prediction.

**Scope of version 1.**

- In scope: an *auxiliary* meeting loss added on top of normal backpropagation; small models; MNIST, then CIFAR-10, then a modular-arithmetic task; efficiency measured as steps and FLOPs to a target quality.
- Out of scope for now: replacing backpropagation entirely, inference-time answer verification, and large-scale training.

### Hypotheses

| ID | Statement | Metric |
|---|---|---|
| **H1** (primary) | Adding the meeting loss reduces the steps and FLOPs needed to reach a target validation accuracy, compared with a tuned baseline at equal total compute. | FLOPs-to-target, steps-to-target |
| **H2** (specificity) | The gain is *not* matched by other auxiliary losses of similar cost at the same layer (deep supervision, autoencoder reconstruction). The backward-derived target is what matters. | Same metrics, arms A to D |
| **H3** (mechanism) | The meeting loss's gradient on the lower layers is positively aligned with the true task gradient, and larger alignment goes with larger speedup. | Cosine similarity, correlation with speedup |
| **H4** (generality) | The effect holds on CIFAR-10, on a modular-arithmetic task, and with more than one optimizer. | Same metrics per setting |

**Null result.** No difference, or a loss, once the backward network's cost is counted.

### Suggested success thresholds (decide and record these in D-003 before the main runs)

- H1 counts as **supported** if the mean FLOPs-to-target is at least 15% lower than the tuned baseline, the paired 95% bootstrap confidence interval across seeds excludes zero, and the result holds on at least 2 different target accuracies.
- H2 counts as **supported** if arm D beats arms B and C under the same criterion.
- H3 counts as **supported** if the median alignment is positive in at least 90% of evaluation points after warm-up.

These numbers are starting suggestions. Change them deliberately, before running, and log the reason.

---

## 2. Method specification (Design B)

Single source of truth for the method: copy this section into `docs/METHOD.md` and update it there when the method changes.

### Notation

- `x`: input, `y`: label, `e_y`: one-hot label vector.
- `F_low`: lower half of the network (input to the middle). `F_up`: upper half (middle to output logits).
- `G`: backward network, mapping from the output space (class-probability vectors) to the middle space.
- `sg(·)`: stop-gradient (treat as a constant).

### Equations

```
h = F_low(x)                       # forward middle state
z = F_up(h)                        # logits
p = softmax(z)                     # predicted class probabilities

L_task = CrossEntropy(z, y)

L_inv  = || G(sg(p)) − sg(h) ||²             # trains G to invert F_up
b      = sg(h) + G(e_y) − G(sg(p))           # per-example backward target for h (a constant)
L_meet = || h − b ||²                         # pulls h toward its backward target

L = L_task + λ_inv · L_inv + λ_meet(t) · L_meet
λ_meet(t) = λ_meet · min(1, t / T_warmup)     # ramp up while G is still poor
```

### Which loss trains which parameters

| Loss | Trains | Why |
|---|---|---|
| `L_task` | `F_low`, `F_up` | Normal backpropagation |
| `L_inv` | `G` only | `G` learns to approximately invert `F_up` |
| `L_meet` | `F_low` only (through `h`) | The target `b` is a constant, so `G` and `F_up` receive no gradient from it |

### Intuition

If `G` were a perfect inverse of `F_up`, then `G(e_y) − G(p)` is how much the middle state would need to change for the prediction `p` to become the true label `e_y`. The meeting loss moves `h` by exactly that amount. Its gradient on `h` is `−2·(G(e_y) − G(p))`, which you can compare with the ordinary backprop gradient on `h`. That comparison is the H3 diagnostic.

### Design choices to record as decisions

| Choice | Default | Alternatives to test later |
|---|---|---|
| Where to meet | One middle layer | Several depths at once |
| Input to `G` | Softmax probabilities | Logits, one-hot only |
| Distance | Mean squared error | Cosine distance, smooth L1 |
| `G` size | Same width as `F_up` | Smaller (cheaper), larger |
| Warm-up | 200 to 1000 steps | None, longer |
| Backprop kept for `L_task` | Yes (auxiliary use) | Weaker reliance on backprop (future work) |

### Known failure modes and how to detect them

- **Collapse:** the middle state shrinks toward a constant so the meeting loss reaches zero trivially. *Detect:* the variance of `h` across a batch falls toward zero while accuracy stalls.
- **Noisy targets early:** `G` is a poor inverse at the start, so targets mislead. *Detect:* negative or near-zero gradient alignment early. *Mitigate:* warm-up.
- **Extrapolation:** `G` is trained on softmax outputs but queried on one-hot vectors. *Detect:* `||G(e_y) − G(p)||` exploding. *Mitigate:* a smoothed one-hot target, or a smaller `λ_meet`.
- **Hidden cost:** `G` adds compute. *Detect:* always count FLOPs per step per arm.

### The H3 diagnostic (gradient alignment)

At regular points during training, compute on the same batch:

- `g_task`: gradient of `L_task` with respect to the `F_low` parameters
- `g_meet`: gradient of `L_meet` with respect to the same parameters

and log `cos(g_task, g_meet)`. A consistently positive value means the meeting signal agrees with the true gradient and can explain a speedup. Near zero or negative values explain a failure.

---

## 3. Repository layout and the living documents

### Layout

```
meeting-loss/
├── README.md
├── CLAUDE.md                 # standing instructions for AI assistants
├── docs/
│   ├── DOCUMENT_MAP.md       # how all documents connect (read first)
│   ├── GUIDE.md              # this guide, with its changelog
│   ├── PLAN.md               # stages, status, gates, next actions
│   ├── HYPOTHESES.md         # H-IDs, metrics, thresholds, status
│   ├── METHOD.md             # source of truth for the method (Section 2)
│   ├── DECISIONS.md          # decision log (D-IDs)
│   ├── EXPERIMENT_LOG.md     # chronological runs (E-IDs)
│   ├── FINDINGS.md           # curated conclusions (F-IDs)
│   ├── LITERATURE.md         # related work and overlap notes
│   ├── RISKS.md              # risk register (R-IDs)
│   ├── DRIFT_LOG.md          # every plan/paper change and what it touched
│   ├── reference/            # frozen background docs (proposal, brain comparison)
│   └── templates/            # blank copies of the templates below
├── paper/
│   ├── PAPER_DRAFT.md        # living draft (Section 7)
│   ├── CLAIMS.md             # claim → evidence map (C-IDs)
│   ├── FIGURES.md            # figure/table → script → data map
│   ├── references.bib
│   └── figures/
├── src/                      # model.py, losses.py, train.py, data.py, flops.py, diagnostics.py, utils.py
├── configs/                  # one YAML per experiment family
├── scripts/                  # run_sweep.py, analyze.py, make_figures.py
├── tests/
├── results/
│   ├── raw/                  # append-only: never edited by hand
│   └── processed/            # produced by scripts from raw
└── notebooks/                # exploration only; nothing cited from here
```

### Document purposes and update triggers

| Document | Purpose | Update when |
|---|---|---|
| `PLAN.md` | Current stage, exit criteria, next 3 actions, pivot conditions | Each stage starts or ends; weekly |
| `HYPOTHESES.md` | What you claim to test and how you will judge it | Before main runs (lock); status after analysis |
| `METHOD.md` | Exact method as implemented | Any change to losses, architecture, detach rules |
| `DECISIONS.md` | Why each choice was made | Every non-trivial choice, immediately |
| `EXPERIMENT_LOG.md` | What was run, with commit, config, seeds, results | After every run group |
| `FINDINGS.md` | What you now believe, with evidence and confidence | After analysis of any run group |
| `LITERATURE.md` | What exists, how it overlaps, what you do differently | When you read or find a paper |
| `RISKS.md` | What could sink the project and how you would notice | Weekly; when a risk changes |
| `DRIFT_LOG.md` | Each deviation and the docs it forced you to update | Whenever a drift trigger fires |
| `paper/*` | Draft, claim map, figure map | Whenever findings change |

### Templates (copy into `docs/templates/` and use for every entry)

**Decision entry (`DECISIONS.md`)**

```
## D-0XX: <short title>
- Date: YYYY-MM-DD
- Status: proposed | accepted | superseded by D-0YY
- Context: what forced this decision
- Decision: what was chosen
- Alternatives considered: ...
- Consequences: which docs, code, or experiments this affects
- Evidence: F-/E- IDs, papers
```

**Hypothesis entry (`HYPOTHESES.md`)**

```
## H<N>: <statement>
- Metric(s):
- Success threshold: <value>   (pre-registered on YYYY-MM-DD, git tag prereg-v1; post-hoc changes marked)
- Status: open | supported | refuted | inconclusive
- Linked findings: F-...
- Paper sections: ...
```

**Experiment entry (`EXPERIMENT_LOG.md`)**

```
## E-0XX: <title>
- Date / Stage / Hypothesis tested (H-ID):
- Git commit / config file(s) / seeds:
- Reproduce with: <exact command>
- Result summary: <numbers> (raw: results/raw/<path>)
- Valid: yes | invalidated by <bug or decision ID>
- Interpretation: one paragraph
- Follow-ups:
```

**Finding entry (`FINDINGS.md`)**

```
## F-0XX: <the finding, as one sentence>
- Status: tentative | supported | refuted | superseded by F-0YY
- Confidence: low | medium | high   (and why: seeds, effect size, replication)
- Evidence: E-..., figure path
- Scope: dataset, model, arm, optimizer
- Implications: hypotheses, claims, paper sections affected
- Found / last re-verified: YYYY-MM-DD
```

**Claims register (`paper/CLAIMS.md`)**

| ID | Claim (exact paper wording) | Type | Evidence (F-/E- IDs) | Status | Paper location | Last checked |
|---|---|---|---|---|---|---|
| C-01 | ... | empirical / theoretical / background | ... | planned / supported / weak / refuted | Sec. 5.1 | YYYY-MM-DD |

**Figure map (`paper/FIGURES.md`)**

| ID | What it shows | Script | Data source | Paper location | Status |
|---|---|---|---|---|---|

**Drift log entry (`DRIFT_LOG.md`)**

```
## DR-0XX: <what drifted>
- Date:
- Trigger: (see Section 4 trigger list)
- What changed: before → after
- Decision recorded: D-0XX
- Docs updated: [ ] PLAN  [ ] HYPOTHESES  [ ] METHOD  [ ] FINDINGS  [ ] LITERATURE  [ ] RISKS  [ ] CLAIMS  [ ] PAPER_DRAFT  [ ] FIGURES
- Experiments invalidated or needing rerun: E-...
```

---

## 4. The drift protocol

"Drift" means the real work has moved away from what the plan, hypotheses, or paper say. Drift is normal in research. Undocumented drift is what causes inconsistent papers.

### Triggers (any one means: stop and run the procedure)

1. A hypothesis, metric, or threshold wording changes.
2. The method changes (loss, architecture, detach rules, hyperparameter structure).
3. The baseline set or tuning protocol changes.
4. The dataset, model, or scale changes.
5. A bug is found that touches earlier results.
6. A result contradicts a hypothesis or an earlier finding.
7. You find related work that overlaps your claim.
8. You are tempted to explore a variant not in the plan.
9. You decide to pivot to a different idea.

### Procedure

1. **Open a drift entry** in `DRIFT_LOG.md`.
2. **Record the decision** in `DECISIONS.md` (new ID, link to the drift entry, mark superseded decisions).
3. **Update documents** using the matrix below, in this order: `HYPOTHESES` → `METHOD` → `PLAN` → `EXPERIMENT_LOG` (invalidations) → `FINDINGS` → `RISKS` → `LITERATURE` → `CLAIMS` → `FIGURES` → `PAPER_DRAFT`.
4. **Re-check the claims register:** every claim whose evidence changed gets a new status.
5. **Rerun** anything invalidated, or mark it clearly as not rerun.
6. **Commit** docs and code together with a message that names the drift ID.

### Change propagation matrix

| If this changes | Also update |
|---|---|
| Hypothesis wording or threshold | HYPOTHESES, DECISIONS, PLAN gates, CLAIMS, paper Intro and Results. Mark as post-hoc if results were already seen. |
| Method (loss, detach, architecture) | METHOD, DECISIONS, EXPERIMENT_LOG (mark old runs superseded), FINDINGS, paper Method. Rerun affected experiments. |
| Baselines or tuning protocol | DECISIONS, EXPERIMENT_LOG (mark invalid comparisons), FINDINGS, paper Experimental Setup. Rerun. |
| Dataset, model, or scale | PLAN, HYPOTHESES (scope), paper Setup and Limitations |
| Metric definition | DECISIONS, analysis scripts, FINDINGS, FIGURES, paper Results |
| New related work | LITERATURE, novelty note in DECISIONS, paper Related Work and Intro, possibly a pivot decision |
| Negative or surprising result | FINDINGS, RISKS, PLAN (branch), paper Limitations and Discussion |
| Bug found | DECISIONS (bug record), EXPERIMENT_LOG (invalidate), FINDINGS, rerun |
| Pivot to another idea | New HYPOTHESES set, archive (do not delete) old docs, rewrite paper title and abstract |

### Weekly review (30 minutes, same day each week)

- Does `PLAN.md` match what you actually did this week?
- Are all runs from this week in `EXPERIMENT_LOG.md`?
- Did any finding change status? If yes, are the claims and paper draft updated?
- Did you try anything not in the plan? If yes, is there a drift entry?
- Update `RISKS.md` and set the next three actions.

### Pre-registration lock

Before the Stage 3 main runs: commit `HYPOTHESES.md`, `METHOD.md`, and decisions D-003 to D-005, then create the git tag `prereg-v1`. Any later change to those files is recorded as post-hoc in the drift log.

### Guard against "forking paths"

Trying many variants until one works produces false positives. Rules: set a maximum number of variants per stage in `PLAN.md`; log every variant tried, including failures; report all of them in the paper's appendix.

### Exploratory versus confirmatory findings (added v1.2, D-015)

Every entry in `EXPERIMENT_LOG.md` and `FINDINGS.md` carries a **Type**:

- `confirmatory`: tests a hypothesis pre-registered under the `prereg-v1` tag, with fresh seeds.
- `exploratory`: anything else, including surprises found while exploring variants or settings. Record it; it may be real.
- `verification`: a deterministic check that tests no hypothesis (a FLOP count, a unit test).

Paper claims (`paper/CLAIMS.md`) may only rest on `confirmatory` or `verification` evidence. An exploratory discovery becomes a claim only after a confirmatory run with fresh seeds, planned and logged **before** that run. The templates in `docs/templates/` include the Type field and take precedence over the template copies in Section 3.

---

## 5. Environment and engineering standards

### Setup

- Python 3.10 or newer, PyTorch (a version that includes `torch.utils.flop_counter`, 2.1 or newer), torchvision, numpy, pandas, matplotlib, pyyaml, pytest. Optional: Weights & Biases for tracking.
- A GPU helps but is not required for MNIST. Free Colab or Kaggle GPUs cover the CIFAR-10 stage; use a university lab GPU if you have access.
- Initialize git on day one. Pin dependencies in `requirements.txt`.

### Reproducibility rules

- Seed everything (Python, NumPy, PyTorch, data-loader workers) and log the seed.
- Every run is launched from a YAML config. Log the config, its hash, and the git commit.
- Name runs `YYYYMMDD_stage_arm_seed`.
- Log at least: step, wall-clock time, FLOPs so far, training loss, validation accuracy, and (for arm D) `L_inv`, `L_meet`, gradient alignment, and the variance of `h`.
- Save raw logs as CSV or JSONL in `results/raw/`. Analysis scripts read raw files and write to `results/processed/`. Nothing is edited by hand.
- Use a fixed train/validation split from the training set (for example 55,000 / 5,000 for MNIST). The test set is touched **once**, at the very end.

### Fair-comparison rules

- **Equal tuning budget.** Every arm gets the same number of tuning trials. Current protocol (v1.2, D-014, supersedes the earlier 20-trial suggestion): 25 random trials per arm, drawing learning rate log-uniformly in [1e-4, 1e-2], weight decay in [1e-6, 1e-2], activation in {relu, tanh}, learning-rate schedule in {constant, cosine}, plus each arm's own loss weights (see D-014 for ranges). Use 2 seeds per trial for selection, then 10 fresh seeds for final results. Also run the **re-tuning noise floor**: tune arm A a second time with fully independent seeds, and require any gain to exceed the shift between the two tunings.
- **Equal compute.** Compare at equal FLOPs, counting the backward network and any auxiliary heads. Report wall-clock time separately.
- **Selection on validation only.**
- **Seeds.** At least 5 for early comparisons, and at least 10 for headline results (MNIST is cheap enough).
- **Statistics.** Report the median and spread of steps-to-target and FLOPs-to-target. Use paired bootstrap confidence intervals across seeds.
- **Targets.** Choose target accuracies that the baseline reaches in *every* seed (for example 97.0%, 97.5%, 98.0% validation accuracy for an MNIST MLP), so the metric is always defined. Record the choice in D-003.

### Measuring FLOPs

Use `torch.utils.flop_counter.FlopCounterMode` around one full training step (forward and backward, including the backward network and heads). Multiply by the step count. It counts mostly matrix multiplications and convolutions, which dominate here. Optimizer FLOPs are small and are noted but ignored. Check the counter on a tiny known case (a single linear layer) as part of the tests.

---

## 6. Stages

Timelines are rough and assume part-time work. Adjust them, and write the real dates in `PLAN.md`.

### Stage 0: Project setup and documents (days 1 to 2)

**Tasks**

- Create the repo and the layout in Section 3.
- Create every document from the templates, with the starting content below.
- Record the starting decisions D-001 to D-008 (Section 9).
- Write `HYPOTHESES.md` from Section 1 and `METHOD.md` from Section 2.
- Start `LITERATURE.md` with the papers in Appendix D (mark each as "not yet read in full").
- Create `paper/PAPER_DRAFT.md`, `CLAIMS.md`, and `FIGURES.md` from Section 7, with all claims status `planned`.
- Fill `RISKS.md` from Section 8.

**Exit criteria:** all documents exist and link to each other; D-001 to D-008 are recorded; `PLAN.md` lists the stages with dates.

### Stage 1: Baseline reproduction (week 1)

**Goal:** a trustworthy baseline you fully understand.

**Tasks**

- Implement data loading, the MLP, the training loop, evaluation, and logging (Appendix A).
- Train arm A (plain AdamW). Plot loss and accuracy curves.
- Run the tuning protocol for arm A. Measure variation across seeds.
- Measure steps-to-target and FLOPs-to-target for the baseline.

**Exit criteria:** MLP reaches about 98% on MNIST validation; curves are reproducible across seeds; the targets for Stage 3 are chosen and recorded (D-003); FLOP counter verified.

**Docs to update:** EXPERIMENT_LOG, FINDINGS (baseline numbers), DECISIONS, PLAN.

### Stage 2: Implement the meeting loss and test it (weeks 1 to 2)

**Tasks**

- Implement arm D (Section 2) and arms B and C (Appendix A).
- Write and pass the unit tests (below).
- Run small diagnostic runs (not for results): confirm `L_inv` falls, `h` variance does not collapse, alignment is logged.

**Required tests**

1. With `λ_inv = λ_meet = 0`, the gradients on `F_low` and `F_up` match arm A exactly.
2. The meeting-loss gradient reaches only `F_low`: `G` and `F_up` gradients are `None` after backpropagating `L_meet` alone.
3. `L_inv` decreases when `G` is trained alone on fixed `h` and `p`.
4. The target `b` carries no gradient.
5. The FLOP counter returns the known value for a single linear layer.
6. The run is bit-for-bit reproducible given the same seed (on CPU).

**Exit criteria:** all tests pass; no collapse in diagnostic runs; `METHOD.md` matches the code.

**Docs to update:** METHOD, EXPERIMENT_LOG, FINDINGS (diagnostic observations, labeled tentative), DECISIONS.

### Stage 3: Main comparison (weeks 2 to 3)

**Lock first:** commit and tag `prereg-v1` (Section 4).

**Arms**

| Arm | Description |
|---|---|
| A | Baseline: plain network, AdamW |
| B | Deep supervision: auxiliary classifier head on the middle state, weight `λ_aux` |
| C | Autoencoder auxiliary: decoder reconstructs `x` from the middle state, weight `λ_aux` |
| D | Meeting loss (Section 2; see `docs/METHOD.md` v1.1 for variants) |
| E | Control: arm A plus a lower-layer learning-rate multiplier (`low_lr_mult`), D-013 |
| F | Control: exact-gradient target (`meet_grad`), D-013 |

**Tasks**

- Run the equal-budget tuning protocol for all four arms.
- Run final seeds for each arm's best configuration.
- Compute steps-to-target, FLOPs-to-target, final accuracy at fixed FLOPs budgets, and wall-clock time.
- Plot validation accuracy against steps and against FLOPs, with confidence bands.
- Evaluate H1 and H2 against the pre-registered thresholds.

**Gate after Stage 3**

| Outcome | Action |
|---|---|
| Thresholds met for H1 and H2 | Proceed to Stage 4 |
| Some gain, but confidence interval crosses zero | Add seeds; check warm-up and `λ` ranges; at most 3 pre-listed variants (log all); re-evaluate |
| Gain matched by arm B or C (H2 fails) | The backward target is not special. Record it. Options: analyze why, test a variant that makes the backward path matter more, or reframe the paper around the finding |
| No gain or worse | Use the alignment diagnostic to explain it. Options: a variant design, switch to another idea from the idea bank in `brain-vs-model-learning-comparison.md`, or write a negative-result paper or workshop submission. Record the choice in DECISIONS |

**Docs to update:** EXPERIMENT_LOG, FINDINGS, HYPOTHESES (status), PLAN, CLAIMS, DRIFT_LOG if the plan changes.

### Stage 4: Diagnostics and ablations (weeks 3 to 4)

**Ablations (each one a logged experiment group)**

- No `L_inv` (G untrained).
- Frozen random `G` (does inverse learning matter?).
- Meeting loss with `G` replaced by the true gradient direction (an upper bound on usefulness).
- No warm-up.
- Meeting depth: earlier, later, and two depths at once.
- `G` capacity: smaller and larger.
- Alternative distances (cosine, smooth L1).
- Detach variants.

**Diagnostics (H3)**

- Gradient alignment over training, per layer.
- Correlation between per-seed alignment and per-seed speedup.
- Variance of `h`, norms of `G(e_y) − G(p)`.

**Exit criteria:** you can state which components matter and why, backed by ablation results; H3 evaluated.

**Docs to update:** FINDINGS, CLAIMS, FIGURES, PAPER_DRAFT (Analysis section).

### Stage 5: Generalization (weeks 4 to 6)

**5a. CIFAR-10 with a small CNN or ResNet.**
Split the network after a middle block. To keep `G` simple, meet in a globally pooled feature space (a vector per example) first. Run arms A to D with the same fairness rules.

**5b. Modular-arithmetic task with a small transformer.**
Generate examples such as `y = ((a + c1) × c2 − c3) mod p` with `p` prime and known nonzero constants, so each step is exactly invertible. This is the setting where a backward path can be exact. Add an extra baseline: **intermediate-step supervision** (predicting each intermediate value), because that is the closest "free" alternative to the backward path. Report honestly if it matches arm D.

**Exit criteria:** H4 evaluated on at least one setting beyond MNIST.

### Stage 6: Robustness and scale (weeks 6 to 7)

- Vary batch size, optimizer (AdamW and SGD with momentum), width, and depth.
- Run a scaling check: increase model size by 4 times and see whether the gain grows, holds, or shrinks.
- Report the failure cases as well.

**Exit criteria:** a table of settings where the method helps, does not help, and hurts.

### Stage 7: Write-up and release (weeks 7 to 10)

- Finish the paper draft (Section 7), with every claim at status `supported` or reworded to match the evidence.
- Run the final test-set evaluation once.
- Clean and document the code; include exact commands to reproduce every figure.
- Run a "fresh machine" check: a new environment reproduces one headline result.
- Post a preprint, then consider a workshop or conference submission (check current deadlines).

---

## 7. The paper draft system

The draft is a **living document**, kept consistent with the evidence by the claims register.

### Rules

1. No numeric or comparative claim appears in the draft without a claim ID like `[C-07]`.
2. Every claim ID has evidence IDs in `CLAIMS.md`.
3. When a finding changes status, check every claim that cites it (Section 4, step 4).
4. Each section carries a status header: `skeleton`, `draft`, or `stable`, and the date it was last checked against `FINDINGS.md`.
5. Write the stable parts early (Method, Experimental Setup). Write Results last.
6. State limitations as soon as you know them, and keep them in the draft.

### `paper/PAPER_DRAFT.md` skeleton

```
# <Working title: e.g. "Meeting in the Middle: A Bidirectional Target Loss for Faster Training">

Status: skeleton | Version: v0.1 | Last checked against FINDINGS: YYYY-MM-DD

## Abstract
Status: skeleton | Depends on: all sections
Problem (1 sentence) → idea (1) → key result with number [C-..] (1) → scope and limitation (1).

## 1. Introduction
Status: skeleton | Depends on: H1-H4, LITERATURE
- The cost of training and the credit-assignment path
- The idea in one paragraph
- Contributions (each one a claim ID): [C-01] ... [C-02] ...

## 2. Related work
Status: skeleton | Depends on: LITERATURE.md
- Target propagation and difference target propagation
- Auxiliary and deep-supervision losses
- Ladder networks, autoencoder and cycle-consistency objectives
- Predictive coding and local-learning rules
- Efficient optimizers
- How this work differs (must match novelty notes in DECISIONS)

## 3. Method
Status: skeleton | Depends on: METHOD.md
- Setup and notation
- The meeting loss and its three components
- Algorithm box
- Cost analysis (extra FLOPs, memory)

## 4. Experimental setup
Status: skeleton | Depends on: D-003..D-008
- Datasets and splits; models; arms A-D
- Tuning protocol and budgets; seeds; metrics (steps-to-target, FLOPs-to-target)
- Compute used

## 5. Results
Status: skeleton | Depends on: F-IDs from Stages 3, 5, 6
- 5.1 Main comparison [C-..]
- 5.2 Comparison with other auxiliary losses [C-..]
- 5.3 Generalization: CIFAR-10, modular arithmetic [C-..]
- 5.4 Robustness and scale [C-..]

## 6. Analysis
Status: skeleton | Depends on: F-IDs from Stage 4
- Ablations; gradient alignment (mechanism); failure cases

## 7. Limitations
Status: skeleton
- Scale; tasks tested; added compute; cases where it hurts; unresolved questions

## 8. Conclusion
Status: skeleton

## References
## Appendix
- A. All hyperparameters and search ranges
- B. All variants tried (including failures)
- C. Extra plots
- D. Reproduction commands and code link
```

### Writing order (suggested)

Method and Setup (stable early) → Related Work (as you read) → Analysis and Results (after each stage) → Introduction and Abstract (last) → Limitations (continuously).

### Reporting checklist (check before submitting)

- [ ] Baselines tuned with the same budget; budgets reported
- [ ] Seeds and spread reported; confidence intervals shown
- [ ] FLOPs including the backward network; wall-clock reported separately
- [ ] All variants tried are listed in the appendix
- [ ] Negative results and failure cases included
- [ ] Test set used once
- [ ] Code and configs released; reproduction commands tested on a clean machine
- [ ] Related work checked within the last 2 weeks before submission
- [ ] Every claim in the paper has a supported entry in `CLAIMS.md`

---

## 8. Initial risk register (copy into `docs/RISKS.md`)

| ID | Risk | Detection | Mitigation |
|---|---|---|---|
| R-01 | Prior work overlaps (difference target propagation, ladder networks, deep supervision) | Literature scan; reviewer-style novelty check | Pin the contribution to the auxiliary use next to backprop, and to specific variants (surprise-weighted, multi-depth); state the overlap openly |
| R-02 | Extra compute from `G` cancels any speedup | FLOPs-to-target comparison | Smaller `G`; share layers; compare at equal FLOPs |
| R-03 | Collapse or noisy targets | Variance of `h`; alignment early in training | Warm-up, detach rules, smaller `λ` |
| R-04 | Gains disappear against properly tuned baselines | Equal-budget tuning | Report honestly; reframe as negative result |
| R-05 | Forking paths (too many variants) | Variant count in PLAN | Pre-listed variants only; report all |
| R-06 | Works only at small scale | Stage 6 scaling check | State the limit; do not generalize |
| R-07 | FLOP accounting error | Unit test on a known layer; cross-check with wall-clock | Fix and rerun |
| R-08 | Time overrun | Weekly review vs PLAN | Cut stages 5b or 6 first; keep the core comparison rigorous |
| R-09 | Classification labels give too little information for useful targets | Alignment near zero in Stage 3 | Move to tasks where the result determines intermediate steps (Stage 5b) |

---

## 9. Starting decisions to record in Stage 0

Write each as a full entry with the decision template. Suggested initial content:

| ID | Decision | Notes |
|---|---|---|
| D-001 | Implementation in PyTorch; Python 3.10 or newer | Reproducibility and FLOP counter support |
| D-002 | Version 1 uses the meeting loss as an auxiliary signal next to backprop (Design B) | Replacing backprop is a separate, harder project |
| D-003 | Metrics and thresholds: steps-to-target and FLOPs-to-target, target accuracies, success criteria for H1 to H3 | Fill in from Section 1 and Stage 1 results; lock before Stage 3 |
| D-004 | Arms A to D as the main comparison | B and C test whether the backward target is specific |
| D-005 | Tuning protocol: equal trial budget per arm, selection on validation | Section 5 |
| D-006 | Stage order: MNIST, then CIFAR-10, then modular arithmetic | Increase difficulty only after the gate |
| D-007 | Seeds: at least 5 for exploration, at least 10 for headline results | Cheap at MNIST scale |
| D-008 | Data split: fixed train/validation split; test set used once at the end | Prevents tuning on test |

---

## Appendix A: Starter code (PyTorch)

**Important:** this code was written for this guide and has **not been run**. Treat it as a starting skeleton. Stage 2's unit tests exist to catch exactly the mistakes untested code can contain, so run them before trusting any result.

### `src/model.py` and `src/losses.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeetNet(nn.Module):
    """MLP split at a middle layer, with extra modules per experimental arm.

    arm: "base" (A), "deepsup" (B), "ae" (C), "meet" (D)
    """

    def __init__(self, d_in=784, d_hid=256, d_mid=128, n_cls=10, arm="meet"):
        super().__init__()
        self.arm, self.n_cls = arm, n_cls
        self.low = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_mid))
        self.up = nn.Sequential(nn.ReLU(), nn.Linear(d_mid, d_hid), nn.ReLU(), nn.Linear(d_hid, n_cls))
        if arm == "meet":
            self.G = nn.Sequential(nn.Linear(n_cls, d_hid), nn.ReLU(), nn.Linear(d_hid, d_mid))
        elif arm == "deepsup":
            self.aux = nn.Linear(d_mid, n_cls)
        elif arm == "ae":
            self.dec = nn.Sequential(nn.Linear(d_mid, d_hid), nn.ReLU(), nn.Linear(d_hid, d_in))

    def forward(self, x):
        return self.up(self.low(x))


def compute_losses(model, x, y, cfg, step):
    """Returns (total_loss, dict_of_parts, h, z). cfg keys: lam_inv, lam_meet,
    warmup_steps, lam_aux."""
    h = model.low(x)
    z = model.up(h)
    parts = {"task": F.cross_entropy(z, y)}
    total = parts["task"]

    if model.arm == "meet":
        h_d = h.detach()
        p = z.detach().softmax(-1)
        e_y = F.one_hot(y, model.n_cls).float()
        gp = model.G(p)                              # trained via L_inv
        L_inv = F.mse_loss(gp, h_d)
        with torch.no_grad():                        # the target is a constant
            b = h_d + model.G(e_y) - gp.detach()
        L_meet = F.mse_loss(h, b)                    # gradient reaches F_low only
        ramp = min(1.0, step / max(1, cfg["warmup_steps"]))
        total = total + cfg["lam_inv"] * L_inv + cfg["lam_meet"] * ramp * L_meet
        parts.update(inv=L_inv, meet=L_meet)

    elif model.arm == "deepsup":
        L_aux = F.cross_entropy(model.aux(h), y)
        total = total + cfg["lam_aux"] * L_aux
        parts["aux"] = L_aux

    elif model.arm == "ae":
        L_aux = F.mse_loss(model.dec(h), x)
        total = total + cfg["lam_aux"] * L_aux
        parts["aux"] = L_aux

    return total, parts, h, z
```

### `src/diagnostics.py` (H3: gradient alignment, and collapse check)

```python
def grad_alignment(model, x, y):
    """Cosine similarity between the task gradient and the meeting gradient
    on the lower layers (arm D only)."""
    params = list(model.low.parameters())
    h = model.low(x)
    z = model.up(h)
    L_task = F.cross_entropy(z, y)
    p = z.detach().softmax(-1)
    with torch.no_grad():
        e_y = F.one_hot(y, model.n_cls).float()
        b = h.detach() + model.G(e_y) - model.G(p)
    L_meet = F.mse_loss(h, b)
    g_task = torch.autograd.grad(L_task, params, retain_graph=True)
    g_meet = torch.autograd.grad(L_meet, params)
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
    return F.cosine_similarity(flat(g_task), flat(g_meet), dim=0).item()


@torch.no_grad()
def mid_state_variance(model, x):
    """Mean per-feature variance of h across the batch (collapse indicator)."""
    return model.low(x).var(dim=0).mean().item()
```

### `src/flops.py`

```python
from torch.utils.flop_counter import FlopCounterMode


def flops_per_step(model, opt, batch, cfg):
    """FLOPs of one full training step (forward + backward) for this arm,
    including G or auxiliary heads. Optimizer FLOPs are ignored (small)."""
    x, y = batch
    opt.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        loss, _, _, _ = compute_losses(model, x, y, cfg, step=10**9)
        loss.backward()
    opt.zero_grad(set_to_none=True)
    return fc.get_total_flops()
```

### `src/train.py` (outline)

```python
import torch, time, json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_data(batch_size, seed, root="data"):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                              transforms.Lambda(lambda t: t.view(-1))])
    full = datasets.MNIST(root, train=True, download=True, transform=tfm)
    g = torch.Generator().manual_seed(0)              # FIXED split across all runs (D-008)
    train, val = random_split(full, [55000, 5000], generator=g)
    return (DataLoader(train, batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed), drop_last=True),
            DataLoader(val, 1000))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(-1).cpu()
        correct += (pred == y).sum().item()
        n += y.numel()
    model.train()
    return correct / n


def run(cfg, device="cpu"):
    torch.manual_seed(cfg["seed"])
    train_loader, val_loader = get_data(cfg["batch_size"], cfg["seed"])
    model = MeetNet(arm=cfg["arm"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    fps = flops_per_step(model, opt, next(iter(train_loader)), cfg)

    log, step, t0 = [], 0, time.time()
    while step < cfg["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss, parts, _, _ = compute_losses(model, x, y, cfg, step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % cfg["eval_every"] == 0:
                row = dict(step=step, flops=step * fps, time=time.time() - t0,
                           val_acc=evaluate(model, val_loader, device),
                           **{k: v.item() for k, v in parts.items()})
                if cfg["arm"] == "meet":
                    row["align"] = grad_alignment(model, x, y)
                    row["h_var"] = mid_state_variance(model, x)
                log.append(row)
            if step >= cfg["max_steps"]:
                break
    return log  # write to results/raw/<run_id>.jsonl


def steps_to_target(log, target):
    for row in log:
        if row["val_acc"] >= target:
            return row["step"], row["flops"]
    return None  # target not reached: record it, never drop the run
```

Suggested starting config: batch size 128, `max_steps` 6000, `eval_every` 50, `lam_inv` 1.0, `lam_meet` 0.1, `warmup_steps` 500, `lam_aux` 0.3 (all of these are starting points for the tuning protocol, not final values).

### `tests/test_meeting_loss.py` (starting tests)

```python
import torch
from src.model import MeetNet, compute_losses

CFG0 = dict(lam_inv=0.0, lam_meet=0.0, warmup_steps=1, lam_aux=0.0)
CFG = dict(lam_inv=1.0, lam_meet=0.1, warmup_steps=1, lam_aux=0.0)


def batch(n=32):
    torch.manual_seed(0)
    return torch.randn(n, 784), torch.randint(0, 10, (n,))


def test_zero_lambda_matches_baseline():
    m = MeetNet(arm="meet")
    base = MeetNet(arm="base")
    base.low.load_state_dict(m.low.state_dict())
    base.up.load_state_dict(m.up.state_dict())
    x, y = batch()
    for net in (m, base):
        loss, *_ = compute_losses(net, x, y, CFG0, step=100)
        loss.backward()
    for a, b in zip(list(m.low.parameters()) + list(m.up.parameters()),
                    list(base.low.parameters()) + list(base.up.parameters())):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_meet_gradient_reaches_only_low():
    m = MeetNet(arm="meet")
    x, y = batch()
    _, parts, _, _ = compute_losses(m, x, y, CFG, step=100)
    parts["meet"].backward()
    assert all(p.grad is not None for p in m.low.parameters())
    assert all(p.grad is None for p in m.up.parameters())
    assert all(p.grad is None for p in m.G.parameters())


def test_inverse_loss_decreases_when_G_trained_alone():
    m = MeetNet(arm="meet")
    x, y = batch(256)
    opt = torch.optim.Adam(m.G.parameters(), lr=1e-2)
    first = last = None
    for i in range(200):
        _, parts, _, _ = compute_losses(m, x, y, CFG, step=100)
        opt.zero_grad()
        parts["inv"].backward()
        opt.step()
        first = parts["inv"].item() if first is None else first
        last = parts["inv"].item()
    assert last < first
```

Add the remaining tests from Stage 2 (target carries no gradient, FLOP counter on a single linear layer, bitwise reproducibility on CPU) yourself, and keep them in the repo.

---

## Appendix B: Kickoff prompt for an AI coding assistant

Paste this at the start of a session, with this guide in the repo:

```
You are my research engineering assistant. Read docs/GUIDE.md fully before doing anything.

Rules:
1. Follow Section 0 (the eight project rules) and Section 4 (the drift protocol) at all times.
2. Start with Stage 0: create the repo layout and every living document (PLAN, HYPOTHESES,
   METHOD, DECISIONS, EXPERIMENT_LOG, FINDINGS, LITERATURE, RISKS, DRIFT_LOG, and the paper/
   files) from the templates, and record decisions D-001 to D-008. Do not write experiment
   code until these exist.
3. Work one stage at a time. At the end of each stage, update the docs, run the drift check,
   and give me a short summary: what was done, what was found, what changed in the plan.
4. After every experiment batch, update EXPERIMENT_LOG; after every analysis, update FINDINGS
   and CLAIMS and, if a claim's evidence changed, the paper draft.
5. Never delete anything; mark entries superseded.
6. Ask me before changing a hypothesis, a threshold, the baselines, or the method.
   Anything you change after results are seen must be labeled post-hoc.
7. The starter code in Appendix A is untested. Run the tests first and fix problems before
   trusting results.
8. Be honest: report negative and inconclusive results as plainly as positive ones, and tell
   me when something looks too good to be true.
```

---

## Appendix C: Drift-check checklist (run at every stage gate and weekly)

- [ ] `PLAN.md` matches reality (stage, dates, next actions)
- [ ] Every run since the last check is in `EXPERIMENT_LOG.md` with commit, config, seeds, and result
- [ ] Every finding has evidence IDs, scope, and confidence
- [ ] Hypothesis statuses match the findings
- [ ] `METHOD.md` matches the code on the main branch
- [ ] Every claim in `CLAIMS.md` whose evidence changed has an updated status
- [ ] Every number in `PAPER_DRAFT.md` has a claim ID, and every claim ID exists
- [ ] Section status headers in the draft are current
- [ ] Anything attempted outside the plan has a drift entry and a decision entry
- [ ] `LITERATURE.md` has been searched for new overlapping work since the last check
- [ ] `RISKS.md` is updated

---

## Appendix D: Starting reading list for this project

Titles, years, and venues below are from memory and may contain small errors. Verify each one when you add it to `LITERATURE.md`, and search for newer related work.

**Target propagation and backward signals**

- Bengio, "How auto-encoders could provide credit assignment in deep networks via target propagation" (2014)
- Lee, Zhang, Fischer and Bengio, "Difference Target Propagation" (2015)
- Meulemans et al., "A Theoretical Framework for Target Propagation" (2020)
- Bartunov et al., "Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures" (2018). Shows target-propagation-style methods struggling at larger scale.
- Lillicrap et al., "Random synaptic feedback weights support error backpropagation for deep learning" (2016)

**Auxiliary and bidirectional objectives**

- Lee et al., "Deeply-Supervised Nets" (2015). Your arm B baseline.
- Rasmus et al., "Semi-Supervised Learning with Ladder Networks" (2015)
- Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (CycleGAN, 2017)
- Hinton, Dayan, Frey and Neal, "The wake-sleep algorithm for unsupervised neural networks" (1995)

**Collapse prevention**

- Grill et al., BYOL (2020); Chen and He, SimSiam (2021); Bardes et al., VICReg (2022)

**Reversible computation (a different use of "backward")**

- Gomez et al., "The Reversible Residual Network: Backpropagation Without Storing Activations" (2017)

**LLM backward reasoning (inference-time relatives)**

- Weng et al., "Large Language Models are Better Reasoners with Self-Verification" (2022)
- Berglund et al., "The Reversal Curse" (2023)
- Chen et al., "Reverse Thinking Makes LLMs Stronger Reasoners" (2024)

**Evaluation rigor**

- Schmidt, Schneider and Hennig, "Descending through a Crowded Valley: Benchmarking Deep Learning Optimizers" (2021)

---

## Changelog

| Version | Date | Change | Decision / drift ID |
|---|---|---|---|
| 1.0 | 2026-10-21 | Initial guide | none |
| 1.1 | 2026-10-21 | Repo created. Layout gains `CLAUDE.md`, `docs/DOCUMENT_MAP.md`, `docs/reference/`. Starter code placed in `src/` and `tests/`. Method now maintained in `docs/METHOD.md` (Section 2 is the v1.0 snapshot). | D-009, D-010, D-011, DR-001 |
| 1.2 | 2026-10-21 | Reading sprint folded in: contribution reframed, arms E and F, variants, 25-trial protocol with schedules and noise floor, exploratory versus confirmatory rule, `docs/notes/`. See `docs/METHOD.md` v1.1. | D-012 to D-016, DR-002 |
