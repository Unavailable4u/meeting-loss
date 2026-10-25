# Mechanism study (Stage 4 diagnostics): overview, gates and how it fits the two studies

**Status:** plan, written BEFORE any run. Type: `exploratory` for every check (D-015). Do not edit a decision rule after seeing results; add a dated note below instead. Revised after the pilot of the selective backward pass (E-010): the project is now organised as one paper with two studies and a shared protocol.

## The two studies and what links them

| Study | Status | Role |
|---|---|---|
| 1. Meeting loss (arm D) | Done: H1 and H6 refuted (E-008, pre-registered, confirmatory) | The negative control: it shows the protocol can say no. The mechanism checks below explain *why*. |
| 2. Selective backward (arm S) | Pilot GO (E-010, F-012); a pre-registered test is still to be written | A conditional positive result, credited to Jiang et al. 2019; the contribution is the rigorous, compute-matched test. |
| Link: check L | Plan only | Asks whether the meeting gradient lives on the same rows that selective backward keeps. |
| Shared protocol | In place | Paired seeds, the two-consecutive crossing rule (D-018), pre-registration, two baseline tunings (R-21, F-011), exact FLOP counting, no seed-adding after "inconclusive". |

Three claims, of different kinds, if the plans go well: (1) a reproducible protocol with evidence that it catches false positives; (2) a negative result with a mechanism; (3) a conditional positive result. Novelty stays modest, so aim for a negative-results, reproducibility or workshop venue.

## Notes in this study

| Note | What it does | Training |
|---|---|---|
| `mechanism-a-direction-geometry.md` (check A) | Parallel/orthogonal split and Gauss-Newton cosine of the meeting signal | 3 runs of arm D, measurement only |
| `linking-measurement-concentration.md` (check L) | E-009 concentration analysis on the meeting gradient (selected vs unselected rows) | shares A's runs, plus 3 runs of arm A |
| `mechanism-dose-response.md` (check G0) | Fixed-hyperparameter dose-response of the learned loss, with rho (relative gradient norm) as the dose | 50 runs, no tuning |
| `mechanism-b-oracle-arm.md` (check B) | Oracle targets (pinv, Gauss-Newton, damped logit-space) plus the scaled-gradient control, cost ignored | oracle arms |
| `mechanism-d-replacement-test.md` (check D) | Is the meeting signal sufficient to train the lower half with backprop cut? | replacement arms |
| `mechanism-c-deep-network-sgd.md` (check C) | Deep plain MLP, SGD with momentum, against deep supervision | a small Stage-3-style study |
| `mechanism-g-and-cost-deferred.md` (Gate 3) | Fixes for G's quality and cost; **blocked** until a step reduction is shown | none until the gate opens |
| `selective-backward-prereg-prep.md` | Regression check, wall-clock, small-batch control, `sel_frac` curve, decisions for `prereg-v2` | 3 small looks |

## Gates and order

- **Gate 0 (cheap, first):** check A, check L, check G0. They reuse existing code and need no new network structure. A and L share the same training runs.
- **Gate 1:** check B (needs A's Jacobian code; A's Q3 tells you whether the Gauss-Newton arm is worth running) and check D (reuses B's oracle). Run G0 before B so the dose interpretation is known.
- **Gate 2:** check C (largest; independent of the others and may be started in parallel).
- **Gate 3 (deferred):** `mechanism-g-and-cost-deferred.md`. Opens only if G0, B or C shows a step reduction large enough to pay the overhead: about 31% fewer steps for +23% per step, 24% for +12%, 20% for +6% (0.85 divided by 1 plus the overhead). If no gate opens, the mechanism result is the contribution and Gate 3 is not run.
- **Track S (independent of the gates above, start it first):** `selective-backward-prereg-prep.md`: regression check, then the three looks, then the decisions and `prereg-v2`. It is the main line for a positive result.

Suggested working order: Step 0 of track S, then check A + L + G0, then the three looks of track S, then B and D, then write `prereg-v2`. C can run alongside any of these.

## Shared conventions

- **Decision.** Record D-020 (or the next free ID): "Stage 4: mechanism diagnostics, exploratory", and update D-021 (open items of the selective line). Commit the plan and the decision BEFORE the first run. A drift entry is needed for changes to `src/` (see "Files to add").
- **Seeds.** Used so far: 100, 101, 1000, 1001, 2000, 2001 (sweeps), 0 to 19 (Stage 3), 500 to 502 (E-009), 600 to 602 (pilot E-010, also check A and check L, which answer different questions). New blocks:

| Check | Seeds |
|---|---|
| A and L | 600 to 602 |
| G0 | 1100 to 1109 |
| B phase 1 (screen) | 700 to 702 |
| B phases 2 and 3 | 710 to 719 |
| C probe | 800, 801 |
| C sweep (selection) | 810, 811 |
| C final | 820 to 829 |
| D tuning | 950, 951 |
| D final | 900 to 909 |
| Track S: timing, small batch | 1200 to 1204 |
| Track S: `sel_frac` curve | 1210 to 1212 |
| `prereg-v2` final (reserved) | 1300 to 1319 |

- **Tuning tags.** `t3` (check C), `t4` (check D); use `t5` for a fresh independent tuning of A in `prereg-v2`.
- **Metric** where a comparison is made: steps-to-target and FLOPs-to-target under the D-018 rule (two consecutive evaluations at or above the target). A run that never reaches a target is recorded with its full-run steps and FLOPs and counted in `n_never`; never dropped. Targets 0.97 and 0.975 (C uses 0.95 and 0.97).
- **Reference baselines:** both tunings of arm A (`t1_base_two.yaml` = A, `t2_base_two.yaml` = A2); fixed-hyperparameter designs (G0) remove tuning noise by construction and are conditional on A's tuned settings, which favour the baseline.
- **Statistics:** paired percentage change by seed with a bootstrap interval, as in `src/sweep.py` and `scripts/analyze.py compare` (check the signatures before reuse).
- **No seed-adding after "inconclusive"**, and no editing of a rule after the data. A positive lead needs a new pre-registered confirmatory run on fresh seeds with a fixed reference baseline (D-015).

## How the outcomes combine

| Outcome | Reading |
|---|---|
| A REDUNDANT and L ALREADY SELECTIVE | The meeting signal is parallel to the task gradient and lives on the same rows; supports the redundancy explanation and predicts that a hybrid adds nothing. |
| G0 DOSE HARMS or FLAT, and B NO HELP | No support for the signal in this setup, with and without a perfect `G`. Strongest form of the mechanism result (not a proof). |
| G0 DOSE HELPS | The E-008 failure was partly a scale problem. B becomes much more interesting; plan a confirmatory test. |
| B HELPS | The direction carries information that backprop lacks. Gate 3 may open if the step reduction is large enough. |
| C SUPPORTED | The premise holds where backprop is weak. Plan a confirmatory run (new seeds, pre-registered). |
| C REFUTED | No support for the premise even where backprop is weak (in this setup). |
| D SUFFICIENT with learned G | The meeting signal can carry the lower half; any benefit is about parallel execution, which this repo does not measure. |
| D INSUFFICIENT with learned G but SUFFICIENT with the oracle | G's quality is the bottleneck (problem 5). Gate 3 is relevant only if a step reduction exists. |
| D INSUFFICIENT with both | The local-inverse signal itself is not enough to train on. |
| Track S looks: BASELINE PROBLEM | The confirmatory test must use the best-batch baseline as reference. |
| Track S looks: DOES NOT TRANSLATE (wall-clock) | The FLOPs saving is real but not a speed saving at this model size; say so, and test a larger width before concluding. |

## What these checks can NOT show

- Anything beyond the MNIST MLP family, one optimizer family per check, and 3,000 steps (6,000 in C if the probe requires it).
- Mechanism in the causal sense: A, L and G0 are geometry and dose; B and D are interventions on one signal. Together they narrow the explanation; they do not prove it.
- A wall-clock or parallelism benefit (single device; D counts FLOPs only; track S measures timing on one machine).

## Files to add

`exploratory/mechanism.py`, `scripts/mechanism_check.py`, `tests/test_mechanism.py` (A and L); a `meet_rho` arm and `scripts/run_dose_response.py` (G0); new arm keys in `src/model.py` and `compute_losses` (B, D); a deep model and an `optimizer` config key (C); extensions of `scripts/pilot_selective.py` and timing and batch-control scripts (track S). Changes under `src/` touch the paths locked by D-019: each needs a drift entry (DR-008 onward) stating that no Stage 3 arm changes, and a regression test that arms A to F and S still reproduce their logged results bit for bit (as in Stage 1).

## Log

Add an `exploratory` entry in `EXPERIMENT_LOG.md` per check (next free E-IDs), with the printed verdict.
