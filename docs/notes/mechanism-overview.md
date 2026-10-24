# Mechanism study (Stage 4 diagnostics): overview of checks A, B, C and D

**Status:** plan, written BEFORE any run. Type: `exploratory` for all four checks (D-015). Do not edit a decision rule after seeing results; add a dated note below instead.

## Why these four checks

E-008 refuted H1: the meeting loss needs about 36% more FLOPs than the tuned baseline, and it also needs about 10% more *steps*. H3 shows its gradient is aligned with the task gradient (about +0.75). The working explanation is **redundancy**: a correct signal that backprop already supplies adds nothing (problem 1 in our problem list; not yet an entry in RISKS.md). That explanation is an interpretation, not a tested mechanism. These four checks test it from four sides.

| Check | Question | Training needed | Note |
|---|---|---|---|
| A | What is the geometry of the meeting signal: how much is parallel to the task gradient, and does it resemble a Gauss-Newton direction? | 3 runs of arm D, measurement only | `mechanism-a-direction-geometry.md` |
| B | If G were a perfect local inverse (or a Gauss-Newton oracle), would the signal help at all? | Oracle arms, cost ignored | `mechanism-b-oracle-arm.md` |
| C | Does the meeting loss help where backprop is weak (a deep plain MLP, SGD with momentum)? | A small Stage-3-style study | `mechanism-c-deep-network-sgd.md` |
| D | Is the meeting signal *sufficient* to train the lower half on its own (backprop cut at the middle state)? | Replacement arms | `mechanism-d-replacement-test.md` |

## Order and dependencies

1. **A first.** It is cheap and its code (`exploratory/mechanism.py`: the top-half Jacobian, the softmax Hessian and the target directions) is reused by B and D.
2. **B second.** It uses A's directions as its oracle targets. A's result also tells you whether B is worth running (see A's decision rule, Q3).
3. **D third.** Its oracle replacement arm is B's oracle with backprop cut. Do D only after B's code is tested.
4. **C last.** It is the largest (a new model, a new optimizer, a tuning sweep) and the one where a positive result is most plausible, so it should be run with the cleanest code.

C does not depend on A, B or D and can be started in parallel if you prefer.

## Shared conventions

- **Decision.** Record one decision before running anything: D-020 (or the next free ID), "Stage 4: mechanism diagnostics A to D, exploratory". Add a drift entry if the PLAN changes (the PLAN lists these as option (a)). Commit the plan and the decision BEFORE the first run.
- **Seeds.** Used so far: 100, 101, 1000, 1001, 2000, 2001 (sweeps), 0 to 19 (Stage 3), 500 to 502 (E-009). New, one block per check:

| Check | Seeds |
|---|---|
| A | 600 to 602 |
| B phase 1 (screen) | 700 to 702 |
| B phases 2 and 3 | 710 to 719 |
| C probe | 800, 801 |
| C sweep (selection) | 810, 811 |
| C final | 820 to 829 |
| D tuning | 950, 951 |
| D final | 900 to 909 |

- **Metric** where a comparison is made: steps-to-target and FLOPs-to-target under the D-018 crossing rule (two consecutive evaluations at or above the target). A run that never reaches a target is recorded with its full-run steps and FLOPs and counted in `n_never`; never dropped.
- **Targets:** 0.97 and 0.975 (as D-019) except in C, where plain SGD on a deep net is slower and the targets are 0.95 and 0.97 (fixed in C's note).
- **Reference baselines:** both tunings of arm A (`configs/tuned/t1_base_two.yaml` = A, `configs/tuned/t2_base_two.yaml` = A2). Any comparison reports against both (R-21, F-011).
- **Statistics:** paired percentage change by seed with a bootstrap interval, as in `src/sweep.py` (`paired_pct_change`, `margin_verdict`, `overall_verdict`) and `scripts/analyze.py compare`. Check the exact function signatures before reuse.
- **No seed-adding after "inconclusive".** If a result is inconclusive, log it as such; a new test on fresh seeds needs its own written plan (D-019 point 2).
- **Exploratory, not confirmatory.** Nothing here can be reported as a confirmed effect. A positive lead needs a new pre-registered confirmatory run on fresh seeds with a fixed reference baseline (D-015).

## How the outcomes combine

| Outcome | Reading |
|---|---|
| A: REDUNDANT, and B: NO HELP | Closed: the signal is redundant even with a perfect G. Strongest form of the mechanism result. |
| B: HELPS | The direction carries information that backprop lacks. Engineering a cheaper G becomes worthwhile; plan a confirmatory test. |
| B: NO HELP but A's Q2 positive | G approximates a curvature direction, but a better direction does not buy steps. Report as a negative mechanism result. |
| C: SUPPORTED | The premise holds where backprop is weak. Plan a confirmatory run (new seeds, pre-registered). |
| C: REFUTED | The signal does not help even where backprop is weak (in this setup). Closes the line for MLPs. |
| D: SUFFICIENT with learned G | The meeting signal can carry the lower half. Any benefit then is about wall-clock or parallel execution, which this repo does not measure. |
| D: INSUFFICIENT with learned G but SUFFICIENT with the oracle | G quality is the bottleneck (problem 5). |
| D: INSUFFICIENT with both | The local-inverse signal itself is not enough to train on. |

## What these checks can NOT show

- Anything beyond the MNIST MLP family, one optimizer family per check, and 3,000 steps (6,000 in C if the probe requires it).
- Mechanism in the causal sense: A is geometry, B and D are interventions on one signal. Together they narrow the explanation; they do not prove it.
- A wall-clock or parallelism benefit (single device; D counts FLOPs only).

## Files to add

`exploratory/mechanism.py`, `scripts/mechanism_check.py`, `tests/test_mechanism.py` (A); new arm keys in `src/model.py` and `compute_losses` (B, D); a deep model and an `optimizer` config key (C). Methods that change `src/` change the locked paths of D-019: the lock applies to the Stage 3 plan, so these additions need a drift entry (DR-005 onward) stating that they do not alter any Stage 3 arm, and a regression test that arms A to F still reproduce their logged results bit for bit (as in Stage 1).

## Log

Add an `exploratory` entry in `EXPERIMENT_LOG.md` per check (next free E-IDs), with the printed verdict.
