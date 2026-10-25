# Selective backward (arm S): housekeeping, three exploratory looks and preparation of `prereg-v2`

**Status:** plan and decision rules, written BEFORE any of the runs below. Type: `exploratory` for the three looks; the final test is a separate, later, pre-registered step. Do not edit a rule after seeing results; add a dated note below instead. Follows the pilot (`selective-backward-pilot.md`, E-010, F-012) and D-021 (proposed, items OPEN).

## Why a preparation step

The pilot gives S30 about 0.61 of A's FLOPs-to-target, with a step ratio of about 0.95. The break-even margin is large: S30 can need up to 1.39 times A's steps and still save 15% (pilot note), so a repeat of the same test on MNIST would very probably pass and teach little. The things that can still change the conclusion are:
1. the baseline is not compute-optimal in **batch size** (it was fixed at 128 in every config and sweep), so a plain smaller batch may already save a lot;
2. **wall-clock time**, which FLOPs ignore (selecting and gathering rows has a cost);
3. **how aggressive** the selection can be (only `sel_frac` 0.3 and 0.5 were tried);
4. a **second setting** and an independent baseline tuning.

## Step 0: housekeeping (before anything else)

**Regression check (E-002 style), proposed by DR-007.** Rerun one Stage 3 arm-A seed on the current code: `python -m src.train --config configs/tuned/t1_base_two.yaml --seed 0` and compare its validation curve with the raw log of `final-t1-base-two` seed 0. Rule (fixed now): **PASS** if the validation accuracy at every evaluation step is identical on the same machine and library versions, or, on different hardware or versions, differs by at most 0.002 at every evaluation and gives the same steps-to-target. Otherwise stop and find the cause before any further run. Record the result as a verification entry.

**Log (2026-10-25):** attempt 1 ended FAIL by this rule (E-011). The rule is unchanged. The likely cause is a different torch thread count from the Stage 3 runs (`run_final.py` uses 1 thread per worker); the script now defaults to `--threads 1` for attempt 2. A second FAIL calls for the code-versus-environment control described in E-011. **Update 2026-10-25:** attempt 2 with `--threads 1` PASSED bit-identically (E-011); Step 0 is done and the thread-count hypothesis is supported. The control is not needed.

Then record D-020 and update D-021 (the open items) so the choices below have an owner.

## Look 1: wall-clock time per step

S30 versus A, same machine, same device (`device: auto`), warm-up of 200 steps excluded, then 1,000 steps timed, 3 repeats, median per step; include the top-k selection and row gather in S30. Descriptive, no seeds needed (use seed 1200). Rule:
- **FLOPS SAVING TRANSLATES:** S30's time per step is at most 0.85 of A's.
- **PARTIAL:** between 0.85 and 1.00.
- **DOES NOT TRANSLATE:** 1.00 or more (the selection and gather cost eat the saving at this model size).

Report the device. A result on CPU does not transfer to a GPU. A "does not translate" at this size is likely to be a property of the small MLP and should be tested at a larger width before concluding anything.

## Look 2: the small-batch control (is A compute-optimal in batch size?)

Arm A with batch size `b` in {32, 48, 64} (128 is the reference). FLOPs per step scale with `b` (check that the FLOP counter takes the batch size). Design (fixed now):
- horizon `max_steps = 3000 x 128 / b` (equal samples seen as A's 3,000 steps);
- tuning: for each `b`, 8 random trials over `lr` (log-uniform 3e-4 to 5e-3) with A's other settings fixed, 2 seeds (1200, 1201), scoring by FLOPs-to-0.97 under the D-017 and D-018 rules;
- evaluation: the best trial per `b` on 3 fresh seeds (1202 to 1204); S30 and A (batch 128, tuned) on the same seeds.

Metric: ratio of mean FLOPs-to-target versus A at 0.97 and 0.975. Rule:
- **BASELINE PROBLEM:** the best small-batch arm has ratio at most 0.85 at both targets. Then A is not compute-optimal, and the confirmatory test must use the best-batch A as the reference.
- **A WAS FINE:** every small-batch arm has ratio at least 1.00 at both targets.
- **PARTIAL:** anything else. Include the best small-batch arm as a reference arm in `prereg-v2` anyway.

Also report whether S30 still beats the best small-batch arm by at least 15% (ratio S30 / best small batch at most 0.85).

## Look 3: how aggressive can the selection be? (`sel_frac` curve)

`sel_frac` in {0.1, 0.2, 0.3, 0.5} with A's tuned hyperparameters, `sel_warmup` 100, on seeds 1210 to 1212, plus A on the same seeds. Exact per-step FLOPs (batch 128, forward 68,812,800 always paid, 673,792 per backward row):

| `sel_frac` | rows `k` | FLOPs per step | vs A |
|---|---|---|---|
| 0.1 | 13 | 77,572,096 | 0.500 |
| 0.2 | 26 | 86,331,392 | 0.557 |
| 0.3 | 39 | 95,090,688 | 0.613 |
| 0.5 | 64 | 111,935,488 | 0.722 |

The forward pass is 44.4% of A's cost, so **no selection can save more than about 56% per step**; the useful range is bounded. Metric: FLOPs ratio and steps ratio versus A at both targets, and misses. Descriptive, with one rule: the **knee** is the smallest `sel_frac` whose ratio is at most 0.85 at both targets with no missed seed. State it in the log and use it, with 0.3, as the pre-registered `sel_frac` candidates (at most two values in the final test).

## Clarifications for the implementation (2026-10-26, written before the three looks run; no rule above is changed)

**Look 1.** The timed step is the whole training step (zero_grad, forward, backward, optimizer and scheduler) on 100 batches prefetched to the device and cycled, so data loading is not timed. The 200 warm-up steps also cover S30's 100 full-backward steps, so the timed steps are all selective. The repeats alternate the order of the two variants (A then S30, S30 then A, ...) to cancel drift. The **verdict uses one torch thread** (the project convention, E-011); torch's default thread count is timed as well and reported as a second, descriptive reading. The device is reported with the result.

**Look 2.** Evaluation spacing is kept at about the same number of samples as arm A (every 50 steps of 128): every 200, 133 and 100 steps for batch 32, 48 and 64. The 8 learning rates per batch size are drawn log-uniformly in [3e-4, 5e-3] by a generator seeded from the tag and the batch size (so the design is reproducible and written to `configs/looks/<tag>/` before any run). All other settings are arm A's tuned config (relu, AdamW, cosine schedule over the scaled horizon, weight decay). A tuning trial is scored by the mean over its two seeds of the FLOPs to reach 0.97 under the two-consecutive rule, a miss at its full-run FLOPs, ties to the higher late validation accuracy, then the lower trial index (the D-017 mechanics, with the target 0.97 as written above). The **best small-batch arm** is the one with the lowest worst-case ratio (the larger of its ratios at 0.97 and 0.975). The ratio of an arm is the mean FLOPs-to-target over arm A's, on the three evaluation seeds. The FLOP counter measures the actual batch, so FLOPs scale with the batch size.

**Look 3.** The variants are named S10, S20, S30 and S50 (top-loss rows, `sel_warmup` 100, arm A's tuned hyperparameters). The knee rule is applied as written; the pilot's GO/STOP/TUNE rule is not applied in this mode.

## The hybrid arm (meeting loss on the selected rows)

Include it in `prereg-v2` **only if** check L (`linking-measurement-concentration.md`) ends NOT SELECTIVE or PARTIAL. If it ends ALREADY SELECTIVE, skip it (predicted null, no information). If included: one arm, labeled exploratory, with the expectation "null" written before the run, outside the primary hypothesis and outside the multiplicity of the primary comparison.

## Decisions needed before `prereg-v2` (recommended defaults, yours to choose)

1. **S's hyperparameters:** inherit A's tuned values as the **primary** (conservative: A was tuned, S was not; the claim reads "bolting selective backward onto a tuned recipe saves compute"). Tuning S with the same budget is a descriptive secondary only.
2. **Baselines:** A1 (`t1_base_two`), A2 (`t2_base_two`), **further independent tunings of A** (tags `t5` to `t8`, proposed in D-022, which makes the number of tunings and the reference rule OPEN there; `t3` and `t4` are reserved by checks C and D), and the best small-batch arm from Look 2. The noise-floor rule of D-019 applies.
3. **Seeds and rule:** about 20 seeds (block 1300 to 1319), a 15% margin, the three-way outcome (supported / refuted / inconclusive) as in D-019, R30 reported and never deciding. The expected effect (about -39%) is far from the 15% margin, so power is not the issue; use the effort on the baselines and the second setting.
4. **Second setting:** an MLP on Fashion-MNIST or flattened CIFAR-10 (the hand-written backward probably carries over; check the input dimension handling). It needs **its own tuned baseline** (a 25-trial, 2-seed tuning of A in that setting, same rules). A CNN needs a new backward pass and can come later.
5. **Wall-clock** as a pre-registered secondary outcome (Look 1's protocol).

**Update (2026-10-25, E-012):** check L ended ALREADY SELECTIVE, so by the rule above the hybrid arm is skipped in `prereg-v2`.

## Draft hypothesis for `HYPOTHESES.md` (edit before use)

> **H7 (draft).** In the MNIST MLP setting of D-008, the selective backward arm S at `sel_frac` = *[from Look 3]*, inheriting the hyperparameters of the tuned baseline A1, reaches validation accuracy 0.97 and 0.975 with at least 15% fewer FLOPs (counted as in `src/flops.py`) than the best of A1, A2, a fresh independent tuning of A, and the best small-batch baseline, on 20 paired seeds. **Supported / refuted / inconclusive** by the rule of D-019 (margin 15%, noise floor from the baseline tunings, both targets must agree).

Then: a lock tag `prereg-v2`, an update of the lock check in `scripts/run_final.py` (it is tied to `prereg-v1`), and a plan file in the style of `configs/stage3_plan.yaml`, all committed before the first final seed.

## Related work to read before any claim (verify the citations)

Jiang et al. 2019, "Accelerating Deep Learning by Focusing on the Biggest Losers" (selective backprop; the abstract reports convergence to a target error up to 3.5 times faster than standard SGD on image datasets, mostly in wall-clock terms, which makes Look 1 and the choice of what is counted important); Katharopoulos and Fleuret 2018 (importance sampling with gradient norms); Loshchilov and Hutter 2015 (online batch selection). The idea is known; the contribution is a rigorous, compute-matched test with the baselines above and tuning noise controlled.

## What these looks can NOT show

- Anything confirmatory: all three are exploratory (D-015), on 3 seeds, and each informs the design of `prereg-v2`, not a claim.
- Look 1 is one machine and one model size; Look 2 tunes only the learning rate; Look 3 inherits A's hyperparameters.
- Generality beyond the MNIST MLP: that is the job of the second setting.

## How to run (once the code exists)

```
python -m pytest -q
git add -A && git commit -m "..." && git push        # BEFORE the looks
python scripts/verify_regression.py                                        # Step 0 regression (done, E-011)
python scripts/time_selective.py --tag sw1                                # Look 1 (a few minutes)
python scripts/run_batch_control.py --tag sb1 --workers 4                 # Look 2 (63 runs, resumable)
python scripts/pilot_selective.py --tag sf1 --sel-fracs 0.1 0.2 0.3 0.5 --seeds 1210 1211 1212   # Look 3 (15 runs)
```

The scripts exist (`scripts/time_selective.py`, `scripts/run_batch_control.py`, `scripts/pilot_selective.py --sel-fracs`; logic in `exploratory/looks.py`, tests in `tests/test_looks.py` and `tests/test_time_selective.py`). Log each look as an `exploratory` entry in `EXPERIMENT_LOG.md`.
