# Component ledger: what each ingredient did, and what to keep, drop or vary

**Status:** living note, written 2026-10-27 from F-001 to F-019 and E-001 to E-022. It adds no result; every number is quoted from the entry named in the evidence column. Labels are the current reading of the evidence and change when a new entry changes it. All results are on one MNIST MLP unless stated. Update this table in the same commit as any finding it touches.

Labels: **KEEP** (use as is), **CONTROL** (always include as a comparison), **VARY** (a lever with measured or plausible headroom), **OPEN** (never tested or not tested properly), **DROP** (no gain shown; do not build on it).

## A. Selective backward pass (arm S)

| Component | What we measured | Evidence | Confidence | Label | What it says for the next step |
|---|---|---|---|---|---|
| Backward only through the highest-loss rows (S30, kept 0.3, warm-up 100) | -35.1% / -38.4% FLOPs to 0.97 / 0.975 against tuned A1, 20 of 20 seeds in the same direction; shift 5.5% / 3.9% | F-019, E-022 (confirmatory) | high within scope | **KEEP** | The reference to beat in the Improve track is Selective-Backprop itself, not A1 |
| Selection by loss against random rows | Random 30% (R30): +26.2% / +14.4% against A1; pilot +41% / +18% | E-022, E-010 | high (control repeated) | **CONTROL** / DROP random | Selection quality is what produces the saving |
| Concentration of the gradient on those rows | Top 30% by loss carries 98% to 99.7% of the gradient norm with cosine about 1.0 after about step 500; random 30% carries 25% to 33% | C-14, E-009 (3 seeds) | medium | explains KEEP | Earlier steps are not summarized in the claim; read the E-009 table by step before building a schedule on it |
| Kept fraction | 0.1 to 0.3 flat in the 3-seed look (F-018); S20 against S30 on the 20 seeds of E-022: -7.9% / -7.2% FLOPs (E-023). On 10 fresh seeds (E-024) against S30 (0.97 / 0.975): S25 -7.4% / -2.2%, S20 -6.0% / -10.9%, S15 -9.6% / -12.9%, S10 -6.0% / -13.4%; S15, S20 and S10 are not separated from each other; step penalty at 0.1 | F-018, F-020, F-022, E-023, E-024 | low to medium | **VARY**: 0.1 to 0.2 beat 0.3 by about 6% to 13% | A constant fraction around 0.15 to 0.2; below 0.15 no further gain and a step penalty at 0.1. Look 2 (E-025, 10 new seeds, against S30): S20 -10.0% / -8.8%, S15 -12.2% / -10.1% (replicated by rule R1); S10 with no warm-up needs +24% / +20% steps (F-023) |
| Warm-up (100 full-backward steps) | Varied in E-024 on S20 (against warm-up 100, 0.97 / 0.975): warm-up 0 -17.8% / -14.6% FLOPs and -11% / -10% steps, cheaper in 10 of 10 seeds; 25 steps -4.7% / -2.4% and 50 steps -7.0% / +0.0% (no clear effect). Without warm-up the first 100 updates are slower (0.9016 against 0.9146 accuracy at update 100) and the run is ahead from about update 200 to 400 | E-024, F-022 | low (best of 14 arms; effect not graded in the length) | **VARY**: replicated once (E-025); about half of it is the saved warm-up FLOPs | Selection from the first update looks cheapest; mechanism unknown Look 2 (E-025, F-023, F-024, 10 new seeds, against S20 warm-up 100, 0.97 / 0.975): warm-up 0 -13.8% / -12.0% FLOPs (replicated; REPLICATED by rule), 5 steps -13.4% / -7.1%, 10 steps -14.0% / -10.4%, 25 steps -3.9% / -4.9%. About half of the w0 saving is the 6.87 GFLOPs that 100 full steps cost over 100 selective steps; the rest is 60 to 90 fewer steps (intervals include 0). Not graded; the AdamW-state and step-at-zero explanations are both unsupported |
| Kept fraction below 0.2 with no warm-up (combination: S15w0, S10w0) | Against S20w0 (0.97 / 0.975): S15w0 -5.8% / +5.7%, S10w0 +11.5% / +7.9% with +24% / +20% steps; accuracy at update 50 falls with the kept fraction when there is no warm-up (0.787, 0.734, 0.649 against 0.881) | E-025, F-023 | low to medium | **DROP** as a combination | The two leads do not add; no combination goes forward (section D, item 5) |
| Gradient scale of the kept rows (`sel_scale: k`, the mean over kept rows) | S20k against S20 -5.4% / -9.2% (0.97 interval includes 0), S20w0k against S20w0 -3.4% / +2.8%; rule R4: SCALE DOES NOT MATTER | E-025, F-024 | low (point changes of S20k lean towards a gain; up to about 10% not excluded) | **DROP** (no effect by the rule) | Not a lever; not used in `prereg-v3` unless a reason appears |
| S's own hyperparameters (tuned S, S30T) | -37.4% / -36.3% against A1, against -35.1% / -38.4% for S30 with A1's settings; tuned lr 0.00126 (A1 0.00130), wd 7.65e-4 (A1 1.31e-4) | E-021, E-022 (descriptive) | low | KEEP inherited | Tuning S adds little; not a priority |
| Wall-clock time | Per step on one CPU: 0.83 of A at one thread, 0.91 at eight, against 0.61 of FLOPs; time-to-target of the final runs not yet analysed | F-016, E-016 | low to medium | **OPEN** | Overhead of selection, gather and small matrices is a real lever and a real limit |

## B. Baselines, tuning and measurement protocol

| Component | What we measured | Evidence | Confidence | Label | Rule it gives |
|---|---|---|---|---|---|
| Re-tuning noise | A2 against A1: +26.2% / +13.4% on 20 paired seeds although the two tied at selection; further tunings t5 to t8 within -5.5% to +6.3% on the final seeds after looking 8% to 15% better at selection; noise floor 1.336 / 1.180 (LARGE) | F-011, E-019, E-022 | medium to high | **CONTROL** | Always report against more than one tuning, on 20 fresh paired seeds; selection on 2 seeds does not predict the ranking |
| Crossing rule | First crossing of the target reorders arms; two consecutive evaluations at or above the target is stable | F-006, D-018 | low to medium | **KEEP** | Use the D-018 rule |
| Targets | 0.97 and 0.975 usable; 0.98 too noisy on a 5,000-sample validation set | F-002 | medium-low | **KEEP** | Report 0.98 only descriptively |
| Batch size | A fully tuned batch 48 (8,000 steps): -7.0% / -4.6% against A1, intervals contain zero | E-022, F-017 | medium | **CONTROL** | A smaller batch does not explain the saving; keep as a control in later plans |
| Activation | relu in the tuned winners of A1, B48 and S30T; in the dose study relu harms grow with the dose, tanh is flat up to rho 1 and breaks at rho 3 | E-020, E-021, F-015 | medium | **VARY** (only with a reason) | Second activation adds cost; not a lever for S so far |

## C. Meeting loss and its relatives (study 1)

| Component | What we measured | Evidence | Confidence | Label | Reading |
|---|---|---|---|---|---|
| Meeting loss (arm D) as an accelerator | +36.9% / +35.5% FLOPs to 0.97 / 0.975 against A, and more steps (1,025 against 920 and 1,510 against 1,370); per-step cost +23% | F-009, E-008 (confirmatory), C-07 | high | **DROP** | H1 refuted |
| Alignment of its gradient with the task gradient | Cosine 0.75 to 0.80 at steps 500 to 2,000; mostly parallel; the learned inverse is closer to backpropagation's direction than to Gauss-Newton's | F-009 (H3 met), C-15, E-012 | medium | background | High alignment did not produce a speed-up: the signal is redundant, not wrong |
| Dose-response at fixed hyperparameters | No dose reduced steps-to-target by 10% or more, in relu or tanh; relu cost +6% to +9% at rho 0.03, +60% to +76% at 0.1, up to +208% at 1 to 3; middle representation grows at large doses | F-014, F-015, E-013 to E-015 | medium | **DROP** (as additive loss) | Closes Gate 3 of `mechanism-overview.md` |
| Where the meeting gradient lives | 97% to 99% of its norm is in the top 30% rows by loss, the rows S keeps | C-16, F-013, E-012 | low to medium | inference only | A "selective meeting loss" would restrict a signal that already sits there; no gain is expected from that alone (this is an inference, not a test) |
| Frozen-G variant, control F | Frozen-G ties A at selection with about zero alignment; control F is nearly inert (scale 2*lam*eta/d) | F-008, F-005 | low to medium / high | background | The ties are not explained by alignment |
| Deep supervision (arm B) | +2.3% / +1.7% against A (H6 refuted); -23.7% / -13.2% against A2 (exploratory); per-step cost +0.6% | F-010, F-011, C-07 | medium | **DROP** as a claim | Its apparent lead came from tuning noise in the baseline |
| Autoencoder auxiliary (arm E) | Per-step cost +116% | C-07 | high (count) | **DROP** | Too expensive to pay back |
| Oracle target, replacement test, deep network with SGD (checks B, D, C) | Not run | `mechanism-b-oracle-arm.md`, `mechanism-d-replacement-test.md`, `mechanism-c-deep-network-sgd.md` | none | **OPEN** | These are what can say why the signal is redundant, and whether it can replace part of backpropagation at lower cost |

## D. What the table suggests (hypotheses, not findings)

1. **Keep:** loss-based selection, the D-018 crossing rule, paired fresh seeds, several tunings of the baseline, a tuned small batch and a random-row control in every comparison.
2. **Drop:** the meeting loss as an additive accelerator, deep supervision and the autoencoder as claims, random selection, first-crossing, the 0.98 target as a decision metric.
3. **Vary (the Improve track's levers, in the order the data supports):** (a) a lower constant kept fraction: the paired S20 against S30 contrast (E-023, F-020) gives -7% to -8% FLOPs with the same steps, and no sign of an early penalty, so a schedule is not supported; the lowest fraction without a step penalty (between 0.1 and 0.2) is still to be found; (b) warm-up length (11% to 12% of the cost, never varied); (c) the selection rule itself (Selective-Backprop's probabilistic rule against our top-k; to be read from the paper before coding); (d) per-step overhead (wall-clock: 0.83 and 0.91 of A's time per step against 0.61 of its FLOPs, F-016), which may be a larger lever than (a) and (b).
4. **Replace:** the hand-set kept fraction 0.3 by a better constant (E-023 suggests 0.2 or lower); a schedule only if a later look shows a step penalty that depends on the stage of training.
5. **Do not do:** combine components until something wins. A component enters a confirmatory plan only with a stated reason from this table, on seeds that were not used to pick it.

## E. Literature-derived levers (added 2026-10-27, D-025; none has been run in our protocol)

| Lever | Source | What it says | Our status |
|---|---|---|---|
| Probabilistic selection by loss percentile with full-size training batches (Selective-Backprop) | Jiang et al. 2019; code read | Costs extra selection forward passes; in our counter 2.33 to 2.78 times a plain update uncached (1.89 to 2.33 cached) | **TESTED (E-024, exploratory): DROP as a FLOP competitor.** Needs 32% to 46% fewer updates than A1 but, at beta 2, 1.9 to 2.3 times the cost per update (more at beta 3); +26% to +84% FLOPs against S30 in every variant; with A1's hyperparameters, FLOPs not time. A tuned Selective-Backprop is the fair reference for a pre-registered test. Tuned in the arm-A protocol (t11, E-026, selection runs): in-sample 209.9 GFLOPs uncached (182.7 cached) to 0.975 with 825 updates, level with A's in-sample winners (197.7 to 232.6) and 62% above tuned S30; final-seed comparison not run |
| Kept fraction that shrinks with training accuracy (q = s, s/2, s/4, s/8, s/16 at 80%, 90%, 95%, 99.5%) | Kawaguchi and Lu 2020 | Default rule of Ordered SGD, aimed at test error | **TESTED (E-024, exploratory): worse.** SOSGD +12.7% / +2.7% FLOPs against S30 and +24.7% / +17.9% against S15 (its first epoch is a full backward pass). Together with E-023: high-then-low not supported |
| Selection pressure decreased over epochs | Loshchilov and Hutter 2015 (MNIST) | Opposite direction to the line above | **TESTED (E-024, exploratory): worse.** Sup (fraction rising 0.1 to 0.5) is +6.8% / +9.3% against S15 |
| Gradient-norm bound as the score (norm of softmax minus one-hot) | Katharopoulos and Fleuret 2018 | Loss is a weaker proxy than the bound early in training; the bound is free in the forward pass | **TESTED (E-024, exploratory): no clear difference** from ranking by loss (-1.9% / +2.0% against S20) |
| Switch selection on only when it pays | Katharopoulos and Fleuret 2018 | A variance-reduction estimate decides when | OPEN. E-024 suggests that the fixed warm-up itself costs (S20w0 is cheapest), so a switch that turns selection on later is not what the data ask for |

Status 2026-10-27 (after D-025 and DR-014): the arms for the first four rows (and the rising-fraction arm) are implemented and tested; the headroom plan is fixed in `docs/notes/improve-headroom-plan.md`; none has been run, so every row stays OPEN.

Status 2026-10-28 (after E-024): the first four rows of this section were run on 10 fresh seeds (exploratory); read the verdicts as leads. The best arms so far are S with a constant fraction of 0.1 to 0.2 and, as a lead to replicate, no warm-up; Selective-Backprop and both schedules lose under FLOPs.

Status 2026-10-28 (look 2 built, not run): replication of the no-warm-up and fraction leads, the S15w0 and S10w0 combinations, a finer warm-up and a gradient-scale test are planned in `docs/notes/improve-look2-plan.md`; the rows for kept fraction and warm-up above stay leads until then.

Status 2026-10-29 (after look 2, E-025): the no-warm-up lead and the kept-fraction lead replicated on new seeds (F-023: S20w0 against S20 -13.8% / -12.0%, S15 against S30 -12.2% / -10.1%), the combinations and the gradient scale did not pay, and the warm-up dose is not graded (F-024). Rule R6 names S20w0 and S15 as the leads for a pre-registered test. Selective-Backprop has not been tuned, so the comparison with it is still against an untuned reference.

Status 2026-10-30 (after t11, E-026): Selective-Backprop has a tuned configuration (`configs/tuned/t11_sb_two.yaml`). In-sample it needs about 42% fewer updates than A at 2.3 times the cost per update, so it stays near tuned A in FLOPs; whether S20w0 beats it on fresh seeds is what H10 of the `prereg-v3` proposal would test.
