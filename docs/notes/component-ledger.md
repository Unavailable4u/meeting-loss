# Component ledger: what each ingredient did, and what to keep, drop or vary

**Status:** living note, written 2026-10-27 from F-001 to F-019 and E-001 to E-022. It adds no result; every number is quoted from the entry named in the evidence column. Labels are the current reading of the evidence and change when a new entry changes it. All results are on one MNIST MLP unless stated. Update this table in the same commit as any finding it touches.

Labels: **KEEP** (use as is), **CONTROL** (always include as a comparison), **VARY** (a lever with measured or plausible headroom), **OPEN** (never tested or not tested properly), **DROP** (no gain shown; do not build on it).

## A. Selective backward pass (arm S)

| Component | What we measured | Evidence | Confidence | Label | What it says for the next step |
|---|---|---|---|---|---|
| Backward only through the highest-loss rows (S30, kept 0.3, warm-up 100) | -35.1% / -38.4% FLOPs to 0.97 / 0.975 against tuned A1, 20 of 20 seeds in the same direction; shift 5.5% / 3.9% | F-019, E-022 (confirmatory) | high within scope | **KEEP** | The reference to beat in the Improve track is Selective-Backprop itself, not A1 |
| Selection by loss against random rows | Random 30% (R30): +26.2% / +14.4% against A1; pilot +41% / +18% | E-022, E-010 | high (control repeated) | **CONTROL** / DROP random | Selection quality is what produces the saving |
| Concentration of the gradient on those rows | Top 30% by loss carries 98% to 99.7% of the gradient norm with cosine about 1.0 after about step 500; random 30% carries 25% to 33% | C-14, E-009 (3 seeds) | medium | explains KEEP | Earlier steps are not summarized in the claim; read the E-009 table by step before building a schedule on it |
| Kept fraction | 0.1 to 0.3 flat in the 3-seed look (0.58 to 0.63 of A at 0.975); 0.5 worse (0.73); S10 needed 27% more steps at 0.97. On the 20 paired seeds of E-022, S20 against S30: -7.9% / -7.2% FLOPs at 0.97 / 0.975 with the same steps (cost per step 86.3 M against 95.1 M); no early step penalty | F-018, E-018, F-020, E-023 | low to medium | **VARY** | A lower constant fraction has about 7% to 8% headroom; the step penalty starts somewhere between 0.1 and 0.2; a high-then-low schedule is not supported by this look |
| Warm-up (100 full-backward steps) | Fixed at 100 in every experiment, never varied; it is 11.5% (S30) and 12.4% (S20) of the FLOPs to 0.975 (15.51 GFLOPs) | E-010, D-023, E-023 | none on its effect | **VARY** | Worth a few percent if a shorter warm-up costs no steps; interacts with the kept fraction |
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
