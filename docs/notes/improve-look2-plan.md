# Improve track, look 2: replication, combinations, warm-up dose and gradient scale (exploratory plan, fixed before any run)

**Status:** written 2026-10-28, committed with the code and the analysis script, before the first run. Type: `exploratory` (D-015, D-024). It decides nothing about the paper and makes no claim; it decides what a pre-registered test (`prereg-v3`, seeds 2300 to 2399) should contain. It follows E-024 / F-022 and uses new seeds (2110 to 2119).

## Questions and what each hypothesis predicts (written before the data)
1. **Is the no-warm-up effect real?** In look 1, S20 without warm-up was 17.8% / 14.6% cheaper than S20 with 100 warm-up steps (0.97 / 0.975), in 10 of 10 seeds, but it was the best of 14 arms, so its size is probably inflated. *If real:* S20w0 is again at least 10% cheaper than S20. *If a winner's-curse artefact:* the gap shrinks or disappears.
2. **Is the constant-fraction lead real, and do the two leads add up?** S15 against S30 (look 1: -9.6% / -12.9%); S15w0 and S10w0 against S20w0. Arithmetic expectation: per-step cost falls by 4.7% from 0.2 to 0.15 (k 26 to 20) and by about 10% from 0.2 to 0.1 (k 13), so a combination gains at most that much unless steps also fall; the threshold for "pays" is therefore set at 3%.
3. **How does the effect depend on the warm-up length?** Look 1 saw an effect at 0 but none at 25 or 50 (not graded). *Hypothesis B (optimizer state):* AdamW (betas 0.9 and 0.999) normalizes by a running mean of squared gradients; full-batch gradients of the warm-up are several times larger than those of S (the sum over the kept rows divided by the batch size, about k/B of the size), so after the switch the second-moment estimate stays inflated for hundreds of updates and the effective step is smaller; a shorter warm-up inflates it less. B predicts a graded dose (FLOPs change monotone: 0 < 5 < 10 < 25 < 100 steps, all improving over 100). *Hypothesis C (a step at zero):* something about selecting from the first update (hard rows from a random network) that a few full steps remove at once. C predicts no effect at 5, 10 and 25 and a clear one at 0.
4. **Does the gradient scale matter?** `sel_scale: "k"` divides the kept rows' gradient by k instead of B (the mean over kept rows, as Ordered SGD does), which removes the size mismatch at the switch. B predicts S20k (warm-up 100) is better than S20 and S20w0k is about the same as S20w0 (no mismatch to remove). C predicts no effect of the scale in either. A different pattern (for example the scale hurts) is reported as it is.

## Design
12 groups, 10 paired seeds each (2110 to 2119), 3,000 updates, A1's hyperparameters inherited. 120 runs. Configs `tracks/improve/configs_look2/`, plan `tracks/improve/look2_plan.yaml` (made by `tracks/improve/make_look2_configs.py`), runner `tracks/improve/run_headroom.py --plan tracks/improve/look2_plan.yaml`, logs `results/raw/improve_look2/`, analysis `tracks/improve/look2_analysis.py`.

| Group | Arm and keys changed from A1 | Question |
|---|---|---|
| A1 | `base` | paired reference |
| S30 | `selective`, kept 0.3, warm-up 100 | reference |
| S20 | kept 0.2, warm-up 100 | reference of 3 and 4 |
| S15 | kept 0.15, warm-up 100 | 2 (replication) |
| S20w0 | kept 0.2, warm-up 0 | 1 (replication) |
| S15w0, S10w0 | kept 0.15 or 0.1, warm-up 0 | 2 (combination) |
| S20w5, S20w10, S20w25 | kept 0.2, warm-up 5, 10, 25 | 3 |
| S20k | kept 0.2, warm-up 100, `sel_scale: k` | 4 |
| S20w0k | kept 0.2, warm-up 0, `sel_scale: k` | 4 |

## Reading rules (implemented in `tracks/improve/look2_analysis.py`; thresholds fixed now)
Quantities, crossing rule and interval as in `improve-headroom-plan.md` (FLOPs and steps to 0.97 and 0.975, D-018; never = whole run; 97.5% paired-bootstrap interval, 10,000 resamples, seed 0). "Improves by T%" means point change at most -T% with the interval below 0, at both targets.
- **R1 replication.** S20w0 against S20: REPLICATED if it improves by 10%; NOT REPLICATED if no interval excludes 0 or both point changes are above -5%; PARTIAL otherwise. S15 against S30: REPLICATED if it improves by 5%; NOT REPLICATED if no interval excludes 0 or both changes are above -3%; PARTIAL otherwise.
- **R2 combinations.** S15w0 and S10w0 against S20w0: COMBINATION PAYS if it improves by 3%; WORSE if it is worse by 3% at both targets (interval above 0); NO GAIN otherwise.
- **R3 warm-up dose** (S20w0, S20w5, S20w10, S20w25 against S20): GRADED if S20w0 improves by 10% and the FLOPs changes rise strictly with the warm-up length (w0 < w5 < w10 < w25 < 0) at both targets; STEP AT ZERO if S20w0 improves by 10% and w5, w10, w25 each change by less than 5% in absolute value at both targets; MIXED otherwise.
- **R4 gradient scale.** S20k against S20 and S20w0k against S20w0: BETTER if it improves by 5%, WORSE if it is worse by 5% at both targets (interval above 0), NO CLEAR EFFECT otherwise. Reading: OPTIMIZER-STATE HYPOTHESIS SUPPORTED if S20k is BETTER and S20w0k is NO CLEAR EFFECT; SCALE DOES NOT MATTER if both are NO CLEAR EFFECT; SCALE MATTERS (other pattern) otherwise.
- **R5** the cheapest S arm by mean FLOPs over the two targets, against A1 and S30. **R6 leads** for `prereg-v3`: S20w0 and S15 if REPLICATED, S15w0 and S10w0 if COMBINATION PAYS, S20k and S20w0k if BETTER; only these may be proposed, with a reason from the ledger.

## Limits and cautions
- 10 seeds per group; 11 S groups and two targets: some differences will arise by chance; a lead needs a pre-registered replication on fresh seeds.
- A mechanism reading (R3 with R4) is a hypothesis test on one network and one optimizer, not a proof; the AdamW explanation is one candidate, and a result that fits neither B nor C stays "MIXED" or "other pattern".
- All arms inherit A1's hyperparameters (no tuning, so an arm that prefers a different learning rate is disadvantaged); FLOPs only; no wall-clock claim; no seed is added and no threshold changed after the results are seen (a bug in the analysis script is fixed only with a drift entry).
- No change to a claim, to C-21 or C-27, follows from this look.

## Running it
```
python tracks/improve/run_headroom.py --plan tracks/improve/look2_plan.yaml --dry-run
nohup python tracks/improve/run_headroom.py --plan tracks/improve/look2_plan.yaml --workers 4 --threads 1 > look2.log 2>&1 &
python tracks/improve/look2_analysis.py          # after all 120 runs have finished
```
My guess is 1 to 1.5 hours on 4 workers (all arms are cheaper than Selective-Backprop; look 1 had 30 Selective-Backprop runs). Log the outcome as E-025 (exploratory) in the same commit as the raw logs.
