# Improve track, look 1: paired contrast of S20 against S30 on the E-022 logs (no new runs)

**Status:** plan and decision rule, written 2026-10-27 BEFORE the script was run. Type: `exploratory` (D-015, D-024). Caveat in the open: the E-022 table (S20 -40.3% / -42.9% and S30 -35.1% / -38.4% against A1) had already been seen when this note was written, so the *direction* is not blind; the rule below fixes how it is read before the paired numbers, the curves and the target grid are seen.

## Question
Is there headroom for a kept fraction that depends on the stage of training? The ledger (`component-ledger.md`) says a constant fraction looked flat between 0.1 and 0.3 (F-018, 3 seeds), that S10 needed more steps at 0.97, and that the gradient is concentrated on the high-loss rows from about step 500 (C-14). If a lower fraction costs steps early and saves FLOPs late, a schedule could beat both constants.

## Data and method
Groups S30 (`prereg2-S30`) and S20 (`prereg2-S20`) of `results/raw/prereg_v2/`, seeds 1300 to 1319, paired by seed, same A1 hyperparameters, 3,000 steps, warm-up 100. Target is reached at the first of two consecutive evaluations at or above it (D-018). Paired percentage change of the mean (S20 against S30), 97.5% paired bootstrap interval (10,000 resamples, seed 0). Quantities: FLOPs-to-target and steps-to-target at a grid of targets 0.90, 0.95, 0.96, 0.97, 0.975; the per-step FLOP cost after warm-up; the share of the warm-up in the cost; the mean validation accuracy at equal steps. Script: `tracks/improve/s20_vs_s30_contrast.py`; output `results/processed/improve_s20_vs_s30.csv`.

## Decision rule (fixed now)
Let r(t) be the paired change in FLOPs-to-target of S20 against S30 at target t, with its interval.
- **CONSTANT FRACTION HEADROOM:** r(0.97) and r(0.975) are both below -5% with intervals excluding 0. Reading: a lower constant fraction is already better; test 0.2 (and 0.15) against 0.3 in the next plan, not a schedule.
- **SCHEDULE LEAD:** r(t) is above 0 with the interval excluding 0 at one or more targets up to 0.96 and below 0 with the interval excluding 0 at 0.97 or 0.975 (a lower fraction costs early and pays late). Reading: a fraction that starts high and decreases is worth a headroom study.
- **NO HEADROOM:** neither of the above (all intervals contain 0, or the differences are under 5%). Reading: the kept fraction is not the lever; look at warm-up, the selection rule and per-step overhead instead.
- Anything else (for example S20 better early and worse late) is reported as such and read as a lead for the opposite schedule only with a stated reason.
Whatever the outcome, this look is exploratory, uses seeds already used for H7a and H7b (so it cannot confirm anything), and any lead needs a new plan with fresh seeds (D-024).
