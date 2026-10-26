# prereg-v2: the plan, the machinery and what is still open

**Status:** plan and machinery written BEFORE any `prereg-v2` run (2026-10-27). Design: D-023 (accepted 2026-10-26), drift entry DR-011. Nothing here is locked yet; the lock is one commit and the tag `prereg-v2` (steps below). Type of the final runs: confirmatory (H7a, H7b), everything else descriptive.

## The hypotheses (drafts; they enter `docs/HYPOTHESES.md` in the lock commit)

See `selective-backward-prereg-prep.md`, dated note of 2026-10-26: **H7a** (S30 against A1, margin 15%, D-019 rule, shift rule) and **H7b** (S30 against the fully tuned batch-48 arm, margin 0%). Same 20 paired seeds (1300 to 1319) in every group; 97.5% paired-bootstrap intervals; both targets (0.97, 0.975) must agree; 0.98 is not used.

## What exists now

| Piece | File |
|---|---|
| Plan (11 groups, 220 runs; B48 and S30T wait for their tunings) | `configs/prereg_v2_plan.yaml` |
| Derived configs S30, S20, R30 (A1 plus selective keys) | `configs/prereg_v2/*.yaml`, written by `scripts/make_prereg2_configs.py`, checked against `src/prereg2.py` |
| Sweep arms for t9 (batch 48) and t10 (selective, 0.3) | `src/sweep.py` (`base_b48`, `selective30`, `ARM_STEPS`, `expected_steps`) |
| Plan with its own seed block, group horizons, output directory, lock tag and locked paths | `src/final.py` (`plan_meta`, `load_plan`), `scripts/run_final.py --plan` |
| Pre-registered analysis (H7a, H7b, shift, noise floor, descriptive tables) | `src/prereg2.py`, `scripts/analyze_prereg2.py` |
| Tests | `tests/test_prereg2.py` |

## Order of work (each step is one commit that is pushed before the next)

1. **t9** (batch 48, 8,000 steps, evaluation every 125 steps): `python scripts/run_sweep.py --tag t9 --arms base_b48 --seeds 9000 9001 --workers 4 --threads 1` (50 runs of 8,000 steps; about 2.7 times the cost of a standard tag), then `python scripts/analyze.py select --tag t9 --arms base_b48 --write-configs`; the winner is `configs/tuned/t9_base_b48_two.yaml`.
2. **t10** (S at 0.3, tuned once): `python scripts/run_sweep.py --tag t10 --arms selective30 --seeds 9100 9101 --workers 4 --threads 1`, then `python scripts/analyze.py select --tag t10 --arms selective30 --write-configs`; the winner is `configs/tuned/t10_selective30_two.yaml`.
3. `python scripts/run_final.py --plan configs/prereg_v2_plan.yaml --dry-run` must list 11 groups and 220 runs with no pending config.
4. **Lock commit:** add H7a and H7b to `docs/HYPOTHESES.md`, fix `shift_rule` in the plan (decision below), commit, `git tag prereg-v2 && git push origin prereg-v2`. The plan, `configs/prereg_v2/`, `configs/tuned/`, `src/`, `HYPOTHESES.md` and `METHOD.md` must not change afterwards (`run_final.py` checks this against the tag).
5. **Final runs:** `python scripts/run_final.py --plan configs/prereg_v2_plan.yaml --workers 4 --threads 1` (resumable; logs in `results/raw/prereg_v2/`).
6. **Analysis:** `python scripts/analyze_prereg2.py` (refuses incomplete groups or unfinished runs; writes `results/processed/prereg_v2.csv`). Log one confirmatory entry, a finding, and update the claims.

## Implementation clarifications (written before any run)

- **Evaluation spacing for t9 and the B48 final runs: 125 steps** (6,000 samples; arm A evaluates every 6,400 samples). The small-batch look used 133 steps, but 133 does not divide 8,000, so the last evaluation falls at step 7,980 and the sweep's completeness check (the last logged step must equal `max_steps`) would reject every run. 125 divides 8,000. The finer spacing than A's by 6% moves the crossing time of B48 by at most one evaluation interval, a small effect that favours B48 by about 3% of its steps at most; it is stated as a limit.
- **Margin 0 in H7b** uses the same function as H7a (`margin_verdict`): supported if the point change is at most 0 and the interval's upper end is below 0; refuted if the lower end is above 0 (strict; the equality case has probability zero for a bootstrap quantile); inconclusive otherwise.
- **The reference set for the noise floor** is A1 plus the shift groups (A2, A5 to A8): the ratio max / min of their mean FLOPs-to-target at each target gives LARGE (at least 1.15 at both), SMALL (at most 1.10 at both) or MIXED, as in D-022. It is descriptive and decides no claim.
- **Descriptive comparisons** against A1 and against the best tuning of A on the same seeds (picked on the same seeds, so optimistic for the baseline) are reported for every group, including S20, the tuned S (S30T) and the random control R30.
- **Files:** the final logs of B48 and S30T are found by their stage and arm names (`final-t9-base_b48-two` / `base`, `final-t10-selective30-two` / `selective`); no group's files can collide with Stage 3 logs because the output directory is separate.
- **Regression:** `src/sweep.py` and `src/final.py` changed and `src/prereg2.py` is new; `src/model.py`, `src/train.py`, `src/selective.py`, the FLOP counter and every Stage 3 arm are untouched. A rerun of `scripts/verify_regression.py` on real MNIST is still the researcher's check after the patch (as in E-011). **Done 2026-10-27: E-011 attempt 4, PASS (bit-identical), on the same `src/` that is locked.**

## DECIDED (2026-10-27): the re-tuning shift rule is `max_gain`

D-023 decision 2 set the default `max_abs`: the gain of S must exceed the **largest absolute** paired change of any further tuning of A (A2, A5 to A8) against A1, at the same target. D-019 used that form for H1 with the one tuning available. The numbers from Stage 3 show what it does here: A2 needs **+34.0%** (0.97) and **+17.2%** (0.975) more FLOPs than A1 on the same 20 seeds (97.5% intervals +26.5 to +42.0 and +6.7 to +27.7). S30's saving against A1 has been about 22% to 38% (E-010, E-017, E-018). Under `max_abs`, H7a could not be supported at 0.97 unless S saves more than 34% there, which A2 alone decides, regardless of how good the other tunings are, because A2 is a *worse* tuning than A1 and a worse tuning does not weaken S's claim.

| Rule | Shift = | What it protects against | Consequence |
|---|---|---|---|
| `max_abs` (D-023 default, D-019's form) | largest absolute change of any further tuning | A1 being unrepresentative in either direction | probably inconclusive at 0.97 by construction (A2 is +34%) |
| `max_gain` | largest improvement of any further tuning over A1 (0 if none is better) | A1 being a weak baseline: S must beat what a better-tuned A achieves | the logically relevant quantity; can still be large if t5 to t8 beat A1 |
| `median_abs` | median absolute change | typical re-tuning spread | between the two |

**Decision:** `max_gain`. The researcher was asked and delegated the choice to the assistant, which recommended `max_gain`; the choice is a judgment made knowing the A2 number, which is why it is flagged here and in D-023. Its justification does not depend on S: the shift rule exists to stop us claiming a gain smaller than what better tuning of the baseline would give, and only better tunings do that. `max_abs` and `median_abs` can be reported next to it (`scripts/analyze_prereg2.py --shift-rule ...` prints them as descriptive). It is fixed in the plan; after the lock it cannot change. If the further tunings t5 to t8 turn out to beat A1 by a lot on the final seeds, `max_gain` makes H7a hard to support, which is the intended behaviour: S would then be compared with a baseline that is easy to improve.
