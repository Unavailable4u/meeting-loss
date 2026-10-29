# prereg-v3: proposal (nothing is decided; the OPEN items are the researcher's; a decision entry D-026 would record them)

**Status:** drafted 2026-10-29 from E-022 to E-025 and F-019 to F-024. It fixes no hypothesis, margin or seed; the plan file, the analysis script and the lock commit are not built. It exists so that the pooled evidence and the choices are written down before the confirmatory plan is made.

## Evidence so far (exploratory except E-022; pooled = look 1 on seeds 2100 to 2109 plus look 2 on 2110 to 2119, 20 paired seeds, 97.5% paired-bootstrap intervals; recomputed from the raw logs on 2026-10-29)
| Comparison (FLOPs to 0.97 / 0.975) | Pooled change | Interval | Other evidence |
|---|---|---|---|
| S30 against A1 | -33.8% / -37.1% | [-37.7, -29.6] / [-42.3, -32.4] | E-022 (confirmatory, 20 seeds) -35.1% / -38.4% |
| S20 against S30 | -8.0% / -9.9% | [-12.1, -3.9] / [-15.0, -4.0] | E-023 -7.9% / -7.2% |
| S15 against S30 | -10.9% / -11.5% | [-14.7, -6.7] / [-15.4, -6.8] | S15 against S20: -3.2% / -1.8%, not separated |
| S20 without warm-up against S30 | -22.6% / -21.9% | [-27.2, -18.4] / [-28.3, -14.3] | |
| S20 without warm-up against A1 | -48.7% / -50.9% | [-52.2, -45.1] / [-55.0, -46.6] | |
| S20 without warm-up against S20 | -15.9% / -13.3% | [-21.2, -9.8] / [-19.4, -6.9] | steps to 0.975 -8.2% [-14.9, -1.3] |
| warm-up 25 against 100 (S20) | -4.3% / -3.6% | [-8.3, +0.0] / [-9.4, +1.9] | w5 and w10 look like w0 (10 seeds, look 2) |
- About 6.87 GFLOPs of the 12 to 15 that S20 without warm-up saves against S20 is the cost of 100 full steps instead of 100 selective steps, a fixed amount; charged like selective steps, the saving is -7.8% [-14.2, -1.3] at 0.975, which comes from 8% fewer steps. Both parts are real FLOP savings; only the second needs a learning explanation, and no mechanism is supported (F-024).
- Selective-Backprop as released (beta 2, A1's hyperparameters) needed 44% / 84% more FLOPs than S30 and was level with A1 at 0.975 (E-024, F-021); a tuned one does not exist yet (t11).
- Not additive: kept fraction below 0.2 without warm-up gives no gain (E-025). Both schedules and the gradient-norm score are not leads.

## Proposed hypotheses (D-019 rule: point change at most -margin, 97.5% paired-bootstrap interval excluding 0, both targets; three-way outcome; 20 paired fresh seeds 2300 to 2319; 3,000 updates; A1's hyperparameters for every S arm)
- **H8 (primary):** S20w0 (kept 0.2, no warm-up, `sel_mode` top) needs at least **15%** fewer FLOPs than A1 at 0.97 and 0.975, and the gain exceeds the re-tuning shift of A (rule `max_gain`, as in H7a, with A2 and A5 to A8 re-run on the same new seeds). Pooled expectation about -49% / -51%: the margin is not tight, it is kept at 15% for comparability with H7a.
- **H9:** S20w0 needs at least **10%** fewer FLOPs than S30, the arm confirmed in prereg-v2. Pooled -22.6% / -21.9%; lower interval ends -18.4% / -14.3%. The reporting must say it combines two changes (kept 0.2 instead of 0.3, and no warm-up) and give the split from the descriptive groups S20 and S20w0.
- **H10:** S20w0 needs at least **15%** fewer FLOPs than the tuned Selective-Backprop SBT (`configs/tuned/t11_sb_two.yaml`), uncached count (the released algorithm) as the decisive one, the cached count reported next to it. Expectation: untuned Selective-Backprop was far behind S30 (E-024); whether tuning changes that is exactly what t11 and H10 test.
- **Descriptive groups (decide nothing):** S30, S20, S15, S20w10 (a short warm-up, to report the dose without a mechanism claim), and R20w0 (random rows at kept 0.2 without warm-up: the control that selection by loss produces the saving).
- **Secondary, reported:** wall-clock time per step and derived time to target at one and eight threads (FLOPs stay primary; time saved much less in F-016); steps to target; FLOPs charged without the warm-up difference.
- **Groups and cost:** A1, A2, A5, A6, A7, A8 (6), S20w0, S30, SBT (3), S20, S15, S20w10, R20w0 (4) = 13 groups x 20 seeds = 260 runs, of which 20 are Selective-Backprop. My estimate is about 3 hours on four workers (look 2: 4,542 s for 120 runs).

## OPEN: decisions for the researcher (defaults in bold)
1. Adopt H8, H9, H10 with margins **15%, 10%, 15%**; or drop H9 (it is the least independent of the three) or H10 (if tuned Selective-Backprop should not be a hypothesis but a descriptive group).
2. Primary arm **S20w0** (strongest, replicated), or the safer S20 with warm-up 100 (smaller effect, no slow start; its pooled gain against S30 is -8% / -10%).
3. Re-run A2, A5 to A8 on the new seeds for the shift rule (**yes**: 100 cheap runs, and it keeps the pre-registered rule of H7a) or use the old tunings' shifts (not comparable: different seeds).
4. A fully tuned batch-48 control for the new arm (**no**: H7b already addresses the batch-size explanation, and S20w0 changes neither batch size nor optimizer).
5. A tuned S20w0 (tag t12) as a descriptive group (**no**: tuning S added little before).
6. The wall-clock secondary on one machine only (**yes**), reported as secondary.

## What has to be built before the lock (none of it touches `src/` beyond DR-014 to DR-016)
`configs/prereg_v3_plan.yaml` and `configs/prereg_v3/*.yaml`; `scripts/analyze_prereg3.py` (H8 to H10, the shift, the noise floor, the descriptive groups, a refusal on incomplete groups, like `analyze_prereg2.py`); tests; the lock commit (H8 to H10 in `docs/HYPOTHESES.md`, the arms in `docs/METHOD.md`, a drift entry); the tag `prereg-v3`. Order: t11 (tuning) -> build -> lock -> run -> analyse. The tag also freezes `src/`, `configs/tuned/`, the plan and the hypotheses; nothing about the arms may change after it.
