# Improve track: headroom study (exploratory plan, fixed before any run)

**Status:** written 2026-10-27, committed together with the code and the analysis script, before the first run. Type: `exploratory` (D-015, D-024). It decides nothing about the paper and makes no claim; its job is to find which levers, if any, deserve a pre-registered test on fresh seeds (`prereg-v3`, seeds 2300 to 2399, D-024). Decisions behind it: D-025 (accepted), `docs/notes/improve-reference-arms.md`, `docs/notes/component-ledger.md`, E-023.

## Questions
1. **Constant kept fraction.** E-023 saw S20 about 7% to 8% cheaper than S30 with the same steps, and F-018 saw a step penalty at 0.1. Where is the best constant fraction, and where does the penalty start?
2. **Warm-up.** It is 11% to 12% of the cost to 0.975 and has never been varied. Does a shorter one cost steps?
3. **Reference arm.** Under our FLOP accounting, does Selective-Backprop as released beat, tie or lose to S30? (The released second forward pass is counted; the cached count is reported next to it.)
4. **Schedules and score.** Does the Ordered-SGD rule for the kept fraction, a fraction that rises (the direction of Loshchilov and Hutter), or ranking by the last-layer gradient norm (Katharopoulos and Fleuret) beat the best constant fraction ranked by loss?

## Design
15 groups, 10 paired seeds each (2100 to 2109), 3,000 updates, evaluation every 50 updates, A1's hyperparameters inherited by every arm (`configs/tuned/t1_base_two.yaml`; only the keys below change). 150 runs. Configs: `tracks/improve/configs/` (made by `tracks/improve/make_headroom_configs.py`), plan `tracks/improve/headroom_plan.yaml`, runner `tracks/improve/run_headroom.py` (no lock tag), logs in `results/raw/improve_headroom/`, analysis `tracks/improve/headroom_analysis.py`.

| Group | Arm and keys changed from A1 | Question |
|---|---|---|
| A1 | `base` | paired reference |
| S30 | `selective`, kept 0.3, warm-up 100, top | reference for the other S arms |
| S25, S20, S15, S10 | kept 0.25, 0.2, 0.15, 0.1 (warm-up 100) | 1 |
| S20w0, S20w25, S20w50 | kept 0.2, warm-up 0, 25, 50 | 2 |
| SB2 | `sb`, beta 2, history 1024, one-epoch priming (as released) | 3 |
| SB2p100 | `sb`, beta 2, priming 100 steps | 3 (isolates the rule from the warm-up) |
| SB3 | `sb`, beta 3 (the released default) | 3 |
| SOSGD | `selective`, `sel_schedule: osgd`, no warm-up | 4 |
| Sup | `selective`, `sel_schedule: linear`, kept 0.1 rising to 0.5 after a 100-step warm-up | 4 |
| S20gn | kept 0.2, `sel_score: gradnorm` | 4 |

Implementation (`src/sb.py`, `src/selective.py`; DR-014): Selective-Backprop keeps each forwarded row with probability (share of the last 1,024 losses strictly below its loss)^beta, queues the kept rows and updates when a full batch is available (at most one update per forward batch); the update runs a second forward pass. Steps always count updates. The priming phase is charged as ordinary steps of arm A. If many losses tie at exactly zero (float32), the kept rate falls below 1/(beta+1); this is as in the released code and the kept rate is logged (`sel_rate`) and printed.

## Quantities
FLOPs-to-target and steps-to-target at 0.97 and 0.975 (D-018: first of two consecutive evaluations at or above the target); a run that never reaches a target is charged its whole run and counted ("never"). Paired percentage change of the mean against a reference on the same seeds; 97.5% paired-bootstrap interval (10,000 resamples, seed 0). Selective-Backprop is reported as `flops` (released algorithm) and as `_cached` (selection forward reused).

## Reading rules (implemented in `tracks/improve/headroom_analysis.py`; thresholds fixed now)
An improvement means: point change at most -T% and the interval below 0, at both targets. T = 5 for R1, R4, R5 and 3 for R3; the step penalty is steps above +10% with the interval above 0.
- **R1 best constant fraction:** the S group with the lowest mean FLOPs-to-target averaged over the two targets. HEADROOM CONFIRMED if it improves on S30; NO FURTHER GAIN if no interval against S30 excludes 0; PARTIAL otherwise.
- **R2 step penalty:** the kept fractions whose steps are above +10% against S30 (interval above 0) at either target.
- **R3 warm-up (against S20):** PAYS if FLOPs improve by at least 3% at both targets; COSTS STEPS if steps rise by more than 10% (interval above 0) at either target; NO CLEAR EFFECT otherwise.
- **R4 Selective-Backprop against S30 (FLOPs, uncached and cached, each of SB2, SB2p100, SB3):** SB WINS if it improves by at least 5% at both targets; SB LOSES if it is worse by at least 5% (interval above 0) at both; UNCLEAR otherwise.
- **R5:** S20gn against S20; SOSGD and Sup against the best constant of R1. BETTER, WORSE or UNCLEAR by the same 5% rule.
- **R6 leads:** the arms with HEADROOM CONFIRMED, PAYS, SB WINS or BETTER. Only these may be proposed for `prereg-v3`, and only with a reason stated from the ledger.

## Limits and cautions
- 10 seeds per group: intervals are wide. With 14 groups, two targets and two references, some "improvements" will arise by chance; this is why R6 only nominates leads and why every lead needs a pre-registered replication on fresh seeds. No claim, no paper sentence and no change of C-21 or C-27 follows from this study.
- All arms inherit A1's hyperparameters; an arm that prefers another learning rate or weight decay is disadvantaged (Selective-Backprop uses full-size updates on selected rows, so it may be). A tuned run of a lead belongs in the pre-registered plan.
- FLOPs, not time; one MLP; one data split.
- No seed is added and no threshold changed after the results are seen. A bug in the analysis script can be fixed only with a drift entry that says what changed and why.

## Running it
```
python tracks/improve/run_headroom.py --dry-run
nohup python tracks/improve/run_headroom.py --workers 4 --threads 1 > headroom.log 2>&1 &
python tracks/improve/headroom_analysis.py          # after all 150 runs have finished
```
Expected time: my estimate is 2 to 3 hours on 4 workers (Selective-Backprop runs cost about twice as much per update); it is a guess. Log the outcome as E-024 (exploratory) and update the ledger, in the same commit as the raw logs.
