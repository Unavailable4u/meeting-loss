# Further independent tunings of arm A (tags t5 to t8): the noise floor of the tuned baseline and the reference set for `prereg-v2`

**Status:** plan and decision rules, written BEFORE any of the runs below. Proposed together with D-022 (proposed; nothing here is accepted until the researcher marks D-022). Type: the selection sweeps are `exploratory` (D-014, D-017: selection runs, no hypothesis tested); the final runs of the tuned configs belong to the `prereg-v2` plan and carry its type. Do not edit a rule after seeing results; add a dated note below instead.

## Why

Two tunings of arm A tied at selection (232.6 GFLOPs each, F-007) but differ by 17% to 34% on 20 final seeds (F-011): the tuned configuration, not the seed, is the dominant noise (the SD of log steps across seeds is only 0.08 to 0.12). Everything compared against "tuned A" inherits that noise (R-21), including the selective-backward test that `prereg-v2` is meant to run (D-021, OPEN item b). Two tunings are a sample of two: they say the spread can be large, not how large. Six tunings (A1, A2 and four new ones) give a distribution and a defensible reference set.

## What is run now (selection only)

Arm `base` only, protocol exactly as D-014, D-017 and D-018: 25 random-search trials, 2 selection seeds per trial, 3,000 steps, score = mean over the two seeds of the FLOPs to reach 0.975 (two consecutive evaluations, misses scored at the full budget), tie-break on late accuracy then trial index. No discretion: the winner is whatever `analyze.py select` returns.

| Tag | Selection seeds | Winner config (written by `select --write-configs`) |
|---|---|---|
| `t5` | 5000, 5001 | `configs/tuned/t5_base_two.yaml` (stage `final-t5-base-two`) |
| `t6` | 6000, 6001 | `configs/tuned/t6_base_two.yaml` |
| `t7` | 7000, 7001 | `configs/tuned/t7_base_two.yaml` |
| `t8` | 8000, 8001 | `configs/tuned/t8_base_two.yaml` |

Why these names and seeds: `t3` and `t4` are reserved for checks C and D (`mechanism-overview.md`); seeds 5000 and above are unused, at least 100 (selection and final seeds stay disjoint) and disjoint from every block in the overview table. The random-search draws of each tag depend on the tag only; they were checked to share none of their 25 trials with t1, t2 or one another.

Cost: 50 runs per tag, 200 runs for four tags. Effective cost per sweep run on 4 workers was about 33 s in E-006 and about 66 s per final run in E-008, so about 2 to 4 hours for all four. Do **not** run these while the wall-clock look (Look 1 of `selective-backward-prereg-prep.md`) is being timed: parallel load contaminates timing.

Convention (E-011): every run uses one torch thread per process (`--threads 1`, the default of `run_sweep.py` and `run_final.py`), which is what makes a rerun bit-identical to a Stage 3 run.

## Not run now

The final runs of the four new configs. They go into the `prereg-v2` plan on its reserved seed block (1300 to 1319), paired with A1, A2, S, the random control R30 and the best small-batch arm, so the noise floor and the reference baseline are measured on the same seeds as the test. Optionally, and exploratory only, the six configs could also be run on seeds 0 to 19 to re-compare D and B with a distribution of baselines (about 80 more runs); it is not proposed.

## Pre-registered outcome for the noise floor (descriptive, no hypothesis test, no verdict on a claim)

On the `prereg-v2` final seeds, per target in {0.97, 0.975}: the mean FLOPs-to-target of each of the six tunings (D-018 rule, misses at full budget), the ratio max / min over the six, and the SD of the log means. Seed noise alone would give a ratio of about 1.06 (six independent means with an SD of log mean of about 0.025, from the final-seed SD of 0.11 over 20 seeds).

- **LARGE:** max / min is at least 1.15 at both targets. Then every statement about a gain is made against the best tuning of the set and against the median tuning, the paper's limits state the spread, and F-011 is a two-tuning observation confirmed.
- **SMALL:** max / min is at most 1.10 at both targets. Then F-011 was a pair-of-tunings accident and A1 alone is an adequate reference.
- **MIXED:** anything else. Report the table and use the best-of-set reference.

## The reference baseline of `prereg-v2` (OPEN: yours to choose, see D-022)

| Option | Reference for H7 | Effect |
|---|---|---|
| (a) best of the six tunings and the best small-batch arm, on the same seeds | strictest; extends the draft H7 wording from "A1, A2, a fresh tuning" to all six | the expected gain of S (about -39% against A1, E-010) shrinks by the gap between the best tuning and A1; if the best is 0.75 to 0.85 of A1, S would sit at about -19% to -28% against it, still beyond the 15% margin but not by much |
| (b) the median tuning | typical baseline | larger measured gain; defensible but weaker |
| (c) A1 only | the Stage 3 reference | weakest; R-21 says it is not enough |

Recommended: (a) as the deciding comparison, (b) and (c) reported next to it. S inherits A1's hyperparameters in the proposed design, so S carries A1's own tuning luck; the best-of-six comparison is the one that removes that advantage.

## Commands

```
python -m pytest -q
git status                                   # clean, D-022 marked accepted and committed first
# one tag at a time; k = 5, 6, 7, 8 with seeds k000 and k001
python scripts/run_sweep.py --tag t5 --arms base --seeds 5000 5001 --dry-run
nohup python scripts/run_sweep.py --tag t5 --arms base --seeds 5000 5001 --workers 4 --threads 1 > sweep_t5.log 2>&1 &
python scripts/analyze.py select --tag t5 --arms base --write-configs
git add -A && git commit -m "Tuning t5 of arm A (D-022)" && git push
# repeat for t6, t7, t8; then log one exploratory entry (next free E-ID) with the four winners
```

## What this can NOT show

- Anything confirmatory by itself: it produces candidates and a plan, the measured spread comes with the `prereg-v2` finals.
- The spread of the other arms: D, B, S and the controls were each tuned once. A comparison of one tuning of an arm against six of A is asymmetric (it favours the arm tuned more often), so any claim about B or S against A states how many tunings each had. Tuning B or S the same number of times is OPEN in D-022 and not proposed.
- Anything beyond arm A on the MNIST MLP with AdamW and 3,000 steps.
