# Root-cause checks, follow-ups: tanh verdict of check B (mb2) and a bounded target for replacement (md2)

Status: planned, not run | Type: exploratory (D-015) | Decision: D-029 | Drift: DR-021 | Follows: `mechanism-root-cause.md` (D-028), E-029, E-030, F-027 to F-029

## Why
Check B ended INCONCLUSIVE in tanh and check D's relu replacement runs collapsed (final accuracy 0.11 to 0.52). Two things stand in the way of reading them:

1. **mb2 (check B in tanh).** The tanh baseline has one tuning, and the only difference in the helpful direction (Q at lr x0.5, -11.8% at 0.97, -4.3% at 0.975) has no arm A at the same learning rate to be compared with. mb2 adds the matched control (A at lr x0.5), a second independent tuning of the baseline in tanh (A2, t14) as the noise reference, and repeats F' and Q on 10 fresh seeds.
2. **md2 (bounded target in relu).** The collapse in relu is probably the unbounded drift of the middle state under a signal with a component along h (F-014, the dose check); this was not tested. md2 removes the component of the meeting gradient along h (`replace_bound: radial`; exact for the oracle arms, first order for the others) and asks whether a relu replacement then trains.

## What is built (additions only; DR-021)
- `src/oracle.py`: `remove_radial(v, h)`.
- `src/model.py`: `replace_bound` (`none` default, `radial`) in `meet_replace` and `meet_replace_frozen`; with `radial` the meeting term is a surrogate whose gradient at h is the unbounded one minus its component along h; `none` is the old code path, bit for bit (test).
- `src/sweep.py`: sweep arm `meet_replace_rb` (learned replacement, bounded, activation fixed to relu; lr, wd, schedule and `lam_inv` searched).
- `exploratory/oracle.py`, `scripts/run_root_cause.py`: tags `mb2` and `md2`, their plans, readings and reports.
- `tests/test_followups.py` (22 tests); the earlier 304 are unchanged.

## Tunings (before the checks)
| Tag | Arm | Selection seeds | Output |
|---|---|---|---|
| t14 | `base_tanh` | 14000, 14001 | `configs/tuned/t14_base_tanh_two.yaml` = A2 in tanh (t13 has 13000, 13001) |
| t15 | `meet_replace_rb` | 15000, 15001 | `configs/tuned/t15_meet_replace_rb_two.yaml` = R1b |

25 trials x 2 seeds each, the D-014, D-017, D-018 protocol.

## mb2: groups (tanh only, seeds 720 to 729, 10 per group)
| Group | What | Read against |
|---|---|---|
| A-tanh | arm A, t13 configuration | |
| A-tanh-lr0p5 | A with lr x0.5 | A-tanh (size of the lr effect) |
| A2-tanh | A with the t14 configuration | A-tanh (size of the tuning noise) |
| F-tanh-rho1 | F' (exact gradient), rho as chosen in mb1 | A-tanh |
| Q-tanh-rho0p3 | Q (damped inverse), rho as chosen in mb1, lr x1 | A-tanh |
| Q-tanh-rho0p3-lr0p5 | Q at lr x0.5 | **A-tanh-lr0p5 (matched)**; also reported against A-tanh |

Rule (the rule of check B phase 2, `decide_oracle`, over Q matched, Q and F', both targets must agree): **TANH ORACLE HELPS** if some entry is at least 10% fewer steps at both targets with a 95% interval excluding 0; **NO HELP** if every interval's lower end is above -10% at both targets; **INCONCLUSIVE** otherwise. A gain that is only the lr effect (Q at lr x0.5 against A at lr x1) is reported and does not decide.

## md2: groups (relu only, seeds 930 to 939, 10 per group, all against arm A in relu)
| Group | Arm | Signal |
|---|---|---|
| A-relu | base | reference |
| R1b-relu | meet_replace | learned G, bounded, tuned in relu (t15) |
| R4b-relu | meet_replace_frozen | frozen random G, bounded (arm A's relu configuration) |
| R2q-relu-rho1 | meet_replace_oracle | oracle Q, unbounded (arm A's relu configuration) |
| R2qb-relu-rho1 | meet_replace_oracle | oracle Q, radial bound |
| R2pb-relu-rho1 | meet_replace_oracle | oracle P, radial bound |

Rule per group: the rule of check D (SUFFICIENT / PARTIAL / INSUFFICIENT / UNCLEAR, both targets, reach fractions, +15% steps, 1 point of accuracy). Reading (`diagnose_bounded`, written before the runs): (a) if R2q is INSUFFICIENT and R2qb is SUFFICIENT or PARTIAL, the bound rescues the replacement: the relu collapse is the unbounded drift of h; if both are INSUFFICIENT, the radial drift is not (only) the cause; if R2q already works, the md1 collapse was specific to the P and random directions; (b) whether a bounded learned or random G reaches 0.97; (c) whether R1b and R4b get the same verdict.

## Limits
One network, AdamW, 10 seeds; `radial` is a first-order bound (h_var is logged); R2q, R2qb, R2pb and R4b use arm A's relu configuration without their own tuning (R1b is tuned); a speed-up is not the question in md2, only whether the lower half trains at all; the mb2 verdict still rests on two tunings of the tanh baseline; nothing here is a confirmatory test.

## Commands (from the repo root)
```
git am 0001-*.patch
python -m pytest -q                        # expect 326 passed
python scripts/verify_regression.py        # real MNIST; no run of it is recorded for DR-019 or DR-021
python scripts/run_sweep.py --tag t14 --arms base_tanh --seeds 14000 14001 --workers 4 --threads 1
python scripts/analyze.py select --tag t14 --arms base_tanh --write-configs
python scripts/run_sweep.py --tag t15 --arms meet_replace_rb --seeds 15000 15001 --workers 4 --threads 1
python scripts/analyze.py select --tag t15 --arms meet_replace_rb --write-configs
git add -A && git commit -m "tuned configs t14, t15"
python scripts/run_root_cause.py --tag mb2 --write-configs      # commit
python scripts/run_root_cause.py --tag mb2 --workers 4
python scripts/run_root_cause.py --tag md2 --write-configs      # commit
python scripts/run_root_cause.py --tag md2 --workers 4
```
Runs: 50 + 50 tuning, 60 for mb2, 60 for md2. The last command of each tag prints the verdict and writes `results/processed/oracle_<tag>_p1.csv`; `--analyze-only` repeats it.
