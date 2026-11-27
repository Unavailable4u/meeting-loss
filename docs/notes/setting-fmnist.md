# Setting study 1: the three method families on Fashion-MNIST (same MLP)

Status: planned, not run | Type: exploratory (D-015), then lock the winners | Decision: D-031 | Drift: DR-024 | Follows: E-027 (Study 2 on MNIST), E-035 (Study 1b), E-008 (Study 1)

## Why
Every result so far is MNIST with one MLP, and on MNIST a single hidden layer is already enough (E-035: the shallow network equals the 4-layer one), so a method that moves cost around has little to win. The paper lists a second setting as a planned check (C-24, C-20). The researcher chose to extend everything, one setting at a time: Fashion-MNIST with the same MLP first, then the deep plain MLP with SGD (check C), CIFAR-10 on flattened pixels and CIFAR-10 with a small CNN (each its own later patch and note). Exploratory first; the comparisons that look real are locked (new tag) afterwards.

## What is built (additions only; DR-024)
- `src/datasets.py`: the dataset registry (`mnist`, `fmnist`; normalisation constants) and the accuracy targets. `src/data.py`: `get_data(..., dataset="mnist")`. The config key `dataset` (default `mnist`) reaches `get_data` only when it is set, so every earlier config and log is unchanged (test). `src/train.py` and `src/sb.py` pass it.
- **Targets are fixed by a probe before any other arm runs on the dataset:** `scripts/probe_dataset.py` runs arm A as tuned on MNIST (unchanged except `dataset`) for 3,000 steps on seeds 3000 to 3002; plateau = the median over seeds of the mean validation accuracy of the last 5 evaluations; high = floor((plateau - 0.005) / 0.005) x 0.005; low = high - 0.005. The rule reproduces the MNIST targets from A's MNIST plateau (0.9804 gives 0.975 and 0.97). The result is written once to `configs/datasets/fmnist_targets.json` and committed; the probe uses the untuned-for-Fashion configuration, so the targets are conservative (a tuned A can only plateau higher).
- `src/sweep.py`: sweep arms `f_base`, `f_sb`, `f_meet`, `f_deepsup`, `f_shallow`, `f_mn_match`, `f_mn_fwd`, `f_mn_table`, `f_mn_match_k2`, `f_mn_match_k3`, `f_mn_fwd_k3` = the MNIST sweep arms with `dataset: fmnist` (same search spaces, 25 trials, 2 selection seeds, 3,000 steps). `analyze.py select` now takes its default target from the arms' dataset (0.975 for MNIST arms, the dataset's high target otherwise; arms of two datasets in one selection are refused).
- `exploratory/setting.py`, `scripts/run_setting.py`: tag `f1` (seeds 2500 to 2509, paired), 13 groups, derived configs in `configs/setting/`. `exploratory/meetnet.py` now takes the targets as a parameter (defaults unchanged). `tests/test_setting.py` (18 tests); full suite 408 passed.

## Groups of `f1` (13 groups x 10 seeds = 130 runs; every arm at its own tuned config)
References: **A** arm A tuned on Fashion-MNIST (t17), **A2** a second independent tuning of A (t18; supplies the re-tuning shift), **S** the shallow network tuned (t17). Study 2: **S20** selective backward pass at 20% of the rows by loss, no warm-up, with A's hyperparameters (the arm confirmed on MNIST; not tuned separately), **SBT** Selective-Backprop tuned (t17). Study 1: **D** the meeting loss (arm D tuned), **B** deep supervision (tuned). Study 1b: **M1m, M1f, M1t, M2m, M3m, M3f** (tuned; the noise and multi-hot controls and M2f of E-035 are dropped: they showed nothing).

## Decision rules (fixed before any run on the dataset; FLOPs to the two targets, rule `two` of D-018, misses charged the whole run, paired change of the mean, 95% bootstrap interval)
- **S20 against A:** the margin rule of D-019: HOLDS if at both targets the point change is <= -15%, the interval excludes 0 and the gain exceeds the re-tuning shift of A (A2 against A, rule `max_gain`); DOES NOT HOLD if at both targets the interval excludes every gain of 15%; else INCONCLUSIVE. **D and B against A:** the same rule, labels HELPS / NO HELP / INCONCLUSIVE. **S20 against SBT:** the same rule without a shift (as H10).
- **Study 1b groups:** the group verdicts and readings of `exploratory/meetnet.py` (EXCELS, FASTER LESS ACCURATE, FAILS, COMPETITIVE, SLOWER, UNCLEAR; chain vs table; shallow control; meeting point; readout), against A and against S, at the dataset's targets.
- Tuning noise between two tunings of one arm was 16% to 27% on MNIST; the re-tuning shift of A in this setting is measured by A2.

## Known risks
1. Fashion-MNIST validation accuracy can peak and then fall (overfitting), so "final accuracy" is the mean of the last 5 evaluations and a run may reach a target and lose it; the crossing rule `two` handles the first, not the second. 2. The targets come from a probe with an untuned A; if tuned A reaches a plateau much higher than the probe's, both targets are easy for it (read the table with that in mind). 3. One tuning per arm; only large effects are detectable. 4. SBT's tuning is the slowest (extra forward passes). 5. S20 reuses A's hyperparameters, as on MNIST; that is part of what C-24 asks (does the MNIST-confirmed arm transfer). 6. The deep MLP, CIFAR-10 (MLP and CNN) are not in this patch.

## Commands, in order (run from the repository root; one at a time, each after the previous one has finished)
```
git am mn1-results-writeup.patch               # if not applied yet
git am setting-fmnist.patch
python -m pytest -q                            # 408 passed
python scripts/verify_regression.py            # src changed (DR-024); needs a fresh log name: see the note in DR-024 if the file exists
python scripts/probe_dataset.py --dataset fmnist --workers 3 --threads 1
git add -A && git commit -m "probe and targets for fmnist (D-031)"
nohup python scripts/run_sweep.py --tag t17 --arms f_base f_sb f_meet f_deepsup f_shallow f_mn_match f_mn_fwd f_mn_table f_mn_match_k2 f_mn_match_k3 f_mn_fwd_k3 --seeds 17000 17001 --workers 4 --threads 1 > t17.log 2>&1 &
```
When `t17.log` shows 550/550:
```
nohup python scripts/run_sweep.py --tag t18 --arms f_base --seeds 18000 18001 --workers 4 --threads 1 > t18.log 2>&1 &
```
When `t18.log` shows 50/50:
```
python scripts/analyze.py select --tag t17 --write-configs     # no "incomplete trials" line may appear; paste the output
python scripts/analyze.py select --tag t18 --write-configs
git add -A && git commit -m "tuned configs t17, t18 (DR-024)"
python scripts/run_setting.py --tag f1 --write-configs
git add -A && git commit -m "f1 configs (DR-024)"
python scripts/run_setting.py --tag f1 --dry-run               # 130 runs, 0 done
nohup python scripts/run_setting.py --tag f1 --workers 4 --threads 1 > f1.log 2>&1 &
python scripts/run_setting.py --tag f1 --analyze-only          # when f1.log is finished
git add -A && git commit -m "f1 results" && git push
```
Estimated run time at the speed of the MNIST tuning (about 1.2 runs per minute with 4 workers): probe 5 to 10 minutes, t17 about 7.5 hours, t18 about 40 minutes, f1 about 2 hours.

## After the runs
E-036 (probe and tunings), E-037 (f1), findings in `docs/FINDINGS.md`, claims only through `paper/CLAIMS.md` with a drift entry (C-24 would be updated from this); then the next setting (deep plain MLP with SGD, check C).
