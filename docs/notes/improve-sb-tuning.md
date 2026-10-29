# Improve track: tuning of Selective-Backprop (tag t11), the fair reference for a pre-registered comparison

**Status:** protocol written 2026-10-29, before the sweep. Type: selection runs under the D-014, D-017 and D-018 protocol (`exploratory`, D-015); nothing is tested. Why: in E-024 Selective-Backprop ran with A1's hyperparameters and lost to S30 in FLOPs (F-021). A comparison that a reviewer will accept needs a Selective-Backprop that was tuned as much as the baseline (arm A: 25 trials, 2 selection seeds, FLOPs-to-0.975).

## Protocol (src/sweep.py, DR-016)
- Sweep arm `sb`: arm `sb` (`src/sb.py`) with `sb_beta` 2.0, `sb_history` 1,024 and one-epoch priming fixed; the search space of arm A (`lr` log 1e-4 to 1e-2, `wd` log 1e-6 to 1e-2, `act` relu or tanh, `lr_schedule` constant or cosine); 25 trials; selection seeds 9200 and 9201 (new; 9000/9001 were t9 and 9100/9101 t10); 3,000 updates; evaluation every 50 updates.
- Score: FLOPs to 0.975 under the D-018 rule, as counted for the released algorithm (`flops`: every selection forward pass and the second forward pass charged); the cached count is in the logs and is not used for selection. Winner by the D-017 rule (lowest mean score over the two seeds; ties by the late validation accuracy).
- Not searched: beta (E-024 found beta 2 and 3 alike), the history length and the priming length (the released values). These are the method's own settings; searching them would give Selective-Backprop a larger search space than arm A had. If a reviewer asks, a beta search is a separate tag.
- t12 (optional, not requested): `selective20w0`, S without warm-up at kept 0.2, same search space, tuned once as S30 was in t10 (S30T was not clearly better than S30 with A1's settings, F-020 context). Run it only if the plan of `prereg-v3` asks for a tuned S.

## Commands
```
python scripts/run_sweep.py --tag t11 --arms sb --seeds 9200 9201 --workers 4 --threads 1 --dry-run     # 50 runs
nohup python scripts/run_sweep.py --tag t11 --arms sb --seeds 9200 9201 --workers 4 --threads 1 > sweep_t11.log 2>&1 &
python scripts/analyze.py select --tag t11 --arms sb --write-configs          # after it finishes
git add -A && git commit -m "Tuning t11: Selective-Backprop (D-025)" && git push
```
Time: my estimate is 1.2 to 2 hours (50 runs at about 2.3 times the cost of a plain update; look 2 took 4,542 s for 120 mostly cheaper runs on the same four workers). Run it with the laptop idle. The winner's config is written to `configs/tuned/t11_sb_two.yaml` and becomes group SBT of `prereg-v3` if H10 is adopted (`prereg-v3-proposal.md`).
