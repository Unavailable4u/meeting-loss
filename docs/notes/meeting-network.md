# Study 1b: the meeting network (a forward chain and a backward chain that meet)

Status: run 2026-11-27 (E-034, E-035; F-032: no group excels) | Type: exploratory (D-015) | Decision: D-030 | Drift: DR-023 | Follows: the Study 1 root-cause checks (`mechanism-root-cause.md`, `mechanism-root-cause-2.md`; E-028 to E-033)

## Why a new study
The root-cause checks showed that arm D, as built, is an auxiliary loss on a fixed split: its meeting signal is about 0.1% of the task gradient and redundant next to backprop (F-027 to F-031). That is not the researcher's idea. The idea is: a FORWARD process goes from the input toward a state, a BACKWARD process goes from the desired output toward a state, and the two meet where they meet; the quantity of interest is the agreement of the two sides ("left side = right side"), and the meeting point depends on cost (the NH3 and H2 pipe: the cheaper chain crosses less of the pipe). Study 1 (arm D) stays in the record as the "auxiliary version" and a negative result; this is a new study, 1b.

## What is built (additions only; DR-023)
- `src/meetnet.py`: `MeetingNet` and `build_model`. Nodes of the 4-layer MNIST MLP: dims (784, 256, 128, 256, 10). Forward chain = layers 0..k-1 (input to node k). Backward chain = the mirrored layers from a label code to node k (k = 1, one-hot: 10 -> 256 -> 128 -> 256).
- **Meeting node k, fixed before training by the equal-cost rule** (`equal_cost_k`): the k at which the cumulative multiply-adds of the forward part (layers 0..k-1) and of the mirrored backward part (layers k..L-1) are closest. For this MLP the layers cost 200,704 / 32,768 / 32,768 / 2,560, so k = 1 (200,704 against 68,096; k = 2 would be 233,472 against 35,328). No chain is run to full depth to find the meeting point. This is the assistant's reading of the pipe analogy (a lighter gas travels farther; here a more expensive chain covers fewer layers) and the researcher agreed to fix it by cost. Graham's law has 1/sqrt(M), not 1/M; the study does not depend on that, and the k sweep below tests the rule directly.
- **Meeting loss:** cosine logits `scale * cos(forward state of the example, backward state of class c)`, cross-entropy over the 10 classes (`mn_scale`, tuned). The other classes' backward states are the negatives that stop both chains collapsing to a constant. Both chains train jointly by this loss, backprop inside each half. The backward chain runs on the 10 class codes of a step, not on every example, and its cost is counted (FlopCounterMode).
- **State = the post-activation output of the last layer of each chain** (`mn_post_act`, default true). This corrects the first sketch given to the researcher (pre-activation states): at k = 1 the forward chain would then be a single linear map, about 92% accuracy on MNIST by construction.
- **Arms (config `arm`):** `mn_match` (prediction = the nearest backward state), `mn_fwd` (as `mn_match`, plus the upper layers k..L-1 trained by cross-entropy on the DETACHED meeting state; both readouts are logged: `val_acc` = forward, `val_acc_match`, `val_acc_fwd`), `mn_table` (ablation: the backward chain replaced by a table of 10 free vectors), `shallow` (baseline: forward chain, activation, Linear(d_k, 10), ordinary cross-entropy, no backward chain).
- **Label codes (`mn_code`):** `onehot` (core); `noise` (one-hot plus 8 Gaussian components, std 0.5, drawn once per class per step, zero at test time); `multihot` (a fixed random +-1 code of 16 components; the same information as one-hot, a control that should change nothing in capacity).
- `src/train.py`, `src/model.py`: the meeting network is built and dispatched through `build_model` and `MeetingNet.mn_losses`; the optimizer skips the lower group when the model has none; every readout is evaluated at every evaluation step. For all existing arms the code path is unchanged (tests).
- `src/sweep.py`: sweep arms `mn_match`, `mn_fwd`, `mn_noise`, `mn_mh`, `mn_table`, `shallow`, and the k sweep `mn_match_k2`, `mn_match_k3`, `mn_fwd_k2`, `mn_fwd_k3` (`mn_k` is written into every config). Search space: the usual four (lr, wd, act, lr_schedule) plus `mn_scale` log-uniform 3 to 30; 25 trials, 2 selection seeds, 3,000 steps (tag t16, seeds 16000 and 16001).
- `exploratory/meetnet.py`, `scripts/run_meeting_net.py`: the plan (tag `mn1`, seeds 1500 to 1509), derived configs (`configs/mnet/`), scoring and the rules below. `tests/test_meetnet.py` (53 tests).

## Groups of `mn1` (each at its own tuned config; 12 groups x 10 seeds = 120 runs)
References: **A** arm A tuned (relu, t1), **S** the shallow network tuned (t16), **D** the old auxiliary arm D as tuned (descriptive). Tests: **M1m** `mn_match` at k = 1, **M1f** `mn_fwd` at k = 1, **M1n** noise code, **M1h** multi-hot code (control), **M1t** table ablation, **M2m M2f M3m M3f** the k sweep (k = 2 and 3, each tuned on its own).

## Decision rules (fixed before any run; FLOPs to 0.97 and 0.975, rule `two` of D-018, misses charged the whole run, paired change of the mean, 95% bootstrap interval)
- **EXCELS** (the researcher's "excels" rule): at least 8 of 10 seeds reach both targets, FLOPs change <= -10% against A AND against S at both targets with the interval below 0, and final accuracy within 0.2 points of A (A minus the group <= 0.002; final accuracy = mean of the last 5 evaluations of the readout the arm predicts with).
- **FASTER, LESS ACCURATE:** the FLOPs conditions hold, the accuracy gap is larger. **FAILS:** fewer than 5 of 10 seeds reach 0.97, or the accuracy is more than 1 point below A. **COMPETITIVE:** at least 8 of 10 reach both targets, accuracy within 0.2 points, and FLOPs within +15% of the cheaper baseline at both targets. **SLOWER:** reaches the targets with the accuracy but is not competitive. **UNCLEAR:** anything else. (Checked in this order.)
- **Readings** (`diagnose_study`): chain vs table (M1t against M1m: CHAIN HELPS / CHAIN NOT NEEDED / UNCLEAR); label richness (M1n and M1h against M1m: RICHER LABEL HELPS / NO EFFECT / UNCLEAR; M1h is a control); shallow control (M1m against S: BETTER / WORSE / SAME AS SHALLOW NET); meeting point (for each readout family, the k with the lowest mean FLOPs to 0.975; EQUAL-COST RULE CONFIRMED if it is the equal-cost k or the two are not distinguishable, else NOT CONFIRMED); readout (M1f: forward minus matching accuracy, FORWARD BETTER / MATCH BETTER / SAME at +-0.2 points).
- Tuning noise between two tunings of one arm was 16% to 27% (E-032), so only large effects are detectable with 10 seeds and one tuning per arm; a result near the thresholds is INCONCLUSIVE in spirit even when a rule fires, and is reported with that caveat.

## Known risks (told to the researcher before the build)
1. With a one-hot code the backward chain only ever produces 10 distinct states, so it is a learned set of 10 prototypes (M1t tests whether the chain adds anything over a table).
2. At the equal-cost node k = 1 the matching-only network is one hidden layer plus a cosine classifier; the shallow baseline S will probably match it on MNIST (M1m against S).
3. The equal-cost rule is an interpretation; M2 and M3 test it, but each k is tuned separately, so a difference mixes the node and its tuning.
4. Only large effects are detectable. 5. No real-MNIST run was possible where the code was written (synthetic-data smoke only; the machinery is tested on 379 tests). 6. Novelty is not checked: related methods are prototype/cosine classifiers, two-tower and contrastive label embeddings, target propagation, direct feedback alignment; a literature entry has to be written before any novelty sentence.
7. The deeper network and other datasets are out of this study by the researcher's decision (the paper already lists them as future work); they would repeat this plan with the tuned arms.

## Commands, in order (run from the repository root)
```
git am meeting-network-study1b.patch            # or apply the patch the way you did before
python -m pytest -q                              # 379 passed
python scripts/verify_regression.py              # real MNIST; no run of it is recorded for DR-019, DR-021 or DR-023
python scripts/run_sweep.py --tag smoke-t16 --arms mn_match mn_fwd mn_table shallow --trials 2 --max-steps 100 --seeds 100 101
python scripts/run_sweep.py --tag t16 --arms mn_match mn_fwd mn_noise mn_mh mn_table shallow mn_match_k2 mn_fwd_k2 mn_match_k3 mn_fwd_k3 --seeds 16000 16001 --workers 4 --threads 1
python scripts/analyze.py select --tag t16 --write-configs
git add -A && git commit -m "tuned configs t16 (DR-023)"
python scripts/run_meeting_net.py --tag mn1 --write-configs
git add -A && git commit -m "mn1 configs (DR-023)"
python scripts/run_meeting_net.py --tag mn1 --dry-run
python scripts/run_meeting_net.py --tag mn1 --workers 4 --threads 1
python scripts/run_meeting_net.py --tag mn1 --analyze-only
git add -A && git commit -m "mn1 results" && git push
```
The tuning is 10 arms x 25 trials x 2 seeds = 500 runs of 3,000 steps; `mn1` is 120 runs. Both resume after a stop (finished runs are skipped).

## After the runs
Log E-034 (t16 tuning) and E-035 (mn1) in `docs/EXPERIMENT_LOG.md`, findings in `docs/FINDINGS.md`, and claims only through `paper/CLAIMS.md` with a drift entry. Nothing in the paper changes before results exist.
