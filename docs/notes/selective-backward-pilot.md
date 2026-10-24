# Pilot: selective backward pass (arm S)

**Status:** plan and decision rule, written BEFORE the first real run. Type: `exploratory`. Do not edit the rule after seeing results; add a dated note below instead.

## Why

E-009 (gradient concentration) found that from step 500 on, the top 30% highest-loss examples of a batch carry 98 to 99.7% of the gradient norm and point the same way as the full gradient (cosine 1.000). The idea: forward the whole batch (this also gives every example's loss), then run the backward pass only for the highest-loss rows. The saved work is real FLOPs, not just a different schedule.

## The arm (`arm: selective`, code in `src/selective.py`)

- Forward on the full batch under `no_grad`, keeping the activations. Per-example losses come for free.
- Select `k = ceil(sel_frac * B)` rows: the highest-loss rows (`sel_mode: top`) or, as a control, random rows of the same count (`sel_mode: random`). The first `sel_warmup` steps use the full backward pass (early on, all examples still matter).
- Hand-written backward for the plain MLP on the selected rows only, reusing the stored activations. The selected rows' gradient is divided by the full batch size `B`, so the update equals the full-batch gradient minus the omitted part. `tests/test_selective.py` checks that it matches autograd exactly for ReLU and tanh and that the FLOP count equals the analytic count.
- Optimizer, schedule, learning rate and weight decay are unchanged.

## Exact FLOPs per training step (batch 128; forward 68,812,800 is always paid)

| Variant | Rows through backward | FLOPs per step | vs A (155,058,176) |
|---|---|---|---|
| A (full backward) | 128 | 155,058,176 | 0% |
| S50, `sel_frac` 0.5 | 64 | 111,935,488 | -27.8% |
| S30, `sel_frac` 0.3 | 39 | 95,090,688 | -38.7% |

Over a 3,000-step run with a 100-step warm-up the total budget is 0.626 of A's for S30 and 0.731 for S50. For S30 to save at least 15% in FLOPs-to-target it may need at most 1.39 times A's steps; at 1.63 times it breaks even.

## The pilot (`scripts/pilot_selective.py`)

Seeds 600, 601, 602 (new). Variants: **A** (tuned baseline `configs/tuned/t1_base_two.yaml`, the better of the two A tunings), **S30**, **S50**, and the control **R30** (random 30%, same cost as S30). Every variant inherits A's tuned hyperparameters; only the backward pass differs. 3,000 steps, targets 0.97 and 0.975 under the shared crossing rule (two consecutive evaluations, D-018). A run that never reaches a target is charged its full-run FLOPs. Runs are saved under `results/raw/pilot/pilot1/` and never overwritten.

## Decision rule (primary comparison: ratio of mean FLOPs-to-target vs A)

- **GO:** S30 or S50 has ratio <= 0.85 at **both** targets and every seed reached both targets. Then build a pre-registered test.
- **STOP:** both S30 and S50 have ratio >= 1.00 at 0.975, or 2 or more seeds missed 0.975. Drop the idea; take the next one from the idea bank.
- **TUNE:** anything else. One more look (for example a different `sel_frac` or `sel_warmup`, at most two new variants), then decide.

R30 never decides the verdict. It is reported: if random rows do about as well as the highest-loss rows (within 5%), then selecting by loss does not matter and the saving would come from subsampling alone, which is a different (and known) claim.

## What the pilot cannot show

- 3 seeds with about 13% run-to-run noise give a ratio uncertainty of roughly 10%: a GO can happen by luck and a STOP can be unlucky. The pilot only decides whether a proper test is worth building.
- The S variants inherit hyperparameters tuned for the full gradient. They may do better (or worse) with their own tuning.
- MNIST MLP only. A hand-written backward for a CNN is more work, and an autograd "re-run the subset" implementation saves less (-26% at 30%).
- Tuning noise is large in this setup: two equally good tunings of A differed by 17 to 34% on fresh seeds (E-008). Comparing against A1, the better tuning, is the conservative choice.

## If the pilot says GO: decisions needed before `prereg-v2`

1. Hyperparameters of S: inherit A's tuned values, or tune S with the same trial budget (it has two extra dimensions, and control E suffered from extra dimensions with 25 trials).
2. Baseline: A1 plus at least one fresh, independent tuning of A, with the same noise-floor rule as before.
3. Seeds (20 for the primary arms), threshold (at least 15% fewer FLOPs), the three-way outcome (supported / refuted / inconclusive), and the control R30.
4. A second setting (for example a small CNN on CIFAR-10) so the claim is not only about one MLP.

## Related work to read before claiming anything (verify the citations)

Jiang et al. 2019, "Accelerating Deep Learning by Focusing on the Biggest Losers" (selective backprop); Katharopoulos and Fleuret 2018, importance sampling with gradient norms; Loshchilov and Hutter 2015, online batch selection. The idea is known; the possible contribution is a rigorous compute-matched test with tuning noise controlled.

## How to run

```
git am --3way selective-backward-pilot.patch
python -m pytest -q
git add -A && git commit -m "..." && git push        # commit the plan BEFORE running
python scripts/pilot_selective.py --tag pilot1       # 4 variants x 3 seeds x 3,000 steps; roughly 30 to 40 minutes
```

It prints the table and the verdict and writes `results/processed/pilot_selective_pilot1.csv`. Log it as an `exploratory` entry in `EXPERIMENT_LOG.md`.

## Result (added 2026-10-25 after the run; the rule above is unchanged)

Run at commit 22f2b3a, seeds 600 to 602, logged as E-010. **Verdict: GO.** S30 FLOPs-to-target ratio 0.634 (0.97) and 0.612 (0.975); S50 0.768 and 0.707; no missed seed; random control R30 1.413 and 1.181 (selection by loss matters: R30 costs +93.1% FLOPs-to-0.975 relative to S30). See `docs/EXPERIMENT_LOG.md` (E-010), `docs/FINDINGS.md` (F-012) and `docs/DECISIONS.md` (D-021, proposed).

Post-hoc remark (written after seeing the result, no effect on the verdict): the STOP clause "or 2 or more seeds missed 0.975" is implemented per variant (STOP needs every main variant to have ratio >= 1.00 or at least 2 misses). The verdict is GO under any reading, because S30 and S50 satisfy the GO condition.
