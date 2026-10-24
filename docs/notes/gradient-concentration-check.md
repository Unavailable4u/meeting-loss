# Check: is the gradient concentrated in the examples the forward pass gets wrong?

**Status:** plan and decision rule, written BEFORE the first real run. Type: `exploratory`. Do not edit the rule after seeing results; if it needs to change, add a dated note below instead.

## The idea being checked (selective backward pass)

The meeting loss failed (E-008): it re-sent a signal backprop already had. A different way to spend fewer FLOPs is to stop doing backward work where the forward pass already agrees with the label. The plan:

1. Forward pass on the whole batch (needed anyway; it also gives every example's loss).
2. Backward pass only for the highest-loss fraction `f` of the batch (for example 30%).

If the rest of the batch contributes little to the gradient, the update barely changes and the step is cheaper. This is close to existing "selective backprop" and online batch selection work, so the novelty is low (to read and verify before any claim: Jiang et al. 2019 "Accelerating Deep Learning by Focusing on the Biggest Losers"; Katharopoulos and Fleuret 2018, importance sampling with gradient norms; Loshchilov and Hutter 2015, online batch selection). The value would be in a rigorous compute-matched test with our pipeline.

## What the check measures

Train the tuned baseline (arm A, `configs/tuned/t1_base_two.yaml`, 3,000 steps) on 3 new seeds (500, 501, 502). At steps 100, 200, 500, 1000, 1500, 2000 and 3000, take 16 fixed batches of 128 **training** examples (the same shuffling granularity the real algorithm would use), compute every example's gradient, and measure for the top-`f` examples by loss (`f` = 10, 20, 30, 50%) and for a random subset of the same size:

| Metric | Meaning |
|---|---|
| `norm_share` | sum of per-example gradient norms in the subset / sum over the batch |
| `cos` | cosine between the subset's gradient sum and the full batch gradient (does it point the same way?) |
| `proj_share` | share of the full gradient's squared norm that the subset accounts for |
| `omitted` | norm of the left-out examples' gradient / norm of the full gradient |
| `loss_share` | share of the total loss held by the subset |

Measuring never changes the training run (separate random generator; tested).

## Decision rule (primary: `f` = 30%, mean over seeds)

- **PROMISING:** `norm_share` >= 0.80 **and** `cos` >= 0.90 at all of steps 500, 1000 and 2000.
- **DROP:** `norm_share` < 0.60 at step 1000.
- **INCONCLUSIVE:** anything else. Then look at the table over fractions (a larger `f` such as 50% may still work) before deciding; this does not by itself justify building the method.

Reference: a random 30% would hold about 30% of the norm on average.

## What each outcome leads to

- **PROMISING:** build a "selective backward" arm and test it with the full pipeline (tuning with the equal-budget protocol, a noise floor with several baseline tunings, FLOPs counted honestly, pre-registered thresholds). This check is only a necessary condition.
- **DROP:** discard the idea; take the next idea from the idea bank.
- **INCONCLUSIVE:** one more look at larger fractions, then decide; do not tune the rule to the data.

## What this check can NOT show

- It does not prove the method saves compute. It measures a model trained with the **full** gradient; training with a selective backward pass follows a different trajectory (it over-weights hard examples, and with a smaller effective batch the gradient is noisier).
- High-loss examples may include mislabeled or ambiguous ones; their large gradients are not necessarily useful.
- Concentration is measured on the MNIST MLP only.

## Achievable savings if the check passes (exact counts, batch 128, from F-001 and F-003)

Forward costs 68.8M FLOPs per step and the backward pass 86.2M (total 155.1M for arm A).

| Fraction sent backward | Hand-written backward on the selected rows | Re-run forward and backward on the subset |
|---|---|---|
| 50% | 111.9M, **-28%** | 146.3M, -6% |
| 30% | 94.7M, **-39%** | 115.3M, -26% |

The selection step itself (sorting the batch's losses) is negligible. A run that needs more steps loses part of this saving, which is why the real test is FLOPs-to-target and not the per-step count.

## How to run

```
git am --3way gradient-concentration-check.patch
python -m pytest -q
git add -A && git commit -m "..." && git push      # commit the plan BEFORE running
python scripts/gradient_concentration.py --tag gc1  # 3 seeds x 3,000 steps, about 15 minutes on a laptop
```

It prints the summary table and the verdict and writes `results/processed/gradient_concentration_gc1.csv`. Seeds 500 to 502 are new (sweeps used 100, 101, 1000, 1001, 2000, 2001; Stage 3 used 0 to 19). Log the result in `EXPERIMENT_LOG.md` as an `exploratory` entry with the printed verdict.
