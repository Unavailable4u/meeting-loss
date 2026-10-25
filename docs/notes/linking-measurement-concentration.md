# Check L: linking measurement, is the meeting gradient already "selective"? (E-009 analysis applied to the meeting loss)

**Status:** plan and decision rule, written BEFORE the first real run. Type: `exploratory`. Do not edit the rule after seeing results; if it needs to change, add a dated note below instead. Part of the mechanism study (`mechanism-overview.md`) and the link between the two studies: the meeting loss (refuted, E-008) and the selective backward pass (pilot GO, E-010).

## The idea being checked

The meeting loss did not help; selective backward did. One possible reason connects them: the meeting target is `h + G(e_y) - G(p)`, which is about `h` for any row the network already gets right, so the meeting gradient may live on the same high-loss rows that carry the task gradient (E-009: the top 30% of rows carry 98% to 99.7% of the gradient norm after about step 500). If so, the meeting loss supplies no signal where the task gradient is small, which would explain why it adds nothing next to backprop and why combining it with row selection cannot recover anything. If instead the meeting gradient is spread over low-loss rows, it carries a different kind of information and a hybrid becomes worth a look.

This reuses the E-009 code and check A's runs; it needs almost no new training.

## What is measured

Train arm D (`configs/tuned/t1_meet_two.yaml`) **and** arm A (`configs/tuned/t1_base_two.yaml`) on seeds 600, 601, 602 (the same runs as check A; train once and measure everything in one pass). At steps 100, 200, 500, 1000, 1500, 2000 and 3000, on 16 fixed batches of 128 **training** examples (the E-009 protocol), define `Sel` = the top 30% of rows by per-example loss (`k = 39`, the S30 selection rule) and compute, with the E-009 definitions (check `exploratory/gradconc.py` for the exact norm-share and cosine):

1. **Task gradient concentration on both models** (all parameters): norm share of `Sel` and cosine with the full gradient. On arm A this reproduces E-009; on arm D it checks that the finding is not specific to the baseline's trajectory.
2. **Meeting gradient concentration on arm D** (lower-layer parameters, unscaled `L_meet`): norm share of `Sel`, the cosine of the `Sel`-only gradient with the full meeting gradient, and the share of the squared norm that lies in the unselected rows.
3. **Row-level coupling:** the Spearman correlation across rows between the per-example loss and (a) the per-example norm of the meeting direction `||G(e_y) - G(p)||` and (b) the per-example norm of the task gradient at `h`.
4. **Overlap:** on the unselected rows, the cosine between the meeting gradient and the task gradient (is the meeting signal pointing the same way where the task gradient is small?).

Rows with `p_y > 0.9999` are kept (they are the point of the measurement) but reported separately as a share.

Measuring never changes the training run (separate random generator; the same kind of test as `tests/test_gradient_concentration.py`).

## Decision rule (mean over seeds, at steps 500, 1000 and 2000)

The thresholds are judgment calls fixed now, before any data.

- **ALREADY SELECTIVE:** the meeting gradient's norm share of `Sel` is at least 0.90 at all three steps **and** the loss correlation of `||G(e_y) - G(p)||` is at least 0.5 at all three steps. Reading: the meeting signal lives on the same rows as the task gradient, which supports the redundancy explanation and predicts that a hybrid adds nothing.
- **NOT SELECTIVE:** the share is at most 0.60 at two of the three steps. Reading: a substantial part of the meeting gradient sits on rows the baseline already handles. Check whether it is signal or noise (use check A's parallel/orthogonal split on those rows) before drawing any conclusion.
- **PARTIAL:** anything else.

Descriptive and never deciding: the task-gradient share on arm D versus arm A (a difference above 0.05 is worth a line in the log), the share of `p_y > 0.9999` rows at each step, and the overlap cosine of item 4.

## What each outcome leads to

- **ALREADY SELECTIVE:** write it into the mechanism section as `exploratory`; do **not** add the hybrid arm to `prereg-v2` (it is predicted to be a null and would only clutter the paper).
- **NOT SELECTIVE:** add the hybrid (meeting loss on the selected rows only) to `prereg-v2` as one clearly labeled *exploratory* arm with the expectation "null" written beforehand. It must not touch the primary hypothesis.
- **PARTIAL:** report as is; the hybrid stays optional.

## Cost accounting for the hybrid (analytic, for the record)

If the meeting loss is applied only to the 39 selected rows, its overhead falls from about 23% of A's per-step FLOPs (E-008 accounting) to about `0.23 x 39/128` = 7.0% (about 10.9M FLOPs per step), **assuming the overhead scales linearly with the number of rows** (check with the FLOP counter before relying on it). The per-step cost of the hybrid would then be about 0.683 of A, which is 1.114 times S30's. Parity with S30 would need about 10% fewer steps than S30, whereas the meeting loss has only ever needed about 10% *more* steps than the baseline (E-008).

## What this check can NOT show

- Whether the hybrid helps: that needs an arm and a test. This is a measurement of where the gradient lives, not a causal result.
- Anything beyond arm D at its tuned configuration (tanh, `lam_meet` = 0.005, where the meeting gradient is small next to the task gradient). Norm shares are scale-free, but the trajectory is dominated by backprop. Optionally repeat on the check B oracle (`P`) later; not part of the rule.
- MNIST MLP, AdamW, one split.

## Cost

No extra training beyond check A (arm D on 3 seeds) plus 3 runs of arm A. Measurement is a few forward and backward passes per checkpoint. Time the smoke run first.

## How to run (once the code exists)

```
python -m pytest -q
git add -A && git commit -m "..." && git push                    # commit the plan BEFORE running
python scripts/mechanism_check.py --tag ma1 --with-linking       # check A and check L in one pass (arms D and A, seeds 600-602)
python scripts/mechanism_check.py --synthetic --max-steps 60 --checkpoints 20 40 60 --seeds 0 --tag smoke
```

Template; extend `scripts/mechanism_check.py` from check A with a `--with-linking` option (or write `scripts/linking_check.py`). Writes `results/processed/linking_ml1.csv`. Log an `exploratory` entry in `EXPERIMENT_LOG.md` with the printed verdict.

## Tests to write first

1. With `sel_frac = 1.0` the `Sel`-only gradient equals the full gradient (share 1.0, cosine 1.0).
2. The row-restricted gradient equals the sum of the per-row gradients over `Sel` (check against per-example autograd on a tiny batch).
3. On arm A the task-gradient norm share reproduces the E-009 values within noise for the same checkpoint.
4. Measuring does not change the training log.
