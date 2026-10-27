# Improve track: reference arms and candidate levers (spec before any code or run)

**Status:** specification written 2026-10-27 after reading the closest prior work (`docs/LITERATURE.md`, section "Selective backward and batch selection") and the official Selective-Backprop code. It fixes what the reference arms are and how their cost is counted. The choices of section 4 were accepted by the researcher on 2026-10-27 (D-025); the arms are implemented (`src/sb.py`, DR-014) and the headroom plan is `docs/notes/improve-headroom-plan.md`.

## 1. What a contribution has to beat
The rule behind our arm S (top-k rows of a forward batch, backward only through them) is Ordered SGD (Kawaguchi and Lu 2020) and, for region proposals, OHEM (Shrivastava et al. 2016). A kept fraction that shrinks during training is the default rule of Ordered SGD (q = s at the start, then s/2, s/4, s/8, s/16 when the training accuracy passes 80%, 90%, 95%, 99.5%). Selective-Backprop (Jiang et al. 2019) skips backward passes by a probabilistic rule at fixed training-batch size. So a method claim needs a comparison against these under our protocol, not only against A1.

## 2. Reference arm SB: Selective-Backprop as released (angelajiang/SelectiveBackprop, commit 7c41937)
Facts read in the code (`lib/SelectiveBackpropper.py`, `lib/calculators.py`, `lib/hist.py`, `lib/backproppers.py`, `examples/train.py`):
- Selection probability of an example = (percentile of its loss among the last `max_history_len` = 1024 forward losses)^beta, floor `sampling_min` = 0; `--prob_pow` (beta) defaults to 3 in `train.py`. If the percentiles are uniform the expected selection rate is 1/(beta+1): about 33% for beta 2 and 25% for beta 3 (my derivation, not from the paper).
- Examples are forwarded in batches of 128; each is kept independently with that probability; kept examples queue until 128 are available, then one update is made. The training batch therefore has the full size, and several forward batches feed one update.
- The backpropper runs a **second forward pass** on the selected examples and backpropagates the mean loss over them (no activation caching).
- `num_images_to_prime` is the number of training images: the first epoch is ordinary training with all backward passes.
- The repo also has top-k, low-k and random-k selector classes, but the shipped training path uses the probabilistic selector (`sample_size = 0`, "only needed for topk, lowk"). It is Python 2.7 and asserts a CUDA device, so it cannot be used as is for our CPU FLOP counting; we reimplement the rule inside our trainer.

**Cost per update in our FLOP counter** (forward 537,600 and backward 673,792 FLOPs per row; plain batch of 128 = 155,058,176; S30 = 95,090,688 after warm-up):
| Arm | Rows forwarded for selection | Training forward | Backward | FLOPs per update | Against A |
|---|---|---|---|---|---|
| SB, beta 2 (about 33% kept), uncached | 384 | 128 | 128 | 361.5 M | 2.33 |
| SB, beta 2, activations cached | 384 | 0 | 128 | 292.7 M | 1.89 |
| SB, beta 3 (about 25% kept), uncached | 512 | 128 | 128 | 430.3 M | 2.78 |
| SB, beta 3, cached | 512 | 0 | 128 | 361.5 M | 2.33 |
| S30 (ours) | 0 (selection is inside the batch) | 0 | 39 | 95.1 M | 0.61 |
So SB breaks even in FLOPs only if it needs fewer than about 43% (beta 2, uncached) to 53% (beta 2, cached) of A's number of updates. Whether it does is an empirical question; its gains in the paper are wall-clock on GPUs and image models. These ratios assume the algorithm as printed and released.

## 3. Candidate arms for a headroom study (exploratory; seeds 2100 to 2199, D-024)
Each arm is a lever from `component-ledger.md` or from the literature; none is combined with another without a stated reason.
1. **SB-faithful** (above): beta 2 and beta 3, history 1024, one-epoch priming, uncached; plus **SB-cached** (activations of the selected rows kept until the update; exact for an MLP).
2. **S, constant fractions** 0.1, 0.15, 0.2, 0.25, 0.3 (E-023 says 0.2 beats 0.3; where does the step penalty start?).
3. **S with the Ordered-SGD schedule** (q shrinking by the training-accuracy thresholds above, updated at the end of each epoch), to test the prior-art schedule under FLOPs; direction opposite to Loshchilov and Hutter, who decrease the selection pressure over epochs (their Eq. 5), so both directions are worth one arm.
4. **S with a different score:** the upper bound on the gradient norm of Katharopoulos and Fleuret (the norm of the last-layer gradient, for cross-entropy the norm of softmax minus one-hot) instead of the loss; free to compute, no extra pass.
5. **S warm-up** 0, 25, 50, 100 steps at the best fraction (the warm-up is 11% to 12% of the cost to 0.975).
6. **Overhead** (wall-clock): kept separate; it is an implementation question.

## 4. Decisions (accepted by the researcher on 2026-10-27; the defaults proposed below were taken as written)
- Primary SB setting: beta 2 or 3; one-epoch priming as released, or 100 steps as in S (the second isolates the selection rule from the warm-up). Default proposed: beta 2, released priming, plus the 100-step variant as secondary.
- Cached or uncached as the primary count. Default proposed: report both; the uncached count is the released algorithm, the cached one is what an ideal implementation costs.
- Whether the headroom study may include arm 3 and 4 before `prereg-v3` (they are exploratory; a confirmatory plan would then pick at most the arms with a stated reason).
- Any change to `src/` for the new arms follows D-024 (additive, drift entry, regression check, tests).

## 5. Implementation notes (`src/sb.py`, written 2026-10-27)
- Follows the released code: history of the last 1,024 forward losses updated before the percentiles are taken, strict "below" percentile, probability percentile^beta with floor 0, independent keep decisions, a queue of kept rows, at most one update per forward batch, a second forward pass for the update, one-epoch priming. Steps count updates.
- Priming steps are charged as ordinary steps of arm A (the released code also runs a redundant second forward pass there); with `sb_prime_steps` equal to the run length the log is bit-identical to arm A's (tested).
- `flops` is the released algorithm's count (every selection forward pass plus a full ordinary step per update); `flops_cached` charges only the backward part of the update. Both come from the same run; the gradient is always computed with a fresh forward pass.
- Ties: float32 cross-entropy is exactly 0 for very confident rows, so many losses can tie and the strict percentile of a tied row is the share strictly below it; the kept rate can then fall below 1/(beta+1) (on a trivially separable synthetic set it fell to 3%). This is the released behaviour; `sel_rate` and the forward batches per update are logged. A run stops with an error if forward batches exceed 200 times the update budget.
- Added to `selective_step`: `sel_score` ("loss" or "gradnorm"), `sel_schedule` ("osgd" or "linear"), and the optional `k`, `frac`, `stats` arguments; the defaults leave every existing run bit-identical (DR-014).

