# Check A: what is the geometry of the meeting signal? (parallel/orthogonal split and Gauss-Newton cosine)

**Status:** plan and decision rule, written BEFORE the first real run. Type: `exploratory`. Do not edit the rule after seeing results; if it needs to change, add a dated note below instead. Part of the mechanism study (`mechanism-overview.md`).

## The idea being checked

The meeting loss failed (E-008) although its gradient is aligned with the task gradient (cosine about +0.75 in all 20 runs). The working explanation is that the signal is **redundant**: backprop already sends the output error to the lower layers (as `J^T r`), and the meeting loss sends roughly the same error through a learned approximate inverse. Two sub-explanations follow, and both are measurable without any new method:

1. **Parallel/orthogonal split.** The part of the meeting gradient parallel to the task gradient only rescales it. The orthogonal part is either useful extra information or approximation noise of `G`.
2. **Curvature.** A perfect local inverse of the top half maps an output change back with a pseudo-inverse (scaling by `1/sigma`), whereas backprop uses the transpose (scaling by `sigma`). The Gauss-Newton (GN) step has the same flavour. If `G` has learned such a direction, the meeting signal differs from backprop in a structured way; if backprop's direction is already close to the GN direction, curvature cannot explain anything.

## Notation (middle state h, 128 dimensions; F_up = act, Linear 128-256, act, Linear 256-10)

Per example `i`: `z` logits, `p = softmax(z)`, `r = p - e_y`, `J = dz/dh` (10 x 128), `S = diag(p) - p p^T` (the softmax Hessian, rank 9).

| Direction (descent direction for h, per example) | Definition | Meaning |
|---|---|---|
| `u_bp` | `-J^T r` | what backprop sends to h |
| `u_meet` | `G(e_y) - G(p)` (arm D's actual G) | what the meeting loss pushes h by (`b - h`) |
| `u_pinv` | `(S J)^+ (e_y - p)` | what a *perfect local inverse* would give (min-norm change of h that moves p to e_y to first order) |
| `u_gn` | `(J^T S J)^+ J^T (e_y - p)` | the Gauss-Newton step of the cross-entropy loss in h; **the same vector as `u_pinv`** (see below) |
| `u_lsq` | `J^T (J J^T + alpha I)^-1 (e_y - p)`, `alpha_i = 0.01 x` mean eigenvalue of `J_i J_i^T` | the damped least-squares inverse in logit space (treats `e_y - p` as the logit change); the genuinely different alternative |

`u_pinv` and `u_gn` are the **same vector** whenever `J` has full row rank (generic here, 128 -> 10): both are the minimum-norm solution in the row space of `J` of `S J J^T w = e_y - p` (checked numerically and by a test; an earlier version of this note wrongly called them different objects). The code therefore computes `u_pinv` (the curvature direction, called `pinv` in the results) and `u_lsq`. The Jacobian has a closed form for this model, `J = W2 diag(act'(a2)) W1 diag(act'(h))` with `a2 = W1 act(h) + b1`; check it against `torch.func.jacrev` in a test.

## What the check measures

Train arm D (`configs/tuned/t1_meet_two.yaml`, tanh, 3,000 steps) on 3 new seeds (600, 601, 602). At steps 100, 200, 500, 1000, 1500, 2000 and 3000, take 16 fixed batches of 128 **training** examples (the E-009 protocol) and compute:

**Part 1, in h-space (per example, then averaged).** Cosines of the pairs (`u_meet`, `u_bp`), (`u_meet`, `u_pinv`), (`u_meet`, `u_lsq`), (`u_bp`, `u_pinv`), (`u_bp`, `u_lsq`), (`u_pinv`, `u_lsq`), and the norm ratio `||u_meet|| / ||u_bp||`. The primary mean is **weighted by `||u_bp||`** (examples that matter for the update); the unweighted mean is also reported. Examples with `p_y > 0.9999` are skipped (their loss is numerically zero and the pseudo-inverse is unstable); report the share skipped at each step.

**Part 2, in parameter space (lower layers `F_low`, as the H3 diagnostic).** On each batch: `g_task`, `g_meet` (the unscaled meeting-loss gradient), then
- `cos(g_task, g_meet)` and the parallel share `cos^2`,
- `||g_perp|| / ||g_task||` with `g_perp = g_meet - (g_meet . g_task / ||g_task||^2) g_task`, and `||g_meet|| / ||g_task||`,
- a **one-step probe**: move `F_low` by a step of fixed length `eps = 0.01 * ||theta_low||` along each of `-g_task`, `-g_meet` and `-g_perp` (unit directions) and record the change of `L_task` on a separate batch of 1,024 training examples that is not one of the 16 batches. The orthogonal direction has zero first-order effect by construction, so this measures its second-order effect, which is the honest question.

Measuring never changes the training run (separate random generator; tested, as in `tests/test_gradient_concentration.py`).

## Decision rule (three separate questions, mean over seeds, at steps 500, 1000 and 2000)

The thresholds are judgment calls fixed now, before any data.

- **Q1, redundancy of the signal in parameter space.** **REDUNDANT** if `cos(g_task, g_meet) >= 0.6` at all three steps **and** `|dL(-g_perp)| <= 0.1 * |dL(-g_task)|` at all three steps. **NOT REDUNDANT** if the second condition fails (the orthogonal part changes the loss by more than 10% of what the task direction does) at two of the three steps. Otherwise **INCONCLUSIVE**.
- **Q2, does G approximate a curvature direction?** **YES** if `cos(u_meet, u_pinv) >= cos(u_meet, u_bp) + 0.10` at steps 500 and 1000. **NO** if it is below `cos(u_meet, u_bp)` at both. Otherwise **UNCLEAR**. Report the same with `u_lsq` (descriptive).
- **Q3, does curvature matter at all for this network?** **BACKPROP ~ GN** if `cos(u_bp, u_pinv) >= 0.9` at all three steps (then a curvature explanation is dead and B's pinv oracle `P` should be expected to behave like its control `F'`). **GN DIFFERS** if `cos(u_bp, u_pinv) <= 0.8` at all three steps. Otherwise **GREY**.

## What each outcome leads to

- **Q1 REDUNDANT:** the redundancy explanation is supported (still an interpretation until B). Write it into FINDINGS as `exploratory`.
- **Q1 NOT REDUNDANT:** the orthogonal part carries something. Look at what (is it correlated with `u_pinv` or `u_lsq`?) before concluding, and treat B as more promising.
- **Q3 BACKPROP ~ GN:** expect B's pinv oracle `P` (which is the Gauss-Newton direction) to behave like the scaled-gradient control `F'`; run `F'` and `Q` first and `P` only as a check.
- **Q3 GN DIFFERS and Q2 YES:** B's oracle arms are well motivated; run them as planned.
- Any mixed outcome: report it as is. Do not tune the thresholds.

## What this check can NOT show

- It is geometry, not causation: a direction can differ from backprop's and still not help training (that is B's job).
- It measures arm D at its tuned configuration only (tanh, `lam_meet` = 0.005). The meeting signal's gradient is tiny at that setting; the cosines are scale-free but the training trajectory is dominated by backprop. Optionally repeat on V3 (frozen `G`) for contrast; it is not part of the rule.
- First-order and local: GN and pseudo-inverse directions are linearizations at the current point.
- MNIST MLP, AdamW, one split of the network.

## Cost

3 runs of 3,000 steps plus measurement. The Jacobians are 128 x (10 x 128) per checkpoint and cheap. Expect a duration close to E-009 (about 15 minutes on a laptop); time the smoke run first.

## How to run (once the code exists)

```
python -m pytest -q
git add -A && git commit -m "..." && git push        # commit the plan BEFORE running
python scripts/mechanism_check.py --tag ma1          # arm D, seeds 600-602, 3,000 steps
python scripts/mechanism_check.py --synthetic --max-steps 60 --checkpoints 20 40 60 --seeds 0 --tag smoke
```

Writes `results/processed/mechanism_a_ma1.csv` (one row per seed, step, quantity). Log an `exploratory` entry in `EXPERIMENT_LOG.md` with the printed answers to Q1, Q2 and Q3.

## Tests to write first

1. The closed-form Jacobian equals `torch.func.jacrev` of `model.up` (relu and tanh).
2. `u_bp` equals minus the autograd gradient of the per-example cross-entropy with respect to `h`.
3. For a linear top half (identity activation) `J` is constant; check that `u_pinv` solves `S J u = e_y - p`, that the Gauss-Newton step equals `u_pinv`, and that `u_lsq` equals the damped form `(J^T J + alpha I)^-1 J^T (e_y - p)` (all in `tests/test_mechanism.py`).
4. `g_perp` is orthogonal to `g_task` to numerical precision, and `g_par + g_perp = g_meet`.
5. Measuring does not change the training log (same run with and without checkpoints).
