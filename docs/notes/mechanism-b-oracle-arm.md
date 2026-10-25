# Check B: does the signal help if G were perfect? (oracle arms, cost ignored)

**Status:** plan and decision rule, written BEFORE the first real run. Type: `exploratory`. Do not edit the rule after seeing results; if it needs to change, add a dated note below instead. Part of the mechanism study (`mechanism-overview.md`). Run it after check A (it reuses A's direction code and A's Q3 tells you whether the GN arm is worth running).

## The idea being checked

Every meeting-loss variant so far had an imperfect, learned `G`. If the signal were redundant for *any* `G`, then a perfect `G` would not help either, and the line is closed cleanly. If a perfect (or curvature-corrected) target does reduce steps, the problem is `G`'s quality and cost, and engineering a cheaper `G` is worth a confirmatory test. This arm is **not a method**: it computes exact quantities from the network's own Jacobian, so its FLOPs are not meaningful (they are logged but ignored).

It also repairs control F (R-19): F was nearly inert because its gradient was scaled by about `0.0016 * lam_meet`. Here the injected gradient is scaled **explicitly** relative to the task gradient.

**Why rho and not `lam_meet`.** The tuned `lam_meet` of arm D is 0.005, the MSE loss is a mean over 128 dimensions (it divides the gradient by 128), and AdamW partly normalizes the scale of an added gradient. A tiny tuned `lam_meet` is therefore not a clean statement about the signal. So the strength is stated as `rho`, the gradient norm at `h` relative to the task gradient's, and the conclusion below is worded "no help in this setup", never "proven redundant".

**Which output change the inverse is applied to (stated explicitly).** `P` works in **probability space** (`e_y - p`, through the softmax Hessian `S`, as `G` does); `N` is the Gauss-Newton step of the cross-entropy; `Q` is a damped least-squares inverse in **logit space** with the gradient step `e_y - p` as the logit change (no softmax weighting).

## The arms (all additive next to normal backprop, F_low receives the extra gradient through h)

For each example `i` let `g_i = ||J_i^T r_i||` (the norm of the task gradient at h, in the units of the mean loss: divide by the batch size N) and let `u_i` be the oracle direction (definitions in check A; descent direction for h). Add the surrogate loss

```
L_oracle = -(rho / N) * sum_i  g_i * <u_i / ||u_i||, h_i>        # u_i, g_i detached
```

whose gradient at `h_i` has norm exactly `rho` times the task gradient's per-example norm at `h_i`, pointing along `u_i`. Examples with `p_y > 0.9999` get `u_i = 0`.

| Arm | Direction `u_i` | What it tests |
|---|---|---|
| `F'` (scaled control) | `-J^T r` (backprop's own direction) | The pure rescale: the lower layers see `(1 + rho)` times backprop's gradient on h. Replaces F under SGD later (R-19). |
| `P` (pinv oracle) | `(S J)^+ (e_y - p)` | The meeting loss with a perfect local inverse `G`. |
| `N` (GN oracle) | `(J^T S J)^+ J^T (e_y - p)` | A Gauss-Newton direction (curvature-corrected). Skip if check A's Q3 is **BACKPROP ~ GN**. |
| `Q` (damped logit-space oracle) | `J^T (J J^T + alpha I)^{-1} (e_y - p)`, `alpha_i = 0.01 * ||J_i||_F^2 / 10` | Damped least-squares inverse of the top half, equal to `(J^T J + alpha I)^{-1} J^T (e_y - p)` (compute the 10 x 10 form). `alpha` removes the instability of the pseudo-inverse where `J` is nearly singular. Robustness: also `alpha` x 10 and x 0.1 on seeds 700 to 702. |

Baselines: arm A (`t1_base_two.yaml`) and A2 (`t2_base_two.yaml`), both re-run on this check's seeds. The oracle arms use **arm A's tuned config unchanged** (relu, AdamW, cosine, `lr` 0.0013). That favours A (it was tuned, the oracle arms were not), so the phase 3 safeguard below exists.

## Protocol

- **Phase 1 (screen).** `rho` in {0.3, 1, 3} for each of `F'`, `P`, `N`, `Q` (12 arms) plus A, on seeds 700 to 702, 3,000 steps. (The dose-response of the *learned* loss is check G0, `mechanism-dose-response.md`, which should be run first.) For each type choose the `rho` with the lowest mean steps-to-0.97 (rule: two). No other tuning.
- **Phase 2 (test).** Seeds 710 to 719: A, A2, and `F'`, `P`, `N`, `Q` at their chosen `rho` (60 runs).
- **Phase 3 (fair-shot safeguard, only if phase 2 would end NO HELP).** For the best of `P`, `N` and `Q`, repeat phase 2 with `lr` multiplied by 0.5 and by 2 (everything else unchanged), on the same seeds (20 runs). The verdict is then taken over the best of the three learning rates for that arm, chosen on phase-2 seeds 710 to 714 and evaluated on 715 to 719. State this selection in the log.

The metric is **steps-to-target** (cost is ignored here) at 0.97 and 0.975 under the D-018 rule; FLOPs-to-target is reported for reference.

## Decision rule (phase 2; paired percentage change in mean steps-to-target vs A, 95% bootstrap interval, `src/sweep.py`; both targets must agree, as D-019)

The 10% margin is about the break-even of the cheapest meeting variants (+12% FLOPs per step needs about 11% fewer steps), so a smaller step gain could not pay for a real `G`.

- **ORACLE HELPS:** for `P`, `N` or `Q`, at both targets the point estimate is at most -10% against A **and** against A2, the interval against A excludes 0, **and** the point estimate is at least 5 points below that of `F'`.
- **NO HELP (in this setup):** for every oracle type that was run (after phase 3), at both targets the lower end of the interval against A is above -10% (the interval excludes a 10% gain).
- **INCONCLUSIVE:** anything else. Do not add seeds to the same test (D-019 point 2). With 10 seeds a true -10% change often ends INCONCLUSIVE (E-007: at a 15% margin only about half of the simulations at a true -15% came out supported), so INCONCLUSIVE is not evidence of no effect.

## What each outcome leads to

- **HELPS:** the information is in the direction, not only in the scale. Plan a confirmatory test and then the deferred note `mechanism-g-and-cost-deferred.md` (the gate there gives the required step reduction for each overhead).
- **NO HELP:** no support for the signal in this setup: even a perfect local inverse, a curvature step or a damped logit-space inverse adds nothing measurable next to backprop at A's tuned settings. Together with check A and check G0 this is the strongest form of the mechanism result; it is not a proof of redundancy (one network, one optimizer, 3,000 steps, 10 seeds).
- **INCONCLUSIVE:** report it as such; run check C (deep network, SGD), where an effect is more likely, before deciding.
- If `F'` beats A: the gain is about scale, not direction; report it and note that it is equivalent to a lower-layer learning-rate change (control E).

## What this check can NOT show

- The oracle is local and first-order (a linearization at the current h). A nonlinear inverse could behave differently.
- Under AdamW, the *scale* of an added gradient is partly normalized away, so `rho` mainly changes the direction mix; the SGD version belongs to check C.
- One tuned configuration of A, one network, MNIST, 3,000 steps. A null here says nothing about deeper networks.
- Noise: with 10 seeds, tuning noise between A and A2 (17% to 34% on final seeds, F-011) is as large as the effect being looked for. That is why both baselines are used.

## Code needed

- Arm keys `meet_oracle_{grad,pinv,gn,logit}` in `src/model.py` / `compute_losses` (config keys `oracle_type`, `oracle_rho`), implemented with the closed-form Jacobian from check A.
- A regression test: with `rho = 0` every oracle arm reproduces arm A's log bit for bit; `F'` with `rho` equals arm A with the lower-layer gradient scaled by `(1 + rho)` (check against a direct hook).
- `configs/stage4_oracle_plan.yaml` listing groups and seeds, with the same lock logic as `scripts/run_final.py` (tag `prereg-v2` before phase 2).

## How to run (once the code exists)

```
python -m pytest -q
git add -A && git commit -m "..." && git tag prereg-v2 && git push      # BEFORE phase 2
python scripts/run_oracle.py --phase 1 --tag mb1
python scripts/run_oracle.py --phase 2 --tag mb1
python scripts/analyze.py compare ...   # as in D-019, margin 10, alpha 0.05, steps metric
```

Log each phase as an `exploratory` entry in `EXPERIMENT_LOG.md`. Check `scripts/analyze.py` options before use; it may need a `--metric steps` flag.
