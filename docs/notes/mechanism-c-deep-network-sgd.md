# Check C: does the meeting loss help where backprop is weak? (deep plain MLP, SGD with momentum)

**Status:** plan and decision rule, written BEFORE the first real run. Type: `exploratory`. Do not edit the rule after seeing results; if it needs to change, add a dated note below instead. Part of the mechanism study (`mechanism-overview.md`). This is the largest check; it does not depend on A, B or D.

## The idea being checked

The meeting loss was sold as a **shorter signal path** to the lower layers. In the current 784-256-128-256-10 MLP the path from the output to the middle state is only two weight layers, so there is almost nothing to shortcut. Gains from a shortcut should appear where backprop's own signal degrades: a deep plain network (no residuals, no normalization). Two further limits of E-008 are covered at the same time: H4's optimizer requirement (AdamW normalizes the scale of an added gradient; SGD with momentum does not, R-19) and the comparison with **deep supervision (arm B)**, which gives the lower layers the same kind of shortcut. The meeting loss must beat B here to be interesting.

## Setting

- **Model.** A new `DeepMeetNet`: `depth` linear layers of width 128 (the first 784-128, the last 128-10), activation per arm (relu or tanh, D-014), PyTorch default initialization, no normalization, no residuals. The split (middle state `h`) is at `depth / 2`. `G` mirrors `F_up` (same depth and width, reversed), as METHOD's "G the same size as F_up" default; record the choice as a decision, since a deep top half makes the inverse harder.
- **Optimizer.** SGD with momentum 0.9, no Nesterov (new config key `optimizer: sgd`, `momentum`), weight decay and schedule (constant or cosine) tuned as before.
- **Arms.** A (plain), B (deep supervision at `h`, companion head, tuned `lam_aux`, constant or linear schedule), D (meeting loss at `h`), `F'` (the scaled exact-gradient control of check B: the lower layers receive `(1 + rho)` times backprop's gradient at `h`; this is the R-19 repair for SGD, record it as its own decision). A2 (second tuning of A) for the noise floor.
- **Targets.** 0.95 and 0.97 (plain SGD on a deep net is slower than the AdamW study; fixed now). Rule `two`.
- **Horizon.** 3,000 steps, unless the depth probe shows arm A cannot reach 0.95 at the chosen depth, in which case 6,000 steps for the whole study (fixed after the probe and before the sweep).

## Phase 0: depth probe (arm A only; seeds 800 and 801)

Depths {8, 16, 24, 32} linear layers, `lr` in {0.01, 0.03, 0.1}, relu, 3,000 steps. Log per-layer gradient norms (mean over steps 100 to 500) and the ratio `first-layer norm / last-layer norm`.

**Depth rule (fixed now).** Choose the smallest depth for which (i) A reaches 0.95 within the horizon in both probe seeds at its best `lr`, and (ii) the gradient-norm ratio is at most 0.1 (backprop is measurably weak). If no depth satisfies (ii), choose the deepest depth that satisfies (i) and state in the log that the weak-backprop premise could not be met here (the check then tests the premise only weakly). If no depth satisfies (i), stop and report; do not weaken the targets.

## Phase 1: equal-budget tuning (new tag `t3`, seeds 810 and 811)

25 random trials per arm (A, B, D, `F'`; plus A2 as a second 25-trial tuning), each on 2 seeds, scoring by FLOPs-to-0.95 with the D-017 and D-018 rules, as in the t1 sweep. Search spaces (new; check `src/sweep.py` for the existing format): `lr` log 1e-3 to 3e-1; `wd` log 1e-6 to 1e-3; `act` in {relu, tanh}; `lr_schedule` in {constant, cosine}; D: `lam_meet` log 1e-3 to 10 (wider than before because scale matters under SGD), `lam_inv` log 1e-2 to 1, `warmup_steps` in {200, 500, 1000}; B: `lam_aux` log 1e-2 to 1, `aux_sched` in {const, linear}; `F'`: `rho` log 0.1 to 10. The FLOP counter must be re-verified against exact analytic counts for the new model (as F-001) before any sweep.

## Phase 2: final runs (seeds 820 to 829, the same seeds in every group)

A, B, D, `F'`, A2 on 10 seeds each (50 runs). Tag `prereg-v3` before the first final seed, lock as in D-019 point 8.

## Decision rule (D vs A and D vs B; paired percentage change of mean FLOPs-to-target, same bootstrap as D-019 with a 97.5% interval because two comparisons share the error budget; both targets must agree)

- **SUPPORTED:** D vs A is *supported* under the D-019 rule (point at most -15%, interval excludes 0, gain larger than the A2-versus-A shift) at both targets, **and** D vs B has a point estimate below 0 with the interval excluding 0 at both targets.
- **REFUTED:** D vs A is *refuted* under the D-019 rule (the interval excludes a 15% gain) at both targets, whatever B does.
- **INCONCLUSIVE:** anything else.

Descriptive and never deciding: `F'` (is the effect only a rescale?), steps-to-target and the final accuracy, the per-layer gradient norms of every arm, the share of runs that never reach a target, A2's shift.

## What each outcome leads to

- **SUPPORTED:** the premise holds where backprop is weak. Write a new pre-registered confirmatory test on fresh seeds (D-015); do not claim it from this check.
- **REFUTED:** the signal does not help even where backprop is weak, in this setup. Together with B this closes the line for plain MLPs.
- **INCONCLUSIVE:** report as such. If B beats A and D does not, report that deep supervision is the shortcut that works here (and that H6 gets a second, independent look).
- **Probe says the weak-backprop premise could not be met:** the result applies only to the depth tested; say so in the paper.

## What this check can NOT show

- Anything about residual networks, normalization, transformers or convolutions; plain MLPs only, and deliberately so.
- That a failure of D is not a failure of this particular `G` (a deep inverse is hard, and `G` mirrors `F_up`). A larger or more tailored `G` is an unexplored variant.
- Wall-clock behaviour; FLOPs only.
- A confirmed effect: exploratory by D-015, and a pre-registered rule does not turn an exploratory design into a confirmatory one.

## Cost

Probe: 4 depths x 3 learning rates x 2 seeds = 24 short runs. Sweep: 5 arms x 25 trials x 2 seeds = 250 runs. Final: 50 runs. This is comparable to the main comparison of Stages 1 to 3 (300 main-arm sweep runs plus 150 final runs, all of 3,000 steps); a deep net is slower per step than the MLP, so time one run before starting. If the total is too large, the smallest cut that keeps the question intact is to drop A2 and `F'` from the sweep (use A's best trial for A2's role as a second look, `rho` = 1 for `F'`), which removes 100 sweep runs; record any such cut as a drift entry.

## How to run (once the code exists)

```
python -m pytest -q
git add -A && git commit -m "..." && git push                  # BEFORE the probe
python scripts/run_depth_probe.py --tag mc0
python scripts/run_sweep.py --tag t3 --arms base deepsup meet meet_grad_scaled ...   # check the CLI
git tag prereg-v3 && git push --tags                           # BEFORE the final runs
python scripts/run_final.py --plan configs/stage4_deep_plan.yaml
python scripts/analyze.py compare ...                          # D-019 settings, alpha 0.025
```

Check `scripts/run_sweep.py`, `scripts/run_final.py` and `scripts/analyze.py` for the real flags; these lines are a template. Log probe, sweep and final as separate `exploratory` entries in `EXPERIMENT_LOG.md`.
