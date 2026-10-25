# Check G0: fixed-hyperparameter dose-response of the learned meeting loss (paired seeds, rho as the dose)

**Status:** plan and decision rule, written BEFORE the first real run. Type: `exploratory`. Do not edit the rule after seeing results; if it needs to change, add a dated note below instead. Part of the mechanism study (`mechanism-overview.md`), Gate 0 (cheap, no new network structure).

## The idea being checked

E-008 compared tuned configurations, and tuning noise between two tunings of the same baseline was 17% to 34% (F-011). That makes it hard to separate "the meeting loss does nothing" from "the tuner picked a different configuration". A dose-response with **everything fixed except the dose** removes tuning noise by construction: same network, same optimizer, same schedule, same seeds, only the strength of the meeting signal changes.

The usual dose, `lam_meet`, is hard to interpret here: the tuned value is 0.005, the MSE loss is a mean over 128 dimensions (it divides the gradient by 128), and AdamW partly normalizes the scale of an added gradient. So the dose is defined directly as **rho, the norm of the meeting gradient at h relative to the task gradient at h**, per example.

## The arm `meet_rho` (learned G, normalized strength)

`G` is trained exactly as in arm D (inverse loss `L_inv`, same `lam_inv`). The meeting direction is the learned one, `u_i = G(e_y) - G(p)` (detached). The meeting signal reaching `F_low` is

```
L_meet_rho = -(rho / N) * sum_i  g_i * <u_i / ||u_i||, h_i>        # u_i, g_i detached
```

where `g_i = ||dL_task/dh_i||` in the units of the mean loss (the same normalization as the oracle arms of check B), so the gradient at `h_i` has norm `rho * g_i` and points along the learned direction. Obtain `g_i` from one extra `torch.autograd.grad(L_task, h, retain_graph=True)`. Examples with `p_y > 0.9999` get `u_i = 0`. `rho` ramps linearly from 0 to its value over the first 200 steps (fixed now, not tuned); `G` is trained from step 0.

## Design

- **Fixed:** arm A's tuned configuration (`configs/tuned/t1_base_two.yaml`: relu, AdamW, cosine schedule, `lr` 0.0013, `wd` 0.00013, batch 128, 3,000 steps). Nothing is tuned in this check.
- **Dose:** rho in {0, 0.1, 0.3, 1, 3}. `rho = 0` must reproduce arm A's validation curve exactly (unit test): `G` is trained but never reaches `F_low`.
- **Seeds:** 1100 to 1109 (new), the same seeds in every group (paired). 5 doses x 10 seeds = 50 runs; the `rho = 0` group doubles as the reference, so no separate A runs are needed.
- **Metric:** steps-to-target at 0.97 and 0.975 under the D-018 rule, and FLOPs-to-target (the `G` overhead is real and counted as in arm D). Paired percentage change versus `rho = 0`, 95% bootstrap interval (`src/sweep.py`); both targets must agree, as D-019. A run that never reaches a target is charged its full-run steps and FLOPs and counted in `n_never`.
- **Also log:** alignment with the task gradient at `F_low` (the H3 diagnostic) at each dose, and `h_var` and the inverse loss (the V2 warning signs seen in the Stage 2 diagnostics).

## Decision rule (steps-to-target; the 10% margin is about the break-even of the cheapest variants, see `mechanism-g-and-cost-deferred.md`)

- **DOSE HELPS:** at least one `rho > 0` has a paired mean change of at most -10% at both targets with the interval excluding 0.
- **DOSE HARMS:** the change is at least +10% at both targets for `rho = 1` and for `rho = 3`, and the point estimates do not decrease with `rho` for `rho >= 0.3`.
- **FLAT:** at every `rho > 0` the change is within +-5% at both targets and the interval includes 0.
- **MIXED:** anything else (report the full curve).

## What each outcome leads to

- **DOSE HELPS:** the signal is useful once its strength is set sensibly; the E-008 failure was partly a scale problem. Then check B's oracle arms become much more interesting, and a confirmatory design is warranted (new seeds, pre-registered, fixed reference baseline).
- **DOSE HARMS or FLAT:** consistent with E-008 and with redundancy; strengthens the negative result with a design that does not depend on the tuner. Continue to Gate 1 (check B) only if you want the oracle bound.
- **MIXED:** report the curve; do not tune.

## What this check can NOT show

- Anything about a better `G` (the direction is the learned one, not an oracle; check B covers that).
- Anything about other optimizers or networks (check C covers SGD and depth).
- The result is conditional on **arm A's tuned settings**, which favour `rho = 0` (they were tuned for the baseline). A null here says "no help at these settings", not "no help at any".
- Ten seeds can detect a change of about 10% only about half the time (E-007 simulations at 15%: roughly half came out supported), so INCONCLUSIVE-like outcomes (MIXED) are not evidence of no effect.

## Cost

50 runs of 3,000 steps with a slightly slower step than arm D (one extra gradient). About the size of the Stage 3 test of one arm pair times 2.5; time one run first.

## How to run (once the code exists)

```
python -m pytest -q                     # includes: rho = 0 reproduces arm A
git add -A && git commit -m "..." && git push        # BEFORE the runs
python scripts/run_dose_response.py --tag mr1        # rho in {0, 0.1, 0.3, 1, 3}, seeds 1100-1109
```

Template. Check `scripts/run_final.py` for a runner pattern (and its lock logic). Log as an `exploratory` entry in `EXPERIMENT_LOG.md`.

## Tests to write first

1. `rho = 0` reproduces arm A's validation curve step for step.
2. The gradient at `h` of `L_meet_rho` has per-example norm `rho * g_i` (compare against an explicit computation on a tiny batch).
3. The meeting direction is detached: no gradient from `L_meet_rho` reaches `G` (G's only loss is `L_inv`).
4. The FLOP counter's overhead for this arm equals the analytic count.

## Implementation notes (2026-10-26, before the first run)

Dated clarifications written while building the arm; none changes the idea, the doses, the seeds, the metric or the decision rule above.

1. **Normalization of `g_i`.** `g_i` is the per-example norm of the gradient of the MEAN task loss with respect to `h` (`torch.autograd.grad(L_task, h)`), so the code is `L_meet_rho = -(rho * ramp) * sum_i g_i * <u_i/||u_i||, h_i>` with no extra `1/N`. The formula above writes `rho / N`, which would count the `1/N` twice; the stated intent ("gradient at `h_i` has norm `rho * g_i`") is what the code and the unit test implement.
2. **`lam_inv`** is arm D's tuned value (0.105792, `configs/tuned/t1_meet_two.yaml`); `sigma_inv` is 0. `G` has the same shape as in arm D and is created after `low` and `up`, so the initial weights of `low` and `up` equal arm A's for a given seed (this is what makes `rho = 0` reproduce arm A).
3. **Ramp key.** `rho_warmup_steps` (200) is a separate key, because arm A's config carries an unused `warmup_steps: 500`.
4. **`rho` is required** (a missing key raises; there is no silent default).
5. **Masking.** Rows with `p_y > 0.9999` get `u_i = 0` (and so no meeting gradient).
6. **Rule overlap.** If DOSE HELPS and DOSE HARMS both hold, the verdict is MIXED. "Interval excluding 0" for HELPS means the upper end is below 0 at both targets. The FLAT and HARMS point-estimate conditions use the paired point estimate, as written.
7. **Cost.** The analytic overhead is 44,564,480 FLOPs per step at batch 128 (G forward twice 18,087,936; G backward for `L_inv` 17,432,576; the extra backward of `L_task` to `h` 9,043,968), +28.7% over arm A (arm D: +22.9%, F-003). It is checked by a unit test and counted in FLOPs-to-target.
8. **Tests** are in `tests/test_dose.py` (written first): `rho = 0` reproduces arm A bit for bit (relu and tanh); gradient norm at `h` equals `rho` times the task norm and points along `u`; confident rows get no meeting gradient; the ramp is linear and 0 at step 0; the meeting gradient reaches only `F_low`; the FLOP overhead matches the analytic count; the decision rule, the analysis and the runner.
9. **Where the results go.** Raw logs in `results/raw/dose/<tag>/` (outside the Stage 3 globs), configs in `configs/dose/` (generated, checked against their derivation in `exploratory/dose.py`), table in `results/processed/dose_response_<tag>.csv`.
10. **Not yet done:** the real-data regression check after the `src/` change (`scripts/verify_regression.py`, needs MNIST) and the real run. Both are run by the researcher; see DR-010.
