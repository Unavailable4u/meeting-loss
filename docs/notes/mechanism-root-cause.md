# Root-cause checks: B (oracle signal), D (replacement), the repaired control F', and a tuned tanh baseline

**Status:** plan and decision rules, written BEFORE the first real run (2026-11-26). Type: `exploratory` (D-015). Decision: D-028. Drift: DR-019. Code: `src/oracle.py`, arms in `src/model.py`, sweep arms in `src/sweep.py`, plans and rules in `exploratory/oracle.py`, runner `scripts/run_root_cause.py`, tests `tests/test_oracle.py`. Do not edit the rules after seeing results; if one has to change, add a dated note at the bottom.

This note implements and extends two earlier plans, `mechanism-b-oracle-arm.md` (check B) and `mechanism-d-replacement-test.md` (check D). Where it differs from them, the differences are listed in "Deviations from the earlier plans" and are decided here, before any run.

## The question

Study 1's meeting loss (arm D) costs +23% FLOPs per step and needs about 10% more steps than arm A; it is +36% FLOPs to target (E-008). The mechanism checks so far (A, G0 to G2) say the signal is a near duplicate of backprop's and, at its tuned weight, about 0.1% of it. Three things stop that from being a root cause:

1. **Never tested with a perfect inverse.** Every meeting variant had an imperfect learned `G`. If even an exact local inverse adds nothing next to backprop, the problem is the signal, not `G`. (check B)
2. **Never tested as a replacement.** The meeting signal was only an add-on. Whether it can train the lower half on its own is untested. (check D)
3. **The control meant to separate direction from scale was inert.** Control F's gradient is about `6e-5` of the task gradient at its tuned weight (F-005, R-19), so "F rules out a step-size explanation" is not supported by F. And the tanh evidence (G2, E-014) used arm A's relu-tuned `lr` and `wd`, so "tanh changes the picture" and "tanh was badly tuned" are not separated.

The checks below are built to answer, in order: is the deficit about **direction**, **scale**, **G's quality**, **sufficiency**, or **tanh tuning**?

## What is built

### Oracle signal (`src/oracle.py`, arm `meet_oracle`)
For each example, an exact direction `u_i` in the middle space `h`, from the closed-form Jacobian `J = dz/dh` of the top half, computed in float64:

| type | direction (descent direction for h) | what it is |
|---|---|---|
| `grad` | `J^T (e_y - p)` | backprop's own direction. With `rho` this is the **repaired control F'**: the lower half sees `(1 + rho)` times backprop's gradient on h. |
| `pinv` (P) | `(S J)^+ (e_y - p)`, `S = diag(p) - p p^T` | a perfect local inverse, as `G` would be if exact; also the Gauss-Newton step |
| `logit` (Q) | `J^T (J J^T + alpha I)^{-1} (e_y - p)` | damped least-squares inverse in logit space (`alpha` = 1% of the mean eigenvalue of `J J^T`) |

The injected signal is the linear surrogate `L = sum_i <c_i, h_i>` with `c_i = -rho * n_i * u_i/||u_i||`, `n_i` the per-example norm of the task gradient at h. So the gradient it sends to h has norm **exactly `rho` times the task gradient's** and points along `u_i`. Strength is set explicitly, not through `lam_meet` (which the dose-response checks showed is a tiny, AdamW-normalised dose). Rows with `p_y > 0.9999` get 0.
- **Bounded scale (`Pr`, `oracle_bound: radial`).** P with the component along `h_i` removed, so to first order the signal cannot stretch or shrink `h`. G0 found that an unbounded linear signal inflated `h` in relu (F-014); `Pr` checks whether P's behaviour is a scale artifact. First-order only; `h_var` is logged to verify.
- Cost is ignored (the oracle uses the true Jacobian); only steps-to-target is the outcome.

### Replacement (`meet_replace`, `meet_replace_frozen`, `meet_replace_oracle`)
`z = F_up(h.detach())`: the task loss trains `F_up` only; the lower half receives **only** the meeting signal, from step `replace_start` (0 by default). Meeting loss as in the check D plan: sum over the 128 dimensions, mean over the batch, weight `lam_meet = 1`, no ramp. `G` is trained from step 0 so it is ready at any switch. `meet_replace_frozen` has a frozen random `G` (the V3 recipe, no inverse loss); `meet_replace_oracle` uses an oracle direction with `rho = 1`.

### Sweep arms (`src/sweep.py`)
- `base_tanh` (tag `t13`): arm A with `act` fixed to tanh; `lr`, `wd`, schedule searched, 25 trials x 2 seeds, D-017 score, rule `two`. This is the **tuned tanh baseline**.
- `meet_replace` (tag `t4`): learned replacement, `act`, `lr`, `wd`, schedule and `lam_inv` searched, 25 x 2 seeds.
Both use `None` in an arm's extra dimensions to remove a common one. Every existing sweep arm's trials and configs are bit-identical (42 pinned digests, `tests/fixtures/sweep_digests_before_dr019.json`).

## Two settings, so activation is not a confound
Check B runs in **relu** (arm A as tuned, `t1_base_two`) and in **tanh** (arm A tuned in tanh, `t13_base_tanh_two`). Phase 2 also runs arm D as tuned (tanh, `t1_meet_two`) on the same seeds as the tanh baseline, descriptive only: it answers whether D's deficit is a tanh/tuning effect or a meeting-loss effect, and whether the tuned tanh baseline is itself slower or faster than relu.

## Protocol

**Tuning first** (selection seeds; same budget as the t1 sweep):
`t13` (tanh baseline) and `t4` (learned replacement). Then `analyze.py select --write-configs` writes `configs/tuned/t13_base_tanh_two.yaml` and `t4_meet_replace_two.yaml`; commit them.

**Check B, tag `mb1`** (`exploratory/oracle.py`, rho grid, seeds and rules are in the module docstring and are the rules of this note):
- Phase 1 (screen), seeds 700 to 702: `rho` in {0.3, 1, 3} for F', P, Q in both settings, plus A. The chosen `rho` per (setting, type) is the one with the lowest mean steps-to-0.97 (ties: the smaller). Written once to `configs/oracle/mb1_chosen_rho.json` and committed before phase 2.
- Phase 2 (test), seeds 710 to 719: A (A2 in relu), F', P, Q, Pr at the chosen `rho` (Pr uses P's); arm D in tanh.
- Phase 3 (learning-rate safeguard), seeds 710 to 719: P and Q at `lr` x 0.5 and x 2. `lr` is **chosen on seeds 710 to 714** among {0.5, 1, 2} and **evaluated on 715 to 719** (a 5-seed test; wide intervals, reported as such).

**Check D, tag `md1`**, seeds 900 to 909, one phase: A (relu; A2 in relu; A in tanh) as references; R1 learned `G` (tuned `t4`), R1w (R1 with 300 steps of backprop first), R2 oracle P with `rho = 1`, R4 frozen random `G`; R2 and R4 in both settings.

Run counts and measured cost (1 thread, 3,000 steps, synthetic-data benchmark of compute only): about 0.2 to 0.6 min per run; phase 1 = 60 runs, phase 2 = 120, phase 3 = 80, check D = 90, tuning = 50 + 50.

## Decision rules (fixed before any run)
Metric: steps-to-target at 0.97 and 0.975, rule `two` (D-018), misses charged the full run, paired change of the mean, 95% bootstrap interval; **both targets must agree**.

**Check B, per setting (phase 2):**
- **ORACLE HELPS:** for some type X in {P, Q, Pr}, at both targets the point change against A is at most -10%, the interval against A excludes 0, the point is at most -10% against A2 too (relu only), and it is at least 5 points below F'.
- **NO HELP:** for every X, at both targets the lower end of the interval is above -10%.
- **INCONCLUSIVE:** anything else. No seeds are added to the same test.
- Phase 3 uses the same two outcomes on seeds 715 to 719.

**Check D (per group against its reference, which is arm A in the same activation):**
- **SUFFICIENT:** reaches both targets in at least 8 of 10 seeds and its paired mean steps are within +15% of the reference at both targets.
- **INSUFFICIENT:** reaches 0.97 in fewer than 5 of 10 seeds, or its mean final accuracy is more than 1 point below the reference's.
- **PARTIAL:** reaches 0.97 in at least 8 of 10 seeds but is not SUFFICIENT. **UNCLEAR:** anything else. (Checked in the order SUFFICIENT, INSUFFICIENT, PARTIAL, UNCLEAR.)
- Pre-written reading (`diagnose_replace`): R1 SUFFICIENT means replacement is viable (sufficiency, not a speedup); R1 short but R2 SUFFICIENT means `G`'s quality is the bottleneck; both short means the local-inverse signal does not carry enough; R4 equal to R1 means `G` adds nothing over a random feedback matrix.

## How to read the outcomes: where the root cause points

| Result | Reading |
|---|---|
| B: F' helps, P/Q do not | the gain is scale (a learning-rate effect, control E), not the meeting direction |
| B: P or Q HELPS, F' does not | the **direction** carries information; the learned `G` is the bottleneck. Next: improve `G` (capacity, input, training), then the cost gate |
| B: NO HELP in both settings | no support for the signal even with a perfect local inverse: the signal is redundant with backprop on this network. Study 1 cannot be improved by a better `G` |
| B: NO HELP in relu, HELPS in tanh (or the reverse) | activation-dependent; look at `h_var` and alignment before claiming anything |
| B: Pr differs from P | P's behaviour is a scale artifact |
| D: R1 SUFFICIENT or PARTIAL | the signal can train the lower half; Study 1's role changes from "add-on" to "substitute" (sufficiency only) |
| D: R2 SUFFICIENT, R1 not | `G` quality is the bottleneck |
| D: R1 and R2 INSUFFICIENT | the local-inverse signal is not a substitute for backprop here |
| tanh baseline vs relu baseline | whether tanh itself is slower; D vs tanh A shows how much of the D deficit was tuning |

## What these checks can NOT show
- A speedup. Replacement saves 5.8% of A's FLOPs per step (D plan), less than `G` costs; any benefit would have to come from parallel execution, which one device cannot measure. B's oracle ignores cost by construction.
- The oracle is local and first-order. A nonlinear inverse could differ.
- One network, one split, MNIST, AdamW, 3,000 steps, 10 seeds. With 10 seeds a true -10% change often ends INCONCLUSIVE (E-007); INCONCLUSIVE is not evidence of no effect. A and A2 differ by 17% to 34% on fresh seeds (F-011), which is why A2 is used in relu. **There is no A2 in tanh**: a tanh verdict rests on one tuned baseline, and a tanh HELPS result is therefore weaker than a relu one.
- The SGD and deeper-network versions (check C) are not run here.

## Deviations from the earlier plans
Decided here, before any run:
1. **tanh setting added** (B and D) with its own tuned baseline (`t13`), and arm D as tuned in tanh on the same seeds (descriptive). The earlier plans used relu only.
2. **`Pr` added** (radial bound), because "bounded scale" was requested and G0 showed an unbounded linear term inflating `h` in relu.
3. **Phase 3 always runs** (the B plan ran it only after a NO HELP), and `lr` is chosen on half the seeds and evaluated on the other half, so that the multiplier is not selected and tested on the same seeds.
4. **The `rho` choice rule** is written here (lowest mean steps-to-0.97, ties to the smaller); the B plan said "chosen" without a rule.
5. **Check D tuning is simpler:** `replace_start` is fixed at 0 for R1 and R1w is the fixed 300-step variant, instead of tuning the switch point in {0, 100, 300, 600}. R1's tuned activation is whatever the sweep selects, and its reference is arm A in that activation. The plan's robustness run of R2 and R4 on R1's tuned config (seeds 900 to 902) is replaced by running R2 and R4 in both activations.
6. **No `prereg-v4` lock tag and no `configs/stage4_*_plan.yaml`.** These are exploratory (D-015), like the dose checks: configs are derived in `exploratory/oracle.py`, written by `--write-configs`, checked against the derivation before every run, and committed before the runs. If a confirmatory test follows, it needs its own lock then.
7. **Per-run directions are computed in float64 and the probabilities recomputed there**: float32 probabilities do not sum to 1 closely enough and make the pseudo-inverse of `S J` blow up (found while comparing with check A's float64 code; guarded by `test_pinv_is_a_float64_computation`).
8. **Report text not touched.** ch06 says the meeting gradient is "16 to 21% of the task gradient's" at the tuned weight; E-012 and F-013 give that as the unscaled ratio (about 0.1% at the tuned weight). This is a correction to make in the report separately, whatever these checks find.

## Verification already done (before any real run)
- `tests/test_oracle.py` (58 tests): rho = 0 reproduces arm A for every oracle type, relu and tanh; replacing backprop by the exact gradient (cut at h, type grad, rho 1) reproduces arm A; the injected gradient has norm `rho` times the task gradient's and points along the oracle direction; F' gives `(1 + rho)` times backprop at h; the radial bound is orthogonal to `h`; directions equal check A's (`exploratory/mechanism.py`); in replacement arms the task loss does not reach `F_low`, the meeting term does, `G` is trained by `L_inv` only, the frozen `G` has no gradient; existing sweep arms are bit-identical; the decision rules and the plan derivation are covered; the runner's config check refuses a hand edit.
- Full suite: 304 passed (the 244 existing tests, 58 new, 2 new parametrisations of existing sweep tests). Two existing tests were extended, not changed in meaning (DR-019).
- An end-to-end smoke of both checks, all phases, on a synthetic dataset (MNIST was not reachable from the build environment): the pipeline runs, resumes and analyses. It says nothing about accuracy. **`scripts/verify_regression.py` on real MNIST has not been run for this change** and is the first command below.

## Commands, in order
Run from the repo root. Commit at each marked point (the configs and the chosen `rho` are fixed before the runs they feed).

```bash
# 0. environment and checks of the code (run these first)
pip install torch torchvision numpy pyyaml scipy pytest
python -m pytest -q                                        # expect 304 passed
python scripts/verify_regression.py                        # real MNIST; arm A seed 0 must still match its Stage 3 log (DR-007)

# 1. tuning (about 50 runs each; selection seeds follow the tag x 1000 convention)
python scripts/run_sweep.py --tag t13 --arms base_tanh   --seeds 13000 13001 --workers 4 --threads 1
python scripts/analyze.py select --tag t13 --arms base_tanh   --write-configs
python scripts/run_sweep.py --tag t4  --arms meet_replace --seeds 4000 4001   --workers 4 --threads 1
python scripts/analyze.py select --tag t4  --arms meet_replace --write-configs
git add configs/tuned results && git commit -m "tuning t13 (tanh baseline) and t4 (learned replacement)"     # COMMIT

# 2. check B, phase 1 (screen): 60 runs
python scripts/run_root_cause.py --tag mb1 --phase 1 --write-configs
git add configs/oracle && git commit -m "mb1 phase 1 configs"                                              # COMMIT
python scripts/run_root_cause.py --tag mb1 --phase 1 --workers 4
python scripts/run_root_cause.py --tag mb1 --phase 1 --analyze-only --write-chosen
git add configs/oracle results && git commit -m "mb1 phase 1 and chosen rho"                               # COMMIT

# 3. check B, phase 2 (test): 120 runs
python scripts/run_root_cause.py --tag mb1 --phase 2 --write-configs
git add configs/oracle && git commit -m "mb1 phase 2 configs"                                              # COMMIT
python scripts/run_root_cause.py --tag mb1 --phase 2 --workers 4

# 4. check B, phase 3 (lr x 0.5 / 2): 80 runs
python scripts/run_root_cause.py --tag mb1 --phase 3 --write-configs
git add configs/oracle results && git commit -m "mb1 phase 2 results, phase 3 configs"                     # COMMIT
python scripts/run_root_cause.py --tag mb1 --phase 3 --workers 4

# 5. check D (replacement): 90 runs
python scripts/run_root_cause.py --tag md1 --write-configs
git add configs/oracle results && git commit -m "mb1 phase 3 results, md1 configs"                         # COMMIT
python scripts/run_root_cause.py --tag md1 --workers 4
```

Every command accepts `--dry-run` (what would run), `--limit N`, `--analyze-only`; a stopped run resumes where it stopped and never overwrites a log. `--settings relu` or `--settings tanh` restricts a phase to one activation. Verdicts print at the end and go to `results/processed/oracle_<tag>_p<phase>.csv`. Raw logs: `results/raw/oracle/<tag>/`.

Log each phase as an `exploratory` entry in `EXPERIMENT_LOG.md` (E-028 onward) and add findings to `FINDINGS.md`.
