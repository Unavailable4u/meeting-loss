# Hypotheses

**Pre-registration status:** READY TO LOCK (finalized in D-019 on 2026-10-23; the lock is the git tag `prereg-v1` on the commit that contains this text, set **before** the Stage 3 main runs). Edits on 2026-10-21 (DR-002), 2026-10-22 (DR-004) and 2026-10-23 (DR-005) were made before the lock and are not post-hoc. After the tag, any change is labeled **post-hoc** and logged in `DRIFT_LOG.md`.

**Outcome rule for H1 and H6 (D-019, point 3):** per target in {0.97, 0.975} (0.98 reported, not decisive), the paired percentage change in mean FLOPs-to-target relative to arm A with a two-sided 97.5% paired-bootstrap interval [lo, hi] (10,000 resamples, bootstrap seed 0): **supported** if the point estimate is at most -15% and hi is below 0; **refuted** if lo is above -15% (the interval excludes a gain of 15% or more); **inconclusive** otherwise. Both primary targets must agree, and a gain must exceed the re-tuning shift of arm A (A2 vs A at the same target), or the verdict is inconclusive. Seeds 0 to 19 for A, D, B and A2; 0 to 9 for every other group. No seeds are added after the results are seen.

**Null result (applies to all):** no difference, or a loss, once the backward network's compute is counted.

---

## H1 (primary): the meeting loss makes training more efficient

- **Statement:** Adding the meeting loss reduces the steps and FLOPs needed to reach a target validation accuracy, compared with a tuned baseline at equal total compute.
- **Metrics:** FLOPs-to-target (decisive) and steps-to-target (reported) at targets 0.97 and 0.975 (0.98 secondary). A target is reached at the first of two consecutive evaluations at or above it (D-018). Arm D (tuned config `t1_meet_two`) against arm A (`t1_base_two`), seeds 0 to 19, 3,000 steps.
- **Pre-registered outcome (D-019, locked with `prereg-v1`):** *supported* if, at both 0.97 and 0.975, the paired change in mean FLOPs-to-target is at most -15% and its 97.5% bootstrap interval excludes zero, and the gain exceeds the A re-tuning shift; *refuted* if, at both targets, the 97.5% interval excludes a gain of 15% or more (lower bound above -15%); *inconclusive* otherwise (including when the two targets disagree). Note: arm D costs about 23% more FLOPs per step (F-003), so it needs about a 19% step reduction just to break even.
- **Expectation recorded before the runs:** a null or negative result is the likely outcome (F-006, F-008; R-20).
- **Status:** open (locked, not yet run)
- **Linked findings:** F-006, F-008 (exploratory context only)
- **Paper sections:** Introduction, 5.1

## H2 (specificity): the backward-derived target is what matters

- **Statement:** The gain is not matched by other auxiliary losses at the same layer: DSN-style deep supervision (arm B, primary implementation B-paper; B-code reported as a secondary implementation) and autoencoder reconstruction (arm C). Costs are compared at equal FLOPs, not equal cost per step.
- **Metrics:** same as H1, arms A to D.
- **Pre-registered threshold:** arm D beats both arm B and arm C under the same three-way rule as H1 (D relative to B and to C, 97.5% intervals, both primary targets). **Gated on H1:** tested only if H1 is supported (D-019); otherwise B and C are reported descriptively. Arm B is run on seeds 0 to 19, arm C on seeds 0 to 9 (so D versus C is paired on seeds 0 to 9). B's own comparison with A is H6.
- **Status:** open (gated on H1)
- **Linked findings:** none yet
- **Paper sections:** 5.2

## H3 (mechanism): the meeting gradient points the right way

- **Statement:** The gradient of the meeting loss on the lower layers is positively aligned with the true task gradient, and larger alignment goes with larger speedup.
- **Metrics:** cosine similarity between `g_task` and `g_meet` (see `docs/METHOD.md`); correlation of per-seed alignment with per-seed speedup.
- **Proposed success threshold:** median alignment positive at at least 90% of evaluation points after warm-up. Expectation from the literature: alignment is likely positive but well below 1 (Lee et al. 2015, Theorem 1 gives only a lower bound under perfect inverses; Meulemans et al. 2020 argue DTP follows Gauss-Newton-like targets).
- **Status:** open
- **Linked findings:** none yet
- **Paper sections:** 6

## H4 (generality): the effect is not specific to one setting

- **Statement:** The effect holds on CIFAR-10, on a modular-arithmetic task, and with more than one optimizer.
- **Metrics:** same as H1 per setting.
- **Proposed success threshold:** supported on at least one setting beyond MNIST, and with both AdamW and SGD with momentum on at least one setting.
- **Status:** open
- **Linked findings:** none yet
- **Paper sections:** 5.3, 5.4

## H5 (confound control): the gain is not just a lower-layer learning-rate effect

- **Statement:** Arm D's gain over arm A is not reproduced by control E (arm A with a tuned lower-layer learning-rate multiplier `low_lr_mult`, tuned with the same trial budget). Control F (exact-gradient target) is reported as a reference.
- **Metrics:** same as H1.
- **Pre-registered threshold:** arm D beats arm E under the H1 three-way rule (97.5% intervals, both primary targets). **Gated on H1** (D-019). If H1 is supported, the E refinement runs first: `low_lr_mult` in {0.5, 1, 2, 4, 8} swept on arm A's tuned config (`t1_base_two.yaml`) with fresh selection seeds (at least 100), the D-017 score and the D-018 rule, and the winner is run on the same seeds as D. Arm E as tuned in the t1 sweep (seeds 0 to 9) is reported either way.
- **Status:** open (gated on H1)
- **Linked findings:** none yet
- **Paper sections:** 5.2, 6
- **Note on control F (corrected 2026-10-22, D-018):** control F is nearly inert as implemented (F-005): its gradient on the middle state is about 0.0016 times `lam_meet` times the task gradient, so it is pre-registered as a **negative control** (arm A plus 5.8% FLOPs per step). No inference is drawn from F under AdamW. Its rescaling is decided at Stage 5, before the SGD experiments (R-19).

## H6 (generated from F-007): deep supervision is more efficient than the tuned baseline

- **Origin:** *generated from an exploratory finding* (F-007: at the selection level arm B-paper scores 191.2 GFLOPs to 0.975 against 232.6 for both tunings of arm A, 18% lower; its runner-up is 6% lower). It is added to the pre-registration on 2026-10-23, before any final seed, so the test is confirmatory; the origin is stated wherever the result is reported.
- **Statement:** Deep supervision on the middle state (arm B, B-paper implementation, tuned config `t1_deepsup_two`) reduces the FLOPs needed to reach a target validation accuracy compared with the tuned baseline at equal compute.
- **Metrics and rule:** exactly as H1 (targets 0.97 and 0.975, 97.5% paired-bootstrap interval, margin 15%, both targets must agree, gain must exceed the A re-tuning shift), arm B against arm A, seeds 0 to 19.
- **Status:** open (locked, not yet run)
- **Linked findings:** F-006, F-007 (exploratory context only)
- **Paper sections:** 5.2

## Roles of the other groups (D-019)

Reported with means, intervals and paired differences, never used to decide a claim: A2 (second tuning of arm A; supplies the re-tuning shift), B-code (`deepsup_code`), C (when H2 is gated in), E, F (negative control), and the variants V1 (`meet_noise`), V2 (`meet_ao`), V3 (`meet_frozen`). H3 (alignment) is a mechanism check using the logged `align` of the arm D, V1, V2 and F runs. H4 is outside Stage 3.

---

## Hypothesis template (for new entries)

```
## H<N>: <statement>
- Metric(s):
- Success threshold: <value>   (pre-registered on YYYY-MM-DD, git tag prereg-v1; post-hoc changes marked)
- Status: open | supported | refuted | inconclusive
- Linked findings: F-...
- Paper sections: ...
```
