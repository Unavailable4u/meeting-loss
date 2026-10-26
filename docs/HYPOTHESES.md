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
- **Status:** **refuted** (confirmatory, 2026-10-23: D needs +36.9% and +35.5% FLOPs at 0.97 and 0.975; status line edited after the lock, see DR-006)
- **Linked findings:** F-009 (result); F-006, F-008 (exploratory context)
- **Paper sections:** Introduction, 5.1

## H2 (specificity): the backward-derived target is what matters

- **Statement:** The gain is not matched by other auxiliary losses at the same layer: DSN-style deep supervision (arm B, primary implementation B-paper; B-code reported as a secondary implementation) and autoencoder reconstruction (arm C). Costs are compared at equal FLOPs, not equal cost per step.
- **Metrics:** same as H1, arms A to D.
- **Pre-registered threshold:** arm D beats both arm B and arm C under the same three-way rule as H1 (D relative to B and to C, 97.5% intervals, both primary targets). **Gated on H1:** tested only if H1 is supported (D-019); otherwise B and C are reported descriptively. Arm B is run on seeds 0 to 19, arm C on seeds 0 to 9 (so D versus C is paired on seeds 0 to 9). B's own comparison with A is H6.
- **Status:** not tested (H1 was not supported, so the gate of D-019 stays closed; descriptive results for B and C in E-008; status line edited after the lock, DR-006)
- **Linked findings:** none
- **Paper sections:** 5.2

## H3 (mechanism): the meeting gradient points the right way

- **Statement:** The gradient of the meeting loss on the lower layers is positively aligned with the true task gradient, and larger alignment goes with larger speedup.
- **Metrics:** cosine similarity between `g_task` and `g_meet` (see `docs/METHOD.md`); correlation of per-seed alignment with per-seed speedup.
- **Proposed success threshold:** median alignment positive at at least 90% of evaluation points after warm-up. Expectation from the literature: alignment is likely positive but well below 1 (Lee et al. 2015, Theorem 1 gives only a lower bound under perfect inverses; Meulemans et al. 2020 argue DTP follows Gauss-Newton-like targets).
- **Status:** supported on the pre-registered alignment threshold (2026-10-23: median alignment of arm D positive at 100% of evaluations after warm-up in 20 of 20 runs, about +0.75); the second half of the statement (larger alignment goes with larger speedup) is not supported: there is no speedup, and across 20 seeds the correlation is 0.15 (exploratory, p = 0.54). "After warm-up" was not defined numerically in advance; steps after 200 and after 1000 give the same result. Status line edited after the lock, DR-006
- **Linked findings:** F-009
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
- **Status:** not tested (H1 was not supported; the E refinement was not run; arm E as tuned is reported descriptively in E-008; status line edited after the lock, DR-006)
- **Linked findings:** none
- **Paper sections:** 5.2, 6
- **Note on control F (corrected 2026-10-22, D-018):** control F is nearly inert as implemented (F-005): its gradient on the middle state is about 0.0016 times `lam_meet` times the task gradient, so it is pre-registered as a **negative control** (arm A plus 5.8% FLOPs per step). No inference is drawn from F under AdamW. Its rescaling is decided at Stage 5, before the SGD experiments (R-19).

## H6 (generated from F-007): deep supervision is more efficient than the tuned baseline

- **Origin:** *generated from an exploratory finding* (F-007: at the selection level arm B-paper scores 191.2 GFLOPs to 0.975 against 232.6 for both tunings of arm A, 18% lower; its runner-up is 6% lower). It is added to the pre-registration on 2026-10-23, before any final seed, so the test is confirmatory; the origin is stated wherever the result is reported.
- **Statement:** Deep supervision on the middle state (arm B, B-paper implementation, tuned config `t1_deepsup_two`) reduces the FLOPs needed to reach a target validation accuracy compared with the tuned baseline at equal compute.
- **Metrics and rule:** exactly as H1 (targets 0.97 and 0.975, 97.5% paired-bootstrap interval, margin 15%, both targets must agree, gain must exceed the A re-tuning shift), arm B against arm A, seeds 0 to 19.
- **Status:** **refuted** (confirmatory, 2026-10-23: B is +2.3% and +1.7% against A at 0.97 and 0.975; status line edited after the lock, see DR-006; against the second tuning of A the sign differs, F-011)
- **Linked findings:** F-010 (result), F-011; F-006, F-007 (exploratory context)
- **Paper sections:** 5.2

## H7a (generated from F-012, F-016 to F-018): the selective backward pass, bolted onto the tuned baseline, reduces the FLOPs needed to reach a target validation accuracy

- **Origin:** *generated from exploratory findings* (F-012: arm S needed 0.61 to 0.63 of A's FLOPs to 0.97 and 0.975 in the pilot, E-010; F-017 and F-018 add two further seed groups, E-017 and E-018, at 22% to 38% fewer FLOPs than A1). It is added to the pre-registration on 2026-10-27, before any seed of the `prereg-v2` plan (1300 to 1319) has been run, so the test is confirmatory; the origin is stated wherever the result is reported. The margin of 15% is the one fixed in D-019 and D-023 and was not changed after any look at S (D-023).
- **Statement:** In the MNIST MLP setting of D-008, arm S (selective backward pass: full-batch forward, backward only through the 30% of rows with the highest loss, `sel_frac` 0.3, `sel_mode` top, `sel_warmup` 100, `configs/prereg_v2/S30.yaml`) with the hyperparameters of the tuned baseline A1 (`configs/tuned/t1_base_two.yaml`) needs fewer FLOPs than A1 to reach a target validation accuracy at equal total compute.
- **Metrics and rule:** FLOPs-to-target (decisive) and steps-to-target (reported) at targets 0.97 and 0.975; a target is reached at the first of two consecutive evaluations at or above it (D-018); FLOPs counted by `src/flops.py` (full-batch forward plus the backward pass of the rows kept; the selection itself and the optimizer are not counted); S30 against A1 on 20 paired seeds (1300 to 1319), 3,000 steps. Outcome as H1 (D-019): *supported* if, at both targets, the paired change in mean FLOPs-to-target is at most -15% and its 97.5% paired-bootstrap interval (alpha 0.025, 10,000 resamples, seed 0, `scripts/analyze_prereg2.py`) excludes zero, **and** the gain exceeds the re-tuning shift of A; *refuted* if, at both targets, the interval excludes a gain of 15% or more (lower bound above -15%); *inconclusive* otherwise, including when the two targets disagree.
- **Re-tuning shift (rule `max_gain`, fixed in `configs/prereg_v2_plan.yaml` and decided in `docs/notes/prereg-v2-plan.md` on 2026-10-27):** at each target the shift is the largest reduction in mean FLOPs-to-target, against A1 on the same 20 seeds, achieved by any of A2 (`t2`), A5, A6, A7 and A8 (tunings `t5` to `t8`), taken from their paired percentage changes (0 if none is below A1). The gain of S (the size of its paired change against A1) must be larger than the shift for a *supported* verdict, otherwise the outcome is inconclusive. The shift is a maximum over five noisy point estimates, so it is biased upward, which makes *supported* harder, not easier. The rules `max_abs` and `median_abs` are printed as descriptive. The rule was chosen by the assistant at the researcher's delegation, knowing that A2 alone changed A1's FLOPs by +34.0% (0.97) and +17.2% (0.975) on the Stage 3 seeds (F-011).
- **Expectation recorded before the runs:** a supported verdict is plausible because S's saving against A1 was 22% to 38% on three seed groups; an inconclusive verdict through the shift is a real possibility, because the four further tunings of A scored 8% to 15% below A1 in-sample (E-019) and selection scores did not predict final-seed rankings before (F-011, R-21). No stronger prediction is made.
- **Status:** **open** (locked 2026-10-27, tag `prereg-v2`)
- **Linked findings:** F-012, F-016, F-017, F-018 (exploratory context); E-019 to E-021 (selection runs)
- **Paper sections:** to be assigned

## H7b (generated from F-017): the saving of the selective backward pass is not only a batch-size effect

- **Origin:** *generated from an exploratory finding* (F-017: against the best small batch with a lighter tuning, S30 was 12% (0.97) and 29% (0.975) below on 3 seeds, one seed at parity). Added on 2026-10-27, before any `prereg-v2` seed, so the test is confirmatory.
- **Statement:** On the same 20 seeds, arm S as in H7a needs fewer FLOPs to reach the targets than the fully tuned small-batch arm B48: arm A at batch size 48, tuned with the full 25-trial search of A (tag `t9`, `configs/tuned/t9_base_b48_two.yaml`), run for 8,000 steps (the same number of samples as 3,000 steps at batch 128) with evaluations every 125 steps.
- **Metrics and rule:** as H7a (targets 0.97 and 0.975, two-consecutive crossing, paired 97.5% interval) but with a margin of **0%** and no re-tuning shift: *supported* if, at both targets, the paired change is at most 0 and the interval's upper end is below 0; *refuted* if, at both targets, the interval's lower end is above 0; *inconclusive* otherwise. H7b rules out a batch-size explanation; it is not a second 15% target.
- **Limits stated in advance:** one batch size (48, the best in E-017) is tuned and none other is searched; B48 was tuned in its own tag with two selection seeds while S inherits A1's hyperparameters (conservative for S); the evaluation spacing of B48 (125 steps = 6,000 samples) is finer than A's (6,400 samples), which can move its crossing by at most one interval in B48's favour.
- **Status:** **open** (locked 2026-10-27, tag `prereg-v2`). If H7a is supported and H7b is not, the paper says so: a bolt-on gain that a batch-size change also obtains in part.
- **Linked findings:** F-017 (exploratory context); E-020 (selection runs)
- **Paper sections:** to be assigned

## Roles of the groups of `prereg-v2` (D-023, `configs/prereg_v2_plan.yaml`)

Seeds 1300 to 1319 in every group (paired), 3,000 steps except B48 (8,000). A1 (`t1_base_two`): reference of H7a. S30: decides H7a and H7b. B48: reference of H7b. A2, A5 to A8 (tunings `t2`, `t5` to `t8` of arm A): supply the re-tuning shift and the noise-floor outcome (LARGE, SMALL or MIXED, as in D-022); reported, they decide nothing else. S20 (`sel_frac` 0.2), S30T (S tuned once, tag `t10`, E-021) and the random control R30 are descriptive and never decide a claim. Wall-clock time per step and a derived time-to-target at one and eight threads are a reported secondary (E-016); FLOPs stay primary and the paper says that time saves less. A second setting (Fashion-MNIST MLP) is outside this lock.

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
